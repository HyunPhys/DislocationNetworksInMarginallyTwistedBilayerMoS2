#!/usr/bin/env python3
"""
Dark-field TEM-like simulation with abTEM (multislice) and an objective aperture in diffraction space.

This script:
  1) Loads an atomic structure using ASE (CIF/XYZ/POSCAR/etc.).
  2) Builds an electrostatic potential (optionally with frozen-phonons).
  3) Propagates a plane wave through the potential (multislice) to obtain an exit wave.
  4) Computes the complex diffraction pattern (fftshifted).
  5) Applies a shifted circular aperture around a chosen Bragg spot (gx, gy) in reciprocal space.
  6) Inverse-transforms to obtain a dark-field (DF) wave and its intensity.
  7) Optionally saves figures and intermediate state for reproducibility.

Units and conventions
---------------------
- Reciprocal coordinates (gx, gy) are in 1/Å and must match the diffraction-pattern axes.
- The diffraction pattern is computed with fftshift=True, so the direct beam is at the center.
- The aperture radius is provided directly in 1/Å (k_radius_1_per_A). 

Dependencies
------------
pip install abtem ase matplotlib numpy

"""
from __future__ import annotations

import argparse
import json
import os, sys
from dataclasses import asdict, dataclass
import time

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import numpy as np
import ase.io
import abtem
from abtem.transfer import energy2wavelength
from abtem.core import backend


import cupy as cp
import dask


# ----------------------------
# Setting device (CPU or GPU)
# ----------------------------

abtem.config.set({"device": "gpu"})
print("backend.cp:", backend.cp, flush=True)
dask.config.set({"num_workers": 2}) 
abtem.config.set({"dask.chunk-size-gpu": "8192 MB"})   # 8 GB
abtem.config.set({"cupy.fft-cache-size": "1024 MB"})   # 1 GB


# Ensure conda env DLLs are found first (Windows)
dll_dir = os.path.join(sys.prefix, "Library", "bin")
os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")

# Optional: make it visible for debugging
os.environ["CONDA_PREFIX"] = os.environ.get("CONDA_PREFIX") or sys.prefix
print("PYTHON:", sys.executable)
print("CONDA_PREFIX:", os.environ.get("CONDA_PREFIX"))
print("DLL_DIR:", dll_dir)

x = cp.ones((4096, 4096), dtype=cp.float32)
cp.cuda.Device().synchronize()
print("cupy device count:", cp.cuda.runtime.getDeviceCount(), flush=True)
print("CuPy alloc OK, MB =", x.nbytes/1024/1024, flush=True)


t0 = time.time()

def mark(msg):
    """Print a simple timestamped message (useful for long multislice runs)."""
    print(f"[{time.time()-t0:8.2f}s][pid {os.getpid()}] {msg}", flush=True)


# ----------------------------
# Reproducibility helpers
# ----------------------------


@dataclass
class DFStateMeta:
    """Minimal metadata for re-plotting or re-applying an aperture to a saved DP."""

    # Core simulation params
    energy_eV: float
    realspace_sampling_A: float
    slice_thickness_A: float

    # Reciprocal space info
    wavelength_A: float
    dk_1_per_A: object          # could be float or (dky, dkx)
    gx: float
    gy: float
    k_radius_1_per_A: float

    # Notes / provenance
    structure_path: str = ""
    frozen_phonons: bool = False
    nphon: int = 0
    sigma: float = 0.0


def to_numpy(x):
    """Convert CuPy arrays (if present) to NumPy; pass NumPy arrays through."""
    return x.get() if hasattr(x, "get") else np.asarray(x)


def save_state(prefix: str, out: dict, meta: DFStateMeta, *, save_figures: bool = True, dp_overlay_fig=None, mask_fig=None, df_fig=None):
    """
    Save intermediate results for transparency and reproducibility.

    Files written:
      - <prefix>_state.npz  : dp_complex, KX, KY, mask, df_intensity
      - <prefix>_meta.json  : metadata (simulation settings)
      - <prefix>_dp_overlay.png, <prefix>_df.png (optional)
    """
    os.makedirs(os.path.dirname(prefix), exist_ok=True) if os.path.dirname(prefix) else None

    # --- Save arrays to NPZ ---
    dp_complex_arr = out["dp_complex"].array  # complex
    np.savez_compressed(
        f"{prefix}_state.npz",
        dp_complex=dp_complex_arr,                # complex128 typically
        KX=out["KX"].astype(np.float32),
        KY=out["KY"].astype(np.float32),
        mask=out["mask"].astype(np.uint8),
        df_intensity=out["df_intensity"].astype(np.float32) if "df_intensity" in out else None,
    )

    # --- Save metadata to JSON ---
    with open(f"{prefix}_meta.json", "w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, indent=2)

    # --- Save figures (PNG) ---
    if save_figures:
        if dp_overlay_fig is not None:
            dp_overlay_fig.savefig(f"{prefix}_dp_overlay.png", dpi=300)
        if mask_fig is not None:
            mask_fig.savefig(f"{prefix}_mask.png", dpi=300)
        if df_fig is not None:
            df_fig.savefig(f"{prefix}_df.png", dpi=300)

    print(f"[Saved] {prefix}_state.npz, {prefix}_meta.json" + (", figures PNG" if save_figures else ""))


def load_state(prefix: str):
    """
    Load arrays + metadata.
    Returns: (arrays_dict, meta_dict)
    """
    data = np.load(f"{prefix}_state.npz", allow_pickle=True)
    with open(f"{prefix}_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    arrays = {
        "dp_complex": data["dp_complex"],
        "KX": data["KX"],
        "KY": data["KY"],
        "mask": data["mask"].astype(bool),
        "df_intensity": None if data["df_intensity"] is None else data["df_intensity"],
    }
    return arrays, meta


# ----------------------------
# Plotting / coordinate utilities
# ----------------------------


def k_grid_from_sampling(shape_yx, dk):
    """
    Build fftshifted reciprocal-space coordinate grids (KX, KY) in 1/Å.

    Parameters
    ----------
    shape_yx : (ny, nx)
    dk : float or (dky, dkx)
        Reciprocal-space sampling [1/Å] per pixel (from abTEM DiffractionPatterns.sampling).
    """
    ny, nx = shape_yx
    if np.isscalar(dk):
        dky, dkx = float(dk), float(dk)
    else:
        dky, dkx = float(dk[0]), float(dk[1])

    ky = (np.arange(ny) - ny // 2) * dky
    kx = (np.arange(nx) - nx // 2) * dkx
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    return KX, KY



def extent_from_kgrid(KX, KY):
    """Convenience function to build imshow extent = [xmin, xmax, ymin, ymax]."""
    xmin, xmax = KX[0, 0], KX[0, -1]
    ymin, ymax = KY[0, 0], KY[-1, 0]
    left, right = (xmin, xmax) if xmin < xmax else (xmax, xmin)
    bottom, top = (ymin, ymax) if ymin < ymax else (ymax, ymin)
    return [left, right, bottom, top]


def ensure_2d_intensity(dp_complex_array):
    """
    Convert a complex DP array into a 2D intensity image for display.

    If the array has leading ensemble axes, they are averaged.
    """
    I = np.abs(dp_complex_array) ** 2
    if I.ndim == 2:
        return I
    # Flatten all leading axes into one and average
    lead = int(np.prod(I.shape[:-2]))
    I2 = I.reshape((lead,) + I.shape[-2:])
    return I2.mean(axis=0)


def pick_g_from_dp_plot(I_log, extent, k_radius, title="Click to pick Bragg spot (gx, gy)"):
    """
    Show a DP plot and let the user click once to pick (gx, gy) in 1/Å.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 6), constrained_layout=True)

    im = ax.imshow(to_numpy(I_log), extent=extent, origin="lower", aspect="equal")  

    ax.set_title(title)
    ax.set_xlabel(r"$k_x$ (1/$\AA$)")
    ax.set_ylabel(r"$k_y$ (1/$\AA$)")
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("log10(I)")

    plt.draw()
    pts = plt.ginput(1, timeout=-1)  # wait until user clicks once
    if not pts:
        raise RuntimeError("No point was clicked. Aborting.")
    gx, gy = pts[0]

    # Visual feedback: draw marker + circle
    ax.plot(gx, gy, marker="x", markersize=10)
    ax.add_patch(Circle((gx, gy), radius=k_radius, fill=False, linewidth=2))
    ax.text(gx, gy, "  g", va="center", ha="left")
    print([gx,gy])
    plt.show()

    return float(gx), float(gy)


def plot_dp_and_mask(dp_complex, KX, KY, mask, gx, gy, k_radius):
    """
    Plot DP (log intensity) with g/aperture overlay + mask plot.
    """
    arr_k = dp_complex.array
    I2 = ensure_2d_intensity(arr_k)
    I_log = np.log10(I2 + 1e-12)

    extent = extent_from_kgrid(KX, KY)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    # DP plot
    ax = axes[0]
    im = ax.imshow(to_numpy(I_log), extent=extent, origin="lower", aspect="equal")

    ax.set_title("DP (log10 intensity) + chosen spot/aperture")
    ax.set_xlabel(r"$k_x$ (1/$\AA$)")
    ax.set_ylabel(r"$k_y$ (1/$\AA$)")
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("log10(I)")

    ax.plot(gx, gy, marker="x", markersize=10)
    ax.text(gx, gy, "  g", va="center", ha="left")
    ax.add_patch(Circle((gx, gy), radius=k_radius, fill=False, linewidth=2))

    label_lines = [f"k-radius: {k_radius:.4g} 1/Å"]
    ax.text(
        0.02, 0.98,
        "\n".join(label_lines),
        transform=ax.transAxes, va="top", ha="left",
        bbox=dict(boxstyle="round", alpha=0.8)
    )

    # Mask plot
    ax2 = axes[1]
    ax2.imshow(mask.astype(float), extent=extent, origin="lower", aspect="equal")
    ax2.set_title("Objective aperture mask (passed=1)")
    ax2.set_xlabel(r"$k_x$ (1/$\AA$)")
    ax2.set_ylabel(r"$k_y$ (1/$\AA$)")
    ax2.set_xlim([-1,1])
    ax2.set_ylim([-1,1])
    ax2.plot(gx, gy, marker="x", markersize=10)
    ax2.add_patch(Circle((gx, gy), radius=k_radius, fill=False, linewidth=2))

    # if show:
    #     plt.show()

    return fig, axes

def plot_df_intensity(df_intensity, title="DF intensity (|psi|^2)", savepath=None, vmin=None, vmax=None):
    """
    Plot DF real-space intensity. If df_intensity has leading axes, average them for display.
    """
    if df_intensity.ndim == 2:
        I = df_intensity
    else:
        lead = int(np.prod(df_intensity.shape[:-2]))
        I = df_intensity.reshape((lead,) + df_intensity.shape[-2:]).mean(axis=0)

    h, w = I.shape  # y, x
    fig, ax = plt.subplots(1, 1, figsize=(4, 4 * h / w*0.7), constrained_layout=True)

    if vmin is not None :
        im = ax.imshow(to_numpy(I), origin="lower", aspect="equal",cmap='gray', vmin=vmin, vmax=vmax)
    else: 
        im = ax.imshow(to_numpy(I), origin="lower", aspect="equal",cmap='gray')

    ax.set_title(title)
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("I (a.u.)")

    if savepath:
        fig.savefig(savepath, dpi=300)

    return fig, ax

def save_exit_wave_intensity(exit_wave, fname="exit_wave_intensity.png"):
    # abTEM Measurement → numpy array
    I = exit_wave.intensity().compute().array

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(to_numpy(I), origin="lower", cmap="gray")
    ax.set_title("Exit wave intensity")
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    fig.colorbar(im, ax=ax, label="Intensity")

    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_dp_blocked(dp_complex, fname="dp_blocked.png"):
    dp = dp_complex.compute() if hasattr(dp_complex, "compute") else dp_complex
    arr = dp.array
    dk = dp.sampling  # can be float OR (dky, dkx)

    # ---- handle dk being scalar or (dky, dkx) ----
    if np.isscalar(dk):
        dky, dkx = float(dk), float(dk)
    else:
        dky, dkx = float(dk[0]), float(dk[1])

    # ---- make 2D display intensity ----
    # If complex → intensity; if already real intensity, keep.
    if np.iscomplexobj(arr):
        I = np.abs(arr) ** 2
    else:
        I = arr

    # If there are leading axes (e.g., ensembles), average them for display
    if I.ndim > 2:
        lead = int(np.prod(I.shape[:-2]))
        I2 = I.reshape((lead,) + I.shape[-2:]).mean(axis=0)
    else:
        I2 = I

    ny, nx = I2.shape[-2], I2.shape[-1]

    ky = (np.arange(ny) - ny // 2) * dky
    kx = (np.arange(nx) - nx // 2) * dkx
    extent = [kx[0], kx[-1], ky[0], ky[-1]]

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    im = ax.imshow(
        to_numpy(I2), 
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap="gray", 
    )

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel(r"$k_x$ (1/$\AA$)")
    ax.set_ylabel(r"$k_y$ (1/$\AA$)")
    ax.set_title("Diffraction pattern (direct beam blocked)")
    fig.colorbar(im, ax=ax)

    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)



# ----------------------------
# Core simulation
# ----------------------------

def df_tem_abtem(
    atoms,
    *,
    energy_eV=80e3,
    realspace_sampling_A=0.05,
    slice_thickness_A=1.0,
    g_center_1_per_A=(0.0, 0.0),
    frozen_phonons=False,
    nphon=8,
    sigma=0.1,
    show_dp=True,
    interactive_pick_g=False,
    save_prefix = None,
    k_radius_ang = None
):
    """
    Run multislice + DP (complex) + shifted circular objective aperture + DF intensity.
    Returns dict with outputs and picked/used (gx, gy).
    """
    # 1) Potential

    mark("build potential start")


    if frozen_phonons:
        frozen = abtem.FrozenPhonons(atoms, nphon, sigmas=sigma)
        potential = abtem.Potential(
            frozen,
            sampling=realspace_sampling_A,
            projection="infinite",
            slice_thickness=slice_thickness_A,
        )
    else:
        potential = abtem.Potential(
            atoms,
            sampling=realspace_sampling_A,
            projection="infinite",
            slice_thickness=slice_thickness_A,
        )

    mark("build potential done")
    mark("build probe start")

    # 2) Plane wave + multislice exit wave
    plane_wave = abtem.PlaneWave(energy=energy_eV, sampling=realspace_sampling_A)
    exit_wave = plane_wave.multislice(potential)
    exit_wave_small = exit_wave.downsample()

    save_exit_wave_intensity(exit_wave, f"{save_prefix}_exit_intensity.png")

    mark("build probe done")


    # 3) Complex diffraction pattern (fftshifted)
    dp_complex = exit_wave_small.diffraction_patterns(
        max_angle=80,
        fftshift=True,
        return_complex=True,
    ).compute()

    save_dp_blocked(dp_complex,f"{save_prefix}_dp_blocked_kpm1.png")

    arr_k = dp_complex.array
    ny, nx = arr_k.shape[-2], arr_k.shape[-1]
    dk = dp_complex.sampling  # [1/Å] per pixel
    KX, KY = k_grid_from_sampling((ny, nx), dk)

    # 4) Objective aperture radius in k-space
    wavelength_A = energy2wavelength(energy_eV)
    k_radius = k_radius_ang

    # 5) (Optional) interactive pick of g on the DP plot
    gx, gy = map(float, g_center_1_per_A)

    if interactive_pick_g:
        I2 = ensure_2d_intensity(arr_k)
        I_log = np.log10(I2 + 1e-12)
        extent = extent_from_kgrid(KX, KY)
        gx, gy = pick_g_from_dp_plot(I_log, extent, k_radius)


    # 6) Build shifted circular mask around (gx, gy)
    # mask = ((KX - gx) ** 2 + (KY - gy) ** 2) <= (k_radius ** 2)
    sigma = 0.3 * k_radius  # edge softness (tune 0.05~0.3)
    R2 = (KX - gx)**2 + (KY - gy)**2
    mask = np.exp(-0.5 * R2 / (sigma**2))

    # 7) Show DP + overlay + mask if requested
    dp_fig = None
    if show_dp:
        dp_fig, _ = plot_dp_and_mask(dp_complex, KX, KY, mask, gx, gy, k_radius, show=True)

    # 8) Apply mask (broadcast across leading axes)

    xp = cp if (cp is not None and hasattr(arr_k, "__cuda_array_interface__")) else np
    mask = xp.asarray(mask)
    masked_k = arr_k * mask

    # 9) Back to real space (DF wave) and intensity
    df_waves_k = abtem.Waves(
        masked_k,
        energy=energy_eV,
        sampling=dk,
        reciprocal_space=True,
    ) # Result: "Complex" wave function
    df_waves_r = df_waves_k.ensure_real_space() # Inverse Fourier transform of "complex" wavefunction (not real one)

    df_intensity = np.abs(df_waves_r.array) ** 2 # Now, this intensity is real quantity.


    return {
        "exit_wave": exit_wave,
        "dp_complex": dp_complex,
        "KX": KX,
        "KY": KY,
        "mask": mask,
        "df_waves_r": df_waves_r,
        "df_intensity": df_intensity,
        "gx": gx,
        "gy": gy,
        "k_radius_1_per_A": float(k_radius),
        "wavelength_A": float(wavelength_A),
        "dk_1_per_A": dk,
        "dp_fig": dp_fig,
    }


def recompute_df_from_saved(prefix: str, *, new_g=None, new_k_radius=None, show_dp=True, show_df=None, idx):
    arrays, meta = load_state(prefix)

    dp_complex_arr = arrays["dp_complex"]
    KX, KY = arrays["KX"], arrays["KY"]
    energy_eV = float(meta["energy_eV"])
    dk = meta["dk_1_per_A"]

    gx, gy = float(meta["gx"]), float(meta["gy"])

    k_radius = new_k_radius

    # Optionally override g or semiangle and rebuild mask
    if new_g is not None:
        gx, gy = map(float, new_g)
    else:
        I2 = ensure_2d_intensity(dp_complex_arr)
        I_log = np.log10(I2 + 1e-12)
        extent = extent_from_kgrid(KX, KY)
        gx, gy = pick_g_from_dp_plot(I_log, extent, k_radius)




    # mask = ((KX - gx) ** 2 + (KY - gy) ** 2) <= (k_radius ** 2)

    sigma = 0.3 * k_radius  # edge softness (tune 0.05~0.3)
    R2 = (KX - gx)**2 + (KY - gy)**2
    mask = np.exp(-0.5 * R2 / (sigma**2))

    # Create a minimal dp_complex-like container for plotting
    class _DP:
        def __init__(self, array, sampling):
            self.array = array
            self.sampling = sampling
    dp_obj = _DP(dp_complex_arr, dk)

    dp_fig = None
    if show_dp:
        dp_save = f'dp_new{idx}.png'
        dp_fig, _ = plot_dp_and_mask(dp_obj, KX, KY, mask, gx, gy, k_radius, show=True)
        dp_fig.savefig(dp_save, dpi=300)


    masked_k = dp_complex_arr * mask
    df_waves_k = abtem.Waves(masked_k, energy=energy_eV, sampling=dk, reciprocal_space=True)
    df_waves_r = df_waves_k.ensure_real_space()
    df_intensity = np.abs(df_waves_r.array) ** 2

    df_fig = None
    if show_df:
        df_fig, _ = plot_df_intensity(df_intensity, show=True)

    return {
        "dp_complex": dp_complex_arr,
        "KX": KX, "KY": KY,
        "mask": mask,
        "df_intensity": df_intensity,
        "gx": gx, "gy": gy,
        "k_radius_1_per_A": float(k_radius),
        "dp_fig": dp_fig,
        "df_fig": df_fig,
    }




# ----------------------------
# CLI / Main
# ----------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="abTEM multislice DF-TEM-like simulation + DP plot + aperture overlay + structure loader"
    )
    p.add_argument("structure", type=str, help="Structure file path (any ASE-readable format)")
    p.add_argument("--supercell_a", type=float, help="(In case of .xyz format) lattice constant a of supercell")
    p.add_argument("--supercell_b", type=float, help="(In case of .xyz format) lattice constant b of supercell")
    p.add_argument("--supercell_c", type=float, help="(In case of .xyz format) lattice constant c of supercell")
    p.add_argument("--energy_eV", type=float, default=80e3, help="Electron energy in eV (default: 80e3)")
    p.add_argument("--realspace_sampling_A", type=float, default=0.05, help="Real-space sampling in Å (default: 0.05)")
    p.add_argument("--slice_thickness_A", type=float, default=1.0, help="Slice thickness in Å (default: 1.0)")
    p.add_argument("--gx", type=float, default=None, help="Chosen Bragg spot center gx in 1/Å")
    p.add_argument("--gy", type=float, default=None, help="Chosen Bragg spot center gy in 1/Å")
    p.add_argument("--pick_g", action="store_true", help="Pick (gx,gy) interactively by clicking on the DP")
    p.add_argument("--no_show_dp", action="store_true", help="Do not show DP/mask plots")
    p.add_argument("--show_df", action="store_true", help="Show DF intensity plot at the end")
    p.add_argument("--k_radius_ang", type=float, default=0.034, help="Objective aperture radius in angstrom")


    # Frozen phonons options
    p.add_argument("--frozen_phonons", action="store_true", help="Enable frozen phonons averaging")
    p.add_argument("--nphon", type=int, default=8, help="Number of frozen phonon configs (default: 8)")
    p.add_argument("--sigma", type=float, default=0.1, help="Frozen phonon displacement sigma (Å-ish)")

    # Saving
    p.add_argument("--save_prefix", type=str, default=None, help="If set, save outputs as <prefix>_*.npy / png")
    p.add_argument("--save_state_prefix", type=str, default=None,help="If set, save full intermediate state so you can resume later")

    import sys
    print("sys.argv =", sys.argv)
    print("CWD =", os.getcwd()) 

    return p.parse_args()


def main():

    args = parse_args()

    if not os.path.exists(args.structure):
        raise FileNotFoundError(f"Structure file not found: {args.structure}")

    # Load structure via ASE

    mark("read structure start")
    atoms = ase.io.read(args.structure)
    if args.supercell_a is not None :
        cell = np.array([
            [args.supercell_a, 0.0, 0.0],
            [0.0, args.supercell_b, 0.0],
            [0.0, 0.0, args.supercell_c],
        ])
        atoms.set_cell(cell)
        atoms.set_pbc([True, True, False])
    mark("read structure done")


    # Determine g center mode
    interactive_pick_g = bool(args.pick_g)
    if not interactive_pick_g:
        if args.gx is None or args.gy is None:
            raise ValueError("Provide --gx and --gy (in 1/Å) OR use --pick_g to select interactively.")
        g_center = (args.gx, args.gy)
    else:
        g_center = (0.0, 0.0)  # placeholder; will be overwritten by click

    out = df_tem_abtem(
        atoms,
        energy_eV=args.energy_eV,
        realspace_sampling_A=args.realspace_sampling_A,
        slice_thickness_A=args.slice_thickness_A,
        g_center_1_per_A=g_center,
        frozen_phonons=args.frozen_phonons,
        nphon=args.nphon,
        sigma=args.sigma,
        show_dp=(not args.no_show_dp),
        interactive_pick_g=interactive_pick_g,
        save_prefix = args.save_prefix,
        k_radius_ang= args.k_radius_ang
    )

    print("\n--- Selected / used parameters ---")
    print(f"Energy (eV):               {args.energy_eV}")
    print(f"Wavelength (Å):            {out['wavelength_A']:.6g}")
    print(f"DP sampling dk (1/Å/pix):   {out['dk_1_per_A']}")
    print(f"Selected g center (1/Å):    gx={out['gx']:.6g}, gy={out['gy']:.6g}")
    print(f"Aperture k-radius (1/Å):    {out['k_radius_1_per_A']:.6g}")

    # # Show DF plot
    # if args.show_df:
    #     savepath = None
    #     if args.save_prefix:
    #         savepath = f"{args.save_prefix}_df.png"
    #     plot_df_intensity(out["df_intensity"], savepath=savepath)

    # ---- NEW: DF fig handle for saving ----
    df_fig = None
    if args.show_df:
        # plot_df_intensity가 (fig, ax) 반환하도록 수정되어 있어야 함
        df_fig, _ = plot_df_intensity(out["df_intensity"], show=True)


    # Save outputs
    if args.save_prefix:
        np.save(f"{args.save_prefix}_df_intensity.npy", out["df_intensity"])
        np.save(f"{args.save_prefix}_mask.npy", out["mask"].astype(np.uint8))
        # For DP, save intensity (averaged display) and log-intensity too
        dpI = ensure_2d_intensity(out["dp_complex"].array)
        np.save(f"{args.save_prefix}_dp_intensity.npy", dpI)
        np.save(f"{args.save_prefix}_dp_log10.npy", np.log10(dpI + 1e-12))
        print(f"\nSaved: {args.save_prefix}_df_intensity.npy, _mask.npy, _dp_intensity.npy, _dp_log10.npy")
        if args.show_df:
            print(f"Saved: {args.save_prefix}_df.png")

    # ---- NEW: full-state 저장(중간부터 재시작용) ----
    if args.save_state_prefix:
        meta = DFStateMeta(
            energy_eV=args.energy_eV,
            realspace_sampling_A=args.realspace_sampling_A,
            slice_thickness_A=args.slice_thickness_A,
            wavelength_A=out["wavelength_A"],
            dk_1_per_A=out["dk_1_per_A"],
            gx=out["gx"],
            gy=out["gy"],
            k_radius_1_per_A=out["k_radius_1_per_A"],
            structure_path=args.structure,
            frozen_phonons=bool(args.frozen_phonons),
            nphon=int(args.nphon) if args.frozen_phonons else 0,
            sigma=float(args.sigma) if args.frozen_phonons else 0.0,
        )

        # out["dp_fig"]는 df_tem_abtem 내부에서 DP를 그렸을 때 들어있음
        save_state(
            args.save_state_prefix,
            out,
            meta,
            save_figures=True,
            dp_overlay_fig=out.get("dp_fig", None),
            df_fig=df_fig,
        )
    plt.close("all")



if __name__ == "__main__":
    
    isInitial = False # If you want to re-process saved data, set this True


    if isInitial == True:

        main()

    else:

        # angle = np.deg2rad(120)
        # d_spacing = 0.6329 # 0.3654, 0.6329
        # new_g = (d_spacing*np.cos(angle), d_spacing*np.sin(angle))

        new_g = [(0.3654*np.cos(np.deg2rad(30)), 0.3654*np.sin(np.deg2rad(30))),
                (0.3654*np.cos(np.deg2rad(90)), 0.3654*np.sin(np.deg2rad(90))),
                (0.3654*np.cos(np.deg2rad(150)), 0.3654*np.sin(np.deg2rad(150))),
                (0.6329*np.cos(np.deg2rad(0)), 0.6329*np.sin(np.deg2rad(0))),
                (0.6329*np.cos(np.deg2rad(60)), 0.6329*np.sin(np.deg2rad(60))),
                (0.6329*np.cos(np.deg2rad(120)), 0.6329*np.sin(np.deg2rad(120)))]

        # 1) 먼저 전부 계산해서 DF intensity를 모으기
        dfs = []
        outs = []

        for idx in range(len(new_g)) :

            out2 = recompute_df_from_saved("run1", new_g= new_g[idx], new_k_radius=0.034,idx=idx+1) 
            outs.append(out2)

            I = out2["df_intensity"]
            I_np = to_numpy(I)  
            dfs.append(I_np)

        # # 2) 공통 vmin/vmax 계산
        # vmin = min(a.min() for a in dfs)
        # vmax = max(a.max() for a in dfs)

        all_vals = np.concatenate([a.ravel() for a in dfs])
        vmin, vmax = np.percentile(all_vals, [1, 99])

        for idx, I_np in enumerate(dfs):

            savepath = f'DF_new_{idx+1}.png'
            plot_df_intensity(I_np,savepath=savepath, vmin=vmin, vmax=vmax)

