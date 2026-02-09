"""
GPU-accelerated kinematical dark-field (DF) image simulation 

What this script does
---------------------
This is a lightweight (kinematical) proxy for DF-TEM-like imaging:

  1) Read an atomic structure (ASE-supported formats: CIF/POSCAR/XYZ/...).
  2) Build a 2D projected "scattering potential" V_proj(x,y) on a grid by depositing
     atomic weights (here: atomic number Z) and applying a Gaussian blur.
  3) Compute a centered FFT: A(k) = FFT{V_proj(x,y)}.
  4) Apply a shifted circular aperture mask M(k) centered at (gx, gy) with radius r.
  5) Inverse FFT to get a DF wave proxy and compute intensity: I_df(x,y) = |IFFT(A*M)|^2.

Notes & limitations
-------------------
- This is **not** a full multislice simulation and does not model dynamical diffraction,
  electron-optical transfer functions, thickness effects beyond simple z-windowing, etc.
- The "scattering strength" is approximated by atomic number Z. For quantitative work,
  use proper electron scattering factors.
- Reciprocal-space units are 1/Å. The reciprocal grid uses dkx = 1/fov_x and dky = 1/fov_y
  (no 2π factor). Ensure your chosen (gx, gy) matches this convention.

Dependencies
------------
pip install numpy ase matplotlib abtem cupy cupyx
"""


import numpy as np
import os, sys
from ase.io import read
from ase.build import make_supercell
from scipy.ndimage import gaussian_filter

import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter as cp_gaussian_filter

# Ensure conda env DLLs are found first (Windows)
dll_dir = os.path.join(sys.prefix, "Library", "bin")
os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")

# Optional: make it visible for debugging
os.environ["CONDA_PREFIX"] = os.environ.get("CONDA_PREFIX") or sys.prefix
print("PYTHON:", sys.executable)
print("CONDA_PREFIX:", os.environ.get("CONDA_PREFIX"))
print("DLL_DIR:", dll_dir)

# ----------------------------
# Utilities
# ----------------------------

def _ensure_outdir(save_prefix: str):
    """Create parent directory for save_prefix, if any."""

    out_dir = os.path.dirname(save_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)


def next_pow2(n: int) -> int:
    """Return the smallest power-of-two >= n (helps FFT performance)."""
    return 1 << (int(n) - 1).bit_length()

def auto_fov_and_grid_from_cell(atoms, dx_target=0.2, dy_target=None, pow2=True,min_n=256,max_n=None):
    """
    Estimate (fov_x, fov_y, nx, ny) from the structure cell.

    Assumption: atoms.cell[0] and atoms.cell[1] are roughly aligned with in-plane x and y.
    """
    if dy_target is None:
        dy_target = dx_target

    a = np.array(atoms.cell[0], float)
    b = np.array(atoms.cell[1], float)

    fov_x = np.linalg.norm(a)  # Å
    fov_y = np.linalg.norm(b)  # Å

    if fov_x <= 0 or fov_y <= 0:
        raise ValueError("Invalid cell vectors: fov_x or fov_y <= 0")

    # --- raw grid size from target resolution ---
    nx = int(np.ceil(fov_x / dx_target))
    ny = int(np.ceil(fov_y / dy_target))

    # --- enforce minimum size ---
    nx = max(nx, min_n)
    ny = max(ny, min_n)

    # --- optional power-of-two rounding ---
    if pow2:
        nx = next_pow2(nx)
        ny = next_pow2(ny)

    # --- optional upper bound ---
    if max_n is not None:
        nx = min(nx, max_n)
        ny = min(ny, max_n)

    return fov_x, fov_y, nx, ny

# ----------------------------
# Core GPU math (centered FFT conventions)
# ----------------------------


def fft2c_gpu(img_gpu):
    """Centered FFT2 on GPU (img_gpu: cp.ndarray)"""
    return cp.fft.fftshift(cp.fft.fft2(cp.fft.ifftshift(img_gpu)))

def ifft2c_gpu(kimg_gpu):
    """Centered iFFT2 on GPU (kimg_gpu: cp.ndarray)"""
    return cp.fft.fftshift(cp.fft.ifft2(cp.fft.ifftshift(kimg_gpu)))


def make_kgrid_gpu(nx, ny, fov_x, fov_y):
    """
    Create reciprocal coordinates KX, KY (1/Å) on GPU.

    Convention: dkx = 1/fov_x, dky = 1/fov_y (no 2π factor).
    """

    dkx = 1.0 / fov_x
    dky = 1.0 / fov_y

    kx = (cp.arange(nx) - nx//2) * dkx
    ky = (cp.arange(ny) - ny//2) * dky
    KX, KY = cp.meshgrid(kx, ky, indexing="xy")
    return KX, KY


def circular_aperture_gpu(KX, KY, center_kx, center_ky, radius):
    """Binary circular aperture mask centered at (gx, gy) with radius (1/Å)."""
    return ((KX - center_kx)**2 + (KY - center_ky)**2) <= radius**2

# ----------------------------
# Projected potential proxy
# ----------------------------



def build_projected_scattering_gpu_deposit_blur(atoms, nx, ny, fov_x, fov_y, sigma=0.2, thickness=None, z0=None):
    """
    Build a simple projected scattering potential V_proj(x,y):
      V_proj(x,y) = sum_j w_j * exp(-((x-xj)^2+(y-yj)^2)/(2 sigma^2))
    where w_j = Z_j.
    
    atoms: ASE Atoms, assumed oriented with beam along z.
    thickness: if not None, include atoms in [z0 - thickness/2, z0 + thickness/2]
    z0: center z for thickness window. If None, uses mean z.
    """
    # Real-space grid (Angstrom)
    x = np.linspace(0, fov_x, nx, endpoint=False)
    y = np.linspace(0, fov_y, ny, endpoint=False)
    gx, gy = np.meshgrid(x, y, indexing="xy")

    pos = atoms.get_positions()
    Z = atoms.get_atomic_numbers() # atomic scattering factor (approximated to atomic number)

    if thickness is not None:
        if z0 is None:
            z0 = float(np.mean(pos[:, 2]))
        zmin = z0 - thickness / 2.0
        zmax = z0 + thickness / 2.0
        mask = (pos[:, 2] >= zmin) & (pos[:, 2] <= zmax)
        pos = pos[mask]
        Z = Z[mask]


    # wrap to FOV
    x = np.mod(pos[:, 0], fov_x)
    y = np.mod(pos[:, 1], fov_y)

    # map to pixel indices
    ix = np.floor(x / fov_x * nx).astype(int)
    iy = np.floor(y / fov_y * ny).astype(int)
    ix = np.clip(ix, 0, nx-1)
    iy = np.clip(iy, 0, ny-1)


    # move to GPU
    ix_g = cp.asarray(ix)
    iy_g = cp.asarray(iy)
    Z_g  = cp.asarray(Z)

    V0 = cp.zeros((ny, nx), dtype=cp.float32)
    # scatter-add
    cp.add.at(V0, (iy_g, ix_g), Z_g)
    

    # sigma in pixels
    dx = fov_x / nx
    dy = fov_y / ny
    sigma_px = float(sigma / dx)
    sigma_py = float(sigma / dy)

    # periodic boundary 조건이면 mode='wrap'이 적합
    V = cp_gaussian_filter(V0, sigma=(sigma_py, sigma_px), mode="wrap")


    # # Optional: normalize (purely for display stability)
    # V -= V.mean()
    # V /= (V.std() + 1e-12)

    return V


# ----------------------------
# Optional plotting / mask debug
# ----------------------------


def _save_mask_images(save_prefix, I_diff, M, kx, ky, title_suffix=""):
    """
    Save:
      - mask only image
      - diff with mask overlay (mask contour)
    """
    try:
        import matplotlib.pyplot as plt

        # Mask image
        plt.figure()
        plt.imshow(
            M.astype(float),
            origin="lower",
            extent=[kx[0], kx[-1], ky[0], ky[-1]],
            aspect="equal"
        )
        plt.title("DF aperture mask" + title_suffix)
        plt.xlabel("kₓ (1/Å)")
        plt.ylabel("kᵧ (1/Å)")
        plt.xlim([-1,1])
        plt.ylim([-1,1])
        plt.savefig(save_prefix + "_mask.png", dpi=300)
        plt.close()

        # Diff with mask overlay
        plt.figure()
        plt.imshow(
            np.log10(I_diff + 1e-6),
            origin="lower",
            extent=[kx[0], kx[-1], ky[0], ky[-1]],
            aspect="equal"
        )
        # overlay mask boundary
        plt.contour(
            M.astype(float),
            levels=[0.5],
            origin="lower",
            extent=[kx[0], kx[-1], ky[0], ky[-1]],
        )
        plt.title("Log Diffraction Intensity + mask" + title_suffix)
        plt.xlabel("kₓ (1/Å)")
        plt.ylabel("kᵧ (1/Å)")
        plt.xlim([-1,1])
        plt.ylim([-1,1])
        plt.savefig(save_prefix + "_diff_with_mask.png", dpi=300)
        plt.close()
    except Exception as e:
        print("Mask image export skipped:", e)


def save_df(
    save_prefix,
    center_k=None,          # (kx, ky) in 1/Å
    radius=0.05,            # 1/Å
    out_suffix=None,
    save_mask_debug=True,
    fov_x=None,
    fov_y=None,
    nx=None,
    ny=None,
    A=None,
    I_df=None,
    M=None,
    vmin=None,
    vmax=None
):
    """
    Load saved complex A(k) and metadata, apply a new circular mask,
    compute new DF image and save.
    """
    _ensure_outdir(save_prefix)

    cx, cy = float(center_k[0]), float(center_k[1])


    if out_suffix is None: 

        # Save DF
        np.save(save_prefix + f"_{out_suffix}_Idf.npy", I_df)
        np.save(save_prefix + f"_{out_suffix}_mask.npy", M.astype(np.float32))

    else: 
        # Also save a PNG
        try:
            import matplotlib.pyplot as plt
            x = np.linspace(0, fov_x, nx, endpoint=False)
            y = np.linspace(0, fov_y, ny, endpoint=False)

            h, w = I_df.shape

            plt.subplots(1,1,figsize=(4, 4 * h / w*1.5))

            if vmin is not None :
                # plt.imshow(I_df, origin="lower", extent=[x[0], x[-1], y[0], y[-1]], aspect="equal",cmap='gray',vmin=vmin, vmax=vmax)
                plt.imshow(I_df, origin="lower", aspect="equal",cmap='gray',vmin=vmin, vmax=vmax)
            else: 
                plt.imshow(I_df, origin="lower", extent=[x[0], x[-1], y[0], y[-1]], aspect="equal",cmap='gray')


            plt.imshow(I_df, origin="lower", extent=[x[0], x[-1], y[0], y[-1]], aspect="equal",cmap='gray')
            # plt.title(f"DF image (reused A): center=({cx:.3f},{cy:.3f}) 1/Å, r={radius:.3f} 1/Å")
            plt.colorbar()
            plt.xlabel("x (Å)")
            plt.ylabel("y (Å)")
            plt.axis('off')
            plt.savefig(save_prefix + f"_{out_suffix}_df.png", dpi=300)
            plt.close()

            kx_plt = (np.arange(nx) - nx//2) / fov_x
            ky_plt = (np.arange(ny) - ny//2) / fov_y

            # Save mask overlay on original diff (from A)
            if save_mask_debug:
                I_diff = (np.abs(A)**2).astype(np.float32)
                _save_mask_images(
                    save_prefix + f"_{out_suffix}",
                    I_diff=I_diff,
                    M=M,
                    kx=kx_plt, ky=ky_plt,
                    title_suffix=f" (center=({cx:.3f},{cy:.3f}) 1/Å, r={radius:.3f} 1/Å)"
                )

        except Exception as e:
            print("re-DF PNG export skipped:", e)

    return {
        "center_k_1_per_A": (cx, cy),
        "radius_1_per_A": radius,
        "Idf": I_df,
        "mask": M
    }


# ----------------------------
# High-level pipelines
# ----------------------------


def simulate_df_kinematical_gpu(
    structure_path,
    fov_x=200.0, fov_y=200.0,      # Angstrom
    nx=1024, ny=1024,
    sigma=0.2,                      # Angstrom: Gaussian blur for each atom
    thickness=None, z0=None,         # Angstrom
    # hk=(1, 0),                       # selected reflection (h,k) in-plane
    g_vec=(0,0),
    aperture_radius=0.05,            # 1/Angstrom
    supercell_repeat=(1, 1, 1),      # repeat to cover FOV if needed
    save_prefix="df_sim",
    auto_grid=True,
    dx_target=0.1, # Å/pixel 목표
    save_fft_for_reuse=True,
    save_mask_debug=True
):
    """
    Kinematical DF simulation pipeline:
      structure -> projected scattering V(x,y) -> FFT -> apply aperture around g(hk) -> iFFT -> |psi|^2

    aperture_radius: in reciprocal units (1/Angstrom).
    """
    
    _ensure_outdir(save_prefix)

    atoms = read(structure_path)



    # Build supercell if needed
    if supercell_repeat != (1, 1, 1):
        P = np.diag(supercell_repeat)
        atoms = make_supercell(atoms, P)

    if auto_grid:
        fov_x, fov_y, nx, ny = auto_fov_and_grid_from_cell(atoms, dx_target=dx_target)


    # Build projected scattering image
    Vproj_g = build_projected_scattering_gpu_deposit_blur(
        atoms, nx=nx, ny=ny, fov_x=fov_x, fov_y=fov_y,
        sigma=sigma, thickness=thickness, z0=z0
    )

    # K-space amplitude (simple: use FFT of Vproj as "scattering amplitude")
    Vproj_g = Vproj_g.astype(cp.float32)
    A_g = fft2c_gpu(Vproj_g).astype(cp.complex64)
    I_diff_g = (cp.abs(A_g)**2).astype(cp.float32)

    # Reciprocal grid
    KX, KY = make_kgrid_gpu(nx, ny, fov_x, fov_y)

    gx, gy = g_vec

    # Apply DF aperture around +g (you can also try -g by flipping sign)
    # M_g = circular_aperture_gpu(KX, KY, gx, gy, aperture_radius).astype(cp.float32)
    sigma = 0.3 * aperture_radius  # edge softness (tune 0.05~0.3)
    R2 = (KX - gx)**2 + (KY - gy)**2
    M_g = np.exp(-0.5 * R2 / (sigma**2))
    A_sel = A_g * M_g

    # DF image wave and intensity
    psi_df_g = ifft2c_gpu(A_sel)
    I_df_g = (cp.abs(psi_df_g)**2).astype(cp.float32)

    # Also export diffraction intensity for inspection
    Vproj = cp.asnumpy(Vproj_g)
    I_diff = cp.asnumpy(I_diff_g.astype(cp.float32))
    I_df = cp.asnumpy(I_df_g)
    M = cp.asnumpy(M_g)

    # Save as numpy arrays (lossless)
    np.save(save_prefix + "_Vproj.npy", Vproj)
    np.save(save_prefix + "_Idiff.npy", I_diff)
    np.save(save_prefix + "_Idf.npy", I_df)

    # Save complex FFT and metadata for reuse
    if save_fft_for_reuse:
        A = cp.asnumpy(A_g)  # complex64
        np.save(save_prefix + "_A.npy", A)  # complex64
        np.savez(
            save_prefix + "_meta.npz",
            nx=nx, ny=ny, fov_x=float(fov_x), fov_y=float(fov_y),
            gx=float(gx), gy=float(gy), aperture_radius=float(aperture_radius)
        )


    # Optional: quick PNG output (requires matplotlib)
    try:
        import matplotlib.pyplot as plt
        kx = (np.arange(nx) - nx//2) / fov_x
        ky = (np.arange(ny) - ny//2) / fov_y
        plt.figure()
        plt.imshow(np.log10(I_diff + 1e-6), origin="lower",extent=[kx[0], kx[-1], ky[0], ky[-1]], aspect="equal")
        plt.title("Log Diffraction Intensity (kinematical proxy)")
        plt.xlabel("kₓ (1/Å)")
        plt.ylabel("kᵧ (1/Å)")
        plt.xlim([-1,1])
        plt.ylim([-1,1])
        plt.savefig(save_prefix + "_diff.png", dpi=300)
        plt.close()

        x = np.linspace(0, fov_x, nx, endpoint=False)
        y = np.linspace(0, fov_y, ny, endpoint=False)

        plt.figure()
        plt.imshow(I_df, origin="lower", extent=[x[0], x[-1], y[0], y[-1]], aspect="equal")
        plt.title(f"DF image (r={aperture_radius:.3f} 1/Å, center=({gx:.3f},{gy:.3f}) 1/Å)")
        plt.xlabel("x (Å)")
        plt.ylabel("y (Å)")
        plt.tight_layout()
        plt.savefig(save_prefix + "_df.png", dpi=300)
        plt.close()

        plt.figure()
        plt.imshow(Vproj, origin="lower",extent=[x[0], x[-1], y[0], y[-1]], aspect="equal")
        plt.title("Projected scattering potential (normalized)")
        plt.xlabel("x (Å)")
        plt.ylabel("y (Å)")
        plt.tight_layout()
        plt.savefig(save_prefix + "_Vproj.png", dpi=300)
        plt.close()
    except Exception as e:
        print("PNG export skipped (matplotlib missing or error):", e)

    # Save mask debug images (mask + overlay)
    if save_mask_debug:
        _save_mask_images(
            save_prefix,
            I_diff=I_diff,
            M=M,
            kx=kx, ky=ky,
            title_suffix=f" (center=({gx:.3f},{gy:.3f}) 1/Å, r={aperture_radius:.3f} 1/Å)"
        )
        np.save(save_prefix + "_mask.npy", M.astype(np.float32))



    return {
        "g_center_1_per_A": (gx, gy),
        "Vproj": Vproj,
        "Idiff": I_diff,
        "Idf": I_df
    }



# ----------------------------
# 2) Reuse saved FFT amplitude A to simulate DF with a NEW mask
# ----------------------------
def df_from_saved_fft(
    save_prefix,
    center_k=None,          # (kx, ky) in 1/Å
    radius=0.05,            # 1/Å
    out_suffix="reDF",
    save_mask_debug=True,
    vmin=None,
    vmax=None
):
    """
    Load saved complex A(k) and metadata, apply a new circular mask,
    compute new DF image and save.
    """
    _ensure_outdir(save_prefix)

    A = np.load(save_prefix + "_A.npy")  # complex
    meta = np.load(save_prefix + "_meta.npz")
    nx = int(meta["nx"])
    ny = int(meta["ny"])
    fov_x = float(meta["fov_x"])
    fov_y = float(meta["fov_y"])


    A_g = cp.asarray(A)

    KX, KY = make_kgrid_gpu(nx, ny, fov_x, fov_y)
    kx = KX[0, :]
    ky = KY[:, 0]

    if center_k is None:
        # fall back to previous saved center if user didn't give one
        gx = float(meta["gx"])
        gy = float(meta["gy"])
        center_k = (gx, gy)

    cx, cy = float(center_k[0]), float(center_k[1])

    # M_g = circular_aperture_gpu(KX, KY, cx, cy, radius).astype(cp.float32)

    sigma = 0.3 * radius  # edge softness (tune 0.05~0.3)
    R2 = (KX - cx)**2 + (KY - cy)**2
    M_g = np.exp(-0.5 * R2 / (sigma**2))
    A_sel = A_g * M_g
    psi_df_g = ifft2c_gpu(A_sel)
    I_df_g = (cp.abs(psi_df_g)**2).astype(cp.float32)

    I_df = cp.asnumpy(I_df_g)
    M = cp.asnumpy(M_g)

    if out_suffix is None: 

        # Save DF
        np.save(save_prefix + f"_{out_suffix}_Idf.npy", I_df)
        np.save(save_prefix + f"_{out_suffix}_mask.npy", M.astype(np.float32))

        # Also save a PNG
        try:
            import matplotlib.pyplot as plt
            x = np.linspace(0, fov_x, nx, endpoint=False)
            y = np.linspace(0, fov_y, ny, endpoint=False)

            plt.figure()
            plt.imshow(I_df, origin="lower", extent=[x[0], x[-1], y[0], y[-1]], aspect="equal",cmap='gray')
            plt.title(f"DF image (reused A): center=({cx:.3f},{cy:.3f}) 1/Å, r={radius:.3f} 1/Å")
            plt.colorbar()
            plt.xlabel("x (Å)")
            plt.ylabel("y (Å)")
            plt.tight_layout()
            plt.savefig(save_prefix + f"_{out_suffix}_df.png", dpi=300)
            plt.close()

            kx_plt = (np.arange(nx) - nx//2) / fov_x
            ky_plt = (np.arange(ny) - ny//2) / fov_y

            # Save mask overlay on original diff (from A)
            if save_mask_debug:
                I_diff = (np.abs(A)**2).astype(np.float32)
                _save_mask_images(
                    save_prefix + f"_{out_suffix}",
                    I_diff=I_diff,
                    M=M,
                    kx=kx_plt, ky=ky_plt,
                    title_suffix=f" (center=({cx:.3f},{cy:.3f}) 1/Å, r={radius:.3f} 1/Å)"
                )

        except Exception as e:
            print("re-DF PNG export skipped:", e)

    return {
        "center_k_1_per_A": (cx, cy),
        "radius_1_per_A": radius,
        "A":A,
        "Idf": I_df,
        "mask": M,
        "fov_x":fov_x,
        "fov_y":fov_y,
        "nx":nx,
        "ny":ny
    }






if __name__ == "__main__":


    isInitial = False

    if isInitial == True:

        res = simulate_df_kinematical_gpu(
            structure_path="/root/kinematical/260128_bilayer_rect_crop_82_40_1.cif",   
            sigma=0.25,                    # Å 
            g_vec=(0,0), # 1/Å
            aperture_radius=3,          # 1/Å
            supercell_repeat=(1,1,1),    
            save_prefix="/root/kinematical/out/sample",
            auto_grid=True,
            dx_target=0.5
        )
        print("Selected g center (1/Å):", res["g_center_1_per_A"])

    else:


        new_g = [(0.3654*np.cos(np.deg2rad(0)), 0.3654*np.sin(np.deg2rad(0))),
                (0.3654*np.cos(np.deg2rad(60)), 0.3654*np.sin(np.deg2rad(60))),
                (0.3654*np.cos(np.deg2rad(120)), 0.3654*np.sin(np.deg2rad(120))),
                (0.6329*np.cos(np.deg2rad(30)), 0.6329*np.sin(np.deg2rad(30))),
                (0.6329*np.cos(np.deg2rad(90)), 0.6329*np.sin(np.deg2rad(90))),
                (0.6329*np.cos(np.deg2rad(150)), 0.6329*np.sin(np.deg2rad(150)))]
        
        dfs = []
        outs = []

        for idx in range(len(new_g)) :


            out2 = df_from_saved_fft(
                "/root/kinematical/out/sample", 
                center_k=new_g[idx],   
                radius=0.034, 
                out_suffix=None 
            )

            I = out2["Idf"]
            dfs.append(I)
            outs.append(out2)

        all_vals = np.concatenate([a.ravel() for a in dfs])
        vmin, vmax = np.percentile(all_vals, [1, 99])

        for idx in range(len(new_g)):

            out_suffix_str = f"g{idx+1}"
            crr_data = outs[idx]

            save_df(
                "/root/kinematical/out/sample",
                    center_k=new_g[idx],          # (kx, ky) in 1/Å
                    radius=0.034,            # 1/Å
                    out_suffix=out_suffix_str,
                    save_mask_debug=True,
                    fov_x=crr_data["fov_x"],
                    fov_y=crr_data["fov_y"],
                    nx=crr_data["nx"],
                    ny=crr_data["ny"],
                    A=crr_data["A"],
                    I_df=crr_data["Idf"],
                    M=crr_data["mask"],
                    vmin=vmin,
                    vmax=vmax
                    )

