# Dark-Field TEM and Domain-Wall Contrast Simulations

This repository contains simulation codes and structure files used in **our manuscript**:

> **“Dislocation networks in marginally twisted bilayer MoS₂”**  
> *Byunghyun Kim‡, Ayoung Yuk‡, Yunyeong Chang‡, Daesung Park, Miyoung Kim\*, Moon Sung Kang\*, and Hyobin Yoo\*.*  
> *(under review)*  
> DOI: To be updated  

The codes are provided to support the **Code Availability** statement and enable reproduction of the key simulation results related to **dark-field TEM contrast and domain-wall physics in layered MoS₂ systems**.

---

## Directory Structure

```
.
├── 0. Simulated structures
├── 1. DF simulation (Multislice)
├── 2. DF simulation (Kinematical)
└── 3. Domain wall contrast simulation (Kinematical)
```

Each directory is described below.

---

## 0. Simulated structures

This directory contains **atomic structure files** used as inputs for the simulations.

- Formats: CIF, POSCAR, XYZ, or other ASE-readable formats
- Structures include:
  - Bilayer / multilayer MoS₂
  - Domain-wall–related geometries


---

## 1. DF simulation (Multislice)

This directory contains **quantitative dark-field TEM simulations** based on **multislice electron scattering**, implemented using **abTEM** (https://github.com/abTEM/abTEM).

**Key features**
- Full multislice propagation of an electron plane wave
- Optional frozen-phonon averaging
- Objective aperture applied in reciprocal space
- Generation of dark-field real-space intensity maps

**Purpose in the paper**
- Used for **quantitative validation** of DF-TEM contrast
- Captures dynamical diffraction effects beyond kinematical approximations


---

## 2. DF simulation (Kinematical)

This directory contains a **kinematical simulation** for DF-TEM imaging.

**Key features**
- FFT-based diffraction intensity
- Circular aperture filtering in reciprocal space

**Purpose in the paper**
- Efficient scanning over aperture position and size
- Qualitative comparison with multislice results

**Notes**
- This approach does **not** include dynamical diffraction
- Intended for **trend analysis**, not absolute intensity matching

---

## 3. Domain wall contrast simulation (Kinematical)

This directory contains a **minimal analytical / kinematical model** used to study
**domain-wall–induced DF contrast** as a function of **interlayer translation (Burgers vector)**.

**Key features**
- Structure-factor–based intensity proxy
- Comparison between:
  - Perfect translation paths
  - Partial + partial (two-step) translation paths
- Generates intensity vs. translation curves for selected reciprocal vectors

**Purpose in the paper**
- Provides **physical intuition** for domain-wall contrast mechanisms
- Supports interpretation of DF-TEM contrast asymmetry and selection rules

---

## Contact

For questions regarding the code or simulations, please contact:

**Byunghyun Kim**  
Ph.D. Candidate  
Department of Materials Science and Engineering  
Seoul National University  
Email: bhkim133@snu.ac.kr
