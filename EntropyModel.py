"""
Title: Entropic penalty model for linker accessibility

Description:
Geometry-based statistical mechanics model to estimate entropic penalties
for linker placement in protein–ligand complexes.

"""

# =========================
# 1. Imports
# =========================
from __future__ import annotations

import numpy as np
import pandas as pd

from math import tan, radians
from itertools import product
from tqdm import tqdm

from typing import Sequence, List, Dict, Any

# =========================
# 2. Global Parameters
# =========================

R_kcal_per_molK = 1.987204258e-3

# =========================
# 3. Geometry Utilities
# =========================


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Zero-length vector")
    return v / n
    
# =========================
# 4. Conformation Generation
# =========================


def build_coords_from_torsions(
    q_rad: np.ndarray,
    cone_half_angle_deg: float,
    n_carbons: int,
    d: float = 1.54,
    tetra_deg: float = 109.5,
    r_bead: float = 1.4,
    r_origin: float | None = None,
):
    """
    Continuous version of the chain builder.

    Inputs
    ------
    q_rad : array shape (n_carbons-1,)
        Torsion angles (radians) used for each step after the first atom.
        For n_carbons=1 -> empty array.
    cone_half_angle_deg : float
        Half angle of cone (degrees). Used only to place origin.
    n_carbons : int
        Number of carbon beads.
    d : float
        Bond length.
    tetra_deg : float
        Tetrahedral angle parameter used exactly as in your enumerator.
    r_bead : float
        Linker bead radius (kept for interface consistency).
    r_origin : float | None
        Radius of the bottom circle of the cone.
        If None, defaults to 2.2.

    Returns
    -------
    coords : np.ndarray, shape (n_carbons, 3)
        Cartesian coordinates of the beads (excluding the anchor p0).
    p0 : np.ndarray, shape (3,)
        Anchor position.
    """

    q_rad = np.asarray(q_rad, dtype=float)

    if n_carbons < 1:
        raise ValueError("n_carbons must be >= 1")

    if n_carbons == 1 and q_rad.size != 0:
        raise ValueError("For n_carbons=1, q_rad must be empty")

    if n_carbons > 1 and q_rad.size != (n_carbons - 1):
        raise ValueError(f"Expected q_rad of length {n_carbons-1}, got {q_rad.size}")

    half_angle = radians(cone_half_angle_deg)

    if not (0.0 < half_angle < (np.pi / 2)):
        raise ValueError("cone_half_angle_deg must be in (0, 90)")

    if r_origin is None:
        r_origin = 2.2

    z0 = float(r_origin) / float(tan(half_angle))
    p0 = np.array([0.0, 0.0, z0], dtype=float)

    p1 = np.array([0.0, 0.0, z0 + float(d)], dtype=float)

    if n_carbons == 1:
        return np.stack([p1], axis=0), p0

    theta = radians(180.0 - tetra_deg)
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))

    axis = _unit(p1 - p0)

    tmp = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])

    e1 = _unit(np.cross(axis, tmp))
    e2 = _unit(np.cross(axis, e1))

    points = [p1]

    for phi in q_rad:

        c = float(np.cos(phi))
        s = float(np.sin(phi))

        perp = c * e1 + s * e2
        b_next_dir = _unit(cos_t * axis + sin_t * perp)

        p_next = points[-1] + float(d) * b_next_dir
        points.append(p_next)

        axis2 = b_next_dir

        e1p = axis - axis2 * np.dot(axis2, axis)

        if np.linalg.norm(e1p) < 1e-9:
            tmp2 = np.array([1.0, 0.0, 0.0]) if abs(axis2[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            e1p = np.cross(axis2, tmp2)

        e1 = _unit(e1p)
        e2 = _unit(np.cross(axis2, e1))
        axis = axis2

    return np.stack(points, axis=0), p0

    
    
def enumerate_linker_conformations_in_cone(
    n_carbons: int,
    cone_half_angle_deg: float,
    d: float = 1.54,
    tetra_deg: float = 109.5,
    r_bead: float = 1.4,
    r_origin: float | None = None,
    dihedrals_deg: tuple[float, float, float] = (0.0, 120.0, 240.0),
    skip_bonds: int = 2,
    max_conformations: int | None = None,
    debug: bool = False,
    cone_depth: float | None = None,
    require_endpoint_above_mouth: bool = True,
    skip_cone_atoms: int = 2,
) -> Dict[str, Any]:
    """
    Enumerate discrete 3-state torsion conformations of an n-carbon chain inside a cone.

    Geometry convention
    -------------------
    r_origin : radius of the bottom circle of the cone
    r_bead   : linker bead radius used for both
               - cone wall fit
               - intralinker self-clash
    """

    if n_carbons < 1:
        return {
            "count": 0,
            "torsions_rad": np.zeros((0, 0)),
            "coords": None,
            "rejections": {"bad_n": 1},
        }

    half_angle = radians(cone_half_angle_deg)
    if not (0.0 < half_angle < (np.pi / 2)):
        return {
            "count": 0,
            "torsions_rad": np.zeros((0, max(0, n_carbons - 1))),
            "coords": None,
            "rejections": {"bad_angle": 1},
        }

    if r_origin is None:
        r_origin = 2.2

    z0 = float(r_origin) / float(tan(half_angle))
    z_mouth = None if cone_depth is None else (z0 + float(cone_depth))

    p0 = np.array([0.0, 0.0, z0], dtype=float)
    p1 = np.array([0.0, 0.0, z0 + float(d)], dtype=float)

    def inside(p: np.ndarray, r: float) -> bool:
        if z_mouth is None:
            return _inside_cone_sphere(p, half_angle, r)
        return _inside_finite_cone_sphere_zmouth(p, half_angle, z_mouth, r)

    if not inside(p0, float(r_origin)):
        return {"count": 0, "torsions_rad": np.zeros((0, max(0, n_carbons - 1))), "coords": None}

    if skip_cone_atoms < 1:
        if not inside(p1, float(r_bead)):
            return {"count": 0, "torsions_rad": np.zeros((0, max(0, n_carbons - 1))), "coords": None}

    theta = radians(180.0 - tetra_deg)
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))

    phis = [radians(x) for x in dihedrals_deg]

    axis = np.array([0.0, 0.0, 1.0], dtype=float)
    e1 = np.array([1.0, 0.0, 0.0], dtype=float)
    e2 = np.array([0.0, 1.0, 0.0], dtype=float)

    torsions_list: List[np.ndarray] = []
    coords_list: List[np.ndarray] = []

    rej = {"cone": 0, "self": 0, "ok": 0}

    target_torsions = max(0, n_carbons - 1)

    def rec(points: List[np.ndarray], phi_path: List[float], axis: np.ndarray, e1: np.ndarray, e2: np.ndarray):
        nonlocal rej, torsions_list, coords_list

        if max_conformations is not None and len(torsions_list) >= max_conformations:
            return

        if len(points) == n_carbons:

            if (
                require_endpoint_above_mouth
                and z_mouth is not None
                and float(points[-1][2]) < float(z_mouth) - 1e-12
            ):
                rej["cone"] += 1
                return

            rej["ok"] += 1
            torsions_list.append(np.array(phi_path, dtype=float))
            coords_list.append(np.stack(points, axis=0))
            return

        p_last = points[-1]

        for phi in phis:

            c = float(np.cos(phi))
            s = float(np.sin(phi))

            perp = (c * e1 + s * e2)
            b_next_dir = _unit(cos_t * axis + sin_t * perp)
            p_next = p_last + float(d) * b_next_dir

            next_atom_index = len(points) + 1

            if next_atom_index > int(skip_cone_atoms):
                if not inside(p_next, float(r_bead)):
                    rej["cone"] += 1
                    continue

            new_points = points + [p_next]

            if not _self_avoiding_ok(new_points, float(r_bead), skip_bonds=skip_bonds):
                rej["self"] += 1
                continue

            axis2 = b_next_dir
            e1p = axis - axis2 * np.dot(axis2, axis)

            if np.linalg.norm(e1p) < 1e-9:
                tmp2 = np.array([1.0, 0.0, 0.0]) if abs(axis2[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
                e1p = np.cross(axis2, tmp2)

            e1p = _unit(e1p)
            e2p = _unit(np.cross(axis2, e1p))

            rec(new_points, phi_path + [phi], axis2, e1p, e2p)

    if n_carbons == 1:
        torsions_list = [np.zeros((0,), dtype=float)]
        coords_list = [np.stack([p1], axis=0)]
        rej["ok"] = 1
    else:
        rec([p1], [], axis, e1, e2)

    torsions = np.stack(torsions_list, axis=0) if torsions_list else np.zeros((0, target_torsions), dtype=float)
    coords = np.stack(coords_list, axis=0) if coords_list else None

    out = {"count": int(torsions.shape[0]), "torsions_rad": torsions, "coords": coords}

    if debug:
        out["rejections"] = rej

    return out

    

def enumerate_feasible_torsions(
    *,
    geom: dict,
    dihedrals_deg: tuple[float, float, float] = (0.0, 120.0, 240.0),
    max_conformations: int | None = None,
    debug: bool = False,
) -> np.ndarray:
    """
    Enumerate discrete torsion states and return feasible torsion tuples in radians.

    Cone geometry convention:
      - r_origin: radius of the bottom circle of the cone
      - r_bead : linker bead radius used for both cone-wall fit and self-clash
    """
    r_bead = float(geom.get("r_bead", 1.4))
    r_origin = float(geom.get("r_origin", 2.2))

    out = enumerate_linker_conformations_in_cone(
        n_carbons=geom["n_carbons"],
        cone_half_angle_deg=geom["cone_half_angle_deg"],
        cone_depth=geom.get("cone_depth", None),
        require_endpoint_above_mouth=geom.get("require_endpoint_above_mouth", True),
        d=geom.get("d", 1.54),
        tetra_deg=geom.get("tetra_deg", 109.5),
        r_bead=r_bead,
        r_origin=r_origin,
        dihedrals_deg=dihedrals_deg,
        skip_bonds=geom.get("skip_bonds", 2),
        skip_cone_atoms=geom.get("skip_cone_atoms", 2),
        max_conformations=max_conformations,
        debug=debug,
    )

    if debug:
        print("count =", out["count"])
        print("rejections =", out.get("rejections"))

    qs = np.asarray(out["torsions_rad"], dtype=float)

    mask = np.array([feasible(q, geom=geom) for q in qs], dtype=bool)
    return qs[mask]
    

# =========================
# 5. Steric / Geometric Filters
# =========================


def _inside_cone_sphere(
    p: np.ndarray,
    half_angle_rad: float,
    r_bead: float = 0.0,
) -> bool:
    """
    Cone tip at (0,0,0), axis +z, half-angle = half_angle_rad.
    Sphere-fit condition: rho + r_bead <= z * tan(half_angle).
    """
    x, y, z = map(float, p)

    if z < 0:
        return False

    rho = (x*x + y*y) ** 0.5
    return (rho + float(r_bead)) <= (z * tan(half_angle_rad) + 1e-12)


def _inside_finite_cone_sphere_zmouth(
    p,
    half_angle_rad: float,
    z_mouth: float,
    r_bead: float = 0.0,
) -> bool:
    """
    Finite-depth cone where the constrained region is 0 <= z <= z_mouth.
    For z > z_mouth: unconstrained (free space).
    """
    x, y, z = map(float, p)

    if z < 0:
        return False

    if z > float(z_mouth):
        return True

    rho = (x*x + y*y) ** 0.5
    return (rho + float(r_bead)) <= (z * tan(half_angle_rad) + 1e-12)
    

def _self_avoiding_ok(points: List[np.ndarray], r_bead: float, skip_bonds: int = 2) -> bool:
    if r_bead <= 0:
        return True

    min_dist2 = (2.0 * r_bead) ** 2
    m = len(points)

    for i in range(m):
        pi = points[i]
        for j in range(i + skip_bonds + 1, m):
            pj = points[j]
            if np.dot(pi - pj, pi - pj) < min_dist2 - 1e-12:
                return False

    return True


def is_allowed_torsions(
    q_rad: np.ndarray,
    n_carbons: int,
    cone_half_angle_deg: float,
    d: float = 1.54,
    tetra_deg: float = 109.5,
    r_bead: float = 1.4,
    r_origin: float | None = None,
    skip_bonds: int = 2,
    cone_depth: float | None = None,
    require_endpoint_above_mouth: bool = True,
    *,
    skip_cone_atoms: int | None = None,
) -> bool:
    """
    Final geometric feasibility check.

    Cone geometry model
    -------------------
    r_origin : radius of the bottom circle of the cone
    r_bead   : linker bead radius used for both
               - cone-wall fit
               - intralinker self-clash
    """

    if r_origin is None:
        r_origin = 2.2

    coords, p0 = build_coords_from_torsions(
        q_rad=q_rad,
        cone_half_angle_deg=cone_half_angle_deg,
        n_carbons=n_carbons,
        d=d,
        tetra_deg=tetra_deg,
        r_bead=r_bead,
        r_origin=r_origin,
    )

    half_angle = radians(cone_half_angle_deg)

    z0 = float(r_origin) / float(tan(half_angle))
    z_mouth = None if cone_depth is None else (z0 + float(cone_depth))

    def inside(p: np.ndarray, sphere_radius: float) -> bool:
        if cone_depth is None:
            return _inside_cone_sphere(p, half_angle, sphere_radius)
        return _inside_finite_cone_sphere_zmouth(p, half_angle, z_mouth, sphere_radius)

    if not inside(p0, float(r_origin)):
        return False

    if skip_cone_atoms is None:
        skip_cone_atoms = 2
    skip_cone_atoms = max(0, int(skip_cone_atoms))

    for i, p in enumerate(coords):
        if i < skip_cone_atoms:
            continue
        if not inside(p, float(r_bead)):
            return False

    if require_endpoint_above_mouth and cone_depth is not None:
        if float(coords[-1][2]) < float(z_mouth):
            return False

    if not _self_avoiding_ok(
        np.asarray(coords, dtype=float),
        float(r_bead),
        skip_bonds=int(skip_bonds),
    ):
        return False

    return True

    
def feasible(q: np.ndarray, *, geom: dict) -> bool:
    """
    Final geometric feasibility check.

    Cone geometry model
    -------------------
    r_origin : radius of the bottom circle of the cone
    r_bead   : linker bead radius used for both
               - cone-wall fit
               - intralinker self-clash
    """

    r_bead = float(geom.get("r_bead", 1.4))

    r_origin = geom.get("r_origin", None)
    if r_origin is None:
        r_origin = 2.2

    return is_allowed_torsions(
        q_rad=q,
        n_carbons=geom["n_carbons"],
        cone_half_angle_deg=geom["cone_half_angle_deg"],
        d=geom.get("d", 1.54),
        tetra_deg=geom.get("tetra_deg", 109.5),
        r_bead=r_bead,
        r_origin=r_origin,
        skip_bonds=geom.get("skip_bonds", 2),
        cone_depth=geom.get("cone_depth", None),
        require_endpoint_above_mouth=geom.get("require_endpoint_above_mouth", True),
        skip_cone_atoms=geom.get("skip_cone_atoms", 2),
    )

    
# =========================
# 6. Energy Model
# =========================


def filter_conformations_by_torsion_energy(
    qs: np.ndarray,
    *,
    dE_cut_kcal: float = 2.5,
    **energy_kwargs,
):
    """
    qs: (K, d) torsion vectors (radians)
    Returns: (qs_keep, E_keep, mask, E_all)
    """
    qs = np.asarray(qs, dtype=float)
    E_all = np.array([torsion_energy_kcal(q, **energy_kwargs) for q in qs], dtype=float)
    Emin = float(np.min(E_all)) if E_all.size else float("inf")
    mask = (E_all - Emin) <= float(dE_cut_kcal)
    return qs[mask], E_all[mask], mask, E_all
    

def torsion_state_from_phi(phi_rad: float, dihedrals_deg=(0.0, 120.0, 240.0)) -> int:
    """
    Map a phi (radians) to the nearest discrete torsion state index {0,1,2}
    corresponding to dihedrals_deg.
    """
    phis = np.deg2rad(np.array(dihedrals_deg, dtype=float))
    x = (phi_rad + np.pi) % (2*np.pi) - np.pi
    d = (phis - x + np.pi) % (2*np.pi) - np.pi
    return int(np.argmin(np.abs(d)))
    

def torsion_energy_kcal(
    q_rad: np.ndarray,
    *,
    dihedrals_deg=(0.0, 120.0, 240.0),
    # single-torsion penalties (kcal/mol) for [0°, +120°, -120°] in that order:
    # Use ~0.54–0.62 for a first gauche; pick a midpoint default.
    U_single=(0.0, 0.8, 0.8),
    # neighbor coupling (kcal/mol) depending on signed gauche adjacency:
    # same-sign adjacent gauche: ~0.22–0.37 (midpoint default)
    g_same_penalty_kcal: float = 0.30,
    # opposite-sign adjacent gauche: typically much larger (model-dependent); set to 0 to disable
    g_opposite_penalty_kcal: float = 2.7,
) -> float:
    """
    Discrete 3-state torsion energy with signed-gauche neighbor coupling.

    States (by default dihedrals_deg):
      state 0: 0°   (trans-like in this simplified model)
      state 1: +120° (g+)
      state 2: 240° == -120° (g-)

    Energy:
      E = sum(U_single[state_i])
        + sum over adjacent pairs:
            + g_same_penalty_kcal      if (g+,g+) or (g-,g-)
            + g_opposite_penalty_kcal  if (g+,g-) or (g-,g+)
            + 0                        otherwise (if any is trans)
    """
    q_rad = np.asarray(q_rad, dtype=float)
    if q_rad.size == 0:
        return 0.0

    states = np.array([torsion_state_from_phi(phi, dihedrals_deg=dihedrals_deg) for phi in q_rad], dtype=int)

    U_single = np.asarray(U_single, dtype=float)
    E = float(np.sum(U_single[states]))

    if states.size >= 2:
        s0 = states[:-1]
        s1 = states[1:]

        both_gauche = (s0 != 0) & (s1 != 0)
        same_sign = both_gauche & (s0 == s1)          # g+g+ or g-g-
        opp_sign  = both_gauche & (s0 != s1)          # g+g- or g-g+

        if g_same_penalty_kcal:
            E += float(g_same_penalty_kcal) * float(np.sum(same_sign))
        if g_opposite_penalty_kcal:
            E += float(g_opposite_penalty_kcal) * float(np.sum(opp_sign))

    return E



def torsion_energies_and_weights(
    qs: np.ndarray,
    *,
    T_K: float = 298.15,
    use_filter: bool = False,
    dE_cut_kcal: float = 2.5,
    energy_kwargs: dict | None = None,
):
    """
    Compute torsional strain energies E_k and Boltzmann weights p_k for qs.

    Parameters
    ----------
    qs : (K, d) array
        Torsion vectors in radians
    T_K : float
        Temperature
    use_filter : bool
        If True, apply filter_conformations_by_torsion_energy before weighting
    dE_cut_kcal : float
        Cutoff used by filter_conformations_by_torsion_energy
    energy_kwargs : dict | None
        Passed into torsion_energy_kcal / filter function

    Returns
    -------
    qs_use : (K', d)
    E_kcal : (K',)
    p      : (K',) normalized weights
    """
    qs = np.asarray(qs)
    if energy_kwargs is None:
        energy_kwargs = {}

    if use_filter:
        qs_use, E_use, mask, E_all = filter_conformations_by_torsion_energy(
            qs, dE_cut_kcal=dE_cut_kcal, **energy_kwargs
        )
        qs_use = np.asarray(qs_use)
        E_kcal = np.asarray(E_use)
    else:
        qs_use = qs
        E_kcal = np.array([torsion_energy_kcal(q, **energy_kwargs) for q in qs_use], dtype=float)

    p = boltzmann_weights_from_energy(E_kcal, T_K=T_K)
    p = np.asarray(p, dtype=float)

    if not np.isclose(p.sum(), 1.0):
        # defensive renormalization
        s = p.sum()
        if s <= 0:
            raise ValueError("Boltzmann weights sum to non-positive value.")
        p = p / s

    return qs_use, E_kcal, p

    
# =========================
# 7. Entropy / Statistical Mechanics
# =========================

def _kd_fold_from_dG(dG_kcal: float, T_K: float) -> float:
    """Fold-increase in Kd from a free-energy penalty dG (kcal/mol)."""
    RT = R_kcal_per_molK * T_K
    return float(np.exp(dG_kcal / RT))

    
def boltzmann_weights_from_energy(E_kcal: np.ndarray, T_K: float = 298.15) -> np.ndarray:
    beta = 1.0 / (R_kcal_per_molK * float(T_K))
    E = np.asarray(E_kcal, dtype=float)
    E0 = np.min(E)
    w = np.exp(-(E - E0) * beta)
    return w / np.sum(w)


def conformational_entropy_from_p(
    p: np.ndarray,
    *,
    T_K: float = 298.15,
):
    """
    Return:
      S_conf (cal/mol/K and kcal/mol/K) and -T S_conf (kcal/mol)

    Uses S = -R Σ p ln p
    """
    p = np.asarray(p, dtype=float)
    s = p.sum()
    if not np.isclose(s, 1.0):
        if s <= 0:
            raise ValueError("p sums to non-positive value.")
        p = p / s

    p_safe = np.clip(p, 1e-300, 1.0)
    S_over_R = -float(np.sum(p_safe * np.log(p_safe)))
    S_kcal_per_mol_K = R_kcal_per_molK * S_over_R
    minus_T_S_kcal = -T_K * S_kcal_per_mol_K
    S_cal_per_mol_K = S_kcal_per_mol_K * 1000.0
    return dict(
        S_over_R=S_over_R,
        S_cal_per_mol_K=S_cal_per_mol_K,
        S_kcal_per_mol_K=S_kcal_per_mol_K,
        minus_T_S_kcal_per_mol=minus_T_S_kcal,
    )


def select_by_entropy_fraction(
    p: np.ndarray,
    *,
    frac: float = 0.80,
    include_ties: bool = True,
):
    """
    Select minimal subset of conformations that cumulatively accounts for >= frac
    of the *conformational entropy* S = -R sum p ln p.

    Selection is based on per-state contributions:
      s_k = -p_k ln p_k

    Parameters
    ----------
    p : (K,) normalized weights
    frac : float
        Fraction of total conformational entropy to cover (0<frac<=1)
    include_ties : bool
        If True, if cutoff falls within equal s_k values, include all tied.

    Returns
    -------
    idx_keep : np.ndarray of indices into p
    info : dict with totals and achieved fraction
    """
    p = np.asarray(p, dtype=float)
    if p.ndim != 1:
        raise ValueError("p must be 1D.")
    if not (0.0 < frac <= 1.0):
        raise ValueError("frac must be in (0, 1].")

    eps = 1e-300
    p_safe = np.clip(p, eps, 1.0)
    s = -p_safe * np.log(p_safe)  # dimensionless contributions (S/R)

    S_total = float(s.sum())
    if S_total <= 0:
        # Degenerate case: all probability mass on one state -> zero entropy
        idx = np.array([int(np.argmax(p))])
        return idx, dict(S_total_over_R=S_total, S_kept_over_R=0.0, frac_achieved=1.0)

    order = np.argsort(s)[::-1]  # descending by contribution
    s_sorted = s[order]
    csum = np.cumsum(s_sorted)

    target = frac * S_total
    k = int(np.searchsorted(csum, target, side="left"))
    k = min(k, len(p) - 1)
    cutoff_s = s_sorted[k]

    idx_keep = order[: k + 1]

    if include_ties:
        # Include any additional entries with the same contribution as cutoff (within tol)
        tol = 1e-15
        tied = order[(s_sorted >= cutoff_s - tol)]
        # tied includes everything above cutoff; but we want all with ~cutoff_s too
        # easiest: include all with s >= cutoff_s - tol
        idx_keep = tied

    # Recompute achieved fraction
    S_kept = float(s[idx_keep].sum())
    frac_achieved = min(1.0, S_kept / S_total) if S_total > 0 else 1.0

    info = dict(
        S_total_over_R=S_total,
        S_kept_over_R=S_kept,
        frac_target=frac,
        frac_achieved=frac_achieved,
        n_total=len(p),
        n_kept=len(idx_keep),
    )
    return np.array(sorted(set(map(int, idx_keep)))), info



def Svib_coupled_from_lambdas(
    lambdas_rad: np.ndarray,
    n_dof: int,
    *,
    V_ref: float,
    T_K: float = 298.15,
):
    """
    Coupled vibrational entropy from directional probing.

    Parameters
    ----------
    lambdas_rad : (n_dirs,) array
        Max step lengths along random unit directions
    n_dof : int
        Number of internal DOFs
    V_ref : float
        Reference configuration-space volume (same proxy)
    T_K : float
        Temperature in Kelvin

    Returns
    -------
    dict with:
        log_V_eff
        dS_over_R
        minus_T_dS_kcal_per_mol
    """

    lambdas_rad = np.asarray(lambdas_rad)

    eps = 1e-12
    lambdas_rad = np.maximum(lambdas_rad, eps)

    # log of effective local volume proxy
    log_V_eff = np.mean(n_dof * np.log(lambdas_rad))

    dS_over_R = log_V_eff - np.log(V_ref)

    minus_T_dS = -R_kcal_per_molK * T_K * dS_over_R

    return dict(
        log_V_eff=log_V_eff,
        dS_over_R=dS_over_R,
        minus_T_dS_kcal_per_mol=minus_T_dS,
    )



def deltaG_total_from_E_and_vib(
    E_tors_kcal: np.ndarray,
    deltaG_vib_kcal: np.ndarray,
    *,
    T_K: float = 298.15,
):
    """
    Total free energy (kcal/mol) from partition function:
      G = -RT ln Σ exp(-β (E_k + ΔG_vib,k))

    This is the cleanest single expression.

    Returns
    -------
    G_total_kcal : float
    """
    E = np.asarray(E_tors_kcal, dtype=float)
    Gv = np.asarray(deltaG_vib_kcal, dtype=float)
    if E.shape != Gv.shape:
        raise ValueError("E_tors_kcal and deltaG_vib_kcal must have same shape.")

    beta = 1.0 / (R_kcal_per_molK * T_K)
    x = -(E + Gv) * beta  # log-weights up to constant

    # log-sum-exp for numerical stability
    x_max = np.max(x)
    Z = np.sum(np.exp(x - x_max))
    if Z <= 0:
        raise ValueError("Partition function underflow or invalid.")
    G_total = -(R_kcal_per_molK * T_K) * (np.log(Z) + x_max)
    return float(G_total)
    
    
def deltaG_vib_total_from_p_and_vib(
    p_tors: np.ndarray,
    deltaG_vib_kcal: np.ndarray,
    *,
    T_K: float = 298.15,
):
    """
    Compute the *ensemble* vibrational contribution relative to torsion-only weights:
      ΔG_vib,total = -RT ln Σ_k p_k * exp(-β ΔG_vib,k)

    Here p_k must be normalized torsion-only Boltzmann weights.
    """
    p = np.asarray(p_tors, dtype=float)
    Gv = np.asarray(deltaG_vib_kcal, dtype=float)
    if p.shape != Gv.shape:
        raise ValueError("p_tors and deltaG_vib_kcal must have same shape.")
    s = p.sum()
    if not np.isclose(s, 1.0):
        if s <= 0:
            raise ValueError("p_tors sums to non-positive value.")
        p = p / s

    beta = 1.0 / (R_kcal_per_molK * T_K)
    x = np.log(np.clip(p, 1e-300, 1.0)) - beta * Gv  # log(p_k) - βΔGv

    # log-sum-exp
    x_max = np.max(x)
    Z = np.sum(np.exp(x - x_max))
    if Z <= 0:
        raise ValueError("Weighted sum underflow or invalid.")
    deltaG = -(R_kcal_per_molK * T_K) * (np.log(Z) + x_max)
    return float(deltaG)

# =========================
# 8. Directional Probing (Local Volume)
# =========================

def max_step_direction(
    q0: np.ndarray,
    u: np.ndarray,
    *,
    geom: dict,
    step_init_rad: float,
    step_max_rad: float,
    tol_rad: float,
) -> float:
    """
    Return max lambda >= 0 such that q0 + lambda * u is feasible.
    u must be a unit vector.
    """

    assert feasible(q0, geom=geom)
    assert np.isclose(np.linalg.norm(u), 1.0)

    def q_at(lam):
        return q0 + lam * u

    # ---- bracketing ----
    lam_lo = 0.0
    lam_hi = step_init_rad

    if not feasible(q_at(lam_hi), geom=geom):
        return 0.0

    while lam_hi < step_max_rad:
        lam_next = min(2.0 * lam_hi, step_max_rad)
        if feasible(q_at(lam_next), geom=geom):
            lam_lo = lam_hi
            lam_hi = lam_next
        else:
            break

    if lam_hi >= step_max_rad and feasible(q_at(lam_hi), geom=geom):
        return step_max_rad

    # ---- bisection ----
    while (lam_hi - lam_lo) > tol_rad:
        lam_mid = 0.5 * (lam_lo + lam_hi)
        if feasible(q_at(lam_mid), geom=geom):
            lam_lo = lam_mid
        else:
            lam_hi = lam_mid

    return lam_lo


def directional_probe(
    q0: np.ndarray,
    *,
    geom: dict,
    n_dirs: int = 200,
    rng_seed: int = 0,
    step_init_deg: float = 5.0,
    step_max_deg: float = 90.0,
    tol_deg: float = 0.1,
):
    """
    Sample random directions in DOF space and compute lambda(u).

    Returns:
      U        : (n_dirs, n_dof) unit vectors
      lambdas  : (n_dirs,) max step lengths (radians)
    """

    rng = np.random.default_rng(rng_seed)
    n_dof = len(q0)

    step_init = np.deg2rad(step_init_deg)
    step_max = np.deg2rad(step_max_deg)
    tol = np.deg2rad(tol_deg)

    U = np.zeros((n_dirs, n_dof))
    lambdas = np.zeros(n_dirs)

    for k in range(n_dirs):
        u = rng.normal(size=n_dof)
        u /= np.linalg.norm(u)

        U[k] = u

        lambdas[k] = max_step_direction(
            q0,
            u,
            geom=geom,
            step_init_rad=step_init,
            step_max_rad=step_max,
            tol_rad=tol,
        )

    return U, lambdas


def deltaG_vib_coupled_for_q0(
    q0: np.ndarray,
    *,
    geom: dict,
    T_K: float = 298.15,
    n_dirs: int = 200,
    rng_seed: int = 0,
    step_init_deg: float = 5.0,
    step_max_deg: float = 90.0,
    tol_deg: float = 0.1,
    V_ref: float | None = None,
):
    """
    Compute ΔG_vib(q0) = -TΔS_vib from directional probing (coupled).

    Reference volume:
      If V_ref is None, use (2π)^n_dof.
    """

    q0 = np.asarray(q0, dtype=float)
    n_dof = len(q0)

    if V_ref is None:
        V_ref = (2.0 * np.pi) ** n_dof

    U, lambdas = directional_probe(
        q0,
        geom=geom,
        n_dirs=n_dirs,
        rng_seed=rng_seed,
        step_init_deg=step_init_deg,
        step_max_deg=step_max_deg,
        tol_deg=tol_deg,
    )

    out = Svib_coupled_from_lambdas(
        lambdas,
        n_dof=n_dof,
        V_ref=V_ref,
        T_K=T_K,
    )

    deltaG = float(out["minus_T_dS_kcal_per_mol"])

    diag = dict(
        n_dirs=int(n_dirs),
        min_lambda_rad=float(np.min(lambdas)) if len(lambdas) else float("nan"),
        mean_lambda_rad=float(np.mean(lambdas)) if len(lambdas) else float("nan"),
        median_lambda_rad=float(np.median(lambdas)) if len(lambdas) else float("nan"),
        V_ref=float(V_ref),
        log_V_eff=float(out.get("log_V_eff", np.nan)),
        dS_over_R=float(out.get("dS_over_R", np.nan)),
    )

    return deltaG, diag
    

# =========================
# 9. Main Driver Functions
# =========================

def run_cone_grid_entropy_free_energy(
    *,
    cone_half_angle_deg: float | Sequence[float],
    cone_depth: float | Sequence[float],
    n_carbons: int | Sequence[int],
    require_endpoint_above_mouth: bool | Sequence[bool] = True,
    d: float = 1.54,
    tetra_deg: float = 109.5,
    r_bead: float = 1.4,
    r_origin: float | None = 2.2,
    skip_bonds: int = 2,
    T_K: float = 298.15,
    dihedrals_deg: tuple[float, float, float] = (0.0, 120.0, 240.0),
    U_single: tuple[float, float, float] = (0.0, 0.8, 0.8),
    g_same_penalty_kcal: float = 0.30,
    g_opposite_penalty_kcal: float = 2.70,
    entropy_frac: float = 0.90,
    include_ties: bool = True,
    use_filter: bool = False,
    dE_cut_kcal: float = 2.5,
    n_dirs: int = 200,
    rng_seed: int = 0,
    step_init_deg: float = 5.0,
    step_max_deg: float = 90.0,
    tol_deg: float = 0.1,
    enum_kwargs: dict | None = None,
    # unrestricted reference geometry for second-level baseline correction
    ref_cone_half_angle_deg: float = 89.9,
    ref_cone_depth: float = 0.1,
) -> pd.DataFrame:
    """
    Run the full pipeline over a grid of cone angles/depths/linker lengths.

    Final intended reference structure
    ----------------------------------
    1) Local vibrational penalties are computed relative to a fixed torsional
       reference basin width of 90° per DOF:
           V_ref = (pi/2)^n_dof

    2) The resulting free energies are then referenced to an effectively
       unrestricted cone:
           cone_half_angle_deg = 89.9
           cone_depth = 0.1 Å

    Cone geometry convention
    ------------------------
    r_origin : radius of the bottom circle of the cone
    r_bead   : linker bead radius used for:
               - cone wall fit
               - intralinker self-clash
    """

    if enum_kwargs is None:
        enum_kwargs = {}

    angles = _as_list(cone_half_angle_deg)
    depths = _as_list(cone_depth)
    lengths = _as_list(n_carbons)
    endpoint_flags = _as_list(require_endpoint_above_mouth)

    if r_origin is None:
        r_origin_eff = 2.2
    else:
        r_origin_eff = float(r_origin)

    energy_kwargs = dict(
        dihedrals_deg=dihedrals_deg,
        U_single=U_single,
        g_same_penalty_kcal=g_same_penalty_kcal,
        g_opposite_penalty_kcal=g_opposite_penalty_kcal,
    )

    def evaluate_one_geometry(geom: dict) -> dict:
        """
        Evaluate one specific geometry and return a dict of thermodynamic results.
        This uses the FIXED torsional reference volume V_ref = (pi/2)^n.
        """
        out = {}

        qs = enumerate_feasible_torsions(
            geom=geom,
            dihedrals_deg=dihedrals_deg,
            **enum_kwargs,
        )

        K = int(qs.shape[0])
        out["K_feasible"] = K

        if K == 0:
            out.update(
                S_conf_cal_per_mol_K=np.nan,
                minusT_S_conf_kcal=np.nan,
                n_kept=0,
                entropy_frac_achieved=np.nan,
                dG_vib_mean_kcal=np.nan,
                dG_vib_total_kcal=np.nan,
                G_total_kcal=np.nan,
                lambda_min_mean_rad=np.nan,
                lambda_mean_mean_rad=np.nan,
                V_ref_fixed=np.nan,
                qs_keep=None,
                E_keep=None,
                p_tors_keep=None,
            )
            return out

        qs_use, E_kcal, p = torsion_energies_and_weights(
            qs,
            T_K=T_K,
            use_filter=use_filter,
            dE_cut_kcal=dE_cut_kcal,
            energy_kwargs=energy_kwargs,
        )

        confS = conformational_entropy_from_p(p, T_K=T_K)
        out["S_conf_cal_per_mol_K"] = confS["S_cal_per_mol_K"]
        out["minusT_S_conf_kcal"] = confS["minus_T_S_kcal_per_mol"]

        idx_keep, info_keep = select_by_entropy_fraction(
            p,
            frac=entropy_frac,
            include_ties=include_ties,
        )

        qs_keep = qs_use[idx_keep]
        E_keep = E_kcal[idx_keep]

        beta = 1.0 / (R_kcal_per_molK * T_K)
        w_tors = np.exp(-beta * E_keep)
        p_tors_keep = w_tors / w_tors.sum()

        out["n_kept"] = int(info_keep["n_kept"])
        out["entropy_frac_achieved"] = float(info_keep["frac_achieved"])

        # Fixed torsional reference volume: 90° basin width per DOF
        n_dof = qs_keep.shape[1]
        V_ref_fixed = (np.pi / 2.0) ** n_dof
        out["V_ref_fixed"] = float(V_ref_fixed)

        dG_vib_keep = np.zeros(len(qs_keep))
        vib_min_lambda = []
        vib_mean_lambda = []

        for k, q0 in enumerate(qs_keep):
            dGv, diag = deltaG_vib_coupled_for_q0(
                q0,
                geom=geom,
                T_K=T_K,
                n_dirs=n_dirs,
                rng_seed=rng_seed,
                step_init_deg=step_init_deg,
                step_max_deg=step_max_deg,
                tol_deg=tol_deg,
                V_ref=V_ref_fixed,
            )

            dG_vib_keep[k] = dGv
            vib_min_lambda.append(diag.get("min_lambda_rad", np.nan))
            vib_mean_lambda.append(diag.get("mean_lambda_rad", np.nan))

        out["dG_vib_mean_kcal"] = float(np.sum(p_tors_keep * dG_vib_keep))
        out["dG_vib_total_kcal"] = float(
            deltaG_vib_total_from_p_and_vib(p_tors_keep, dG_vib_keep, T_K=T_K)
        )
        out["lambda_min_mean_rad"] = float(np.nanmean(vib_min_lambda))
        out["lambda_mean_mean_rad"] = float(np.nanmean(vib_mean_lambda))
        out["G_total_kcal"] = float(
            deltaG_total_from_E_and_vib(E_keep, dG_vib_keep, T_K=T_K)
        )

        out["qs_keep"] = qs_keep
        out["E_keep"] = E_keep
        out["p_tors_keep"] = p_tors_keep
        return out

    rows = []
    grid = list(product(angles, depths, lengths, endpoint_flags))

    # Precompute unrestricted references once per (n_carbons, endpoint flag)
    ref_cache = {}

    for ang, dep, nc, req_exit in tqdm(grid, desc="Cone grid", unit="geom"):

        geom = dict(
            n_carbons=int(nc),
            cone_half_angle_deg=float(ang),
            cone_depth=None if dep is None else float(dep),
            require_endpoint_above_mouth=bool(req_exit),
            d=float(d),
            tetra_deg=float(tetra_deg),
            r_bead=float(r_bead),
            r_origin=float(r_origin_eff),
            skip_bonds=int(skip_bonds),
            skip_cone_atoms=3,
        )

        row = dict(
            cone_half_angle_deg=float(ang),
            cone_depth=geom["cone_depth"],
            n_carbons=int(nc),
            require_endpoint_above_mouth=geom["require_endpoint_above_mouth"],
            r_bead=geom["r_bead"],
            r_origin=geom["r_origin"],
            entropy_frac_target=float(entropy_frac),
            step_max_deg=float(step_max_deg),
            ref_cone_half_angle_deg=float(ref_cone_half_angle_deg),
            ref_cone_depth=float(ref_cone_depth),
        )

        # ---- actual geometry ----
        try:
            res = evaluate_one_geometry(geom)
        except Exception as e:
            row.update(
                status="error",
                error_step="evaluate_one_geometry(actual)",
                error=f"{type(e).__name__}: {e}",
            )
            rows.append(row)
            continue

        row["K_feasible"] = res["K_feasible"]

        if res["K_feasible"] == 0:
            row.update(
                status="no_feasible_conformations",
                S_conf_cal_per_mol_K=np.nan,
                minusT_S_conf_kcal=np.nan,
                n_kept=0,
                entropy_frac_achieved=np.nan,
                dG_vib_mean_kcal=np.nan,
                dG_vib_total_kcal_raw=np.nan,
                G_total_kcal_raw=np.nan,
                dG_vib_total_ref_kcal=np.nan,
                G_total_ref_kcal=np.nan,
                dG_vib_corr_kcal=np.nan,
                G_total_corr_kcal=np.nan,
                Kd_fold_from_dG_vib=np.nan,
                Kd_fold_from_G_total=np.nan,
            )
            rows.append(row)
            continue

        row["S_conf_cal_per_mol_K"] = res["S_conf_cal_per_mol_K"]
        row["minusT_S_conf_kcal"] = res["minusT_S_conf_kcal"]
        row["n_kept"] = res["n_kept"]
        row["entropy_frac_achieved"] = res["entropy_frac_achieved"]
        row["dG_vib_mean_kcal"] = res["dG_vib_mean_kcal"]
        row["dG_vib_total_kcal_raw"] = res["dG_vib_total_kcal"]
        row["G_total_kcal_raw"] = res["G_total_kcal"]
        row["lambda_min_mean_rad"] = res["lambda_min_mean_rad"]
        row["lambda_mean_mean_rad"] = res["lambda_mean_mean_rad"]
        row["V_ref_fixed"] = res["V_ref_fixed"]

        # ---- unrestricted reference geometry (cached) ----
        ref_key = (int(nc), bool(req_exit))

        if ref_key not in ref_cache:
            geom_ref = dict(
                n_carbons=int(nc),
                cone_half_angle_deg=float(ref_cone_half_angle_deg),
                cone_depth=float(ref_cone_depth),
                require_endpoint_above_mouth=bool(req_exit),
                d=float(d),
                tetra_deg=float(tetra_deg),
                r_bead=float(r_bead),
                r_origin=float(r_origin_eff),
                skip_bonds=int(skip_bonds),
                skip_cone_atoms=3,
            )

            try:
                ref_cache[ref_key] = evaluate_one_geometry(geom_ref)
            except Exception as e:
                row.update(
                    status="error",
                    error_step="evaluate_one_geometry(reference)",
                    error=f"{type(e).__name__}: {e}",
                )
                rows.append(row)
                continue

        res_ref = ref_cache[ref_key]

        row["K_feasible_ref"] = res_ref["K_feasible"]
        row["dG_vib_total_ref_kcal"] = res_ref["dG_vib_total_kcal"]
        row["G_total_ref_kcal"] = res_ref["G_total_kcal"]

        # ---- second-level baseline correction ----
        dG_vib_corr = res["dG_vib_total_kcal"] - res_ref["dG_vib_total_kcal"]
        G_total_corr = res["G_total_kcal"] - res_ref["G_total_kcal"]

        row["dG_vib_corr_kcal"] = float(dG_vib_corr)
        row["G_total_corr_kcal"] = float(G_total_corr)

        row["Kd_fold_from_dG_vib"] = _kd_fold_from_dG(dG_vib_corr, T_K)
        row["Kd_fold_from_G_total"] = _kd_fold_from_dG(G_total_corr, T_K)

        row["status"] = "ok"
        rows.append(row)

    return pd.DataFrame(rows)

# =========================
# 10. Optional Utilities / Debug
# =========================

def _as_list(x):
    if isinstance(x, (list, tuple, np.ndarray, pd.Series)):
        return list(x)
    return [x]

    
# =========================
# 12. Execution Guard (optional)
# =========================

if __name__ == "__main__":
    try:
        df_8_add = run_cone_grid_entropy_free_energy(
            cone_half_angle_deg=[27, 45], #Add cone angles
            cone_depth=[4.7, 9.7], #Add cone depths
            entropy_frac=0.80, # Define cutoff value for entropy to be considered
            n_carbons=[8], #Define length of linker
            n_dirs=100, # Define number of directions to be explored
            require_endpoint_above_mouth=False,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()