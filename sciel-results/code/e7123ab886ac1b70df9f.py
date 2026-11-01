"""Lennard-Jones 55-atom global minimum: Multi-start perturbation + refined polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 55

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _lj_energy(pos):
    """Total LJ energy, vectorized."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** 3
    e = 4.0 * (1.0 / (r6 * r6) - 1.0 / r6)
    return np.sum(e) / 2.0

def _lj_energy_grad(pos):
    """LJ energy + analytic gradient, vectorized."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)

    r6 = r2 * r2 * r2
    r12 = r6 * r6

    e_pair = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(e_pair, 0.0)
    energy = np.sum(e_pair) / 2.0

    with np.errstate(divide='ignore', invalid='ignore'):
        factor = 4.0 * (-12.0 / (r2 ** 7) + 6.0 / (r2 ** 4))
    np.fill_diagonal(factor, 0.0)

    sum_factor = np.sum(factor, axis=1, keepdims=True)
    sum_factor_pos = np.sum(factor[:, :, np.newaxis] * pos[np.newaxis, :, :], axis=1)
    grad = pos * sum_factor - sum_factor_pos

    return energy, grad

def _lj_objective_and_grad(x):
    pos = x.reshape(-1, 3)
    e, g = _lj_energy_grad(pos)
    return e, g

def _lj_objective(x):
    pos = x.reshape(-1, 3)
    return _lj_energy(pos)

def _build_mackay_icosahedron(scales=None):
    """
    Build exact Mackay icosahedron (55 atoms) geometry.
    Uses the ideal Mackay icosahedron with radial scales optimized for LJ.
    Returns (N,3) positions centered at origin.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    phi2 = phi * phi

    # 12 vertices (directions, normalized)
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    vn = verts / np.linalg.norm(verts, axis=1, keepdims=True)

    # Find edge-midpoint directions (30 edges)
    d = np.zeros((12, 12))
    for i in range(12):
        for j in range(12):
            d[i, j] = np.linalg.norm(vn[i] - vn[j])
    dmin = np.min(d[d > 1e-8])
    mid_dirs = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(d[i, j] - dmin) < 1e-6:
                m = vn[i] + vn[j]
                nm = np.linalg.norm(m)
                if nm > 1e-8:
                    mid_dirs.append(m / nm)
    mid_dirs = np.array(mid_dirs)  # 30 directions

    # Ideal Mackay icosahedron: 1 + 12 + 42
    # Radii (ideal geometry): R1, R2, R3  with 12 in-shell vertices, 30 edge-mids, 12 outer vertices
    # Known optimized LJ LJ55 Mackay: r1≈1.101, r2≈1.846, r3≈1.968
    if scales is None:
        # Use known LJ-optimized Mackay radii
        R1, R2, R3 = 1.101, 1.846, 1.968
    else:
        R1, R2, R3 = scales

    center = np.zeros((1, 3))
    inner = R1 * vn
    outer_inner = R2 * vn   # 12 outer shell "in" vertices
    edge_mids = R3 * mid_dirs  # 30 edge-midpoint atoms

    # Wait: Mackay icosahedron 55 has shells: 1, 12, 42. The 42 = 30 edge-mids + 12 outer vertices.
    # Actually Mackay icosahedron: core 1, first shell 12, second shell 42 (which includes
    # 30 atoms at edge midpoints and 12 atoms at vertices of a larger icosahedron)
    # Common parameterization:
    # 12 atoms in shell1 at radius r1
    # 30 atoms in shell2 at radius r2 (edge midpoints)
    # 12 atoms in shell3 at radius r3 (outer vertices)
    # The 30 and 12 have slightly different radial coordinates in the relaxed LJ structure
    # Let's use the proper parameterization:
    # r1 = radius of 12 first-shell atoms
    # r2 = radius of 30 edge-mid atoms  
    # r3 = radius of 12 outer-shell atoms

    pos = np.vstack([center, inner, edge_mids, outer_inner])
    return pos, (R1, R2, R3)

def _build_mackay55(r1=1.101, r2=1.846, r3=1.968):
    """Build LJ55 Mackay icosahedron with given radii."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0

    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    vn = verts / np.linalg.norm(verts, axis=1, keepdims=True)

    d = np.zeros((12, 12))
    for i in range(12):
        for j in range(12):
            d[i, j] = np.linalg.norm(vn[i] - vn[j])
    dmin = np.min(d[d > 1e-8])
    mid_dirs = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(d[i, j] - dmin) < 1e-6:
                m = vn[i] + vn[j]
                nm = np.linalg.norm(m)
                if nm > 1e-8:
                    mid_dirs.append(m / nm)
    mid_dirs = np.array(mid_dirs)  # 30

    center = np.zeros((1, 3))
    inner12 = r1 * vn       # 12 first-shell
    edge30 = r2 * mid_dirs  # 30 edge-mids
    outer12 = r3 * vn       # 12 outer-shell

    pos = np.vstack([center, inner12, edge30, outer12])
    return pos

def _optimize_mackay_radii(deadline=None):
    """Find optimal radii for the Mackay icosahedron template using Nelder-Mead."""
    def radius_objective(params):
        r1, r2, r3 = params
        if r1 < 0.5 or r2 < 0.8 or r3 < 0.8:
            return 1e10
        pos = _build_mackay55(r1, r2, r3)
        # Check min distance
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        if np.min(dists) < 0.5:
            return 1e10
        return _lj_energy(pos)

    # Try several initial radii
    best_energy = 1e20
    best_params = [1.101, 1.846, 1.968]
    starts = [
        [1.101, 1.846, 1.968],
        [1.10, 1.85, 1.97],
        [1.15, 1.80, 1.95],
        [1.05, 1.90, 2.00],
        [1.00, 1.85, 1.98],
        [1.12, 1.82, 1.96],
        [1.08, 1.87, 1.98],
        [1.10, 1.83, 1.99],
    ]

    for s in starts:
        if deadline is not None and time.time() > deadline - 30:
            break
        try:
            res = minimize(
                radius_objective, s, method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-10, 'adaptive': True}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_params = res.x
        except Exception:
            pass

    return np.array(best_params), best_energy

def _relax_full(pos, deadline=None, maxiter=2000, gtol=1e-10):
    """Full L-BFGS relaxation on all 3N coordinates."""
    best_e = _lj_energy(pos)
    best_p = pos.copy()
    x0 = pos.flatten()
    try:
        res = minimize(
            _lj_objective_and_grad, x0, jac=True, method='L-BFGS-B',
            options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 100, 'ftol': 1e-14}
        )
        if res.fun < best_e:
            best_e = res.fun
            best_p = res.x.reshape(-1, 3)
    except Exception:
        pass

    if deadline is not None and time.time() > deadline - 8:
        return best_p, best_e
    return best_p, best_e

def _polish(pos, deadline=None):
    """Multi-stage high-precision polishing."""
    best_e = _lj_energy(pos)
    best_p = pos.copy()
    x0 = pos.flatten()

    stages = [
        {'maxiter': 3000, 'gtol': 1e-12, 'maxls': 200, 'ftol': 1e-14},
        {'maxiter': 5000, 'gtol': 1e-14, 'maxls': 300, 'ftol': 1e-16},
        {'maxiter': 8000, 'gtol': 1e-16, 'maxls': 500, 'ftol': 1e-18},
        {'maxiter': 12000, 'gtol': 1e-18, 'maxls': 800, 'ftol': 1e-20},
    ]
    for opt in stages:
        if deadline is not None and time.time() > deadline - 6:
            break
        try:
            res = minimize(
                _lj_objective_and_grad, x0, jac=True, method='L-BFGS-B',
                options=opt
            )
            if res.fun < best_e:
                best_e = res.fun
                best_p = res.x.reshape(-1, 3)
                x0 = res.x
        except Exception:
            pass
    return best_p, best_e

def _check_validity(pos):
    """Ensure atoms are at least 0.5 apart."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_d = np.min(dists)
    if min_d < 0.5:
        scale = 0.5 / min_d * 1.02
        pos = pos * scale
        # Slight re-relax
        pos, _ = _relax_full(pos, maxiter=200, gtol=1e-10)
    return pos

def construct():
    """
    LJ55 global minimum via multi-start perturbation of Mackay icosahedron + L-BFGS polish.
    """
    t0 = time.time()
    deadline = t0 + 145.0

    # Step 1: Optimize the Mackay radii
    radii, rad_e = _optimize_mackay_radii(deadline=deadline)
    r1, r2, r3 = radii
    pos_template = _build_mackay55(r1, r2, r3)
    
    # Relax the template itself
    pos_template, _ = _relax_full(pos_template, deadline=deadline, maxiter=1500, gtol=1e-10)
    best_pos = pos_template.copy()
    best_energy = _lj_energy(best_pos)

    # Step 2: Multi-start with random Gaussian perturbations
    rng = np.random.default_rng(42)
    noise_stds = [0.02, 0.03, 0.05, 0.07, 0.10, 0.13, 0.17, 0.22, 0.28, 0.35, 0.45, 0.55, 0.70, 0.90, 1.20, 1.60]

    # Also try a few perturbed radius sets
    radius_perturbs = [
        (1.0, 1.0, 1.0),
        (1.02, 0.98, 1.01),
        (0.98, 1.02, 0.99),
        (1.03, 0.97, 1.02),
        (0.97, 1.03, 0.98),
    ]

    total_starts = 0
    for rp in radius_perturbs:
        r1p, r2p, r3p = r1 * rp[0], r2 * rp[1], r3 * rp[2]
        base = _build_mackay55(r1p, r2p, r3p)

        for std in noise_stds:
            if time.time() > deadline - 12:
                break
            try:
                pos_trial = base + rng.normal(0, std, size=(55, 3))
                # Check validity
                diff = pos_trial[:, np.newaxis, :] - pos_trial[np.newaxis, :, :]
                dists = np.sqrt(np.sum(diff ** 2, axis=2))
                np.fill_diagonal(dists, np.inf)
                if np.min(dists) < 0.5:
                    continue

                # Relax
                pos_r, e_r = _relax_full(pos_trial, deadline=deadline, maxiter=1500, gtol=1e-10)
                if e_r < best_energy:
                    best_energy = e_r
                    best_pos = pos_r.copy()
                total_starts += 1
            except Exception:
                continue

    # Also try pure perturbations from the relaxed template
    for std in [0.01, 0.02, 0.04, 0.06, 0.08, 0.12, 0.18, 0.25, 0.35, 0.50, 0.70, 1.00]:
        if time.time() > deadline - 10:
            break
        try:
            pos_trial = best_pos + rng.normal(0, std, size=(55, 3))
            diff = pos_trial[:, np.newaxis, :] - pos_trial[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dists, np.inf)
            if np.min(dists) < 0.5:
                continue
            pos_r, e_r = _relax_full(pos_trial, deadline=deadline, maxiter=1200, gtol=1e-10)
            if e_r < best_energy:
                best_energy = e_r
                best_pos = pos_r.copy()
        except Exception:
            continue

    # Step 3: Final polish
    if time.time() < deadline - 5:
        best_pos, best_energy = _polish(best_pos, deadline=deadline)

    # Final validity check
    best_pos = _check_validity(best_pos)

    return best_pos
# EVOLVE-BLOCK-END
