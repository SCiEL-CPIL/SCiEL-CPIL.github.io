"""55-atom LJ cluster: Mackay icosahedron construction with optimized shell radii,
full Cartesian L-BFGS relaxation, and a final high-precision polish.
Keeps within the current design family (parametric Mackay + Cartesian relaxation)."""
import numpy as np
import time
from scipy.optimize import minimize

N = 55
DEADLINE_SECONDS = 110.0
_BH_SEED = 42

def _lj_energy(pos):
    """Total LJ energy: 4*sum_{i<j}(r^-12 - r^-6)."""
    rij = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(rij**2, axis=2)
    np.fill_diagonal(r2, 1.0)
    r6 = r2**(-3)
    r12 = r6**2
    # sum over all ordered pairs = 2 * sum over unordered pairs
    # energy = 4 * sum_{i<j} (r^-12 - r^-6) = 2 * sum_{i!=j} (r^-12 - r^-6)
    return 2.0 * np.sum(r12 - r6)

def _lj_energy_grad(pos):
    """Gradient of LJ energy."""
    rij = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(rij**2, axis=2)
    np.fill_diagonal(r2, 1.0)
    # E = 2*sum_{i!=j}(r2^-6 - r2^-3)
    # dE/dx_i = 4*sum_{j!=i}(3*r2^-4 - 6*r2^-7)*(x_i - x_j)
    coeff = 4.0 * (3.0 * r2**(-4) - 6.0 * r2**(-7))
    np.fill_diagonal(coeff, 0.0)
    return np.sum(coeff[:, :, np.newaxis] * rij, axis=1)

def _get_mackay_directions():
    """Return (verts_unit (12,3), edge_mids_unit (30,3))."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / norms

    rij = verts_unit[:, np.newaxis, :] - verts_unit[np.newaxis, :, :]
    dists = np.sqrt(np.sum(rij**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    edge_mask = dists < min_dist * 1.001

    edge_mids = []
    for i in range(12):
        for j in range(i + 1, 12):
            if edge_mask[i, j]:
                mid = (verts_unit[i] + verts_unit[j]) / 2.0
                edge_mids.append(mid / np.linalg.norm(mid))
    edge_mids = np.array(edge_mids)
    return verts_unit, edge_mids

_VERTS, _EDGES = _get_mackay_directions()

def _mackay_pos(r_in, r_edge, r_vert):
    """Build 55-atom Mackay icosahedron with three shell radii."""
    pts = [np.zeros(3)]
    for v in _VERTS:
        pts.append(r_in * v)
    for v in _EDGES:
        pts.append(r_edge * v)
    for v in _VERTS:
        pts.append(r_vert * v)
    return np.array(pts, dtype=float)

def _optimize_radii():
    """Optimize the three shell radii via Nelder-Mead."""
    def obj(p):
        ri, re, rv = p
        if ri < 0.3 or re < 0.5 or rv < 0.5:
            return 1e10
        if re < ri * 0.85 or rv < ri * 0.85:
            return 1e10
        return _lj_energy(_mackay_pos(ri, re, rv))

    r0 = np.array([1.054, 1.755, 2.000])
    res = minimize(obj, r0, method='Nelder-Mead',
                   options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14,
                            'adaptive': True})
    return res.x, res.fun

def _relax_cartesian(pos0, maxiter=8000, gtol=1e-12, ftol=1e-16, deadline=None):
    """L-BFGS-B Cartesian relaxation with optional deadline guard."""
    def ef(x):
        return _lj_energy(x.reshape(-1, 3))
    def gf(x):
        return _lj_energy_grad(x.reshape(-1, 3)).flatten()
    opts = {'maxiter': maxiter, 'gtol': gtol, 'ftol': ftol,
            'maxls': 40, 'iprint': -1}
    if deadline is not None:
        opts['maxfun'] = 2000
    res = minimize(ef, pos0.flatten(), jac=gf, method='L-BFGS-B', options=opts)
    return res.x.reshape(-1, 3), res.fun

def _basin_hopping(pos0, t0, deadline, rng):
    """Basin-hopping: rigid-body perturbations + L-BFGS relaxation.
    Each step displaces the entire cluster by a small random vector,
    then relaxes to the nearest local minimum. Accepts lower-energy
    basins with Metropolis criterion at fixed temperature."""
    best_pos = pos0.copy()
    best_e = _lj_energy(pos0)
    current_pos = pos0.copy()
    current_e = best_e
    T = 0.5  # Metropolis temperature (LJ energy units)
    max_step = 0.15  # max rigid displacement magnitude
    
    elapsed = time.time() - t0
    while time.time() - t0 < deadline - 5.0:
        # Random rigid-body displacement
        disp = rng.standard_normal(3) * max_step
        perturbed = current_pos + disp
        
        # Relax to local minimum
        try:
            new_pos, new_e = _relax_cartesian(perturbed, maxiter=2000,
                                               gtol=1e-10, ftol=1e-14)
        except Exception:
            break
        
        # Check minimum distance constraint
        rij = new_pos[:, np.newaxis, :] - new_pos[np.newaxis, :, :]
        r2 = np.sum(rij**2, axis=2)
        np.fill_diagonal(r2, np.inf)
        if np.min(r2) < 0.25:  # 0.5^2
            continue
        
        # Metropolis acceptance
        delta_e = new_e - current_e
        if delta_e < 0 or (T > 0 and rng.random() < np.exp(-delta_e / T)):
            current_pos = new_pos
            current_e = new_e
        
        # Update best
        if new_e < best_e:
            best_e = new_e
            best_pos = new_pos.copy()
        
        # Gradually reduce temperature (annealing)
        T *= 0.95
        
        elapsed = time.time() - t0
        if elapsed > deadline - 10:
            break
    
    return best_pos, best_e

def construct():
    t0 = time.time()
    rng = np.random.default_rng(_BH_SEED)

    # Step 1: optimize shell radii
    try:
        radii, _ = _optimize_radii()
    except Exception:
        radii = np.array([1.054, 1.755, 2.000])

    pos0 = _mackay_pos(radii[0], radii[1], radii[2])

    # Step 2: main Cartesian relaxation
    elapsed = time.time() - t0
    if elapsed > DEADLINE_SECONDS - 30:
        return pos0

    try:
        pos1, e1 = _relax_cartesian(pos0, maxiter=4000, gtol=1e-10, ftol=1e-14)
    except Exception:
        pos1, e1 = pos0, _lj_energy(pos0)

    # Step 3: Basin-hopping with rigid-body perturbations
    elapsed = time.time() - t0
    remaining = DEADLINE_SECONDS - elapsed - 8
    if remaining > 15:
        try:
            bh_pos, bh_e = _basin_hopping(pos1, t0, DEADLINE_SECONDS - 8, rng)
            if bh_e < e1:
                pos1, e1 = bh_pos, bh_e
        except Exception:
            pass

    # Step 4: high-precision polish
    elapsed = time.time() - t0
    remaining = DEADLINE_SECONDS - elapsed - 3
    if remaining > 5:
        try:
            pos2, e2 = _relax_cartesian(pos1, maxiter=4000, gtol=1e-13, ftol=1e-16)
            if e2 < e1:
                pos1, e1 = pos2, e2
        except Exception:
            pass

    return pos1
