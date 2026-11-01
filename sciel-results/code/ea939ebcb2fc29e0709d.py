"""Combined Lennard-Jones cluster global minimum, 55 atoms.
Uses a coarse-grained radial parameterization of the Mackay icosahedron 
to find a near-optimal structure, followed by high-precision Cartesian polish."""
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

N = 55
DEADLINE_SECONDS = 140.0
_start_time = time.time()

def _remaining():
    return DEADLINE_SECONDS - (time.time() - _start_time)

# Precompute icosahedral directions
_PHI = (1.0 + np.sqrt(5.0)) / 2.0

def _get_icosahedral_directions():
    """
    Returns three sets of unit vectors:
    1. v_inner: 12 vertices of the icosahedron
    2. v_edge: 30 edge midpoints
    3. v_outer: 12 vertices (same directions as v_inner)
    """
    verts = np.array([
        [0, 1, _PHI], [0, -1, _PHI], [0, 1, -_PHI], [0, -1, -_PHI],
        [1, _PHI, 0], [-1, _PHI, 0], [1, -_PHI, 0], [-1, -_PHI, 0],
        [_PHI, 0, 1], [-_PHI, 0, 1], [_PHI, 0, -1], [-_PHI, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    v_inner = verts / norms
    
    # Find edges
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(v_inner[i] - v_inner[j])
            dists[i, j] = d
            dists[j, i] = d
    min_dist = np.min(dists[dists > 0])
    edge_mask = dists < min_dist * 1.001

    v_edge = []
    for i in range(12):
        for j in range(i + 1, 12):
            if edge_mask[i, j]:
                mid = (v_inner[i] + v_inner[j]) / 2.0
                mid_unit = mid / np.linalg.norm(mid)
                v_edge.append(mid_unit)
    v_edge = np.array(v_edge)
    
    # Outer vertices have same directions as inner in the standard Mackay model
    v_outer = v_inner.copy()
    
    return v_inner, v_edge, v_outer

V_INNER, V_EDGE, V_OUTER = _get_icosahedral_directions()

def _lj_energy(pos):
    """Compute total Lennard-Jones energy for N atoms."""
    n = len(pos)
    # Use vectorized calculation for speed and precision
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0) # Avoid division by zero
    r6 = r2 ** (-3)
    r12 = r6 ** 2
    energy = 4.0 * np.sum(r12 - r6)
    return energy

def _lj_energy_grad(pos):
    """Compute gradient of Lennard-Jones energy w.r.t. Cartesian coords."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    r6 = r2 ** (-3)
    r12 = r6 ** 2
    # Gradient of 4(r^-12 - r^-6) is 4(-12r^-14 + 6r^-8) * (r_vec / r)
    # = 4(-12 r^-14 + 6 r^-8) * r_vec * r^-1
    # = 4(-12 r^-15 + 6 r^-9) * r_vec ... wait, let's use r2 form
    # d/dr 4(r^-12 - r^-6) = 4(-12r^-13 + 6r^-7)
    # grad = sum_j 4(-12 r_ij^-13 + 6 r_ij^-7) * (r_i - r_j) / r_ij
    # = sum_j 4(-12 r_ij^-14 + 6 r_ij^-8) * (r_i - r_j)
    # In terms of r2: r^-14 = r2^-7, r^-8 = r2^-4
    coeff = 4.0 * (-12.0 * r2**(-7) + 6.0 * r2**(-4))
    np.fill_diagonal(coeff, 0.0)
    # grad_i = sum_j coeff_ij * (r_i - r_j)
    # We can compute this as: sum_j coeff_ij * r_i - sum_j coeff_ij * r_j
    # = r_i * sum_j coeff_ij - sum_j (coeff_ij * r_j)
    sum_coeff = np.sum(coeff, axis=1, keepdims=True)
    sum_coeff_pos = np.sum(coeff[:, :, np.newaxis] * pos[np.newaxis, :, :], axis=1)
    grad = pos * sum_coeff - sum_coeff_pos
    return grad

def _params_to_cartesian(params):
    """
    Convert 3 radial parameters to 55 Cartesian coordinates.
    params: [r_inner, r_edge, r_outer]
    """
    r_inner, r_edge, r_outer = params
    center = np.zeros((1, 3))
    pos_inner = V_INNER * r_inner
    pos_edge = V_EDGE * r_edge
    pos_outer = V_OUTER * r_outer
    return np.vstack([center, pos_inner, pos_edge, pos_outer])

def _energy_from_params(params):
    """Energy as function of 3 radial parameters."""
    pos = _params_to_cartesian(params)
    # Check min distance constraint (soft penalty if violated, though unlikely in this range)
    from scipy.spatial.distance import pdist
    d = pdist(pos)
    if d.min() < 0.5:
        return 1e10
    return _lj_energy(pos)

def _optimize_radial_params():
    """
    Optimize the 3 radial parameters using a multi-start Nelder-Mead approach.
    """
    if _remaining() < 10.0:
        return None, np.inf
        
    # Initial guesses based on literature values for LJ55
    # Typical radii: inner ~1.04, edge ~1.76, outer ~1.98
    starts = [
        [1.04, 1.76, 1.98],
        [1.00, 1.70, 1.90],
        [1.10, 1.80, 2.00],
        [1.05, 1.75, 1.95],
        [0.95, 1.65, 1.85],
        [1.04, 1.70, 1.98],
        [1.04, 1.80, 1.98],
        [1.135, 1.162, 1.283], # From current program's guess, might be local min
    ]
    
    best_params = None
    best_energy = np.inf
    
    for start in starts:
        if _remaining() < 5.0:
            break
        try:
            res = minimize(
                _energy_from_params,
                start,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_params = res.x
        except Exception:
            continue
            
    if best_params is None:
        return np.array([1.04, 1.76, 1.98]), _energy_from_params([1.04, 1.76, 1.98])
        
    return best_params, best_energy

def _polish_cartesian(pos0, maxiter=20000, gtol=1e-14, ftol=1e-16):
    """High-precision L-BFGS polish in full 165D Cartesian space."""
    if _remaining() < 5.0:
        return pos0, _lj_energy(pos0)

    def energy_func(x):
        return _lj_energy(x.reshape(-1, 3))

    def grad_func(x):
        return _lj_energy_grad(x.reshape(-1, 3)).flatten()

    try:
        result = minimize(
            energy_func,
            pos0.flatten(),
            jac=grad_func,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'gtol': gtol, 'ftol': ftol, 'maxls': 50}
        )
        return result.x.reshape(-1, 3), result.fun
    except Exception:
        return pos0, _lj_energy(pos0)

def _check_min_dist(pos, min_d=0.5):
    """Check all pair distances >= min_d."""
    from scipy.spatial.distance import pdist
    d = pdist(pos)
    return d.min() >= min_d - 1e-9

def construct():
    global _start_time
    _start_time = time.time()
    
    best_pos = None
    best_energy = np.inf
    
    # Step 1: Optimize the 3 radial parameters
    params, e_params = _optimize_radial_params()
    
    if params is not None:
        pos_initial = _params_to_cartesian(params)
        if _check_min_dist(pos_initial):
            best_energy = e_params
            best_pos = pos_initial.copy()
    
    # Fallback if radial optimization fails or yields invalid structure
    if best_pos is None:
        # Use a reasonable default Mackay structure
        default_params = np.array([1.04, 1.76, 1.98])
        best_pos = _params_to_cartesian(default_params)
        best_energy = _lj_energy(best_pos)
        
    # Step 2: High-precision Cartesian polish
    # This refines the structure to the exact local minimum
    if _remaining() > 10.0:
        pos_polished, e_polished = _polish_cartesian(best_pos, maxiter=10000, gtol=1e-13, ftol=1e-15)
        if _check_min_dist(pos_polished) and e_polished < best_energy:
            best_energy = e_polished
            best_pos = pos_polished.copy()
            
    # Step 3: Even tighter polish if time permits
    if _remaining() > 5.0:
        pos_final, e_final = _polish_cartesian(best_pos, maxiter=20000, gtol=1e-15, ftol=1e-17)
        if _check_min_dist(pos_final) and e_final < best_energy:
            best_energy = e_final
            best_pos = pos_final.copy()
            
    # Final safety check
    if not _check_min_dist(best_pos):
        # If constraint violated, try to fix by scaling or return fallback
        # For LJ55 Mackay, this should not happen
        from scipy.spatial.distance import pdist
        d = pdist(best_pos)
        min_d = d.min()
        if min_d < 0.5:
            scale = 0.5 / min_d * 0.99
            best_pos = best_pos * scale
            
    return best_pos
