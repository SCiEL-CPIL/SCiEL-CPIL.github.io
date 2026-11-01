"""Mackay icosahedron 55-atom LJ cluster: analytic construction + high-precision L-BFGS polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 55
DEADLINE_SECONDS = 115.0

# Golden ratio
_PHI = (1.0 + np.sqrt(5.0)) / 2.0

def _get_icosahedral_directions():
    """Return 54 unit direction vectors for the two-shell Mackay icosahedron.
    Order: 12 inner vertices, 30 edge midpoints, 12 outer vertices."""
    verts = np.array([
        [0, 1, _PHI], [0, -1, _PHI], [0, 1, -_PHI], [0, -1, -_PHI],
        [1, _PHI, 0], [-1, _PHI, 0], [1, -_PHI, 0], [-1, -_PHI, 0],
        [_PHI, 0, 1], [-_PHI, 0, 1], [_PHI, 0, -1], [-_PHI, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / norms

    # Find edges by minimum distance
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts_unit[i] - verts_unit[j])
            dists[i, j] = d
            dists[j, i] = d
    min_dist = np.min(dists[dists > 0])
    edge_mask = dists < min_dist * 1.001

    edge_mids = []
    for i in range(12):
        for j in range(i + 1, 12):
            if edge_mask[i, j]:
                mid = (verts_unit[i] + verts_unit[j]) / 2.0
                mid_unit = mid / np.linalg.norm(mid)
                edge_mids.append(mid_unit)
    edge_mids = np.array(edge_mids)

    return np.vstack([verts_unit, edge_mids, verts_unit])

_DIRS = _get_icosahedral_directions()

def _radial_to_cartesian(radii):
    return radii[:, np.newaxis] * _DIRS

def _cartesian_to_full(pos54):
    return np.vstack([np.zeros(3), pos54])

def _lj_energy(pos):
    """Total LJ energy using vectorized pair computation."""
    n = len(pos)
    # Compute all pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff**2, axis=2)
    np.fill_diagonal(r2, 1.0)  # Avoid division by zero
    r6 = r2**(-3.0)
    r12 = r6**2
    return 4.0 * np.sum(r12 - r6) / 2.0  # Divide by 2 to avoid double counting

def _lj_energy_grad(pos):
    """Gradient of LJ energy w.r.t. Cartesian coordinates."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff**2, axis=2)
    np.fill_diagonal(r2, 1.0)
    r6 = r2**(-3.0)
    r12 = r6**2
    # dE/dr2 = 4*(-6*r2^(-7) + 3*r2^(-4)) for each pair, times 2 for both i,j
    coeff = 4.0 * (-6.0 * r2**(-7.0) + 3.0 * r2**(-4.0))
    np.fill_diagonal(coeff, 0.0)
    grad = 2.0 * np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return grad

def _radial_energy(radii):
    pos54 = _radial_to_cartesian(radii)
    pos_full = _cartesian_to_full(pos54)
    return _lj_energy(pos_full)

def _radial_grad(radii):
    pos54 = _radial_to_cartesian(radii)
    pos_full = _cartesian_to_full(pos54)
    cart_grad = _lj_energy_grad(pos_full)
    grad = np.sum(cart_grad[1:] * _DIRS, axis=1)
    return grad

def _optimize_radial_manifold(r0=None, maxiter=10000, gtol=1e-13, ftol=1e-15):
    """Optimize radial distances on the icosahedral manifold."""
    if r0 is None:
        # Better initial estimates based on known Mackay structure
        r_inner = 1.054
        r_outer_mid = 1.755
        r_outer_vert = 2.0
        r0 = np.concatenate([
            np.full(12, r_inner),
            np.full(30, r_outer_mid),
            np.full(12, r_outer_vert)
        ])

    lb = np.concatenate([np.full(12, 0.5), np.full(30, 0.5), np.full(12, 0.5)])
    ub = np.concatenate([np.full(12, 3.0), np.full(30, 3.5), np.full(12, 3.5)])

    result = minimize(
        _radial_energy,
        r0,
        jac=_radial_grad,
        bounds=list(zip(lb, ub)),
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'gtol': gtol, 'ftol': ftol, 'maxls': 50, 'maxcor': 30}
    )
    return result.x, result.fun

def _polish_cartesian(pos0, maxiter=20000, gtol=1e-14, ftol=1e-16):
    """High-precision L-BFGS polish in full Cartesian space."""
    def energy_func(x):
        return _lj_energy(x.reshape(-1, 3))

    def grad_func(x):
        return _lj_energy_grad(x.reshape(-1, 3)).flatten()

    result = minimize(
        energy_func,
        pos0.flatten(),
        jac=grad_func,
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'gtol': gtol, 'ftol': ftol, 'maxls': 50, 'maxcor': 30}
    )
    return result.x.reshape(-1, 3), result.fun

def _construct_mackay_icosahedron():
    """Construct the analytic Mackay icosahedron with refined initial radii."""
    # Use slightly adjusted radii as starting point for better convergence
    r_inner = 1.054
    r_outer_mid = 1.755
    r_outer_vert = 2.0
    radii = np.concatenate([
        np.full(12, r_inner),
        np.full(30, r_outer_mid),
        np.full(12, r_outer_vert)
    ])
    pos54 = _radial_to_cartesian(radii)
    return _cartesian_to_full(pos54)

def construct():
    start_time = time.time()
    best_pos = None
    best_energy = np.inf

    # Step 1: Construct Mackay icosahedron
    try:
        pos_mackay = _construct_mackay_icosahedron()
        e_mackay = _lj_energy(pos_mackay)
        best_pos = pos_mackay
        best_energy = e_mackay
    except Exception:
        return None

    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS - 30:
        return best_pos

    # Step 2: Radial manifold optimization (faster, 54D)
    try:
        radii_opt, rad_energy = _optimize_radial_manifold(maxiter=5000, gtol=1e-13, ftol=1e-15)
        pos_from_radial = _cartesian_to_full(_radial_to_cartesian(radii_opt))
        if rad_energy < best_energy:
            best_energy = rad_energy
            best_pos = pos_from_radial
    except Exception:
        pass

    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS - 20:
        return best_pos

    # Step 3: Full Cartesian polish (high precision)
    try:
        pos_polished, e_polished = _polish_cartesian(best_pos, maxiter=20000, gtol=1e-14, ftol=1e-16)
        if e_polished < best_energy:
            best_energy = e_polished
            best_pos = pos_polished
    except Exception:
        pass

    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS - 10:
        return best_pos

    # Step 4: Final ultra-high-precision polish
    try:
        pos_final, e_final = _polish_cartesian(best_pos, maxiter=10000, gtol=1e-15, ftol=1e-17)
        if e_final < best_energy:
            best_energy = e_final
            best_pos = pos_final
    except Exception:
        pass

    return best_pos
