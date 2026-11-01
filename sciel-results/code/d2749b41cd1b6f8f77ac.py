"""Combined: 54D radial manifold optimization for 55-atom Mackay icosahedron,
with multi-start perturbation exploration and high-precision L-BFGS polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 55
DEADLINE_SECONDS = 112.0

_PHI = (1.0 + np.sqrt(5.0)) / 2.0

def _get_icosahedral_directions():
    """Return 54 unit direction vectors: 12 vertex, 30 edge-mid, 12 vertex."""
    verts = np.array([
        [0, 1, _PHI], [0, -1, _PHI], [0, 1, -_PHI], [0, -1, -_PHI],
        [1, _PHI, 0], [-1, _PHI, 0], [1, -_PHI, 0], [-1, -_PHI, 0],
        [_PHI, 0, 1], [-_PHI, 0, 1], [_PHI, 0, -1], [-_PHI, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / norms

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

    dirs = np.vstack([verts_unit, edge_mids, verts_unit])
    return dirs

_DIRS = _get_icosahedral_directions()

def _radial_to_cartesian(radii):
    return radii[:, np.newaxis] * _DIRS

def _cartesian_to_full(pos54):
    return np.vstack([np.zeros(3), pos54])

def _lj_energy(pos):
    n = len(pos)
    rij = np.zeros((n, n, 3))
    for i in range(n):
        rij[i, i + 1:] = pos[i] - pos[i + 1:]
        rij[i + 1:, i] = -rij[i, i + 1:]
    r2 = np.sum(rij**2, axis=2)
    np.fill_diagonal(r2, 1.0)
    r6 = r2**(-3)
    r12 = r6**2
    return 4.0 * np.sum(r12 - r6)

def _lj_energy_grad(pos):
    n = len(pos)
    rij = np.zeros((n, n, 3))
    for i in range(n):
        rij[i, i + 1:] = pos[i] - pos[i + 1:]
        rij[i + 1:, i] = -rij[i, i + 1:]
    r2 = np.sum(rij**2, axis=2)
    np.fill_diagonal(r2, 1.0)
    r6 = r2**(-3)
    r12 = r6**2
    coeff = 4.0 * (-12.0 * r2**(-7) + 6.0 * r2**(-4))
    np.fill_diagonal(coeff, 0.0)
    grad = 2.0 * np.sum(coeff[:, :, np.newaxis] * rij, axis=1)
    return grad

def _radial_energy(radii):
    pos54 = _radial_to_cartesian(radii)
    pos_full = _cartesian_to_full(pos54)
    return _lj_energy(pos_full)

def _radial_grad(radii):
    pos54 = _radial_to_cartesian(radii)
    pos_full = _cartesian_to_full(pos54)
    cart_grad = _lj_energy_grad(pos_full)
    return np.sum(cart_grad[1:] * _DIRS, axis=1)

def _optimize_radial_manifold(r0=None, maxiter=5000):
    if r0 is None:
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
        _radial_energy, r0, jac=_radial_grad,
        bounds=list(zip(lb, ub)),
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'gtol': 1e-13, 'ftol': 1e-16, 'maxls': 50}
    )
    return result.x, result.fun

def _polish_cartesian(pos0, maxiter=10000, gtol=1e-15, ftol=1e-18):
    def energy_func(x):
        return _lj_energy(x.reshape(-1, 3))
    def grad_func(x):
        return _lj_energy_grad(x.reshape(-1, 3)).flatten()
    result = minimize(
        energy_func, pos0.flatten(), jac=grad_func,
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'gtol': gtol, 'ftol': ftol, 'maxls': 50}
    )
    return result.x.reshape(-1, 3), result.fun

def construct():
    start_time = time.time()
    best_pos = None
    best_energy = np.inf

    # Step 1: Radial manifold optimization
    try:
        radii_opt, rad_energy = _optimize_radial_manifold()
        best_pos = _cartesian_to_full(_radial_to_cartesian(radii_opt))
        best_energy = rad_energy
    except Exception:
        r0 = np.concatenate([
            np.full(12, 1.054), np.full(30, 1.755), np.full(12, 2.0)
        ])
        best_pos = _cartesian_to_full(_radial_to_cartesian(r0))
        best_energy = _lj_energy(best_pos)

    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS - 20:
        return best_pos

    # Step 2: Cartesian L-BFGS polish
    try:
        pos_polished, e_polished = _polish_cartesian(best_pos, maxiter=10000, gtol=1e-14, ftol=1e-16)
        if e_polished < best_energy:
            best_energy = e_polished
            best_pos = pos_polished
    except Exception:
        pass

    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS - 10:
        return best_pos

    # Step 3: Multi-start perturbation exploration
    np.random.seed(42)
    remaining = DEADLINE_SECONDS - elapsed - 8
    if remaining > 5:
        # Try several small perturbations of individual atoms
        n_attempts = min(8, int(remaining / 8))
        for attempt in range(n_attempts):
            if time.time() - start_time > DEADLINE_SECONDS - 5:
                break
            # Perturb a random subset of atoms
            n_perturb = np.random.randint(1, 6)
            indices = np.random.choice(N, size=n_perturb, replace=False)
            noise_scale = 0.03 + 0.02 * attempt
            pos_perturbed = best_pos.copy()
            pos_perturbed[indices] += np.random.normal(0, noise_scale, (n_perturb, 3))
            
            try:
                pos_trial, e_trial = _polish_cartesian(
                    pos_perturbed, maxiter=3000, gtol=1e-13, ftol=1e-15
                )
                if e_trial < best_energy - 1e-12:
                    best_energy = e_trial
                    best_pos = pos_trial
            except Exception:
                pass

    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS - 4:
        return best_pos

    # Step 4: Final ultra-precise polish
    try:
        pos_final, e_final = _polish_cartesian(best_pos, maxiter=5000, gtol=1e-16, ftol=1e-18)
        if e_final < best_energy:
            best_energy = e_final
            best_pos = pos_final
    except Exception:
        pass

    return best_pos
