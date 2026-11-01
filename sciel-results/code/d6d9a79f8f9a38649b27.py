"""Combined: 54D radial manifold optimization for 55-atom Mackay icosahedron,
with Basin-Hopping rigid-body displacement exploration and high-precision L-BFGS polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 55
DEADLINE_SECONDS = 115.0

# Precompute icosahedral directions
_PHI = (1.0 + np.sqrt(5.0)) / 2.0

def _get_icosahedral_directions():
    """Return 54 unit direction vectors: 12 vertex dirs, 30 edge-mid dirs, 12 vertex dirs again."""
    verts = np.array([
        [0, 1, _PHI], [0, -1, _PHI], [0, 1, -_PHI], [0, -1, -_PHI],
        [1, _PHI, 0], [-1, _PHI, 0], [1, -_PHI, 0], [-1, -_PHI, 0],
        [_PHI, 0, 1], [-_PHI, 0, 1], [_PHI, 0, -1], [-_PHI, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / norms

    # Find edges
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
    edge_mids = np.array(edge_mids)  # (30, 3)

    # Order: 12 inner-vertex dirs, 30 edge-mid dirs, 12 outer-vertex dirs
    dirs = np.vstack([verts_unit, edge_mids, verts_unit])
    return dirs  # shape (54, 3)

_DIRS = _get_icosahedral_directions()  # precomputed at module load

def _radial_to_cartesian(radii):
    """Convert 54 radial distances to 54 Cartesian positions (excluding center at origin)."""
    return radii[:, np.newaxis] * _DIRS  # shape (54, 3)

def _cartesian_to_full(pos54):
    """Add center atom at origin to make full 55-atom array."""
    return np.vstack([np.zeros(3), pos54])

def _lj_energy(pos):
    """Compute total Lennard-Jones energy for N atoms."""
    n = len(pos)
    rij = np.zeros((n, n, 3))
    for i in range(n):
        rij[i, i + 1:] = pos[i] - pos[i + 1:]
        rij[i + 1:, i] = -rij[i, i + 1:]
    r2 = np.sum(rij**2, axis=2)
    np.fill_diagonal(r2, 1.0)
    r6 = r2**(-3)
    r12 = r6**2
    energy = 4.0 * np.sum(r12 - r6)
    return energy

def _lj_energy_grad(pos):
    """Compute gradient of Lennard-Jones energy w.r.t. Cartesian coords."""
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
    """Energy as function of 54 radial distances (center fixed at origin)."""
    pos54 = _radial_to_cartesian(radii)
    pos_full = _cartesian_to_full(pos54)
    return _lj_energy(pos_full)

def _radial_grad(radii):
    """Gradient of energy w.r.t. radial distances via chain rule."""
    pos54 = _radial_to_cartesian(radii)
    pos_full = _cartesian_to_full(pos54)
    cart_grad = _lj_energy_grad(pos_full)  # shape (55, 3)
    # grad w.r.t. radii[i] = dot(cart_grad[i+1], dir[i])
    grad = np.sum(cart_grad[1:] * _DIRS, axis=1)
    return grad

def _optimize_radial_manifold(r0=None, maxiter=5000):
    """Optimize all 54 radial distances on the icosahedral direction manifold."""
    if r0 is None:
        # Initial guess: standard Mackay radii
        r_inner = 1.054
        r_outer_mid = 1.755
        r_outer_vert = 2.0
        r0 = np.concatenate([
            np.full(12, r_inner),
            np.full(30, r_outer_mid),
            np.full(12, r_outer_vert)
        ])

    # Bounds: ensure minimum separation >= 0.5
    lb = np.concatenate([np.full(12, 0.5), np.full(30, 0.5), np.full(12, 0.5)])
    ub = np.concatenate([np.full(12, 3.0), np.full(30, 3.5), np.full(12, 3.5)])

    result = minimize(
        _radial_energy,
        r0,
        jac=_radial_grad,
        bounds=list(zip(lb, ub)),
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'gtol': 1e-13, 'ftol': 1e-16, 'maxls': 50}
    )
    return result.x, result.fun

def _polish_cartesian(pos0, maxiter=10000, gtol=1e-14, ftol=1e-16):
    """High-precision L-BFGS polish in full 165D Cartesian space."""
    def energy_func(x):
        return _lj_energy(x.reshape(-1, 3))

    def grad_func(x):
        return _lj_energy_grad(x.reshape(-1, 3)).flatten()

    result = minimize(
        energy_func,
        pos0.flatten(),
        jac=grad_func,
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'gtol': gtol, 'ftol': ftol, 'maxls': 50}
    )
    return result.x.reshape(-1, 3), result.fun

def _basin_hopping_step(pos, rng, step_scale=0.05):
    """
    Perform a single basin-hopping step:
    1. Apply a rigid-body displacement (translation) to the entire cluster.
    2. Relax to the nearest local minimum using L-BFGS.
    Returns the relaxed position and energy.
    """
    # Rigid body displacement: small random translation
    displacement = rng.normal(0, step_scale, 3)
    pos_displaced = pos + displacement
    
    # Relax to nearest local minimum
    try:
        pos_relaxed, e_relaxed = _polish_cartesian(
            pos_displaced, 
            maxiter=2000, 
            gtol=1e-12, 
            ftol=1e-14
        )
        return pos_relaxed, e_relaxed
    except Exception:
        return pos, _lj_energy(pos)

def construct():
    start_time = time.time()
    best_pos = None
    best_energy = np.inf

    # Step 1: Optimize on the 54D radial manifold
    try:
        radii_opt, rad_energy = _optimize_radial_manifold()
        pos_from_radial = _cartesian_to_full(_radial_to_cartesian(radii_opt))
        best_pos = pos_from_radial
        best_energy = rad_energy
    except Exception:
        # Fallback: use standard Mackay radii
        r_inner, r_outer_mid, r_outer_vert = 1.054, 1.755, 2.0
        r0 = np.concatenate([
            np.full(12, r_inner),
            np.full(30, r_outer_mid),
            np.full(12, r_outer_vert)
        ])
        best_pos = _cartesian_to_full(_radial_to_cartesian(r0))
        best_energy = _lj_energy(best_pos)

    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS - 20:
        return best_pos

    # Step 2: High-precision Cartesian polish
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

    # Step 3: Basin-Hopping with rigid-body displacements
    # This explores the landscape by moving the entire cluster and relaxing
    # to find potentially better local minima
    np.random.seed(42)
    rng = np.random.RandomState(42)
    
    remaining_time = DEADLINE_SECONDS - (time.time() - start_time) - 5
    if remaining_time > 10:
        # Adaptive step size: start moderate, decrease if no improvement
        step_scale = 0.05
        no_improve_count = 0
        max_no_improve = 3
        
        # Estimate how many steps we can afford
        # Each step takes roughly 2-5 seconds for relaxation
        estimated_steps_per_second = 0.3  # conservative
        n_attempts = min(12, int(remaining_time * estimated_steps_per_second))
        
        for attempt in range(n_attempts):
            if time.time() - start_time > DEADLINE_SECONDS - 5:
                break
            
            # Adaptive step size: increase if stuck, decrease if improving
            if no_improve_count >= max_no_improve:
                step_scale *= 1.5
                no_improve_count = 0
            elif attempt > 0:
                step_scale *= 0.9  # slight decay
            
            pos_trial, e_trial = _basin_hopping_step(best_pos, rng, step_scale)
            
            if e_trial < best_energy - 1e-12:
                best_energy = e_trial
                best_pos = pos_trial
                no_improve_count = 0
            else:
                no_improve_count += 1

    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS - 5:
        return best_pos

    # Step 4: Final ultra-precise polish
    try:
        pos_final, e_final = _polish_cartesian(best_pos, maxiter=5000, gtol=1e-15, ftol=1e-17)
        if e_final < best_energy:
            best_energy = e_final
            best_pos = pos_final
    except Exception:
        pass

    return best_pos
