"""Lennard-Jones 55-atom cluster: radial manifold optimization + high-precision Cartesian polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 55

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

N = 55

# Deadline guard: evaluator kills at 150s wall-clock; use 115s as safety margin
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
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, 1.0)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    energy = 4.0 * (r12 - r6).sum() / 2.0
    return energy

def _lj_energy_grad(pos):
    """Compute gradient of Lennard-Jones energy w.r.t. Cartesian coords."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, 1.0)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    coeff = 4.0 * (-12.0 * r2 ** (-7) + 6.0 * r2 ** (-4))
    np.fill_diagonal(coeff, 0.0)
    grad = np.einsum('ij,ijk->ik', coeff, diff)
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

def _optimize_radial_manifold(r0=None, maxiter=5000, gtol=1e-12, ftol=1e-15):
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
        options={'maxiter': maxiter, 'gtol': gtol, 'ftol': ftol, 'maxls': 50}
    )
    return result.x, result.fun

def _optimize_radial_two_pass():
    """Two-pass radial optimization: first inner shell, then outer shell, then joint."""
    # Pass 1: Optimize inner shell radii (12 atoms) with outer fixed at Mackay values
    r_inner_init = np.full(12, 1.054)
    r_outer_fixed = np.concatenate([np.full(30, 1.755), np.full(12, 2.0)])
    
    def inner_energy(r_in):
        r_full = np.concatenate([r_in, r_outer_fixed])
        return _radial_energy(r_full)
    
    def inner_grad(r_in):
        r_full = np.concatenate([r_in, r_outer_fixed])
        g_full = _radial_grad(r_full)
        return g_full[:12]
    
    res1 = minimize(inner_energy, r_inner_init, jac=inner_grad,
                    bounds=[(0.5, 3.0)] * 12, method='L-BFGS-B',
                    options={'maxiter': 3000, 'gtol': 1e-12, 'ftol': 1e-15})
    r_inner_opt = res1.x
    
    # Pass 2: Optimize outer shell radii (42 atoms) with inner fixed at optimized values
    r_inner_fixed = r_inner_opt
    r_outer_init = np.concatenate([np.full(30, 1.755), np.full(12, 2.0)])
    
    def outer_energy(r_out):
        r_full = np.concatenate([r_inner_fixed, r_out])
        return _radial_energy(r_full)
    
    def outer_grad(r_out):
        r_full = np.concatenate([r_inner_fixed, r_out])
        g_full = _radial_grad(r_full)
        return g_full[12:]
    
    res2 = minimize(outer_energy, r_outer_init, jac=outer_grad,
                    bounds=[(0.5, 3.5)] * 42, method='L-BFGS-B',
                    options={'maxiter': 3000, 'gtol': 1e-12, 'ftol': 1e-15})
    r_outer_opt = res2.x
    
    # Pass 3: Joint optimization from the two-pass result
    r_combined = np.concatenate([r_inner_opt, r_outer_opt])
    lb = np.concatenate([np.full(12, 0.5), np.full(30, 0.5), np.full(12, 0.5)])
    ub = np.concatenate([np.full(12, 3.0), np.full(30, 3.5), np.full(12, 3.5)])
    
    res3 = minimize(_radial_energy, r_combined, jac=_radial_grad,
                    bounds=list(zip(lb, ub)), method='L-BFGS-B',
                    options={'maxiter': 5000, 'gtol': 1e-12, 'ftol': 1e-15})
    
    return res3.x, res3.fun

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

def _make_mackay_icosahedron():
    """Generate a Mackay icosahedral cluster with 55 atoms using radial manifold."""
    r_inner = 1.054
    r_outer_mid = 1.755
    r_outer_vert = 2.0
    r0 = np.concatenate([
        np.full(12, r_inner),
        np.full(30, r_outer_mid),
        np.full(12, r_outer_vert)
    ])
    pos54 = _radial_to_cartesian(r0)
    return _cartesian_to_full(pos54)

def _make_perturbed_mackay(seed, scale=0.05):
    """Generate a perturbed Mackay icosahedron."""
    base = _make_mackay_icosahedron()
    rng = np.random.RandomState(seed)
    pos = base + rng.randn(N, 3) * scale
    return pos

def _make_cubic_lattice():
    """Generate a cubic lattice fragment."""
    side = int(np.ceil(N ** (1 / 3)))
    points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
    points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
    return np.array(points[:N], dtype=float) * 1.12

def _make_random_cluster(seed):
    """Generate a random cluster."""
    rng = np.random.RandomState(seed)
    pos = rng.randn(N, 3) * 2.0
    return pos

def _relax(pos, maxiter=500):
    """Relax using L-BFGS-B."""
    def fun(x):
        x = x.reshape(N, 3)
        e = _lj_energy(x)
        g = _lj_energy_grad(x).flatten()
        return e, g
    
    try:
        result = minimize(fun, pos.reshape(-1), method='L-BFGS-B', jac=True, 
                         options={'maxiter': maxiter, 'gtol': 1e-8, 'ftol': 1e-10})
        return result.x.reshape(N, 3), result.fun
    except Exception:
        return pos, _lj_energy(pos)

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    best_pos = _make_mackay_icosahedron()
    best_e = _lj_energy(best_pos)
    
    # Step 1: Optimize on the 54D radial manifold (most promising path)
    try:
        radii_opt, rad_energy = _optimize_radial_manifold(maxiter=5000, gtol=1e-12, ftol=1e-15)
        pos_from_radial = _cartesian_to_full(_radial_to_cartesian(radii_opt))
        if rad_energy < best_e:
            best_e = rad_energy
            best_pos = pos_from_radial
    except Exception:
        pass
    
    # Step 1b: Two-pass radial optimization (inner/outer decoupling)
    try:
        radii_opt2, rad_energy2 = _optimize_radial_two_pass()
        pos_from_radial2 = _cartesian_to_full(_radial_to_cartesian(radii_opt2))
        if rad_energy2 < best_e:
            best_e = rad_energy2
            best_pos = pos_from_radial2
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 30:
        return best_pos
    
    # Step 2: High-precision Cartesian polish on the radial-optimized structure
    try:
        pos_polished, e_polished = _polish_cartesian(best_pos, maxiter=15000, gtol=1e-15, ftol=1e-16)
        if e_polished < best_e:
            best_e = e_polished
            best_pos = pos_polished
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 20:
        return best_pos
    
    # Step 3: Try a few perturbed Mackay structures and relax them
    for s in range(8):
        if time.time() > deadline - 10:
            break
        try:
            scale = 0.03 if s < 4 else 0.08
            pos_pert = _make_perturbed_mackay(s, scale=scale)
            pos_relaxed, e_relaxed = _relax(pos_pert, maxiter=400)
            if e_relaxed < best_e:
                best_e = e_relaxed
                best_pos = pos_relaxed.copy()
        except Exception:
            pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        return best_pos
    
    # Step 4: Try cubic lattice and random clusters
    try:
        pos_cubic = _make_cubic_lattice()
        pos_relaxed, e_relaxed = _relax(pos_cubic, maxiter=300)
        if e_relaxed < best_e:
            best_e = e_relaxed
            best_pos = pos_relaxed.copy()
    except Exception:
        pass
    
    if time.time() > deadline - 5:
        return best_pos
    
    # Step 5: Final high-precision polish
    try:
        pos_final, e_final = _polish_cartesian(best_pos, maxiter=10000, gtol=1e-15, ftol=1e-16)
        if e_final < best_e:
            best_e = e_final
            best_pos = pos_final
    except Exception:
        pass
    
    return best_pos

# EVOLVE-BLOCK-END
