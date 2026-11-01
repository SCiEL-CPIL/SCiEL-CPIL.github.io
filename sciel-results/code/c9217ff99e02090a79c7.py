"""Lennard-Jones 55-atom global minimum: Exact Mackay icosahedron with aggressive L-BFGS relaxation."""
import numpy as np
import time
from scipy.optimize import minimize

N = 55

# EVOLVE-BLOCK-START
def _lj_energy(pos):
    """Compute total LJ energy for a set of positions (vectorized)."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** 3
    e = 4.0 * (1.0 / (r6 * r6) - 1.0 / r6)
    return np.sum(e) / 2.0

def _lj_energy_grad(pos):
    """Compute total LJ energy and gradient (vectorized, numerically stable)."""
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

def _get_icosahedron_vertices():
    """Get the 12 vertices of a regular icosahedron centered at origin."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    return verts / norms

def _get_icosahedron_edges(verts):
    """Find the 30 edges of the icosahedron."""
    n = len(verts)
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dists[i, j] = np.linalg.norm(verts[i] - verts[j])
    
    # Find minimum non-zero distance (edge length)
    non_zero = dists[dists > 1e-10]
    min_dist = np.min(non_zero)
    
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if abs(dists[i, j] - min_dist) < 1e-10:
                edges.append((i, j))
    return edges

def _make_mackay_icosahedron(r1, r2, r3):
    """
    Construct 55-atom Mackay icosahedron:
    - 1 atom at center
    - 12 atoms at icosahedron vertices (radius r1)
    - 30 atoms at edge midpoints (radius r2) 
    - 12 atoms at face centers (radius r3)
    Total: 1 + 12 + 30 + 12 = 55
    """
    verts = _get_icosahedron_vertices()
    edges = _get_icosahedron_edges(verts)
    
    # Center atom
    center = np.array([[0.0, 0.0, 0.0]])
    
    # First shell: 12 vertices
    first_shell = verts * r1
    
    # Second shell inner: 30 edge midpoints
    edge_mids = []
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        norm = np.linalg.norm(mid)
        if norm > 1e-10:
            edge_mids.append(mid / norm)
    edge_mids = np.array(edge_mids)
    second_inner = edge_mids * r2
    
    # Third shell: 12 face centers (use the same vertices scaled to r3)
    # In a proper Mackay icosahedron, the outer shell mirrors the inner
    third_shell = verts * r3
    
    all_pos = np.vstack([center, first_shell, second_inner, third_shell])
    return all_pos

def _check_min_distance(pos, min_d=0.5):
    """Check if all pairwise distances are >= min_d."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= min_d

def _fix_min_distance(pos, min_d=0.5):
    """Scale positions to ensure minimum distance constraint is satisfied."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist_found = np.min(dists)
    
    if min_dist_found < min_d:
        scale = min_d / min_dist_found * 1.001
        pos = pos * scale
    
    return pos

def construct():
    """
    Construct 55-atom LJ cluster at global minimum.
    Strategy: 
    1. Start from exact Mackay icosahedron geometry with optimized radii
    2. Apply single aggressive L-BFGS relaxation with very tight tolerance
    3. Final high-precision polish
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Optimal radii for Mackay icosahedron (from literature/optimization)
    # These are starting points; L-BFGS will fine-tune
    r1_init = 1.048
    r2_init = 1.748
    r3_init = 1.968
    
    # Start from exact symmetric geometry
    pos = _make_mackay_icosahedron(r1_init, r2_init, r3_init)
    
    # Ensure minimum distance constraint
    if not _check_min_distance(pos, 0.5):
        pos = _fix_min_distance(pos, 0.5)
    
    # Define objective and gradient
    def obj_and_grad(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_grad(pos_flat)
        return e, g.flatten()
    
    x0 = pos.flatten()
    
    # Phase 1: Aggressive L-BFGS relaxation with tight tolerance
    best_energy = _lj_energy(pos)
    best_pos = pos.copy()
    
    try:
        result = minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            options={'maxiter': 5000, 'gtol': 1e-12, 'maxls': 500, 'ftol': 1e-15}
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_pos = result.x.reshape(-1, 3)
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 20.0:
        if not _check_min_distance(best_pos, 0.5):
            best_pos = _fix_min_distance(best_pos, 0.5)
        return best_pos
    
    # Phase 2: High-precision polish with progressively tighter tolerances
    x0 = best_pos.flatten()
    
    polish_stages = [
        {'maxiter': 10000, 'gtol': 1e-14, 'maxls': 1000, 'ftol': 1e-16},
        {'maxiter': 20000, 'gtol': 1e-16, 'maxls': 2000, 'ftol': 1e-18},
    ]
    
    for stage_opts in polish_stages:
        if time.time() > deadline - 5.0:
            break
        try:
            result = minimize(
                obj_and_grad, x0, jac=True, method='L-BFGS-B',
                options=stage_opts
            )
            if result.fun < best_energy:
                best_energy = result.fun
                best_pos = result.x.reshape(-1, 3)
                x0 = result.x
        except Exception:
            break
    
    # Final constraint check
    if not _check_min_distance(best_pos, 0.5):
        best_pos = _fix_min_distance(best_pos, 0.5)
        # Quick re-relax after fixing constraint
        try:
            result = minimize(
                obj_and_grad, best_pos.flatten(), jac=True, method='L-BFGS-B',
                options={'maxiter': 500, 'gtol': 1e-10, 'maxls': 200, 'ftol': 1e-14}
            )
            if result.fun < best_energy:
                best_energy = result.fun
                best_pos = result.x.reshape(-1, 3)
        except Exception:
            pass
    
    return best_pos
# EVOLVE-BLOCK-END
