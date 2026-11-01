import numpy as np
from scipy.optimize import minimize
import time

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

def _get_icosahedron_vectors():
    """
    Returns the 12 unit vectors to the vertices of a regular icosahedron
    and the 30 unit vectors to the midpoints of the edges.
    """
    phi = (1 + np.sqrt(5)) / 2
    
    verts = [
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ]
    verts = np.array(verts, dtype=float)
    norms = np.linalg.norm(verts, axis=1)
    verts_unit = verts / norms[:, np.newaxis]
    
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(12):
            dists[i, j] = np.linalg.norm(verts[i] - verts[j])
    
    min_dist = np.min(dists[dists > 1e-6])
    
    edge_mids = []
    for i in range(12):
        for j in range(i+1, 12):
            if abs(dists[i, j] - min_dist) < 1e-6:
                mid = (verts[i] + verts[j]) / 2.0
                norm_mid = np.linalg.norm(mid)
                if norm_mid > 1e-6:
                    edge_mids.append(mid / norm_mid)
                    
    edge_mids = np.array(edge_mids)
    return verts_unit, edge_mids

def _make_mackay_icosahedron(r1, r2, r3):
    """
    Construct a three-shell Mackay icosahedron for 55 atoms.
    1 center + 12 first shell (vertices) + 30 second shell (edge midpoints) + 12 third shell (vertices)
    r1: radius of inner shell (12 atoms)
    r2: radius of middle shell (30 atoms)
    r3: radius of outer shell (12 atoms)
    """
    verts_unit, edge_mids = _get_icosahedron_vectors()
    
    center = np.array([[0.0, 0.0, 0.0]])
    first_shell = verts_unit * r1
    second_shell = edge_mids * r2
    third_shell = verts_unit * r3
    
    all_pos = np.vstack([center, first_shell, second_shell, third_shell])
    return all_pos

def _relax_lbfgs(pos, deadline=None, maxiter=500, gtol=1e-8):
    """Relax using L-BFGS-B."""
    start_time = time.time()
    best_energy = _lj_energy(pos)
    best_pos = pos.copy()
    
    def obj_and_grad(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_grad(pos_flat)
        return e, g.flatten()
    
    x0 = pos.flatten()
    
    try:
        result = minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 100,
                     'ftol': 1e-14}
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_pos = result.x.reshape(-1, 3)
    except Exception:
        pass
    
    return best_pos, best_energy

def _polish_bfgs(pos, deadline=None, maxiter=2000, gtol=1e-12):
    """
    High-precision polish using BFGS.
    BFGS maintains a full inverse Hessian approximation, allowing it to converge
    to much tighter tolerances than L-BFGS-B which uses limited memory.
    """
    start_time = time.time()
    best_energy = _lj_energy(pos)
    best_pos = pos.copy()
    
    def obj_and_grad(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_grad(pos_flat)
        return e, g.flatten()
    
    x0 = pos.flatten()
    
    try:
        result = minimize(
            obj_and_grad, x0, jac=True, method='BFGS',
            options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 500,
                     'maxfun': 5000, 'disp': False}
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_pos = result.x.reshape(-1, 3)
    except Exception:
        pass
    
    return best_pos, best_energy

def construct():
    """Construct 55-atom LJ cluster near global minimum."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # 1. Generate Mackay Icosahedron candidates with multi-start stochastic search
    # The optimal radii are close to the integer shell positions but slightly relaxed.
    # Standard Mackay: r1 ~ 1.0, r2 ~ 1.73, r3 ~ 2.0 (approx)
    
    base = _make_mackay_icosahedron(1.0, 1.732, 2.0)
    
    # Start with the exact geometry and several perturbations
    candidates = [base.copy()]
    
    # Multiple perturbation scales and seeds for diversity
    np.random.seed(0)
    for scale in [0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]:
        for seed in range(3):
            np.random.seed(seed * 100 + int(scale * 1000))
            perturbed = base + np.random.normal(0, scale, size=base.shape)
            candidates.append(perturbed)
    
    best_pos = None
    best_energy = np.inf
    
    # 2. Relax each candidate with L-BFGS, prioritizing better energies
    # Use adaptive maxiter: fewer iterations for later candidates if time is tight
    n_candidates = len(candidates)
    for i, cand in enumerate(candidates):
        elapsed = time.time() - start_time
        remaining = deadline - elapsed - 15.0  # Reserve 15s for polishing
        
        if remaining < 1.0:
            break
        
        # Scale maxiter based on remaining time and candidate index
        # First few candidates get more iterations
        if i < 3:
            maxiter = 300
        elif i < 10:
            maxiter = 200
        else:
            maxiter = 150
        
        # If very tight on time, use fewer iterations
        if remaining < 10.0:
            maxiter = min(maxiter, 100)
        if remaining < 5.0:
            maxiter = min(maxiter, 50)
        
        relaxed, energy = _relax_lbfgs(cand, deadline=deadline, maxiter=maxiter, gtol=1e-6)
        
        if energy < best_energy:
            best_energy = energy
            best_pos = relaxed.copy()
    
    if best_pos is None:
        # Fallback: return the first candidate
        best_pos = candidates[0]
        best_energy = _lj_energy(best_pos)
    
    # 3. Polish the best candidate with multi-stage high-precision passes
    remaining = deadline - time.time()
    
    if remaining > 8.0:
        best_pos, best_energy = _polish_bfgs(best_pos, deadline=deadline, maxiter=2000, gtol=1e-12)
        
    remaining = deadline - time.time()
    if remaining > 4.0:
        best_pos, best_energy = _polish_bfgs(best_pos, deadline=deadline, maxiter=1000, gtol=1e-14)
        
    remaining = deadline - time.time()
    if remaining > 2.0:
        best_pos, best_energy = _polish_bfgs(best_pos, deadline=deadline, maxiter=500, gtol=1e-15)

    # 4. Final check for minimum distance constraint
    diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    if min_dist < 0.5:
        # Nudge atoms apart if needed
        for _ in range(100):
            diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            min_dist = np.min(dists)
            if min_dist >= 0.5:
                break
            # Find the closest pair and push them apart
            min_idx = np.unravel_index(np.argmin(dists), dists.shape)
            i, j = min_idx
            direction = best_pos[i] - best_pos[j]
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction = direction / norm
            else:
                direction = np.array([1.0, 0.0, 0.0])
            shift = (0.5 - min_dist + 0.01) / 2.0
            best_pos[i] += shift * direction
            best_pos[j] -= shift * direction

    return best_pos
# EVOLVE-BLOCK-END
