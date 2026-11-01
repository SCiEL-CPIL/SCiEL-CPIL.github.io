"""Lennard-Jones 55-atom cluster: global optimization with symmetry-constrained manifold."""
import numpy as np
import time
from scipy.optimize import minimize

N = 55
DEADLINE = 142.0  # Safety margin before 150s kill

def lj_energy(pos):
    """Compute total LJ energy for N atoms (vectorized)."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    e = 4.0 * (r12 - r6).sum() / 2.0
    return e

def lj_energy_grad(pos):
    """Compute LJ energy and gradient (vectorized)."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    e = 4.0 * (r12 - r6).sum() / 2.0
    coeff = 4.0 * (-12.0 * r2 ** (-7) + 6.0 * r2 ** (-4))
    np.fill_diagonal(coeff, 0.0)
    grad = np.einsum('ij,ijk->ik', coeff, diff)
    return e, grad

def _get_mackay_directions():
    """Get the 55 unit direction vectors for a Mackay icosahedron."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    
    # 12 icosahedron vertices
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    # Find edges of icosahedron
    edge_len = np.linalg.norm(verts[0] - verts[1])
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if np.isclose(d, edge_len, atol=1e-10):
                edges.append((i, j))
    
    # 30 edge midpoints (normalized)
    midpoints = []
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        mid = mid / np.linalg.norm(mid)
        midpoints.append(mid)
    midpoints = np.array(midpoints)
    
    # Total: 1 (center) + 12 (shell 1) + 30 (shell 2) + 12 (shell 3) = 55
    directions = np.zeros((55, 3))
    shell_ids = np.zeros(55, dtype=int)
    
    # Atom 0: center
    directions[0] = [0.0, 0.0, 0.0]
    shell_ids[0] = 0
    
    # Atoms 1-12: Shell 1 (icosahedron vertices)
    for i in range(12):
        directions[1+i] = verts[i]
        shell_ids[1+i] = 1
    
    # Atoms 13-42: Shell 2 (edge midpoints)
    for i in range(30):
        directions[13+i] = midpoints[i]
        shell_ids[13+i] = 2
    
    # Atoms 43-54: Shell 3 (icosahedron vertices, outer)
    for i in range(12):
        directions[43+i] = verts[i]
        shell_ids[43+i] = 3
    
    return directions, shell_ids

def _make_rotation_matrix(euler):
    """Create a rotation matrix from Euler angles (xyz convention)."""
    rx, ry, rz = euler
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    
    return Rz @ Ry @ Rx

def _make_mackay_from_params(params):
    """
    Generate Mackay icosahedron positions from parameters.
    params: [r1, r2, r3, rx, ry, rz]
    """
    r1, r2, r3 = np.maximum(params[:3], 0.5)
    R = _make_rotation_matrix(params[3:6])
    
    directions, shell_ids = _get_mackay_directions()
    
    pos = np.zeros((55, 3))
    for i in range(1, 55):
        shell = shell_ids[i]
        if shell == 1:
            pos[i] = R @ (directions[i] * r1)
        elif shell == 2:
            pos[i] = R @ (directions[i] * r2)
        elif shell == 3:
            pos[i] = R @ (directions[i] * r3)
    
    return pos

def relax_positions(pos, maxiter=200, gtol=1e-6):
    """Relax atomic positions using L-BFGS-B."""
    def fun(x):
        x = x.reshape(N, 3)
        e, g = lj_energy_grad(x)
        return e, g.reshape(-1)
    
    try:
        result = minimize(fun, pos.reshape(-1), method='L-BFGS-B', jac=True, 
                         options={'maxiter': maxiter, 'gtol': gtol, 'ftol': 1e-10, 'maxls': 50})
        return result.x.reshape(N, 3), result.fun
    except Exception:
        return pos, lj_energy(pos)

def optimize_manifold_params(params_init, maxiter_nm=80, maxiter_lbfgs=150):
    """
    Optimize manifold parameters (3 radii + 3 rotation angles) using Nelder-Mead.
    Each evaluation relaxes the full 165-dimensional position space.
    """
    def objective(params):
        pos = _make_mackay_from_params(params)
        pos_relaxed, e = relax_positions(pos, maxiter=maxiter_lbfgs, gtol=1e-6)
        return e
    
    try:
        res = minimize(objective, params_init, method='Nelder-Mead', 
                       options={'maxiter': maxiter_nm, 'xatol': 1e-5, 'fatol': 1e-5,
                                'maxfev': maxiter_nm * 2})
        best_params = res.x
        best_pos = _make_mackay_from_params(best_params)
        best_pos, best_e = relax_positions(best_pos, maxiter=300, gtol=1e-8)
        return best_pos, best_e, best_params
    except Exception:
        return None, np.inf, params_init

def construct():
    start_time = time.time()
    
    def remaining():
        return DEADLINE - (time.time() - start_time)
    
    # Phase 1: Optimize on the symmetry-constrained manifold
    # Parameters: [r1, r2, r3, rx, ry, rz]
    # Start with zero rotation (identity) and various radius guesses
    param_candidates = [
        np.array([1.0, 1.8, 2.6, 0.0, 0.0, 0.0]),
        np.array([1.1, 1.9, 2.7, 0.0, 0.0, 0.0]),
        np.array([0.9, 1.7, 2.5, 0.0, 0.0, 0.0]),
        np.array([1.0, 2.0, 2.8, 0.0, 0.0, 0.0]),
        np.array([1.05, 1.85, 2.65, 0.0, 0.0, 0.0]),
        np.array([1.0, 1.75, 2.55, 0.0, 0.0, 0.0]),
    ]
    
    best_pos = None
    best_e = np.inf
    
    for params_init in param_candidates:
        if remaining() < 25:
            break
        try:
            pos, e, params = optimize_manifold_params(params_init, maxiter_nm=60, maxiter_lbfgs=120)
            if pos is not None and e < best_e:
                best_e = e
                best_pos = pos.copy()
        except Exception:
            continue
    
    # Fallback if no valid result
    if best_pos is None:
        fallback_params = np.array([1.0, 1.8, 2.6, 0.0, 0.0, 0.0])
        best_pos = _make_mackay_from_params(fallback_params)
        best_e = lj_energy(best_pos)
        best_pos, best_e = relax_positions(best_pos, maxiter=300, gtol=1e-7)
    
    # Phase 2: Basin-Hopping global optimization on full space
    rng = np.random.RandomState(42)
    bh_steps = 20
    current_pos = best_pos.copy()
    current_e = best_e
    
    # Adaptive step size
    step_scales = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]
    
    for step in range(bh_steps):
        if remaining() < 12:
            break
        
        # Choose step scale: start small, increase periodically
        scale_idx = min(step // 2, len(step_scales) - 1)
        scale = step_scales[scale_idx]
        
        try:
            # Rigid-body displacement + small per-atom perturbations
            displacement = rng.randn(3) * scale
            perturbed = current_pos + displacement
            small_pert = rng.randn(*current_pos.shape) * (scale * 0.3)
            perturbed = perturbed + small_pert
            
            # Relax to nearest local minimum
            new_pos, new_e = relax_positions(perturbed, maxiter=120, gtol=1e-6)
            
            # Metropolis acceptance
            delta = new_e - current_e
            if delta < 0:
                current_pos = new_pos
                current_e = new_e
                if current_e < best_e:
                    best_e = current_e
                    best_pos = current_pos.copy()
            else:
                kT = 0.1
                if rng.rand() < np.exp(-delta / kT):
                    current_pos = new_pos
                    current_e = new_e
        except Exception:
            continue
    
    # Phase 3: Final high-precision polish
    if remaining() > 5:
        try:
            final_pos, final_e = relax_positions(best_pos, maxiter=500, gtol=1e-10)
            if final_e < best_e:
                best_e = final_e
                best_pos = final_pos
        except Exception:
            pass
    
    return best_pos
