import numpy as np
import time
from scipy.optimize import minimize

N = 55

def _lj_energy_and_grad(pos):
    """Compute LJ energy and gradient."""
    pos = np.asarray(pos)
    n = pos.shape[0]
    # Compute all pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    # Avoid division by zero
    dist_sq[dist_sq < 1e-12] = 1e-12
    dist = np.sqrt(dist_sq)
    
    r6 = dist_sq ** -3
    r12 = r6 ** 2
    
    # Energy: 4 * sum(r^-12 - r^-6) over all pairs
    # We sum over all i,j and divide by 2 to count each pair once
    energy = 4.0 * (np.sum(r12) - np.sum(r6)) / 2.0
    # Subtract self-terms (i==j) which are infinite/undefined but we set dist_sq to 1e-12
    # Actually, r12 and r6 for i==j are huge. We should mask them.
    
    # Let's redo with masking for correctness
    mask = ~np.eye(n, dtype=bool)
    r6_masked = np.where(mask, r6, 0.0)
    r12_masked = np.where(mask, r12, 0.0)
    
    energy = 4.0 * (np.sum(r12_masked) - np.sum(r6_masked))
    # Note: sum over all i,j counts each pair twice. The formula 4*sum(r^-12-r^-6) usually implies sum over pairs.
    # If we sum over all i!=j, we get 2 * sum_{i<j}.
    # So energy = 4.0 * (np.sum(r12_masked) - np.sum(r6_masked)) / 2.0
    
    # Gradient
    # dE/dr = 4 * (-12*r^-13 + 6*r^-7)
    # dE/dx_i = sum_{j != i} dE/dr * dr/dx_i
    # dr/dx_i = (x_i - x_j) / r
    # dE/dx_i = sum_{j != i} 4 * (-12*r^-13 + 6*r^-7) * (x_i - x_j) / r
    # = sum_{j != i} 4 * (-12*r^-14 + 6*r^-8) * (x_i - x_j)
    
    # Vectorized gradient
    # factor = 4 * (-12 * r^-14 + 6 * r^-8)
    # r^-14 = r12 / r^2 = r12 / dist_sq
    # r^-8 = r6 / r^2 = r6 / dist_sq
    r14 = r12 / dist_sq
    r8 = r6 / dist_sq
    
    factor = 4.0 * (-12.0 * r14 + 6.0 * r8)
    factor = np.where(mask, factor, 0.0)
    
    # grad_i = sum_j factor_ij * (x_i - x_j)
    # diff_ij = x_i - x_j
    # grad = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
    
    grad = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad

def _generate_mackay_icosahedron():
    """Generate initial positions for a 55-atom Mackay icosahedron."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    # Icosahedron vertices
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    verts /= np.linalg.norm(verts, axis=1)[:, np.newaxis]
    
    # Inner shell: 12 vertices at radius R1
    R1 = 1.12
    inner = verts * R1
    
    # Edge centers: 30 points
    # Find edges: pairs of vertices with dot product ~ -1/phi
    dot_prods = verts @ verts.T
    edge_mask = (np.abs(dot_prods + 1.0/phi) < 1e-6) & ~np.eye(12, dtype=bool)
    edge_indices = np.argwhere(edge_mask)
    # Each edge appears twice (i,j) and (j,i), so take half
    edge_indices = edge_indices[edge_indices[:, 0] < edge_indices[:, 1]]
    
    edge_centers = []
    for i, j in edge_indices:
        ec = (verts[i] + verts[j]) / 2.0
        ec /= np.linalg.norm(ec)
        edge_centers.append(ec)
    edge_centers = np.array(edge_centers)
    
    # Radius R2 for edge centers
    R2 = 1.5
    middle = edge_centers * R2
    
    # Outer shell: 12 vertices at radius R3
    R3 = 2.2
    outer = verts * R3
    
    # Center
    center = np.array([[0.0, 0.0, 0.0]])
    
    pos = np.vstack([center, inner, middle, outer])
    return pos

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Leave 20s margin
    
    # Generate initial guess
    pos = _generate_mackay_icosahedron()
    
    # Relax
    best_pos = pos
    best_energy, _ = _lj_energy_and_grad(pos)
    
    # Use L-BFGS-B
    try:
        res = minimize(
            _lj_energy_and_grad, 
            pos, 
            method='L-BFGS-B', 
            jac=True, 
            options={'maxiter': 1000, 'gtol': 1e-10, 'disp': False}
        )
        
        if res.success or res.fun < best_energy:
            best_pos = res.x
            best_energy = res.fun
            
    except Exception as e:
        pass
    
    # Check deadline
    if time.time() > deadline:
        return best_pos
    
    # Optional: Second pass with tighter tolerance if time permits
    if time.time() < deadline - 10.0:
        try:
            res2 = minimize(
                _lj_energy_and_grad, 
                best_pos, 
                method='L-BFGS-B', 
                jac=True, 
                options={'maxiter': 2000, 'gtol': 1e-12, 'disp': False}
            )
            if res2.fun < best_energy:
                best_pos = res2.x
                best_energy = res2.fun
        except Exception as e:
            pass
            
    return best_pos
