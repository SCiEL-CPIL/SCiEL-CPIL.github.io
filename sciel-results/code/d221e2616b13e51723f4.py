import numpy as np
from scipy.optimize import minimize
import time

N = 38

def _lj_energy_and_grad(pos):
    """
    Calculate Lennard-Jones energy and gradient for a set of positions.
    pos: (N, 3) array of coordinates.
    Returns: energy, gradient (N, 3)
    """
    # Compute pairwise differences
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    
    # Avoid division by zero for i==j
    np.fill_diagonal(dist_sq, 1.0)
    
    # Inverse distances
    inv_r2 = 1.0 / dist_sq
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    
    # Energy: 4 * sum_{i<j} (r^-12 - r^-6)
    # Sum over all i!=j is 2 * sum over i<j
    # So E = 2 * sum_{i!=j} (r^-12 - r^-6)
    pair_energy = inv_r12 - inv_r6
    np.fill_diagonal(pair_energy, 0.0)
    energy = 2.0 * np.sum(pair_energy)
    
    # Gradient
    # dE/dx_i = sum_{j!=i} 4 * (-12 r^-14 + 6 r^-8) * (x_i - x_j)
    inv_r8 = inv_r2 ** 4
    inv_r14 = inv_r2 ** 7
    
    grad_factor = 4.0 * (-12.0 * inv_r14 + 6.0 * inv_r8)
    np.fill_diagonal(grad_factor, 0.0)
    
    grad = np.sum(grad_factor[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad

def _make_fcc_cluster():
    """
    Generate an FCC lattice fragment of 38 atoms forming a truncated octahedron.
    """
    # FCC basis vectors with nearest-neighbor distance = 1
    # In FCC, nearest neighbors are at distance a/sqrt(2) where a is cubic lattice constant
    # So a = sqrt(2) for NN distance = 1
    a = np.sqrt(2.0)
    v1 = np.array([a, a, 0.0]) / 2.0
    v2 = np.array([a, 0.0, a]) / 2.0
    v3 = np.array([0.0, a, a]) / 2.0
    
    points = []
    for n1 in range(-3, 4):
        for n2 in range(-3, 4):
            for n3 in range(-3, 4):
                p = n1 * v1 + n2 * v2 + n3 * v3
                points.append(p)
                
    points = np.array(points)
    
    # Center at origin
    centroid = np.mean(points, axis=0)
    points = points - centroid
    
    # Select 38 atoms closest to origin to form truncated octahedron
    dists = np.linalg.norm(points, axis=1)
    sorted_indices = np.argsort(dists)
    
    cluster = points[sorted_indices[:N]]
    
    # Re-center
    centroid = np.mean(cluster, axis=0)
    cluster = cluster - centroid
    
    return cluster

def _try_relax(pos, deadline):
    """
    Try to relax positions using L-BFGS-B.
    Returns (success, best_pos, best_e)
    """
    if time.time() > deadline:
        return False, pos, float('inf')
    
    try:
        e0, _ = _lj_energy_and_grad(pos)
        best_pos = pos.copy()
        best_e = e0
        
        res = minimize(
            _lj_energy_and_grad, 
            pos, 
            jac=True, 
            method='L-BFGS-B', 
            options={'maxiter': 800, 'gtol': 1e-10, 'ftol': 1e-12}
        )
        
        if res.fun < best_e:
            best_pos = res.x
            best_e = res.fun
            
        return True, best_pos, best_e
    except Exception:
        return False, pos, float('inf')

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Start with FCC cluster
    pos = _make_fcc_cluster()
    
    # Initial relaxation
    success, best_pos, best_e = _try_relax(pos, deadline)
    
    if not success or best_e > -100:
        # If initial relaxation failed, try a different starting point
        # Add some noise and retry
        for attempt in range(8):
            if time.time() > deadline:
                break
            
            noisy_pos = pos + 0.08 * np.random.randn(N, 3)
            success, pos_candidate, e_candidate = _try_relax(noisy_pos, deadline)
            
            if success and e_candidate < best_e:
                best_pos = pos_candidate
                best_e = e_candidate
    
    # Multi-start with structured perturbations
    # Use decreasing perturbation scales for refinement
    scales = [0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
    for scale in scales:
        if time.time() > deadline:
            break
        
        # Create perturbed version of the best solution
        perturbed = best_pos + scale * np.random.randn(N, 3)
        
        success, pos_candidate, e_candidate = _try_relax(perturbed, deadline)
        
        if success and e_candidate < best_e:
            best_pos = pos_candidate
            best_e = e_candidate
    
    # Final high-precision polish with tighter tolerances
    if time.time() < deadline - 5:
        try:
            res = minimize(
                _lj_energy_and_grad, 
                best_pos, 
                jac=True, 
                method='L-BFGS-B', 
                options={'maxiter': 2000, 'gtol': 1e-14, 'ftol': 1e-16}
            )
            
            if res.fun < best_e:
                best_pos = res.x
                best_e = res.fun
        except Exception:
            pass
    
    return best_pos
