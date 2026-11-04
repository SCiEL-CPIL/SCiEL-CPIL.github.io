"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, Bounds
from scipy.spatial.distance import pdist, squareform
import time

N = 100
DEADLINE = 140.0  # wall-clock safety margin (evaluator kills at 150s)

# EVOLVE-BLOCK-START

def _compute_radius(centers):
    """Compute the feasible common radius for a set of centers in [-0.5, 0.5]^3."""
    if centers.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    # Boundary constraint: min distance from any center to the cube faces
    dist_to_face = 0.5 - np.abs(centers)
    r_boundary = np.min(dist_to_face)
    if r_boundary < 0:
        return 0.0
    # Pairwise constraint
    d = pdist(centers, 'euclidean')
    if d.size == 0:
        r_pair = np.inf
    else:
        r_pair = np.min(d) / 2.0
    r = min(r_boundary, r_pair)
    return r


def _energy_and_grad(centers_flat, s=10.0, boundary_penalty=1.0):
    """
    Riesz s-energy with s > 3 plus boundary penalty.
    Energy = sum_{i<j} 1/d_ij^s + boundary_penalty * sum_i sum_k 1/(0.5 - |c_ik|)^s
    Minimize energy -> maximize minimum distance.
    """
    n = centers_flat.size // 3
    centers = centers_flat.reshape(n, 3)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dist_sq, np.inf)
    dist = np.sqrt(dist_sq)
    
    # Avoid division by zero
    eps = 1e-12
    dist_safe = np.maximum(dist, eps)
    
    # Pairwise energy: sum over i<j of 1/dist^s
    # Use upper triangle to avoid double counting
    iu = np.triu_indices(n, k=1)
    pair_energy = np.sum(1.0 / dist_safe[iu[0], iu[1]] ** s)
    
    # Boundary penalty: for each center and each axis, penalty = 1/(0.5 - |c|)^s
    # This goes to infinity as center approaches boundary
    abs_c = np.abs(centers)
    boundary_dist = 0.5 - abs_c
    boundary_dist_safe = np.maximum(boundary_dist, eps)
    boundary_energy = boundary_penalty * np.sum(1.0 / boundary_dist_safe ** s)
    
    total_energy = pair_energy + boundary_energy
    
    # Gradient
    grad = np.zeros_like(centers_flat)
    
    # Pairwise gradient: d/dx_i (1/|x_i - x_j|^s) = -s * (x_i - x_j) / |x_i - x_j|^(s+2)
    # For each pair (i,j), contribution to grad[i] += -s * (c_i - c_j) / d^(s+2)
    # and grad[j] += -s * (c_j - c_i) / d^(s+2) = s * (c_i - c_j) / d^(s+2)
    # Using vectorized approach:
    # grad_i = sum_j s * (c_i - c_j) / |c_i - c_j|^(s+2)
    # = s * sum_j (c_i - c_j) / dist^(s+2)
    
    # dist_safe[:, :, :] has shape (n, n)
    # factor[i,j] = s / dist_safe[i,j]^(s+2)
    factor = s / dist_safe ** (s + 2)
    np.fill_diagonal(factor, 0.0)
    
    # grad[i] = sum_j factor[i,j] * (c_i - c_j)
    # = s * c_i * sum_j factor[i,j] - sum_j factor[i,j] * c_j
    sum_factor = np.sum(factor, axis=1)  # shape (n,)
    grad_pair = s * centers * sum_factor[:, np.newaxis] - (factor @ centers)
    
    # Boundary gradient: d/dx_ik [1/(0.5 - |x_ik|)^s] = s * sign(x_ik) / (0.5 - |x_ik|)^(s+1)
    # Only if x_ik != 0; if x_ik == 0, gradient is 0 (kink)
    sign_c = np.sign(centers)
    boundary_grad = boundary_penalty * s * sign_c / boundary_dist_safe ** (s + 1)
    # Where centers are 0, the subgradient is in [-s, s]; use 0
    boundary_grad[centers == 0] = 0.0
    
    grad = grad_pair + boundary_grad
    grad_flat = grad.flatten()
    
    return total_energy, grad_flat


def _energy_only(centers_flat, s=10.0, boundary_penalty=1.0):
    """Energy function only, for use with minimize."""
    e, _ = _energy_and_grad(centers_flat, s, boundary_penalty)
    return e


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Initial guess: simple cubic grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    best_centers = grid[:N].copy()
    best_radius = _compute_radius(best_centers)
    
    # Try L-BFGS-B optimization with different parameters
    s_values = [6.0, 8.0, 10.0, 12.0]
    boundary_penalties = [0.5, 1.0, 2.0]
    
    for s in s_values:
        if time.time() > deadline - 5.0:
            break
        for bp in boundary_penalties:
            if time.time() > deadline - 5.0:
                break
            
            # Start from the current best
            x0 = best_centers.flatten()
            
            try:
                result = minimize(
                    _energy_only,
                    x0,
                    args=(s, bp),
                    method='L-BFGS-B',
                    jac=True,  # We'll provide gradient
                    bounds=[(-0.49, 0.49)] * (N * 3),
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if result.success or result.fun < _energy_only(x0, s, bp):
                    new_centers = result.x.reshape(N, 3)
                    new_radius = _compute_radius(new_centers)
                    if new_radius > best_radius:
                        best_centers = new_centers
                        best_radius = new_radius
            except Exception:
                pass
    
    # Final local refinement using a higher s for sharper energy landscape
    if time.time() < deadline - 10.0:
        s_final = 15.0
        bp_final = 1.0
        x0 = best_centers.flatten()
        try:
            result = minimize(
                _energy_only,
                x0,
                args=(s_final, bp_final),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-0.49, 0.49)] * (N * 3),
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result.success:
                new_centers = result.x.reshape(N, 3)
                new_radius = _compute_radius(new_centers)
                if new_radius > best_radius:
                    best_centers = new_centers
                    best_radius = new_radius
        except Exception:
            pass
    
    # Ensure all centers are strictly inside the cube
    best_centers = np.clip(best_centers, -0.499999999, 0.499999999)
    
    return best_centers

# EVOLVE-BLOCK-END
