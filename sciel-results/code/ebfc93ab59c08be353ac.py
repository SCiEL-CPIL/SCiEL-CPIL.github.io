"""Physics-based circle packing using L-BFGS-B optimization."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 200
START_TIME = time.time()
DEADLINE = START_TIME + 110  # 110s safety margin before 120s limit

def _compute_radius(centers):
    """Compute the maximum common radius for a set of centers in [-0.5,0.5]^2."""
    n = len(centers)
    if n == 0:
        return 0.0
    # Boundary distance
    dist_boundary = np.min(np.min(0.5 - np.abs(centers), axis=1))
    if dist_boundary <= 0:
        return 0.0
    # Pairwise distances via KDTree
    tree = cKDTree(centers)
    # Use a radius larger than expected minimum distance to ensure we catch all close pairs
    # If r is ~0.03, min dist is ~0.07. 0.1 is safe.
    pairs = tree.query_pairs(r=0.1, output_type='ndarray')
    if len(pairs) == 0:
        return dist_boundary
    diffs = centers[pairs[:, 0]] - centers[pairs[:, 1]]
    dists = np.sqrt(np.sum(diffs * diffs, axis=1))
    min_pair = np.min(dists) / 2.0
    return min(dist_boundary, min_pair)

def _energy_and_grad(x):
    """
    Compute energy and gradient for the physics-based optimization.
    x is a flattened array of shape (N, 2).
    Energy = sum of repulsive potentials (1/d^2) + boundary penalties.
    """
    centers = x.reshape(N, 2)
    
    # 1. Boundary Penalties
    # Penalty for being outside or too close to boundary
    # We want to push centers into [-0.5, 0.5]
    # Let d be distance to boundary. If d < 0, penalty is high.
    # We use a smooth penalty: max(0, 0.5 - |c|)^2 is not differentiable at 0.5.
    # Better: Penalty = sum( max(0, 0.5 - |c_i| - margin)^2 ) ? 
    # Actually, we just want to maximize min distance to boundary.
    # Let's use a soft constraint: E_wall = sum( (max(0, 0.5 - |c_ik|))^2 )
    # This penalizes any point that is strictly inside the "safe zone" less than the edge?
    # No, we want to maximize the distance to the edge.
    # Let dist_to_edge = 0.5 - |c|. We want to maximize min(dist_to_edge).
    # Minimize -min(dist_to_edge) is hard.
    # Let's minimize sum( (0.5 - |c_ik|)^2 )? No, that pushes to corners.
    # We want to maximize the minimum distance to boundary.
    # A common trick: minimize sum( exp(-k * dist_to_edge) ) or similar.
    # Or simply: minimize sum( max(0, 0.5 - |c| - target)^2 )?
    
    # Let's stick to a simple repulsive force between particles and a boundary force.
    # Boundary force: push inward if |c| > 0.5 - epsilon? 
    # Actually, the problem allows centers up to 0.5. 
    # The score is min( min_pair_dist/2, min_boundary_dist ).
    # To maximize this, we want to maximize both.
    
    # Let's define Energy as:
    # E = sum_{i<j} (1/d_ij^2)  (Repulsion)
    # E += sum_i (max(0, 0.5 - |c_i| - 0.01)^2) ? No.
    
    # Let's use a potential that is high when circles overlap or touch boundary.
    # Overlap: d_ij < 2r. We don't know r.
    # Let's just minimize the sum of inverse squared distances. This naturally pushes them apart.
    # And add a penalty for violating the boundary: (max(0, |c_i| - 0.5))^2.
    # This keeps them inside. The optimizer will find a configuration where they are as far apart as possible while staying inside.
    
    centers = x.reshape(N, 2)
    
    # Boundary Penalty: Push back if outside [-0.5, 0.5]
    # dist_out = max(0, |c| - 0.5)
    abs_c = np.abs(centers)
    out_x = np.maximum(0, abs_c[:, 0] - 0.5)
    out_y = np.maximum(0, abs_c[:, 1] - 0.5)
    E_wall = np.sum(out_x**2) + np.sum(out_y**2)
    grad_wall = np.zeros_like(x)
    grad_wall[:, 0] = 2 * out_x * np.sign(centers[:, 0])
    grad_wall[:, 1] = 2 * out_y * np.sign(centers[:, 1])
    
    # Repulsive Energy
    # We only need to consider pairs that are close to avoid O(N^2) cost.
    # But for gradient, we need all pairs that contribute to the gradient.
    # If we use 1/d^2, the gradient is significant only for close pairs.
    # Let's use a cutoff.
    tree = cKDTree(centers)
    # Use a cutoff radius. If we expect r~0.03, min dist ~0.07.
    # Let's use cutoff 0.15.
    pairs = tree.query_pairs(r=0.15, output_type='ndarray')
    
    E_rep = 0.0
    grad_rep = np.zeros_like(x)
    
    if len(pairs) > 0:
        for i, j in pairs:
            d_vec = centers[i] - centers[j]
            d_sq = np.dot(d_vec, d_vec)
            if d_sq < 1e-12:
                # Perturb if coincident
                d_vec = np.array([1e-6, 1e-6])
                d_sq = np.dot(d_vec, d_vec)
            
            # E = 1/d^2 = d^-2
            # dE/dd = -2 d^-3
            # grad E wrt x_i = dE/dd * (d/d x_i) d = -2/d^3 * (d_vec/d) = -2/d^4 * d_vec
            # Wait, d = ||x_i - x_j||.
            # d/d x_i ||x_i - x_j|| = (x_i - x_j)/d.
            # d/d x_i (1/||x_i - x_j||^2) = -2 * ||x_i - x_j||^-3 * (x_i - x_j)/||x_i - x_j|| = -2 (x_i - x_j) / ||x_i - x_j||^4.
            
            factor = -2.0 / (d_sq * d_sq)
            grad_rep[i] += factor * d_vec
            grad_rep[j] -= factor * d_vec
            E_rep += 1.0 / d_sq
    
    E_total = E_rep + E_wall
    grad_total = grad_rep + grad_wall
    
    return E_total, grad_total

def construct():
    start_time = time.time()
    
    # Initialize with a grid
    m = int(np.ceil(np.sqrt(N)))
    # Add some jitter to break symmetry
    np.random.seed(42)
    grid_x = (np.arange(m) + 0.5) / m - 0.5
    grid_y = (np.arange(m) + 0.5) / m - 0.5
    centers = np.array([(x, y) for y in grid_y for x in grid_x])
    centers = centers[:N]
    centers += np.random.normal(0, 0.01, centers.shape)
    centers = np.clip(centers, -0.49, 0.49)
    
    x0 = centers.flatten()
    
    # Constraints: None (boundaries handled by penalty)
    # Bounds: [-0.5, 0.5] for each coordinate
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    best_x = x0
    best_r = _compute_radius(centers)
    
    # Run L-BFGS-B
    # maxiter is a hard stop, but we also check time.
    # L-BFGS-B is fast.
    try:
        result = minimize(
            _energy_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
        )
        if result.x is not None:
            candidate = result.x.reshape(N, 2)
            # Check validity
            if np.all(np.isfinite(candidate)) and np.all(candidate >= -0.5) and np.all(candidate <= 0.5):
                r_cand = _compute_radius(candidate)
                if r_cand > best_r:
                    best_r = r_cand
                    best_x = result.x
    except Exception as e:
        pass
    
    # If time remains, try a second run with a different initialization or restart
    if time.time() < DEADLINE - 5:
        # Jitter the current best solution and run again
        x_restart = best_x + np.random.normal(0, 0.005, best_x.shape)
        x_restart = np.clip(x_restart, -0.49, 0.49)
        try:
            result2 = minimize(
                _energy_and_grad,
                x_restart,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            if result2.x is not None:
                candidate2 = result2.x.reshape(N, 2)
                if np.all(np.isfinite(candidate2)) and np.all(candidate2 >= -0.5) and np.all(candidate2 <= 0.5):
                    r_cand2 = _compute_radius(candidate2)
                    if r_cand2 > best_r:
                        best_r = r_cand2
                        best_x = result2.x
        except Exception:
            pass
            
    final_centers = best_x.reshape(N, 2)
    
    # Final safety check
    if not np.all(np.isfinite(final_centers)):
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        final_centers = np.array([(x, y) for y in ticks for x in ticks])[:N]
        
    return final_centers
