"""Packing 50 equal circles in a unit circle using non-convex optimization."""
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import minimize
import warnings

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import minimize
import warnings
import time

def _get_delaunay_edges(x):
    """Get edges from Delaunay triangulation of points x."""
    try:
        tri = Delaunay(x)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                j = (i + 1) % 3
                edge = tuple(sorted((simplex[i], simplex[j])))
                edges.add(edge)
        return list(edges)
    except Exception:
        # Fallback to all pairs if triangulation fails (e.g., collinear points)
        return [(i, j) for i in range(len(x)) for j in range(i+1, len(x))]

def _objective_and_grad(x_flat, tau=0.01):
    """
    Objective: Minimize -soft_min(clearances).
    Clearances are from Delaunay neighbors and boundary.
    """
    x = x_flat.reshape(-1, 2)
    n = len(x)
    
    # Get Delaunay edges
    edges = _get_delaunay_edges(x)
    
    # Compute clearances
    clearances = []
    # Gradients for each clearance wrt x
    # We will accumulate grad_x
    
    # Preallocate gradient
    grad_x = np.zeros_like(x)
    
    # Weights for soft-min
    # v_k are the clearances
    # w_k = exp(-v_k/tau) / sum(exp(-v_j/tau))
    
    # First pass: compute v_k and store info for gradient
    v_vals = []
    edge_info = [] # (i, j, dist) for pairs, or ('b', i, radius) for boundary
    
    for i, j in edges:
        d = np.linalg.norm(x[i] - x[j])
        if d < 1e-10:
            d = 1e-10
        v = d / 2.0
        v_vals.append(v)
        edge_info.append(('p', i, j, d))
        
    for i in range(n):
        r = np.linalg.norm(x[i])
        if r < 1e-10:
            r = 1e-10
        v = 1.0 - r
        v_vals.append(v)
        edge_info.append(('b', i, i, r))
        
    v_arr = np.array(v_vals)
    
    # Compute soft min
    # g(x) = -tau * log(sum(exp(-v/tau)))
    # To avoid overflow, use log-sum-exp trick
    max_neg_v = np.max(-v_arr / tau)
    sum_exp = np.sum(np.exp(-v_arr / tau - max_neg_v))
    log_sum_exp = max_neg_v + np.log(sum_exp)
    soft_min = -tau * log_sum_exp
    
    # Objective is -soft_min
    obj = -soft_min
    
    # Weights w_k
    # w_k = exp(-v_k/tau) / sum(exp(-v_j/tau))
    # = exp(-v_k/tau - max_neg_v) / sum_exp
    exp_terms = np.exp(-v_arr / tau - max_neg_v)
    weights = exp_terms / sum_exp
    
    # Compute gradient of soft_min wrt x
    # d(soft_min)/dx = sum_k w_k * (d v_k / dx)
    
    for k, info in enumerate(edge_info):
        w = weights[k]
        if w < 1e-10:
            continue
            
        if info[0] == 'p':
            _, i, j, d = info
            # v = d/2
            # d(v)/dx_i = (x_i - x_j) / (2*d)
            # d(v)/dx_j = (x_j - x_i) / (2*d)
            diff = x[i] - x[j]
            grad_term = diff / (2.0 * d)
            grad_x[i] += w * grad_term
            grad_x[j] -= w * grad_term
        else:
            _, i, _, r = info
            # v = 1 - r
            # d(v)/dx_i = -x_i / r
            grad_term = -x[i] / r
            grad_x[i] += w * grad_term
    
    # Gradient of objective = -gradient of soft_min
    grad_obj = -grad_x.flatten()
    
    return obj, grad_obj

def _initial_hexagonal():
    """Generate initial hexagonal packing."""
    points = []
    # Try to fit 50 points in hexagonal lattice
    # Radius of container 1.
    # Approximate number of points in hex lattice of radius R: ~ (2/sqrt(3)) * (R/d)^2
    # We want 50 points.
    # Let's just generate a large grid and pick the best 50.
    
    # Spacing such that we get roughly 50 points in unit circle
    # Area of unit circle = pi. Area per point = pi/50 ~ 0.0628.
    # Hexagon area = sqrt(3)/2 * d^2 ~ 0.866 * d^2.
    # d^2 ~ 0.0628 / 0.866 ~ 0.0725 -> d ~ 0.27
    spacing = 0.25 
    
    max_iter = 100
    for k in range(max_iter):
        for i in range(-max_iter, max_iter):
            for j in range(-max_iter, max_iter):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x*x + y*y <= 0.9:
                    points.append([x, y])
    
    points = np.array(points)
    if len(points) >= N:
        # Pick N points closest to center to ensure they are well distributed
        radii = np.sqrt(np.sum(points**2, axis=1))
        idx = np.argsort(radii)[:N]
        return points[idx]
    else:
        # Fallback to random
        return _initial_random()

def _initial_random():
    """Generate random initial configuration inside unit circle."""
    np.random.seed(42)
    points = []
    for _ in range(N):
        while True:
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(-1, 1)
            if x*x + y*y <= 0.81:
                points.append([x, y])
                break
    return np.array(points)

def _compute_radius(x):
    """Compute the packing radius for a given configuration."""
    x = x.reshape(-1, 2)
    n = len(x)
    # Use Delaunay for min distance check? No, for exact score we need global min.
    # But for optimization we used Delaunay. For final score, we must check all pairs.
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    radii = np.sqrt(np.sum(x**2, axis=1))
    min_bound_clear = np.min(1.0 - radii)
    return min(min_pair_dist / 2.0, min_bound_clear)

def construct():
    """Construct packing of N equal circles in unit circle."""
    # Start with hexagonal initial configuration
    x = _initial_hexagonal()
    
    # Optimization using L-BFGS-B with smooth objective
    # We use a sequence of decreasing tau to converge to the hard min
    current_tau = 0.05
    x_flat = x.flatten()
    
    # Main optimization loop
    for step in range(5):
        # Define objective for this tau
        def obj_grad(x_flat):
            return _objective_and_grad(x_flat, tau=current_tau)
        
        # Use L-BFGS-B
        # Bounds: x, y in [-1, 1] (actually constrained by circle, but box is safe)
        bounds = [(-1.0, 1.0)] * (2 * N)
        
        try:
            res = minimize(obj_grad, x_flat, method='L-BFGS-B', jac=True, bounds=bounds,
                           options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
            x_flat = res.x
        except Exception as e:
            # If minimization fails, keep current x
            pass
            
        current_tau *= 0.5
        
    x = x_flat.reshape(-1, 2)
    
    # Polish with exact gradient ascent on hard min
    # This helps to fine-tune the positions based on the exact minimum constraint
    for _ in range(100):
        x = x.reshape(-1, 2)
        n = len(x)
        
        # Compute all clearances to find exact min
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        pair_clear = dist / 2.0
        radii = np.sqrt(np.sum(x**2, axis=1))
        bound_clear = 1.0 - radii
        all_clear = np.concatenate([pair_clear.flatten(), bound_clear])
        r_min = np.min(all_clear)
        
        # Find active constraints (within small epsilon)
        eps = 1e-5
        active_mask = all_clear < (r_min + eps)
        
        grad = np.zeros_like(x)
        
        # Pairwise
        # We need to map flat index to pair (i,j)
        # pair_clear is n x n. Flatten is row-major.
        # Index k corresponds to i = k // n, j = k % n
        # But we only want i < j.
        # Let's just iterate over active pairs.
        
        # Efficiently:
        # active_pairs_mask is same shape as pair_clear (n x n)
        active_pair_mask = active_mask[:n*n].reshape(n, n)
        
        for i in range(n):
            for j in range(i+1, n):
                if active_pair_mask[i, j]:
                    d = dist[i, j]
                    if d < 1e-10:
                        continue
                    direction = (x[i] - x[j]) / d
                    grad[i] += direction / 2.0
                    grad[j] -= direction / 2.0
        
        # Boundary
        active_bound_mask = active_mask[n*n:]
        for i in range(n):
            if active_bound_mask[i]:
                r = radii[i]
                if r < 1e-10:
                    continue
                direction = -x[i] / r
                grad[i] += direction
        
        # Normalize and step
        norm = np.linalg.norm(grad)
        if norm > 1e-10:
            grad /= norm
            # Adaptive step size
            step_size = 0.01
            x_new = x + step_size * grad
            
            # Project to circle
            radii_new = np.sqrt(np.sum(x_new**2, axis=1))
            for i in range(n):
                if radii_new[i] > 0.999:
                    x_new[i] *= 0.999 / radii_new[i]
            
            # Check if improvement
            r_new = _compute_radius(x_new)
            if r_new > r_min:
                x = x_new
            else:
                # Reduce step size if no improvement
                step_size *= 0.5
                if step_size < 1e-6:
                    break
        else:
            break
            
    return x
# EVOLVE-BLOCK-END
