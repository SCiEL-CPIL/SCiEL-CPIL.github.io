"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

def _compute_radius(pts):
    """Compute the max radius for given centers in [-0.5, 0.5]^2."""
    if len(pts) < 2:
        return 0.0
    # Min distance between pairs
    tree = cKDTree(pts)
    d, _ = tree.query(pts, k=2)
    min_pair = d[:, 1].min()
    # Distance to boundary
    bounds = 0.5 - np.abs(pts).max(axis=1)
    min_bound = bounds.min()
    r = min(min_pair / 2.0, min_bound)
    return r

def _generate_hex_initialization(seed, scale_factor=1.0):
    """Generate a hexagonal lattice initialization with some jitter."""
    np.random.seed(seed)
    
    # Estimate spacing based on area
    # Area = 1.0, N=200. Area per point ~ 0.005.
    # Hex cell area = s^2 * sqrt(3)/2.
    # s^2 = 0.005 * 2 / sqrt(3) ~ 0.00577 -> s ~ 0.076
    # We want to fit N points.
    # Let's try a few scales to ensure we get enough points inside.
    
    best_pts = None
    best_score = -1
    
    for s in [0.07, 0.075, 0.08]:
        v1 = np.array([s, 0])
        v2 = np.array([s/2, s*np.sqrt(3)/2])
        
        # Generate grid
        max_idx = 15
        indices = np.arange(-max_idx, max_idx + 1)
        i, j = np.meshgrid(indices, indices, indexing='ij')
        x = i * v1[0] + j * v2[0]
        y = i * v1[1] + j * v2[1]
        pts_grid = np.column_stack((x.ravel(), y.ravel()))
        
        # Filter inside square
        mask = (np.abs(pts_grid[:, 0]) <= 0.5) & (np.abs(pts_grid[:, 1]) <= 0.5)
        pts_inside = pts_grid[mask]
        
        if len(pts_inside) < N:
            continue
            
        # Select N closest to center
        dists_to_center = np.sum(pts_inside**2, axis=1)
        idx_sorted = np.argsort(dists_to_center)
        pts = pts_inside[idx_sorted[:N]].copy()
        
        # Add jitter
        pts += np.random.randn(N, 2) * 0.002
        pts = np.clip(pts, -0.499, 0.499)
        
        score = _compute_radius(pts)
        if score > best_score:
            best_score = score
            best_pts = pts
            
    if best_pts is None:
        # Fallback: random
        best_pts = np.random.uniform(-0.4, 0.4, (N, 2))
        
    return best_pts

def _objective(x):
    """
    Objective function for SLSQP.
    We want to maximize the minimum radius r.
    Equivalent to minimizing -r.
    However, r is defined by the minimum of many constraints.
    To make it differentiable and smooth for SLSQP, we can use a barrier-like approach
    or simply minimize the negative of the minimum distance.
    
    Actually, SLSQP works best with explicit constraints.
    Let's reformulate:
    Maximize r
    Subject to:
      ||x_i - x_j|| >= 2r  for all i < j
      x_i in [-0.5+r, 0.5-r] for all i
    
    This is a non-convex problem due to the product r * x in constraints? No, 
    ||x_i - x_j|| >= 2r is convex in (x, r) for fixed r? No, it's non-convex.
    But we can treat r as a variable and minimize -r.
    
    Wait, standard approach:
    Minimize -r
    Constraints:
      (x_i - x_j)^2 + (y_i - y_j)^2 - (2r)^2 >= 0
      x_i - r + 0.5 >= 0
      0.5 - x_i - r >= 0
      etc.
    
    This is a Non-Linear Programming (NLP) problem. SLSQP can handle it.
    Variables: x (400 coords) + r (1 scalar) = 401 variables.
    """
    x = x[:-1].reshape(-1, 2)
    r = x[-1]
    
    # Objective: minimize -r
    return -r

def _constraints(x):
    """
    Return a list of constraint dictionaries for SLSQP.
    """
    x = x[:-1].reshape(-1, 2)
    r = x[-1]
    
    cons = []
    
    # Boundary constraints:
    # x_i >= -0.5 + r  => x_i - r + 0.5 >= 0
    # x_i <= 0.5 - r   => 0.5 - x_i - r >= 0
    # y_i >= -0.5 + r  => y_i - r + 0.5 >= 0
    # y_i <= 0.5 - r   => 0.5 - y_i - r >= 0
    
    # We can combine these into arrays.
    # SLSQP accepts arrays for 'ineq'.
    
    # Lower bounds for x: x[:,0] - r + 0.5 >= 0
    cons.append({'type': 'ineq', 'fun': lambda x, r=r: x[:-1].reshape(-1,2)[:,0] - x[-1] + 0.5})
    # Upper bounds for x: 0.5 - x[:,0] - r >= 0
    cons.append({'type': 'ineq', 'fun': lambda x, r=r: 0.5 - x[:-1].reshape(-1,2)[:,0] - x[-1]})
    # Lower bounds for y: x[:,1] - r + 0.5 >= 0
    cons.append({'type': 'ineq', 'fun': lambda x, r=r: x[:-1].reshape(-1,2)[:,1] - x[-1] + 0.5})
    # Upper bounds for y: 0.5 - x[:,1] - r >= 0
    cons.append({'type': 'ineq', 'fun': lambda x, r=r: 0.5 - x[:-1].reshape(-1,2)[:,1] - x[-1]})
    
    # Pairwise distance constraints:
    # ||x_i - x_j|| >= 2r
    # (x_i - x_j)^2 + (y_i - y_j)^2 - (2r)^2 >= 0
    
    # This is O(N^2) constraints. 200*199/2 = ~20k constraints.
    # SLSQP might struggle with 20k constraints and 400 variables.
    # It might be slow or fail.
    # Alternative: Use a penalty method or reduce constraints?
    # Or use a different solver?
    # SciPy has 'trust-constr' which can handle many constraints better?
    # Let's try SLSQP first, but it might be too slow for 200 points with 20k constraints.
    
    # Given the time limit (150s), SLSQP on 400 vars and 20k constraints might be too slow per iteration.
    # Let's try a different approach: 
    # Minimize a smooth approximation of the minimum distance.
    # Or, use the "soft" constraint approach in the objective?
    
    # Actually, for N=200, the number of constraints is large.
    # Let's check if we can use a simpler objective:
    # Minimize sum of squared violations of the minimum distance?
    
    # Let's stick to the NLP formulation but be careful.
    # If SLSQP is too slow, we can fall back to the force-directed method 
    # but use the NLP result as a better initial guess or local refinement.
    
    # For now, let's add the pairwise constraints.
    # To make it faster, we can only add constraints for pairs that are close?
    # No, SLSQP needs all active/near-active constraints.
    
    # Let's try to generate the constraint functions.
    # This will be a list of 20k lambdas, which is inefficient.
    # Better: One lambda that returns a vector of all violations.
    
    # Let's restructure:
    # Instead of a list of constraints, we can have one 'ineq' constraint that returns a large vector.
    
    # Let's clear the previous constraints and do it properly.
    cons = []
    
    # Boundary constraints as a single vector
    def bound_constraints(x_full):
        x = x_full[:-1].reshape(-1, 2)
        r = x_full[-1]
        # x >= -0.5 + r
        # -x + r - 0.5 <= 0  (SLSQP ineq is fun >= 0)
        # So: x - r + 0.5 >= 0
        cx = x[:, 0] - r + 0.5
        # -x + r - 0.5 <= 0 => 0.5 - x - r >= 0
        cx2 = 0.5 - x[:, 0] - r
        cy = x[:, 1] - r + 0.5
        cy2 = 0.5 - x[:, 1] - r
        return np.concatenate([cx, cx2, cy, cy2])
    
    cons.append({'type': 'ineq', 'fun': bound_constraints})
    
    # Pairwise constraints
    def pair_constraints(x_full):
        x = x_full[:-1].reshape(-1, 2)
        r = x_full[-1]
        n = len(x)
        # Compute all pairwise distances squared
        # diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        # dist_sq = np.sum(diff**2, axis=2)
        # We need dist_sq >= (2r)^2
        # dist_sq - 4r^2 >= 0
        
        # To save memory/time, we can compute only upper triangle?
        # But SLSQP expects a flat vector.
        # Let's compute the full matrix and flatten the upper triangle.
        
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        
        # Mask upper triangle (excluding diagonal)
        i, j = np.triu_indices(n, k=1)
        violations = dist_sq[i, j] - 4 * r * r
        return violations
    
    cons.append({'type': 'ineq', 'fun': pair_constraints})
    
    return cons

def construct():
    t_start = time.time()
    deadline = t_start + 140.0
    
    best_pts = None
    best_r = 0.0
    
    # Multi-start with different seeds
    # We have 140s. Each SLSQP run might take 10-30s?
    # Let's try 3-5 starts.
    
    num_starts = 5
    seeds = [42, 123, 456, 789, 101]
    
    for k in range(num_starts):
        if time.time() > deadline:
            break
            
        print(f"Starting optimization {k+1}/{num_starts}")
        
        # 1. Generate initialization
        init_pts = _generate_hex_initialization(seeds[k])
        
        # 2. Prepare variables for SLSQP
        # x = [x_1, y_1, ..., x_N, y_N, r]
        r_init = _compute_radius(init_pts)
        # Start with a slightly smaller r to be feasible
        r_start = r_init * 0.9
        
        x0 = np.concatenate([init_pts.ravel(), [r_start]])
        
        # 3. Define bounds for variables
        # x, y in [-0.5, 0.5]
        # r in [0, 0.5]
        bounds = [(-0.5, 0.5)] * (2 * N) + [(0, 0.5)]
        
        # 4. Run SLSQP
        # maxiter: limit iterations to avoid infinite loops
        # ftol: tolerance for function change
        
        try:
            res = minimize(
                _objective, 
                x0, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=_constraints(x0), # Note: constraints are static here? 
                # SLSQP calls the constraint functions with the current x.
                # The lambdas inside _constraints capture r from the closure? 
                # No, _constraints returns a list of dicts. The 'fun' functions take x_full.
                # So we should pass the list of constraint dicts.
                options={'maxiter': 100, 'ftol': 1e-8}
            )
            
            if res.success:
                new_pts = res.x[:-1].reshape(-1, 2)
                new_r = _compute_radius(new_pts)
                
                # Ensure validity
                if np.all(np.abs(new_pts) <= 0.5) and np.all(np.isfinite(new_pts)):
                    if new_r > best_r:
                        best_r = new_r
                        best_pts = new_pts.copy()
                        print(f"  Improved: r={new_r}")
            else:
                print(f"  Optimization failed: {res.message}")
                
        except Exception as e:
            print(f"  Exception: {e}")
            continue
            
    # If no valid result from SLSQP, fallback to a simple force-directed or hex grid
    if best_pts is None:
        init_pts = _generate_hex_initialization(42)
        best_pts = init_pts
        best_r = _compute_radius(best_pts)
        
    return best_pts

# EVOLVE-BLOCK-END
