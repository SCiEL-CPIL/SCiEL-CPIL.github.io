"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_radius(centres):
    """Compute the maximum common radius for the given centres."""
    # Minimum distance between any pair
    n = len(centres)
    if n < 2:
        return 0.5
    # Compute all pairwise distances
    diffs = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    # Minimum distance to boundary
    boundary_dists = np.minimum(
        np.minimum(0.5 - centres[:, 0], 0.5 + centres[:, 0]),
        np.minimum(0.5 - centres[:, 1], 0.5 + centres[:, 1])
    )
    min_boundary = np.min(boundary_dists)
    
    return min(min_pair / 2.0, min_boundary)


def _penalty_function(centres, target_r):
    """
    Penalty function: sum of violations.
    For each pair, violation = max(0, 2*target_r - dist_ij)
    For each circle, violation = max(0, target_r - boundary_dist)
    We want to minimize this.
    """
    n = len(centres)
    penalty = 0.0
    
    # Pairwise constraints
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((centres[i] - centres[j])**2))
            if d < 2 * target_r:
                penalty += (2 * target_r - d)**2
    
    # Boundary constraints
    for i in range(n):
        x, y = centres[i]
        for coord in [x, y]:
            dist_to_edge = min(0.5 - coord, 0.5 + coord)
            if dist_to_edge < target_r:
                penalty += (target_r - dist_to_edge)**2
    
    return penalty


def _initial_grid():
    """Create an initial grid configuration."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _initial_random(seed=42):
    """Create a random initial configuration inside the square."""
    rng = np.random.RandomState(seed)
    # Place points in a slightly shrunken square to avoid boundary issues
    margin = 0.05
    centres = rng.uniform(-0.5 + margin, 0.5 - margin, size=(N, 2))
    return centres


def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115s to leave safety margin
    
    best_centres = _initial_grid()
    best_radius = _compute_radius(best_centres)
    
    # Try multiple initial configurations with gradient-based optimization
    # using binary search on radius
    
    # First, get a good lower bound from the grid
    r_lower = best_radius
    
    # Upper bound: theoretical max for 50 circles
    # For a grid of m x m, radius ~ 1/(2*m)
    # For 50 circles, roughly 7x7 grid, radius ~ 1/14 ~ 0.0714
    r_upper = 0.08
    
    # Try several random initializations and optimize
    n_trials = 8
    
    for trial in range(n_trials):
        if time.time() > deadline - 30:
            break
        
        # Choose initial configuration
        if trial == 0:
            centres = _initial_grid()
        else:
            centres = _initial_random(seed=trial * 100 + 7)
        
        # Binary search on radius
        lo = r_lower
        hi = r_upper
        
        for bs_iter in range(12):  # 12 iterations of binary search
            if time.time() > deadline - 25:
                break
            
            mid_r = (lo + hi) / 2.0
            
            # Try to find a valid configuration with radius mid_r
            # using gradient descent on the penalty function
            success = False
            max_inner_iters = 300
            
            current_centres = centres.copy()
            
            # Use L-BFGS-B to minimize penalty
            def objective(x):
                c = x.reshape(N, 2)
                return _penalty_function(c, mid_r)
            
            # Gradient via finite differences is slow; use analytic gradient
            def objective_and_grad(x):
                c = x.reshape(N, 2)
                penalty = 0.0
                grad = np.zeros_like(c)
                
                # Pairwise
                for i in range(N):
                    for j in range(i + 1, N):
                        diff = c[i] - c[j]
                        d = np.sqrt(np.sum(diff**2))
                        if d < 1e-10:
                            d = 1e-10
                        if d < 2 * mid_r:
                            violation = 2 * mid_r - d
                            penalty += violation**2
                            # d(penalty)/d(c_i) = 2*violation * d(violation)/d(c_i)
                            # violation = 2r - d, d(violation)/d(c_i) = -diff/d
                            grad[i] += 2 * violation * (-diff / d)
                            grad[j] += 2 * violation * (diff / d)
                
                # Boundary
                for i in range(N):
                    for axis in range(2):
                        coord = c[i, axis]
                        # Distance to nearest edge in this axis
                        left_dist = 0.5 - coord
                        right_dist = 0.5 + coord
                        if left_dist < right_dist:
                            edge_dist = left_dist
                            edge_grad = -1.0  # d(edge_dist)/d(coord) = -1
                        else:
                            edge_dist = right_dist
                            edge_grad = 1.0   # d(edge_dist)/d(coord) = 1
                        
                        if edge_dist < mid_r:
                            violation = mid_r - edge_dist
                            penalty += violation**2
                            # d(penalty)/d(coord) = 2*violation * d(violation)/d(coord)
                            # violation = r - edge_dist, d(violation)/d(coord) = -edge_grad
                            grad[i, axis] += 2 * violation * (-edge_grad)
                
                return penalty, grad.flatten()
            
            # Optimize
            result = minimize(
                objective_and_grad,
                current_centres.flatten(),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < 1e-10:
                new_centres = result.x.reshape(N, 2)
                actual_r = _compute_radius(new_centres)
                if actual_r >= mid_r - 1e-9:
                    success = True
                    centres = new_centres
                    # Check if we improved best
                    if actual_r > best_radius:
                        best_radius = actual_r
                        best_centres = new_centres.copy()
                    lo = mid_r
                else:
                    hi = mid_r
            else:
                # Check if the result is still valid enough
                new_centres = result.x.reshape(N, 2)
                actual_r = _compute_radius(new_centres)
                if actual_r >= mid_r - 1e-8:
                    success = True
                    centres = new_centres
                    if actual_r > best_radius:
                        best_radius = actual_r
                        best_centres = new_centres.copy()
                    lo = mid_r
                else:
                    hi = mid_r
        
        # After binary search, update lower bound
        if lo > r_lower:
            r_lower = lo
    
    # Final polish: try to improve best_centres with more iterations
    if time.time() < deadline - 10:
        final_r = _compute_radius(best_centres)
        
        def final_objective_and_grad(x):
            c = x.reshape(N, 2)
            penalty = 0.0
            grad = np.zeros_like(c)
            
            # We want to maximize the minimum radius, so minimize the negative
            # of the minimum constraint violation
            # Use a smooth approximation: minimize sum of max(0, 2r - d)^2 
            # where r is current best radius
            r = final_r
            
            for i in range(N):
                for j in range(i + 1, N):
                    diff = c[i] - c[j]
                    d = np.sqrt(np.sum(diff**2))
                    if d < 1e-10:
                        d = 1e-10
                    if d < 2 * r:
                        violation = 2 * r - d
                        penalty += violation**2
                        grad[i] += 2 * violation * (-diff / d)
                        grad[j] += 2 * violation * (diff / d)
            
            for i in range(N):
                for axis in range(2):
                    coord = c[i, axis]
                    left_dist = 0.5 - coord
                    right_dist = 0.5 + coord
                    if left_dist < right_dist:
                        edge_dist = left_dist
                        edge_grad = -1.0
                    else:
                        edge_dist = right_dist
                        edge_grad = 1.0
                    
                    if edge_dist < r:
                        violation = r - edge_dist
                        penalty += violation**2
                        grad[i, axis] += 2 * violation * (-edge_grad)
            
            return penalty, grad.flatten()
        
        result = minimize(
            final_objective_and_grad,
            best_centres.flatten(),
            method='L-BFGS-B',
            jac=True,
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        
        if result.x is not None:
            new_centres = result.x.reshape(N, 2)
            new_r = _compute_radius(new_centres)
            if new_r > best_radius:
                best_radius = new_r
                best_centres = new_centres.copy()
    
    # Ensure validity
    best_centres = np.clip(best_centres, -0.5, 0.5)
    
    return best_centres
# EVOLVE-BLOCK-END
