import numpy as np
import time
import scipy.optimize
import scipy.spatial.distance

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) points.

    """
    n = 16
    d = 2
    start_time = time.time()
    deadline = 600 - 10  # 590 seconds to be safe
    
    def get_score(points):
        """Calculate the actual score R = (d_min/d_max)^2"""
        pts = points.reshape(n, d)
        dists = scipy.spatial.distance.pdist(pts)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def smoothed_objective(x, beta=50.0):
        """
        Smoothed objective for L-BFGS-B.
        Uses soft-min and soft-max to create a differentiable approximation of the min/max ratio.
        """
        points = x.reshape(n, d)
        dists = scipy.spatial.distance.pdist(points)
        max_d = np.max(dists)
        
        # Avoid numerical issues if all points are identical
        if max_d < 1e-12:
            return 0.0

        # Soft min: d_min_soft = max_d - (1/beta) * log(sum(exp(-beta * (dists - max_d))))
        # Note: exp(-beta * (dists - max_d)) = exp(beta * (max_d - dists))
        # Let u = beta * (max_d - dists). Max u is 0.
        term_min = np.exp(beta * (max_d - dists))
        d_min_soft = max_d - (1.0 / beta) * np.log(np.sum(term_min))
        
        # Soft max: d_max_soft = max_d + (1/beta) * log(sum(exp(beta * (dists - max_d))))
        # Let v = beta * (dists - max_d). Max v is 0.
        term_max = np.exp(beta * (dists - max_d))
        d_max_soft = max_d + (1.0 / beta) * np.log(np.sum(term_max))
        
        if d_max_soft == 0:
            return 0.0
            
        # We want to maximize (d_min / d_max). 
        # So we minimize -(d_min_soft / d_max_soft).
        return -(d_min_soft / d_max_soft)

    # 1. Generate a diverse set of initial candidates
    # We use a mix of structured and random points.
    # Normalizing initial points to have d_max ~ 1 helps L-BFGS-B convergence.
    
    def normalize(pts):
        dists = scipy.spatial.distance.pdist(pts)
        d_max = np.max(dists)
        if d_max > 0:
            return pts / d_max
        return pts

    candidates = []
    
    # Candidate 1: 4x4 Grid
    grid_x, grid_y = np.meshgrid(np.linspace(-1.5, 1.5, 4), np.linspace(-1.5, 1.5, 4))
    grid_pts = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    candidates.append(normalize(grid_pts))
    
    # Candidate 2: Regular 16-gon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    circle_pts = np.column_stack((np.cos(angles), np.sin(angles)))
    candidates.append(normalize(circle_pts))
    
    # Candidate 3: Hexagonal-like packing
    hex_pts = grid_pts.copy()
    for i in range(1, 4, 2):
        hex_pts[i*4:(i+1)*4, 0] += 0.5
    candidates.append(normalize(hex_pts))
    
    # Candidate 4: Multi-ring (1 center, 5 inner, 10 outer? 1+5+10=16)
    # Center
    c1 = np.array([0, 0])
    # Inner ring (5 points)
    a2 = np.linspace(0, 2*np.pi, 5, endpoint=False) + np.pi/5
    r2 = 1.0
    inner = np.column_stack((r2 * np.cos(a2), r2 * np.sin(a2)))
    # Outer ring (10 points)
    a3 = np.linspace(0, 2*np.pi, 10, endpoint=False)
    r3 = 2.0
    outer = np.column_stack((r3 * np.cos(a3), r3 * np.sin(a3)))
    ring_pts = np.vstack([c1, inner, outer])
    candidates.append(normalize(ring_pts))
    
    # Add 10 random candidates
    np.random.seed(42)
    for _ in range(10):
        rand_pts = np.random.uniform(-2, 2, (n, d))
        candidates.append(normalize(rand_pts))
    
    best_points = None
    best_score = -np.inf
    
    # Evaluate initial candidates
    for c in candidates:
        s = get_score(c)
        if s > best_score:
            best_score = s
            best_points = c.copy()
            
    # 2. Iterative Local Optimization with Adaptive Restarting
    # We will run L-BFGS-B on a population of points.
    # If the best score doesn't improve for a few rounds, we add more random perturbations.
    
    pop_size = 20
    current_pop = [c.copy() for c in candidates[:pop_size]]
    if len(current_pop) < pop_size:
        # Pad with random if needed
        while len(current_pop) < pop_size:
            rand_pts = np.random.uniform(-2, 2, (n, d))
            current_pop.append(normalize(rand_pts))
            
    stagnation_counter = 0
    max_stagnation = 3
    beta_start = 20.0
    beta_end = 100.0
    
    while time.time() < deadline - 10:
        improved_in_round = False
        
        # For each point in the population, run a local optimization
        # We use a slightly higher beta for later rounds to sharpen the objective
        # But for now, we use a fixed beta for simplicity, or adapt it based on stagnation
        # Let's use beta=50 as a good balance.
        beta = 50.0
        
        for i in range(len(current_pop)):
            if time.time() > deadline - 5:
                break
                
            try:
                # Run L-BFGS-B
                # We bound the search to prevent divergence, though L-BFGS-B usually stays close
                # if the objective is well-behaved.
                # Since we normalized, points are within [-1, 1] roughly.
                # Let's allow [-5, 5] just in case.
                bounds = [(-5, 5)] * (n * d)
                
                res = scipy.optimize.minimize(
                    smoothed_objective, 
                    current_pop[i].flatten(), 
                    args=(beta,),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if res.success:
                    polished_pts = res.x.reshape(n, d)
                    polished_score = get_score(polished_pts)
                    
                    # Update population
                    current_pop[i] = polished_pts
                    
                    # Update global best
                    if polished_score > best_score:
                        best_score = polished_score
                        best_points = polished_pts.copy()
                        improved_in_round = True
                        
            except Exception:
                pass

        if improved_in_round:
            stagnation_counter = 0
        else:
            stagnation_counter += 1
            
        # If stagnated, inject new diversity
        if stagnation_counter >= max_stagnation:
            # Add some random perturbations to the worst half of the population
            # Or just replace some with new random points
            num_new = pop_size // 2
            for _ in range(num_new):
                # Perturb the best point found so far
                if best_points is not None:
                    noise = np.random.normal(0, 0.1, (n, d))
                    new_pt = normalize(best_points + noise)
                else:
                    new_pt = normalize(np.random.uniform(-2, 2, (n, d)))
                
                # Replace a random member of the population
                idx = np.random.randint(0, len(current_pop))
                current_pop[idx] = new_pt
                
            stagnation_counter = 0

    # Final Polish: Run L-BFGS-B on the best point with a very high beta to sharpen
    if best_points is not None and time.time() < deadline - 2:
        try:
            res_final = scipy.optimize.minimize(
                smoothed_objective, 
                best_points.flatten(), 
                args=(200.0,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-10}
            )
            if res_final.success:
                final_pts = res_final.x.reshape(n, d)
                final_score = get_score(final_pts)
                if final_score > best_score:
                    best_points = final_pts
        except Exception:
            pass

    if best_points is None:
        best_points = candidates[0]
        
    return best_points
