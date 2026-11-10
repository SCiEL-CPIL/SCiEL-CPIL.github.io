# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time
from scipy.optimize import minimize

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) of the 16 points.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # Safety margin for 630s limit
    
    n_points = 16

    def compute_score(points):
        """Compute the actual metric R = (d_min / d_max)^2"""
        if points.shape != (n_points, 2):
            return -np.inf
        d = pdist(points)
        if d.size == 0:
            return 0.0
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max < 1e-12:
            return 0.0
        ratio = d_min / d_max
        return ratio * ratio

    def normalize_points(points):
        """Scale points such that the maximum pairwise distance is 1."""
        d = pdist(points)
        d_max = np.max(d)
        if d_max > 1e-12:
            return points / d_max
        return points

    def soft_min_objective(x, beta):
        """
        Objective function to minimize: -soft_min(distances).
        The points are normalized to unit diameter internally to fix the scale.
        """
        points = x.reshape(n_points, 2)
        # Normalize to unit diameter
        d = pdist(points)
        d_max = np.max(d)
        if d_max < 1e-12:
            return 0.0
        points = points / d_max
        
        # Recalculate distances after normalization
        d = pdist(points)
        
        # Soft min calculation
        # soft_min = -1/beta * log(sum(exp(-beta * d)))
        # To avoid overflow/underflow, we use the log-sum-exp trick
        exponents = -beta * d
        m = np.max(exponents)
        if m < -700:
            # All distances are very large relative to beta, or numerical issue
            # If beta is high and d is not tiny, exp(-beta*d) is tiny.
            # Actually, if m is very negative, sum_exp will be tiny.
            # Let's just return a large penalty if the soft min is effectively 0
            # But wait, we want to maximize min dist. 
            # If d_min is large, -beta * d is very negative.
            # The term with the smallest d (largest exponent) dominates.
            # If the smallest d is, say, 0.5, and beta=1000, exp(-500) is 0.
            # We need to be careful with numerical stability.
            
            # Let's use a safer formulation.
            # We want to minimize -soft_min.
            # soft_min approximates min(d).
            # If min(d) is 0, soft_min is 0.
            # If min(d) is 0.1, soft_min is approx 0.1.
            
            # Standard stable log-sum-exp:
            # log(sum(exp(x_i))) = m + log(sum(exp(x_i - m)))
            # Here x_i = -beta * d_i.
            # The max x_i corresponds to the min d_i.
            
            shifted = exponents - m
            sum_exp = np.sum(np.exp(shifted))
            if sum_exp == 0:
                return 0.0
            
            soft_min_val = -(m + np.log(sum_exp)) / beta
            
            # We want to maximize soft_min, so we minimize -soft_min
            return -soft_min_val

    def optimize_candidate_powell(points, time_budget=10.0):
        """
        Optimizes a candidate using Powell with increasing beta.
        """
        points = points.copy()
        points = normalize_points(points)
        
        best_points = points.copy()
        best_score = compute_score(best_points)
        
        # Beta schedule: start smooth, end sharp
        betas = [10, 50, 200, 1000, 5000, 10000]
        
        start_opt = time.time()
        
        for beta in betas:
            if time.time() - start_opt > time_budget:
                break
            
            def obj(x):
                return soft_min_objective(x, beta)
            
            try:
                res = minimize(
                    obj, 
                    points.flatten(), 
                    method='Powell', 
                    options={'maxiter': 500, 'maxfev': 5000, 'xtol': 1e-12, 'ftol': 1e-12, 'disp': False}
                )
                
                opt_points = res.x.reshape(n_points, 2)
                opt_points = normalize_points(opt_points)
                opt_score = compute_score(opt_points)
                
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_points.copy()
                    
                # Use the improved points as the start for the next, sharper beta
                points = best_points.copy()
                
            except Exception:
                pass
                
            if time.time() - start_opt > time_budget:
                break
                
        return best_points, best_score

    def generate_structured_candidates():
        """
        Generate a list of diverse, symmetry-based initial candidates.
        """
        candidates = []
        
        # 1. 4x4 Grid (Distorted)
        # Standard grid
        x, y = np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-1, 1, 4))
        grid = np.column_stack((x.ravel(), y.ravel()))
        grid -= np.mean(grid, axis=0)
        candidates.append(grid)
        
        # Radial distortions of the grid
        r = np.linalg.norm(grid, axis=1)
        theta = np.arctan2(grid[:, 1], grid[:, 0])
        for alpha in [0.5, 0.7, 0.8, 1.0, 1.2, 1.5]:
            r_new = r ** alpha
            warped = np.column_stack((r_new * np.cos(theta), r_new * np.sin(theta)))
            candidates.append(warped)

        # 2. Concentric Rings (Flower Patterns)
        # 16 points can be split into rings in various ways.
        
        # Pattern A: 1 Center, 7 Inner, 8 Outer
        for r_inner in [0.4, 0.5, 0.6, 0.7]:
            center = np.array([[0.0, 0.0]])
            ring1_angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
            ring1_points = np.column_stack([r_inner * np.cos(ring1_angles), r_inner * np.sin(ring1_angles)])
            
            # Outer ring 8 points. Offset to maximize distance from inner ring.
            # Inner ring has 7-fold symmetry. Outer has 8.
            # We try a few phase offsets.
            for phase in [0, np.pi/8, np.pi/16]:
                ring2_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False) + phase
                ring2_points = np.column_stack([np.cos(ring2_angles), np.sin(ring2_angles)])
                flower = np.vstack([center, ring1_points, ring2_points])
                candidates.append(flower)

        # Pattern B: 1 Center, 6 Inner, 9 Outer
        for r_inner in [0.4, 0.5, 0.6, 0.7]:
            center = np.array([[0.0, 0.0]])
            ring1_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
            ring1_points = np.column_stack([r_inner * np.cos(ring1_angles), r_inner * np.sin(ring1_angles)])
            
            for phase in [0, np.pi/9, np.pi/18]:
                ring2_angles = np.linspace(0, 2 * np.pi, 9, endpoint=False) + phase
                ring2_points = np.column_stack([np.cos(ring2_angles), np.sin(ring2_angles)])
                flower = np.vstack([center, ring1_points, ring2_points])
                candidates.append(flower)

        # Pattern C: 2 Rings: 8 Inner, 8 Outer
        for r_inner in [0.5, 0.6, 0.7, 0.8]:
            ring1_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
            ring1_points = np.column_stack([r_inner * np.cos(ring1_angles), r_inner * np.sin(ring1_angles)])
            
            # Outer ring offset by half-step to maximize distance?
            for phase in [0, np.pi/8]:
                ring2_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False) + phase
                ring2_points = np.column_stack([np.cos(ring2_angles), np.sin(ring2_angles)])
                flower = np.vstack([ring1_points, ring2_points])
                candidates.append(flower)

        # Pattern D: 3 Rings: 1 Center, 5 Mid, 10 Outer? (1+5+10=16)
        for r_mid in [0.4, 0.5, 0.6]:
            center = np.array([[0.0, 0.0]])
            ring1_angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)
            ring1_points = np.column_stack([r_mid * np.cos(ring1_angles), r_mid * np.sin(ring1_angles)])
            
            for phase in [0, np.pi/10]:
                ring2_angles = np.linspace(0, 2 * np.pi, 10, endpoint=False) + phase
                ring2_points = np.column_stack([np.cos(ring2_angles), np.sin(ring2_angles)])
                flower = np.vstack([center, ring1_points, ring2_points])
                candidates.append(flower)

        # 3. Hexagonal Lattice Subset
        # Generate a large hex lattice and pick the 16 closest to center
        s = 1.0
        h = s * np.sqrt(3) / 2
        lattice_points = []
        for i in range(-4, 5):
            for j in range(-4, 5):
                x = i * s + (j % 2) * s / 2
                y = j * h
                lattice_points.append([x, y])
        lattice_points = np.array(lattice_points)
        centroid = np.mean(lattice_points, axis=0)
        dists_to_centroid = np.sum((lattice_points - centroid) ** 2, axis=1)
        idx = np.argsort(dists_to_centroid)[:n_points]
        candidates.append(lattice_points[idx])
        
        # 4. Random Points (for diversity, just a few)
        np.random.seed(42)
        for seed in range(5):
            np.random.seed(seed * 100 + 7)
            random_points = np.random.randn(n_points, 2)
            candidates.append(random_points)

        return candidates

    best_points = None
    best_score = -np.inf

    candidates = generate_structured_candidates()
    
    # Shuffle candidates to ensure we don't just process the first few if time runs out
    # But keep structured ones prioritized? No, random order is safer for time cutoffs.
    np.random.shuffle(candidates)

    processed_results = []

    # Phase 1: Optimize all candidates with a reasonable time budget
    for i, cand in enumerate(candidates):
        if time.time() > deadline - 30:
            break
            
        # Dynamic time budget: if we have lots of time left, give more to early candidates?
        # Or just fixed small budget per candidate.
        # Let's say 5 seconds per candidate.
        opt_cand, opt_score = optimize_candidate_powell(cand, time_budget=5.0)
        
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_cand.copy()
            
        processed_results.append((opt_score, opt_cand))

    # Sort results by score
    processed_results.sort(key=lambda x: x[0], reverse=True)
    
    # Phase 2: Refine the top candidates with a much higher time budget and sharper betas
    # We take the top 5 and try to push them further.
    top_k = 5
    for score, cand in processed_results[:top_k]:
        if time.time() > deadline - 10:
            break
            
        # Give these more time, e.g., 20 seconds
        # We might also want to re-run with even higher betas or more iterations
        # Let's create a "deep" optimizer call
        def deep_optimize(points, time_budget=20.0):
            points = points.copy()
            points = normalize_points(points)
            best_points = points.copy()
            best_score = compute_score(best_points)
            
            # Higher betas and more iterations
            betas = [500, 2000, 5000, 10000, 20000]
            start_opt = time.time()
            
            for beta in betas:
                if time.time() - start_opt > time_budget:
                    break
                
                def obj(x):
                    return soft_min_objective(x, beta)
                
                try:
                    res = minimize(
                        obj, 
                        points.flatten(), 
                        method='Powell', 
                        options={'maxiter': 1000, 'maxfev': 10000, 'xtol': 1e-14, 'ftol': 1e-14, 'disp': False}
                    )
                    
                    opt_points = res.x.reshape(n_points, 2)
                    opt_points = normalize_points(opt_points)
                    opt_score = compute_score(opt_points)
                    
                    if opt_score > best_score:
                        best_score = opt_score
                        best_points = opt_points.copy()
                        points = best_points.copy()
                except Exception:
                    pass
                    
            return best_points, best_score

        opt_cand, opt_score = deep_optimize(cand, time_budget=20.0)
        
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_cand.copy()

    # Fallback if nothing worked (should not happen)
    if best_points is None:
        best_points = candidates[0]

    return best_points

# EVOLVE-BLOCK-END
