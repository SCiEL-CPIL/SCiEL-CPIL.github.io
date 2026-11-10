import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) coordinates.

    """
    n = 16
    d = 2
    start_time = time.time()
    deadline = start_time + 590  # Safety margin

    def get_score(points):
        """Calculate the exact score R = (d_min / d_max)^2"""
        if points.shape != (n, d):
            return 0.0
        dists = pdist(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def smooth_objective(flat_points):
        """
        Smooth surrogate objective to minimize.
        We want to maximize (d_min / d_max).
        Let d_max_soft = (1/tau) * log(sum(exp(tau * d_i)))
        Let d_min_soft = -(1/tau) * log(sum(exp(-tau * d_i)))
        We maximize log(d_min_soft) - log(d_max_soft) which is equivalent to maximizing the ratio.
        To handle numerical stability, we can just maximize d_min_soft / d_max_soft.
        Actually, minimizing -log(d_min_soft) + log(d_max_soft) is better for gradients.
        """
        points = flat_points.reshape(n, d)
        # Compute pairwise distances
        # Using cdist for clarity, though pdist is faster, we need the full matrix or at least the logic for soft ops
        # For n=16, 120 distances.
        dists = pdist(points)
        
        # Avoid division by zero or log(0)
        dists = np.maximum(dists, 1e-12)
        
        # Temperature parameter for soft min/max
        # Higher tau -> closer to hard min/max
        # We want a balance. Let's try a few taus or adapt.
        # Fixed tau for now.
        tau = 20.0
        
        # Soft Max
        # log(sum(exp(tau * d))) / tau
        # To be stable: tau * (max(d) + log(sum(exp(tau * (d - max(d))))) ) / tau
        d_max_val = np.max(dists)
        log_sum_exp_max = np.log(np.sum(np.exp(tau * (dists - d_max_val))))
        d_max_soft = d_max_val + log_sum_exp_max / tau
        
        # Soft Min
        # -log(sum(exp(-tau * d))) / tau
        d_min_val = np.min(dists)
        log_sum_exp_min = np.log(np.sum(np.exp(-tau * (dists - d_min_val))))
        d_min_soft = d_min_val - log_sum_exp_min / tau
        
        if d_max_soft < 1e-12:
            return 0.0
            
        # We want to maximize d_min_soft / d_max_soft
        # Minimizing -log(d_min_soft) + log(d_max_soft) is a common smooth approximation for maximizing ratio
        # Or simply - (d_min_soft / d_max_soft)
        # The log version is often smoother.
        obj = -np.log(d_min_soft) + np.log(d_max_soft)
        
        return obj

    def exact_objective(flat_points):
        """Objective for local polish using exact metric"""
        points = flat_points.reshape(n, d)
        score = get_score(points)
        return -score

    best_points = None
    best_score = -np.inf

    def update_best(points):
        nonlocal best_points, best_score
        score = get_score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()

    # Generate initial candidates
    candidates = []

    # 1. Hexagonal lattice subset
    # Center at (0,0).
    # Generate a dense grid and pick 16 closest
    grid_hex = []
    for j in range(-3, 4):
        for i in range(-3, 4):
            x = i * np.sqrt(3) + j * np.sqrt(3) / 2
            y = j * 1.5
            grid_hex.append([x, y])
    grid_hex = np.array(grid_hex)
    dists_to_center = np.sum(grid_hex**2, axis=1)
    indices = np.argsort(dists_to_center)[:n]
    hex_candidate = grid_hex[indices]
    candidates.append(hex_candidate)

    # 2. Square grid 4x4
    sq = np.linspace(-1.5, 1.5, 4)
    xx, yy = np.meshgrid(sq, sq)
    sq_candidate = np.column_stack((xx.ravel(), yy.ravel()))
    candidates.append(sq_candidate)

    # 3. Points on a circle
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    rad_candidate = np.column_stack((np.cos(angles), np.sin(angles)))
    candidates.append(rad_candidate)

    # 4. Random points (multiple seeds)
    for seed in [42, 123, 456, 789]:
        np.random.seed(seed)
        rand_candidate = np.random.randn(n, d) * 2
        candidates.append(rand_candidate)
        
    # 5. Two rings
    # Inner ring 4 pts, Outer ring 12 pts?
    # Or 8 and 8?
    # Let's try a specific structure: 1 center, 5 inner, 10 outer?
    # Just random perturbations of hex/square
    for cand in candidates[:2]:
        np.random.seed(1)
        perturbed = cand + np.random.randn(n, d) * 0.1
        candidates.append(perturbed)

    # Evaluate initial candidates
    for cand in candidates:
        update_best(cand)

    # Local Optimization using L-BFGS-B on the smooth objective
    # We will run this from each candidate
    # Since L-BFGS-B is fast, we can afford many restarts.
    
    # Time check
    time_per_opt = 2.0 # Estimated time per L-BFGS-B run
    num_restarts = int((deadline - time.time()) / time_per_opt)
    num_restarts = max(10, min(num_restarts, 100)) # Cap restarts
    
    # We will iterate through candidates and run L-BFGS-B
    # To ensure diversity, we might shuffle or just go through list
    # If we have more time, we can do random restarts
    
    idx = 0
    while time.time() < deadline - 60: # Leave 60s for final polish
        # Get a starting point
        if idx < len(candidates):
            start_pt = candidates[idx].ravel()
            idx += 1
        else:
            # Random restarts
            if time.time() < deadline - 100:
                np.random.seed(int(time.time() * 1000) % 10000)
                start_pt = np.random.randn(n, d) * 2
            else:
                break
                
        try:
            res = minimize(
                smooth_objective, 
                start_pt, 
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.success or res.fun < 100: # Heuristic check
                update_best(res.x.reshape(n, d))
        except Exception:
            pass

    # Final Polish with Nelder-Mead or Powell on the best exact point
    if best_points is not None and time.time() < deadline - 10:
        try:
            # Use Nelder-Mead for robustness on non-smooth exact metric
            res_nm = minimize(
                exact_objective, 
                best_points.ravel(), 
                method='Nelder-Mead', 
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if -res_nm.fun > best_score:
                best_score = -res_nm.fun
                best_points = res_nm.x.reshape(n, d).copy()
        except Exception:
            pass
        
        # Try Powell as well, it can be very precise
        if time.time() < deadline - 5:
            try:
                res_pw = minimize(
                    exact_objective, 
                    best_points.ravel(), 
                    method='Powell', 
                    options={'maxiter': 1000, 'xtol': 1e-10, 'ftol': 1e-12}
                )
                if -res_pw.fun > best_score:
                    best_score = -res_pw.fun
                    best_points = res_pw.x.reshape(n, d).copy()
            except Exception:
                pass

    if best_points is None:
        best_points = np.random.randn(n, d)
        
    return best_points
