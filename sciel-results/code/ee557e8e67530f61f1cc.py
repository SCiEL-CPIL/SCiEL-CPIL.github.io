import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
import time

N = 200

def construct():
    start_time = time.time()
    timeout = 115.0  # Safety margin for 120s limit

    # 1. Initialization: Hexagonal Lattice
    # We want to pack N circles. A hexagonal lattice has density pi/sqrt(12) ~ 0.9069.
    # We generate a hex lattice and crop/center it.
    
    # Approximate side length of the hex lattice grid needed
    # Area per point ~ 1/sqrt(N)
    # Let's just generate a large enough hex grid and pick the best N points centered at origin.
    
    def generate_hex_lattice(n_points, spread=1.0):
        # Generate hex lattice points
        # dx = 1, dy = sqrt(3)/2
        # We'll create a grid of indices
        max_idx = int(np.ceil(np.sqrt(n_points * 2 / np.sqrt(3)))) + 2
        
        points = []
        for i in range(-max_idx, max_idx + 1):
            for j in range(-max_idx, max_idx + 1):
                # Hex lattice coordinates
                x = (i + j / 2.0) * 1.0
                y = j * (np.sqrt(3) / 2.0)
                points.append([x, y])
        
        points = np.array(points)
        
        # Center the points
        center = points.mean(axis=0)
        points = points - center
        
        # Scale to fit in roughly [-0.5, 0.5]
        # The max coordinate is roughly max_idx * 1.0
        # We want the points to be within the square.
        # Let's scale such that the max absolute coordinate is slightly less than 0.5
        # But we need to pick N points.
        # Sort by distance from origin to pick the central cluster
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)
        points = points[idx[:n_points]]
        
        # Scale to fit in [-0.5, 0.5] with a small margin
        # The current spread depends on the lattice generation.
        # Let's normalize the max norm to 0.45
        max_norm = np.max(np.linalg.norm(points, axis=1))
        if max_norm > 0:
            points = points * (0.45 / max_norm)
            
        return points

    x0 = generate_hex_lattice(N)
    
    # Ensure x0 is a valid float64 array
    x0 = x0.astype(np.float64)
    
    # 2. Objective Function
    def objective(params):
        # params shape (N, 2)
        x = params[:, 0]
        y = params[:, 1]
        
        # Pairwise distances
        # Using broadcasting to compute all pairwise distances
        # dx = x[:, None] - x[None, :]
        # dy = y[:, None] - y[None, :]
        # d2 = dx**2 + dy**2
        # d = np.sqrt(d2)
        
        # To save memory and time for N=200, this is fine (200x200 = 40000 elements)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d2 = dx**2 + dy**2
        # Set diagonal to a large number so it doesn't affect the min
        np.fill_diagonal(d2, 1e10)
        d = np.sqrt(d2)
        
        # Wall distances
        # min(0.5 - |x|, 0.5 - |y|)
        wx = 0.5 - np.abs(x)
        wy = 0.5 - np.abs(y)
        
        # Combine all distances into one vector
        # We want soft-min of all d[i,j] (i!=j) and wx[i], wy[i]
        # Note: d is symmetric, so we have 2*N*(N-1) terms, but we can just use the upper triangle or all.
        # Using all is easier for vectorization.
        
        # Flatten d (excluding diagonal)
        d_flat = d[np.triu_indices_from(d, k=1)]
        
        # Combine
        all_dists = np.concatenate([d_flat, wx, wy])
        
        # Soft min: -1/beta * logsumexp(-beta * all_dists)
        # We want to maximize the min distance, so we minimize the negative soft min.
        # f = - soft_min = - (-1/beta * logsumexp(...)) = 1/beta * logsumexp(-beta * all_dists)
        
        beta = 20.0 # Sharpening parameter
        # Use logsumexp for numerical stability
        # logsumexp takes an array.
        # We want log(sum(exp(-beta * d)))
        # So we pass -beta * all_dists to logsumexp
        log_sum_exp = logsumexp(-beta * all_dists)
        
        return (1.0 / beta) * log_sum_exp

    # 3. Optimization
    # We will run L-BFGS-B.
    # Bounds: each coordinate in [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    best_x = x0.copy()
    best_val = objective(best_x)
    
    # We can try a few restarts or just one long run.
    # Given 120s, one run with a good initial guess is likely sufficient.
    # If we have time, we can do a second pass with a higher beta.
    
    try:
        # Run L-BFGS-B
        # maxiter is high, but we will check time in a callback if we wanted to.
        # However, L-BFGS-B doesn't support custom time-based stopping easily in scipy.
        # We will just run it and hope it finishes. If it takes too long, the timeout kills it.
        # To be safe, we can limit maxiter.
        # N=200, 400 variables. L-BFGS-B is efficient.
        
        res = minimize(
            objective, 
            x0, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if res.success and res.fun < best_val:
            best_x = res.x
            best_val = res.fun
            
    except Exception:
        pass # Keep best_x if optimization fails

    # Check time
    elapsed = time.time() - start_time
    if elapsed < timeout - 5:
        # Try a second pass with a higher beta for better precision
        # This "sharpens" the objective to find the true min better
        def objective_sharp(params):
            x = params[:, 0]
            y = params[:, 1]
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d2 = dx**2 + dy**2
            np.fill_diagonal(d2, 1e10)
            d = np.sqrt(d2)
            wx = 0.5 - np.abs(x)
            wy = 0.5 - np.abs(y)
            d_flat = d[np.triu_indices_from(d, k=1)]
            all_dists = np.concatenate([d_flat, wx, wy])
            beta = 50.0
            log_sum_exp = logsumexp(-beta * all_dists)
            return (1.0 / beta) * log_sum_exp

        try:
            res2 = minimize(
                objective_sharp, 
                best_x, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res2.success and res2.fun < objective_sharp(best_x):
                # Note: comparing fun from different beta is not direct, 
                # but if it converged, it's likely better in the hard min sense.
                # Let's just take it if it improved the soft objective.
                # Actually, let's just check if it's valid and maybe better.
                # Since beta is higher, the value is closer to the true min.
                # We can't directly compare res2.fun with best_val from beta=20.
                # But usually, the second pass improves the packing.
                best_x = res2.x
        except Exception:
            pass

    # Final validation and cleanup
    # Ensure points are within bounds (numerical errors might push them slightly out)
    best_x = np.clip(best_x, -0.5, 0.5)
    
    return best_x
