"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin

    # 1. Initial Configuration: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Flatten for optimizer
    x_flat = x0.flatten()

    # 2. Define Objective Function
    # We use a set of random points on the sphere to approximate the covering radius
    # using a log-sum-exp of squared distances, which is a smooth approximation of the max.
    np.random.seed(42)
    M = 2000  # Number of query points
    # Generate random points on sphere
    u = np.random.uniform(-1, 1, (M, 3))
    norms = np.linalg.norm(u, axis=1, keepdims=True)
    u = u / norms
    # Ensure no points are exactly at poles or equator issues, though random is fine
    
    beta = 10.0 # Sharpness of max approximation

    def objective(x_flat):
        x = x_flat.reshape(N, 3)
        # Normalize to unit sphere
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0 # Avoid division by zero
        x = x / norms
        
        # Compute squared angular distances between query points u and config points x
        # cos_dist = u @ x.T  (M x N)
        # ang_dist = arccos(cos_dist)
        # We want min over i for each j.
        # To avoid arccos issues, we can minimize squared chordal distance or just use cos_dist?
        # Minimizing squared angular distance is better.
        # ang_dist^2 = arccos(c)^2. 
        # For small angles, ang_dist^2 ~ (1-c)^2.
        # Let's use (1 - c)^2 as a proxy for angular distance squared, it's smooth and monotonic for c in [-1, 1].
        
        cos_dists = u @ x.T  # Shape (M, N)
        # Clip to [-1, 1] to avoid numerical errors in sqrt/arccos if we used them
        cos_dists = np.clip(cos_dists, -1.0, 1.0)
        
        # Squared chordal distance is 2 * (1 - cos_dists).
        # This is proportional to sin^2(theta/2), which is monotonic with theta for theta in [0, pi].
        # Minimizing this is equivalent to minimizing theta.
        chordal_dists_sq = 2.0 * (1.0 - cos_dists)
        
        # Find min chordal dist sq for each query point
        min_dists_sq = np.min(chordal_dists_sq, axis=1)
        
        # Log-sum-exp of beta * min_dists_sq
        # To avoid overflow, subtract max
        scaled = beta * min_dists_sq
        max_scaled = np.max(scaled)
        lse = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
        
        return lse / beta

    # 3. Optimization
    best_x = x0.copy()
    best_val = objective(x_flat)
    
    try:
        # Use Powell method, which is derivative-free and robust
        res = scipy.optimize.minimize(
            objective, 
            x_flat, 
            method='Powell',
            options={'maxiter': 5000, 'maxfev': 5000, 'disp': False}
        )
        
        if res.x is not None:
            res_val = objective(res.x)
            if res_val < best_val:
                best_val = res_val
                best_x = res.x.reshape(N, 3)
        
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 5.0:
        # Return current best
        pass
    else:
        # Try a second round with a different initial guess or finer search?
        # Or just refine further.
        # Let's try a local search with smaller steps if time permits.
        # Actually, Powell usually converges well. 
        # We can try running it again from the result to see if it improves.
        try:
            res2 = scipy.optimize.minimize(
                objective, 
                best_x.flatten(), 
                method='Powell',
                options={'maxiter': 2000, 'maxfev': 2000, 'disp': False}
            )
            if res2.x is not None:
                res2_val = objective(res2.x)
                if res2_val < best_val:
                    best_val = res2_val
                    best_x = res2.x.reshape(N, 3)
        except Exception:
            pass

    # Final normalization
    norms = np.linalg.norm(best_x, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    best_x = best_x / norms
    
    return best_x
# EVOLVE-BLOCK-END
