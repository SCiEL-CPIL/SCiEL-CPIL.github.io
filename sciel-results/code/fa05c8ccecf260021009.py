import numpy as np
import time
import scipy.optimize as opt

N = 14
D = 3

# Pre-compute indices for pairwise distances
# We need i < j pairs
pair_indices = []
for i in range(N):
    for j in range(i + 1, N):
        pair_indices.append((i, j))
pair_indices = np.array(pair_indices)
num_pairs = len(pair_indices)

def compute_distances(X):
    """
    X: (N, D) array
    Returns: array of pairwise distances (num_pairs,)
    """
    # X[pair_indices[:, 0]] - X[pair_indices[:, 1]]
    diff = X[pair_indices[:, 0]] - X[pair_indices[:, 1]]
    # Squared distances
    d2 = np.sum(diff**2, axis=1)
    # Avoid division by zero in gradient if points coincide
    d = np.sqrt(d2 + 1e-12)
    return d

def objective(X):
    """
    Variance of pairwise distances.
    """
    d = compute_distances(X)
    # Variance
    return np.var(d)

def gradient(X):
    """
    Analytic gradient of the variance of pairwise distances.
    V = (1/M) * sum(d_ij^2) - (mean(d_ij))^2
    
    grad_xi V = (2/M) * sum_{j!=i} (x_i - x_j) - (2 * mean_d / M) * sum_{j!=i} (x_i - x_j) / d_ij
    """
    d = compute_distances(X)
    mean_d = np.mean(d)
    
    # We need sum_{j!=i} (x_i - x_j) and sum_{j!=i} (x_i - x_j) / d_ij
    # Let's compute these for all i efficiently or just loop since N=14 is small.
    grad = np.zeros_like(X)
    
    for i in range(N):
        # Find all j != i
        # We can use the precomputed pairs, but it's easier to just loop j
        sum_vec = np.zeros(D)
        sum_vec_inv_d = np.zeros(D)
        
        for j in range(N):
            if i == j:
                continue
            diff = X[i] - X[j]
            d_ij = np.linalg.norm(diff)
            if d_ij < 1e-12:
                # If points are very close, the gradient of 1/d is unstable.
                # But variance gradient term 1 has (x_i-x_j) which is 0.
                # Term 2 has (x_i-x_j)/d_ij which is 0/0.
                # Limit of (x_i-x_j)/||x_i-x_j|| is undefined direction.
                # However, if they are identical, the force should be zero or random.
                # We'll assume they are distinct enough or add a small epsilon.
                continue
            
            sum_vec += diff
            sum_vec_inv_d += diff / d_ij
            
        grad[i] = (2.0 / num_pairs) * sum_vec - (2.0 * mean_d / num_pairs) * sum_vec_inv_d
        
    return grad

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initial guess: Random points in a box, or better, a jittered grid
    # Let's try a few different initializations and pick the best one quickly
    # Or just use one good one.
    
    # A spherical distribution is often a good start for distance packing
    # But random is fine for basin hopping.
    
    x0 = np.random.randn(N, D)
    
    # Define the local minimizer
    # BFGS with analytic gradient
    minimizer_kwargs = {
        'method': 'BFGS',
        'jac': gradient,
        'options': {'maxiter': 500, 'gtol': 1e-8}
    }
    
    best_x = x0
    best_obj = objective(x0)
    
    # Run Basin Hopping
    # We need to be careful with the number of iterations.
    # Each basin hop involves a random step and a local minimization.
    # Local minimization with BFGS on 42 vars is fast.
    
    # We will run a loop of basin hopping manually to allow time checking
    # or use scipy's basinhopping with a callback.
    
    # Let's use scipy.optimize.basinhopping
    # It has a 'niter' parameter. We can estimate how many we can afford.
    # One iteration of basinhopping (step + minimize) might take ~0.1s?
    # Let's assume 50-100 iterations are safe.
    
    # We'll run a loop to be safe and check time.
    
    current_x = x0.copy()
    
    # Simple manual basin hopping loop
    # This allows us to break early if time is up.
    
    # Step size for random perturbation
    stepsize = 1.0 
    
    # We will perform a sequence of random perturbations and local optimizations.
    # This is effectively basin hopping.
    
    # To improve exploration, we can try multiple random restarts if time permits.
    # But first, let's do a long run from one start.
    
    # Actually, let's use scipy's basinhopping for robustness, but wrap it in a time check?
    # Scipy's basinhopping doesn't have a built-in time limit.
    # So manual loop is better.
    
    # Let's do a manual loop:
    # 1. Local Optimize current_x
    # 2. Record best
    # 3. Random Step
    # 4. Check Time
    
    # However, simple random steps might not escape deep local minima.
    # Basin hopping uses a specific acceptance criterion (Metropolis).
    # Let's implement a simple version or just do many random restarts + local polish.
    
    # Strategy: 
    # 1. Try 5 different random initializations.
    # 2. For each, run BFGS to local minimum.
    # 3. Keep the best.
    # 4. If time remains, do basin hopping on the best one.
    
    # Let's start with multiple random restarts.
    
    best_x = None
    best_score = np.inf
    
    num_restarts = 10
    
    for k in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Random initialization
        # Scale to unit sphere roughly?
        x0 = np.random.randn(N, D)
        # Normalize to avoid large/small scales affecting optimizer step size
        # The objective is scale invariant? 
        # Variance of distances scales with scale^2.
        # Ratio is scale invariant.
        # So we can normalize the initial points.
        # Let's project to unit sphere or just normalize center of mass and scale.
        
        # Center
        x0 -= np.mean(x0, axis=0)
        # Scale
        norm = np.linalg.norm(x0, axis=1)
        norm[norm < 1e-12] = 1e-12
        x0 /= norm[:, np.newaxis]
        
        # Local Optimization
        try:
            res = opt.minimize(objective, x0, jac=gradient, method='BFGS', options={'maxiter': 200, 'gtol': 1e-8})
            if res.success:
                obj_val = res.fun
                if obj_val < best_score:
                    best_score = obj_val
                    best_x = res.x
        except Exception:
            pass
            
        # If we have plenty of time, we could do basin hopping here,
        # but let's first see if simple restarts are enough.
        # Given 120s, 10 restarts with BFGS (200 iters) should be fast (<1s per restart).
        
    # If we have time left, do Basin Hopping on the best solution found
    if time.time() < deadline - 10.0 and best_x is not None:
        # Implement a simple basin hopping loop
        current_x = best_x.copy()
        current_obj = best_score
        
        # Metropolis parameters
        T = 0.1 # Temperature
        
        # Run for a fixed number of steps or until time limit
        max_hops = 500
        for i in range(max_hops):
            if time.time() > deadline - 5.0:
                break
                
            # Random Step
            # Perturb each point
            step = np.random.randn(N, D) * 0.5
            new_x = current_x + step
            
            # Local Optimize new_x
            try:
                res = opt.minimize(objective, new_x, jac=gradient, method='BFGS', options={'maxiter': 100, 'gtol': 1e-8})
                if res.success:
                    new_obj = res.fun
                    # Accept if better or with probability exp(-(new_obj - current_obj)/T)
                    if new_obj < current_obj:
                        current_x = res.x
                        current_obj = new_obj
                        if current_obj < best_score:
                            best_score = current_obj
                            best_x = current_x
                    else:
                        prob = np.exp(-(new_obj - current_obj) / T)
                        if np.random.rand() < prob:
                            current_x = res.x
                            current_obj = new_obj
            except Exception:
                pass

    # Final check: Ensure points are distinct
    if best_x is None:
        # Fallback to random
        best_x = np.random.randn(N, D)
        
    # Optional: Final polish with higher precision if time allows
    if time.time() < deadline - 2.0:
        try:
            res = opt.minimize(objective, best_x, jac=gradient, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
            if res.success and res.fun < best_score:
                best_x = res.x
        except Exception:
            pass

    return best_x

# EVOLVE-BLOCK-END
