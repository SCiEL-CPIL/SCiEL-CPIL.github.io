"""Pack 50 equal circles in a unit square, maximize common radius."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 135.0  # seconds, safety margin before 150s kill

def _radius(centers):
    """Compute the max common radius for given centers."""
    # Wall radius = min of (0.5 - |coord|)
    wall_r = np.min(0.5 - np.abs(centers))
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    return min(wall_r, min_pair / 2.0)

def _soft_min_radius(centers, beta=50.0):
    """
    Smooth approximation of the minimum radius.
    Uses log-sum-exp for numerical stability.
    r = min(wall_r, min_pair_dist / 2)
    Let d_i be the list of all constraints:
    - For each point i and wall k: d_{ik} = 0.5 - |c_{ik}|
    - For each pair (i, j): d_{ij} = ||c_i - c_j|| / 2
    
    We want min(d).
    soft_min(d) = -1/beta * log( sum( exp(-beta * d_i) ) )
    """
    # Wall distances
    # c is (N, 2). 
    # dist to left: c[:,0] + 0.5
    # dist to right: 0.5 - c[:,0]
    # dist to bottom: c[:,1] + 0.5
    # dist to top: 0.5 - c[:,1]
    
    d_left = centers[:, 0] + 0.5
    d_right = 0.5 - centers[:, 0]
    d_bottom = centers[:, 1] + 0.5
    d_top = 0.5 - centers[:, 1]
    
    wall_dists = np.concatenate([d_left, d_right, d_bottom, d_top])
    
    # Pairwise distances
    # To avoid N^2 memory if N was huge, but N=50 is small.
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair_dists = dist / 2.0
    
    # Combine all distances
    # We only need the lower triangle for pairs to avoid duplicates, but exp(-beta*inf)=0 so it doesn't matter much, 
    # except for efficiency. N=50, 50*49/2 = 1225 pairs. 200 walls. Total 1425 values.
    # Let's just take the lower triangle.
    idx = np.triu_indices(N, 1)
    pair_dists_lower = pair_dists[idx]
    
    all_dists = np.concatenate([wall_dists, pair_dists_lower])
    
    # Log-sum-exp trick for stability
    # m = -1/beta * log(sum(exp(-beta * d)))
    # Let u = -beta * d
    # max_u = max(u)
    # sum(exp(u)) = exp(max_u) * sum(exp(u - max_u))
    # log(sum(exp(u))) = max_u + log(sum(exp(u - max_u)))
    # m = -1/beta * (max_u + log(sum(exp(u - max_u))))
    
    u = -beta * all_dists
    max_u = np.max(u)
    # If max_u is very negative, exp(u-max_u) is safe.
    sum_exp = np.sum(np.exp(u - max_u))
    log_sum = max_u + np.log(sum_exp)
    
    soft_min = -log_sum / beta
    
    return soft_min

def _objective_smooth(x, beta=50.0):
    """Negative soft-min radius for minimization."""
    centers = x.reshape(N, 2)
    r = _soft_min_radius(centers, beta)
    return -r

def _make_grid(n_per_side, N):
    """Create a grid of N points in [-0.5, 0.5]^2."""
    ticks = (np.arange(n_per_side) + 0.5) / n_per_side - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()

def _jitter(centers, scale):
    """Add random jitter to break symmetries."""
    return centers + np.random.normal(0, scale, size=centers.shape)

def _local_refine_smooth(centers, beta=50.0, time_limit=5.0):
    """
    Use L-BFGS-B to minimize the negative soft-min radius.
    """
    x0 = centers.ravel()
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    start_time = time.time()
    
    # Use L-BFGS-B with bounds
    # maxiter needs to be tuned. 200-500 iterations is usually enough for N=50.
    # We check time in callback.
    def callback(xk):
        if time.time() - start_time > time_limit:
            raise RuntimeError("Time limit exceeded")
        return False

    try:
        res = minimize(
            _objective_smooth, x0, args=(beta,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False}
        )
        # Note: L-BFGS-B doesn't support callback in older scipy versions easily?
        # Actually, it does support 'callback' option in recent scipy.
        # Let's wrap it to be safe or just rely on maxiter.
        # 1000 iterations * 50 evals/iter? No, L-BFGS-B is efficient.
        # 1000 iterations should take < 1s for N=50.
        
        if res.success or res.x is not None:
            return res.x.reshape(N, 2)
    except RuntimeError:
        pass
    except Exception:
        pass
    
    return centers

def construct():
    global DEADLINE
    start_time = time.time()
    
    best_centers = None
    best_r = 0.0
    
    # Generate initial configurations
    inits = []
    
    # 1. 7x7 Grid (49 pts) + 1 center, jittered
    c1 = _make_grid(7, 49)
    c1 = np.vstack([c1, np.array([0.0, 0.0])])
    c1 = _jitter(c1, 0.05)
    inits.append(c1)
    
    # 2. 8x8 Grid (64 pts) -> take 50, jittered
    c2 = _make_grid(8, 50)
    c2 = _jitter(c2, 0.05)
    inits.append(c2)
    
    # 3. Random points
    np.random.seed(42)
    c3 = np.random.uniform(-0.45, 0.45, size=(N, 2))
    inits.append(c3)
    
    # 4. Another random seed
    np.random.seed(123)
    c4 = np.random.uniform(-0.45, 0.45, size=(N, 2))
    inits.append(c4)
    
    # 5. 5x10 Grid? 5*10=50.
    # Ticks x: 5, Ticks y: 10
    # This creates a rectangular grid.
    ticks_x = (np.arange(5) + 0.5) / 5 - 0.5
    ticks_y = (np.arange(10) + 0.5) / 10 - 0.5
    grid_5x10 = np.array([(x, y) for y in ticks_y for x in ticks_x])
    c5 = _jitter(grid_5x10, 0.05)
    inits.append(c5)
    
    # 6. 10x5 Grid
    ticks_x = (np.arange(10) + 0.5) / 10 - 0.5
    ticks_y = (np.arange(5) + 0.5) / 5 - 0.5
    grid_10x5 = np.array([(x, y) for y in ticks_y for x in ticks_x])
    c6 = _jitter(grid_10x5, 0.05)
    inits.append(c6)
    
    # Refine each initial configuration
    # We have ~130s. 6 inits. ~20s each.
    # L-BFGS-B is fast. We can afford more inits or longer runs.
    
    for i, init in enumerate(inits):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        time_left = DEADLINE - 10 - (time.time() - start_time)
        # Allocate time proportionally or fixed per init?
        # Let's give each init 15s max.
        alloc_time = min(15.0, time_left / max(1, len(inits) - i))
        
        # Start with a lower beta (smoother) then increase beta (sharper)
        # Two-stage optimization
        refined1 = _local_refine_smooth(init, beta=20.0, time_limit=alloc_time * 0.5)
        r1 = _radius(refined1)
        
        if r1 > best_r:
            best_r = r1
            best_centers = refined1.copy()
        
        # Second stage with higher beta
        refined2 = _local_refine_smooth(refined1, beta=100.0, time_limit=alloc_time * 0.5)
        r2 = _radius(refined2)
        
        if r2 > best_r:
            best_r = r2
            best_centers = refined2.copy()
            
        # Final polish: simple coordinate descent or just return
        # Let's try a few steps of "push apart" if time permits
        # For now, L-BFGS-B should be sufficient.
        
    # If we have time left, do more random restarts
    while time.time() - start_time < DEADLINE - 5:
        np.random.seed(int(time.time() * 1000) % 10000)
        c_rand = np.random.uniform(-0.45, 0.45, size=(N, 2))
        
        # Quick refine
        time_left = DEADLINE - 5 - (time.time() - start_time)
        if time_left < 2: break
        
        refined = _local_refine_smooth(c_rand, beta=50.0, time_limit=time_left * 0.8)
        r = _radius(refined)
        
        if r > best_r:
            best_r = r
            best_centers = refined.copy()
            
    return best_centers
