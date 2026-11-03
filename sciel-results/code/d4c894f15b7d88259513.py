import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
import time

N = 50
DEADLINE = 145.0  # safety margin before 150s

def _score(centers):
    """Compute the actual (hard) radius for a set of centers."""
    n = len(centers)
    # Pairwise distances
    min_pair = np.inf
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(centers[i] - centers[j]) / 2.0
            if d < min_pair:
                min_pair = d
    # Wall distances
    min_wall = np.inf
    for i in range(n):
        for k in range(3):
            w = 0.5 - abs(centers[i][k])
            if w < min_wall:
                min_wall = w
    return min(min_pair, min_wall)

def _soft_min_obj(params, sigma):
    """
    params: flat array of shape (N*3,)
    Returns (-soft_min_value, gradient) where soft_min approximates min of all distances.
    We minimize -soft_min to maximize soft_min.
    """
    centers = params.reshape(N, 3)
    
    # Compute all pairwise distances / 2
    # d_ij = |c_i - c_j| / 2
    # Wall distances: w_ik = 0.5 - |c_ik|
    
    # For efficiency, compute all values in a vector
    # Pairwise: N*(N-1)/2 values
    # Wall: N*3 values
    
    vals = []
    grads_pair = np.zeros((N, 3))
    grads_wall = np.zeros((N, 3))
    
    # Pairwise distances
    for i in range(N):
        for j in range(i+1, N):
            diff = centers[i] - centers[j]
            d = np.linalg.norm(diff)
            if d < 1e-12:
                vals.append(1e-12)
                # Gradient undefined at 0, use small value
                g = np.zeros(3)
            else:
                vals.append(d / 2.0)
                # d/2 = |diff|/2, grad w.r.t. c_i = diff/|diff|, w.r.t. c_j = -diff/|diff|
                g = diff / d  # this is grad of |diff|, so grad of d/2 = g/2... wait
                # d = |diff|, so d/2 = |diff|/2
                # grad_{c_i} (|diff|/2) = (diff/|diff|) / 2 = diff/(2*|diff|)
                # grad_{c_j} (|diff|/2) = -diff/(2*|diff|)
            vals = vals  # will process below
    
    # This approach is too slow with Python loops. Let me vectorize.
    # Actually, for 50 spheres, N*(N-1)/2 = 1225 pairs, N*3 = 150 walls = 1375 total values.
    # That's manageable but let's be careful.
    
    # Let me restructure: compute all values, then soft_min, then gradient via chain rule.
    
    # Actually, let me just compute the soft min and its gradient directly.
    # soft_min(v, sigma) = -sigma * log(sum_k exp(-v_k / sigma))
    # d(soft_min)/dv_k = exp(-v_k/sigma) / sum_j exp(-v_j/sigma)
    # So gradient w.r.t. params: sum_k (d(soft_min)/dv_k) * (dv_k/dparams)
    
    # Let me build the values array and the Jacobian.
    
    # For performance, let me use a different approach: compute in chunks.
    
    pass  # placeholder, will rewrite

def construct():
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    best_centers = None
    best_score = -np.inf
    
    # Initial configurations
    def make_grid_init(perturb=0.0, seed=42):
        rng = np.random.RandomState(seed)
        m = int(np.ceil(N ** (1.0/3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        centers = grid[:N].copy()
        if perturb > 0:
            centers += rng.uniform(-perturb, perturb, size=centers.shape)
        # Clip to valid range
        centers = np.clip(centers, -0.49, 0.49)
        return centers
    
    def make_random_init(seed=0):
        rng = np.random.RandomState(seed)
        return rng.uniform(-0.45, 0.45, size=(N, 3))
    
    def make_fcc_like_init(seed=0):
        """Generate FCC-like lattice points."""
        rng = np.random.RandomState(seed)
        # FCC basis vectors
        a = 0.25  # lattice parameter (rough)
        basis = [
            [0, 0, 0],
            [0, a/2, a/2],
            [a/2, 0, a/2],
            [a/2, a/2, 0],
        ]
        pts = []
        for ix in range(-1, 4):
            for iy in range(-1, 4):
                for iz in range(-1, 4):
                    for b in basis:
                        p = np.array([ix*a, iy*a, iz*a]) + np.array(b)
                        if -0.48 <= p[0] <= 0.48 and -0.48 <= p[1] <= 0.48 and -0.48 <= p[2] <= 0.48:
                            pts.append(p)
        pts = np.array(pts)
        # Shuffle and take first N
        rng.shuffle(pts)
        return pts[:N]
    
    init_configs = [
        make_grid_init(perturb=0.0, seed=42),
        make_grid_init(perturb=0.02, seed=1),
        make_grid_init(perturb=0.03, seed=2),
        make_random_init(seed=10),
        make_random_init(seed=20),
        make_fcc_like_init(seed=30),
        make_grid_init(perturb=0.01, seed=3),
        make_random_init(seed=40),
    ]
    
    def optimize_from(centers0, sigma_schedule, max_iter=2000):
        """Optimize starting from centers0 with given sigma schedule."""
        current = centers0.copy().flatten()
        
        for sigma in sigma_schedule:
            if elapsed() > DEADLINE:
                break
            
            def obj_and_grad(params):
                c = params.reshape(N, 3)
                
                # Compute all distance values
                # Pairwise
                vals = []
                # We'll compute gradient as we go
                
                # For soft_min, we need:
                # f = -sigma * logsumexp(-v/sigma)
                # df/dv_k = exp(-v_k/sigma) / sum_j exp(-v_j/sigma)
                # Then df/dparams = sum_k (df/dv_k) * (dv_k/dparams)
                
                # Build v array
                v_list = []
                
                # Pairwise distances
                for i in range(N):
                    for j in range(i+1, N):
                        diff = c[i] - c[j]
                        d = np.linalg.norm(diff)
                        if d < 1e-10:
                            v_list.append(1e-10)
                        else:
                            v_list.append(d / 2.0)
                
                # Wall distances
                for i in range(N):
                    for k in range(3):
                        w = 0.5 - abs(c[i][k])
                        v_list.append(max(w, 1e-10))
                
                v = np.array(v_list)
                
                # Soft min
                # f = -sigma * logsumexp(-v/sigma)
                neg_v_over_sigma = -v / sigma
                # For numerical stability, subtract max
                max_val = np.max(neg_v_over_sigma)
                log_sum = max_val + np.log(np.sum(np.exp(neg_v_over_sigma - max_val)))
                f = -sigma * log_sum
                
                # Gradient: df/dv_k = exp(-v_k/sigma) / sum_j exp(-v_j/sigma)
                # = exp(neg_v_over_sigma_k - max_val) / sum_j exp(neg_v_over_sigma_j - max_val)
                weights = np.exp(neg_v_over_sigma - max_val)
                weights /= np.sum(weights)
                
                # Now compute df/dparams
                grad = np.zeros_like(params)
                
                # Pairwise gradients
                idx = 0
                for i in range(N):
                    for j in range(i+1, N):
                        diff = c[i] - c[j]
                        d = np.linalg.norm(diff)
                        if d < 1e-10:
                            gi = np.zeros(3)
                            gj = np.zeros(3)
                        else:
                            # dv/dc_i = diff/(2*d) where d=|diff|, v=d/2
                            # dv/dc_j = -diff/(2*d)
                            gi = diff / (2.0 * d)
                            gj = -diff / (2.0 * d)
                        grad[i*3:(i+1)*3] += weights[idx] * gi
                        grad[j*3:(j+1)*3] += weights[idx] * gj
                        idx += 1
                
                # Wall gradients
                for i in range(N):
                    for k in range(3):
                        w = 0.5 - abs(c[i][k])
                        if w < 1e-10:
                            gw = np.zeros(3)
                        else:
                            gw = np.zeros(3)
                            if c[i][k] >= 0:
                                gw[k] = -1.0
                            else:
                                gw[k] = 1.0
                        grad[i*3:(i+1)*3] += weights[idx] * gw
                        idx += 1
                
                return f, grad
            
            # Use L-BFGS-B
            try:
                res = minimize(
                    obj_and_grad,
                    current,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': max_iter, 'gtol': 1e-6, 'maxls': 50}
                )
                current = res.x
            except Exception:
                pass
            
            # Project back to valid region (clip)
            c = current.reshape(N, 3)
            c = np.clip(c, -0.4999, 0.4999)
            current = c.flatten()
        
        return current.reshape(N, 3)
    
    # Sigma schedule: start large, decrease
    sigma_schedules = [
        [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01],
        [0.5, 0.25, 0.1, 0.05, 0.02, 0.01],
        [0.25, 0.1, 0.05, 0.02, 0.01, 0.005],
    ]
    
    for init_idx, init_centers in enumerate(init_configs):
        if elapsed() > DEADLINE - 5:
            break
        
        # Choose sigma schedule based on init type
        if init_idx < 3:
            schedule = sigma_schedules[0]
        elif init_idx < 6:
            schedule = sigma_schedules[1]
        else:
            schedule = sigma_schedules[2]
        
        try:
            result = optimize_from(init_centers, schedule, max_iter=1500)
        except Exception:
            continue
        
        if result is not None and np.all(np.isfinite(result)):
            s = _score(result)
            if s > best_score:
                best_score = s
                best_centers = result.copy()
    
    # Final polish with very small sigma on best result
    if best_centers is not None and elapsed() < DEADLINE - 10:
        try:
            polished = optimize_from(best_centers, [0.005, 0.002, 0.001, 0.0005], max_iter=3000)
            if polished is not None and np.all(np.isfinite(polished)):
                s = _score(polished)
                if s > best_score:
                    best_score = s
                    best_centers = polished.copy()
        except Exception:
            pass
    
    if best_centers is None:
        # Fallback to grid
        m = int(np.ceil(N ** (1.0/3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N]
    
    return best_centers
