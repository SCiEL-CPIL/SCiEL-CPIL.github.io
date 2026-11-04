import numpy as np
import time
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin from 150s limit
    
    N = 100
    target_sum = N / 2.0
    
    # Compute correlations C(k) for k=1..N-1
    # C(k) = sum_{i=0}^{N-1-k} h[i] * (1 - h[i+k])
    # Note: The problem asks for overlap of h with shifted 1-h.
    # Shift 0 is excluded.
    def get_correlations_linear(h):
        # Using FFT for speed? For N=100, direct is fine but let's be safe.
        # Direct computation vectorized:
        # For each k, we need dot(h[:N-k], 1-h[k:])
        # This is hard to vectorize over k without loops or matrix tricks.
        # np.correlate is O(N^2) but in C.
        g = 1.0 - h
        # np.correlate(h, g, 'full') gives shifts from -(N-1) to N-1
        # Index N-1 is shift 0.
        c_full = np.correlate(h, g, mode='full')
        # We want shifts 1..N-1 and -(N-1)..-1
        # c_full[:N-1] are negative shifts (h shifted right? or g shifted left?)
        # correlate(h, g)[k] = sum_i h[i+k] g[i] ? No.
        # numpy correlate(a, v): sum_i a[i+k] v[i]
        # So c_full[N-1+k] = sum_i h[i+k] g[i] = sum_i h[i+k] (1-h[i])
        # This is C(-k) if we define C(k) = sum h[i] (1-h[i+k]).
        # Let's check:
        # C(k) = sum_{i=0}^{N-1-k} h[i] (1-h[i+k])
        # Let j = i+k => i = j-k. Sum_{j=k}^{N-1} h[j-k] (1-h[j])
        # This matches correlate(h, g)[N-1+k] = sum_j h[j-k] g[j] ?
        # numpy correlate(a, v, mode='full')[N-1+k] = sum_i a[i+k] v[i]
        # So yes, c_full[N:] corresponds to positive shifts k=1..N-1 of h relative to g?
        # Wait, standard definition:
        # r[k] = sum_n a[n+k] v[n]
        # If a=h, v=g=1-h:
        # r[k] = sum_n h[n+k] (1-h[n])
        # This is sum of h at later indices times (1-h) at earlier indices.
        # The task: "overlap of h with shifted 1-h".
        # Usually overlap of f and g shifted by k is sum f[i] g[i+k].
        # Here f=h, g=1-h.
        # Overlap(k) = sum_i h[i] (1-h[i+k]).
        # This is sum_i h[i] g[i+k].
        # This is correlate(g, h)[N-1+k]?
        # correlate(g, h)[N-1+k] = sum_i g[i+k] h[i] = sum_i (1-h[i+k]) h[i]. Yes.
        # So we need correlate(1-h, h, 'full').
        
        c_full = np.correlate(g, h, mode='full')
        # c_full[N-1] is shift 0.
        # c_full[N:] are shifts 1..N-1.
        # c_full[:N-1] are shifts -1..-(N-1).
        # Due to symmetry of the max operation, we just need the unique shifts.
        # But let's keep all non-zero shifts to be safe.
        c_nonzero = np.concatenate([c_full[:N-1], c_full[N:]])
        return c_nonzero

    def max_overlap(h):
        c = get_correlations_linear(h)
        return np.max(c)
    
    # Smooth max using log-sum-exp with temperature
    def smooth_max(c, temp=0.05):
        scaled = c / temp
        m = np.max(scaled)
        if m > 700:  # Avoid overflow
            return m * temp
        return temp * (m + np.log(np.sum(np.exp(scaled - m))))
    
    def objective_smooth(h, temp=0.05):
        c = get_correlations_linear(h)
        val = smooth_max(c, temp)
        # Penalty for sum constraint to help optimizer stay close to feasible
        # L-BFGS-B handles bounds, but sum is linear equality.
        # We will use a penalty method or just project?
        # Minimize doesn't support linear equality easily with L-BFGS-B.
        # We can use SLSQP or just rely on the fact that SA provides a good start
        # and the penalty keeps it close.
        sum_err = np.sum(h) - target_sum
        return val + 10.0 * sum_err**2

    # Projection onto feasible set: 0 <= h_i <= 1, sum(h) = N/2
    def project(h):
        h = np.clip(h, 0, 1)
        # Adjust sum
        for _ in range(20):
            diff = target_sum - np.sum(h)
            if abs(diff) < 1e-12:
                break
            h += diff / N
            h = np.clip(h, 0, 1)
        return h
    
    # Swendsen-Wang style: identify clusters of similar heights
    def find_clusters(h, threshold=0.1):
        """Find contiguous blocks where heights are within threshold of each other"""
        clusters = []
        i = 0
        while i < N:
            j = i
            base_val = h[i]
            while j < N and abs(h[j] - base_val) < threshold:
                j += 1
            if j > i:
                clusters.append((i, j))
            i = j
        return clusters
    
    def cluster_move(h, clusters, rng):
        """Perform a non-local move on a cluster: shift all values in the cluster"""
        if not clusters:
            return h.copy()
        
        # Pick a random cluster
        idx = rng.integers(0, len(clusters))
        i, j = clusters[idx]
        
        # Generate a new value for the entire cluster
        h_new = h.copy()
        
        # Strategy: shift the cluster's values by a random amount
        shift = rng.uniform(-0.2, 0.2)
        h_new[i:j] += shift
        
        # Project back to feasible set
        h_new = project(h_new)
        return h_new
    
    def cluster_flip(h, clusters, rng):
        """Flip a cluster: replace values with 1 - values (antiferromagnetic flip)"""
        if not clusters:
            return h.copy()
        
        idx = rng.integers(0, len(clusters))
        i, j = clusters[idx]
        
        h_new = h.copy()
        h_new[i:j] = 1.0 - h[i:j]
        h_new = project(h_new)
        return h_new
    
    # Initialize
    rng = np.random.default_rng(42)
    
    # Start with constant solution
    h_curr = np.full(N, 0.5)
    best_h = h_curr.copy()
    best_score = max_overlap(best_h)
    
    # Simulated annealing with cluster moves
    T_init = 0.1
    T_final = 0.001
    n_iterations = 5000
    
    for it in range(n_iterations):
        if time.time() - start_time > time_limit - 20:  # Leave 20s for polish
            break
        
        # Temperature schedule
        progress = it / n_iterations
        T = T_init * (T_final / T_init) ** progress
        
        # Find clusters
        clusters = find_clusters(h_curr, threshold=0.05)
        
        # Choose move type
        move_type = rng.random()
        if move_type < 0.4:
            h_prop = cluster_move(h_curr, clusters, rng)
        elif move_type < 0.7:
            h_prop = cluster_flip(h_curr, clusters, rng)
        else:
            # Single-point perturbation
            h_prop = h_curr.copy()
            i = rng.integers(0, N)
            h_prop[i] += rng.uniform(-0.1, 0.1)
            h_prop = project(h_prop)
        
        # Acceptance criterion
        score_curr = max_overlap(h_curr)
        score_prop = max_overlap(h_prop)
        
        delta = score_prop - score_curr
        if delta < 0 or rng.random() < np.exp(-delta / T):
            h_curr = h_prop
        
        # Track best
        curr_score = max_overlap(h_curr)
        if curr_score < best_score:
            best_score = curr_score
            best_h = h_curr.copy()
    
    # Final refinement: Use scipy.optimize.minimize on the smooth objective
    if time.time() - start_time < time_limit - 5:
        h_init = best_h.copy()
        
        # Define bounds
        bounds = [(0, 1)] * N
        
        # Try multiple temperatures for the smooth max to balance exploration/exploitation
        for temp in [0.1, 0.05, 0.01]:
            if time.time() - start_time > time_limit - 2:
                break
            
            # Reshape objective for scipy
            def obj_scipy(x, t=temp):
                return objective_smooth(x, t)
            
            try:
                res = minimize(obj_scipy, h_init, method='L-BFGS-B', bounds=bounds, 
                               options={'maxiter': 1000, 'maxfun': 10000, 'gtol': 1e-8})
                h_opt = res.x
                # Project to ensure exact feasibility
                h_opt = project(h_opt)
                score_opt = max_overlap(h_opt)
                
                if score_opt < best_score:
                    best_score = score_opt
                    best_h = h_opt.copy()
                    h_init = h_opt.copy()
            except Exception:
                pass

    # Ensure final solution is valid
    best_h = project(best_h)
    
    return best_h
# EVOLVE-BLOCK-END
