import numpy as np
import time

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    # Precompute all arithmetic progressions as index lists (0-based)
    # An AP: start a (0-based), step s >= 1, length k >= 1
    # indices: a, a+s, a+2s, ..., a+(k-1)s where a+(k-1)s < n
    progs = []
    for a in range(n):
        for s in range(1, n):
            idx = list(range(a, n, s))
            if len(idx) >= 1:
                progs.append(idx)
    
    Np = len(progs)
    
    # For efficient vectorized local search, precompute a sparse incidence structure
    # We need to update sums quickly.
    # Instead of iterating over progs for each position, let's keep the sums array
    # and update it based on the position flipped.
    
    # pos_to_progs[i] = list of prog_ids containing i
    pos_to_progs = [[] for _ in range(n)]
    for pid, idx in enumerate(progs):
        for i in idx:
            pos_to_progs[i].append(pid)
    
    # To make vectorization easier for "try all flips", we can precompute
    # for each position i, the list of prog indices.
    # However, the lengths of these lists vary.
    # A fully vectorized "try all 256 flips" in one go is hard with variable length lists
    # unless we use a dense matrix (256 x Np) which is 256 * ~20000 = 5M entries.
    # 5M * 8 bytes (float) = 40MB. This is feasible.
    
    # Let's build a dense incidence matrix M where M[i, p] = 1 if i in prog p, else 0.
    # Then sums = signs @ M.T (if signs is 1D and M is 2D)
    # Actually, signs is vector of length n. sums is vector of length Np.
    # sums[p] = sum_{i in p} signs[i] = dot(signs, M[:, p])
    # So sums = M.T @ signs
    
    M = np.zeros((n, Np), dtype=np.float32)
    for pid, idx in enumerate(progs):
        for i in idx:
            M[i, pid] = 1.0
            
    # M.T is (Np, n). signs is (n,). sums = M.T @ signs -> (Np,)
    
    def compute_sums(signs):
        return M.T @ signs
        
    def max_abs_sum(sums):
        return np.max(np.abs(sums))
    
    # Initialize with random signs
    rng = np.random.RandomState(42)
    signs = rng.choice([-1, 1], size=n).astype(np.float32)
    sums = compute_sums(signs)
    
    best_signs = signs.copy()
    best_score = max_abs_sum(sums)
    
    # Simulated Annealing parameters
    # Start temperature high enough to accept bad moves, decay exponentially
    T_start = 10.0
    T_end = 0.01
    # We want to do as many iterations as possible.
    # Each iteration:
    # 1. Compute delta for all 256 flips.
    #    If we flip i, signs[i] becomes -signs[i].
    #    New sum for prog p: sums[p] - 2 * signs[i] * M[i, p]
    #    This is not easily vectorized for "all i at once" to get a single "new score"
    #    because the "new score" depends on the max over p of the new sums.
    #    
    #    Alternative: Just do random single flips (standard SA) but optimized.
    #    Or, use a "kick" strategy: random restarts if stuck.
    
    # Given the complexity of vectorizing the max-abs-sum change for all positions simultaneously
    # (it's an L-infinity norm change which is non-separable), let's stick to single flips
    # but make them extremely fast using precomputed arrays.
    
    # Precompute: for each i, the indices in pos_to_progs[i]
    # And maybe the M[i, :] slice? No, M is dense.
    
    # Let's try a different angle: 
    # The current program's bottleneck is the Python loop over 256 positions 
    # and the inner loop over pos_to_progs[i] (which can be large).
    
    # If we use the dense matrix M:
    # To evaluate flipping position i:
    # delta_sums = -2 * signs[i] * M[i, :]  (vector of length Np)
    # new_sums = sums + delta_sums
    # new_score = max_abs_sum(new_sums)
    
    # This is O(Np) per flip. 256 * Np ops per full pass.
    # Np is approx sum_{a,s} (n-a)/s.
    # For n=256, Np is roughly 256 * log(256) ~ 1600? No.
    # Number of APs:
    # s=1: 256
    # s=2: 255
    # ...
    # s=k: 256-k+1
    # Sum_{k=1}^{255} (256-k) = 255*256/2 = 32640.
    # So Np ~ 32640.
    # 256 * 32640 ~ 8.3M operations per full pass.
    # In NumPy, this is fast. 8.3M float ops is < 1ms.
    # So we can do thousands of full passes.
    
    # Let's implement SA with single random flips, but using the dense matrix for fast evaluation.
    
    current_signs = signs.copy()
    current_sums = sums.copy()
    current_score = best_score
    
    # SA Loop
    # We'll do a fixed number of iterations or until time runs out.
    # Let's aim for ~50,000 iterations.
    # Each iter: pick random i, compute new_score, accept/reject.
    
    n_iters = 100000
    
    for it in range(n_iters):
        if time.time() > deadline - 2:
            break
            
        # Decay temperature
        # T = T_start * (T_end/T_start)^(it/n_iters)
        # Or simple exponential: T = T_start * exp(-cooling * it)
        # Let's use linear decay for simplicity and robustness
        T = T_start * (1.0 - it / float(n_iters)) + T_end
        
        # Pick random position
        i = rng.randint(n)
        
        # Compute delta
        # delta_sums = -2 * current_signs[i] * M[i, :]
        # new_sums = current_sums + delta_sums
        # new_score = max(abs(new_sums))
        
        # To avoid allocating new_sums array every time (memory bandwidth), 
        # we can compute the change in max.
        # But max is non-linear. 
        # Let's just compute it. 32k floats is small.
        
        sign_i = current_signs[i]
        delta = -2.0 * sign_i * M[i, :]
        new_sums = current_sums + delta
        new_score = np.max(np.abs(new_sums))
        
        delta_score = new_score - current_score
        
        # Acceptance criterion
        if delta_score <= 0 or rng.rand() < np.exp(-delta_score / T):
            # Accept
            current_signs[i] = -sign_i
            current_sums = new_sums
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_signs = current_signs.copy()
        
        # If we get stuck for a long time, we might restart?
        # SA usually handles this with T decay.
        
    return [int(x) for x in best_signs]
