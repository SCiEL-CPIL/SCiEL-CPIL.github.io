"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _max_overlap(h):
    """
    Compute the maximum overlap of h with shifted (1-h).
    Overlap for shift k is sum_{i} h[i] * (1 - h[(i+k)%N]).
    """
    N = len(h)
    max_ov = 0.0
    for k in range(N):
        # Overlap is dot(h, 1 - roll(h, -k))
        # Note: The problem statement says "shifted 1-h".
        # Usually overlap of h and g is integral h*g.
        # Here g(x) = 1 - h(x+s).
        # So overlap(s) = integral h(x) (1-h(x+s)) dx.
        # In discrete: sum h[i] * (1 - h[(i+k)%N]) * dw
        # We can ignore dw as it's constant for ranking, but let's keep it for correctness if needed.
        # Actually, the score is the raw value. The uniform case gives N/4 * dw = 0.5 * dw?
        # Let's check the uniform case value.
        # h=0.5. sum 0.5*(1-0.5) = sum 0.25 = N*0.25.
        # If the evaluator uses the raw sum, uniform is 25.0 for N=100.
        # If it uses integral, it's 25 * (2/100) = 0.5.
        # The prompt says "score is the upper bound on C5" and "evaluator measures the raw value".
        # Usually in these tasks, if not specified as integral, it's the sum or the max sum.
        # Let's assume the evaluator computes the max sum.
        # The seed program returns h. The evaluator computes the score.
        # We just need to return h that minimizes the metric the evaluator uses.
        # Assuming the metric is max_k sum_i h[i](1-h[i+k]).
        shifted = np.roll(h, -k)
        ov = np.dot(h, 1.0 - shifted)
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _soft_max_overlap(h, beta):
    """
    Smooth approximation of max overlap using log-sum-exp.
    """
    N = len(h)
    overlaps = np.zeros(N)
    for k in range(N):
        shifted = np.roll(h, -k)
        overlaps[k] = np.dot(h, 1.0 - shifted)
    
    # Log-sum-exp trick for stability
    m = np.max(overlaps)
    if m == -np.inf:
        return 0.0
    return m + np.log(np.sum(np.exp((overlaps - m) * beta)) / beta)

def _generate_structural_seed(seed_type, N):
    """
    Generate a binary-like structural seed with sum N/2.
    """
    if seed_type == 0:
        # Uniform
        return np.full(N, 0.5)
    elif seed_type == 1:
        # Alternating 0, 1 pattern (scaled to mean 0.5)
        # 0.5, 1.0, 0.5, 1.0 ... sum is N/2 if N even?
        # 0.5*50 + 1.0*50 = 25+50=75 != 50.
        # We need sum = N/2 = 50.
        # Let's use 0.0 and 1.0.
        # 50 zeros, 50 ones.
        # Pattern: 0, 1, 0, 1...
        h = np.zeros(N)
        h[1::2] = 1.0
        # Sum is 50. Correct.
        return h
    elif seed_type == 2:
        # Block pattern: 0...0 1...1
        h = np.zeros(N)
        h[N//2:] = 1.0
        return h
    elif seed_type == 3:
        # Pseudo-random balanced binary
        np.random.seed(123)
        h = np.random.choice([0.0, 1.0], size=N)
        # Ensure sum is exactly N/2
        # Count ones
        ones = np.sum(h)
        if ones > N/2:
            # Turn off some ones
            idx_ones = np.where(h == 1.0)[0]
            np.random.shuffle(idx_ones)
            to_flip = int(ones - N/2)
            h[idx_ones[:to_flip]] = 0.0
        elif ones < N/2:
            # Turn on some zeros
            idx_zeros = np.where(h == 0.0)[0]
            np.random.shuffle(idx_zeros)
            to_flip = int(N/2 - ones)
            h[idx_zeros[:to_flip]] = 1.0
        return h
    else:
        # m-sequence like (maximal length sequence)
        # For N=100, we can't have a perfect m-seq, but we can use a balanced random sequence with low autocorrelation heuristic.
        # Let's just use a different random seed.
        np.random.seed(456)
        h = np.random.choice([0.0, 1.0], size=N)
        ones = np.sum(h)
        if ones > N/2:
            idx_ones = np.where(h == 1.0)[0]
            np.random.shuffle(idx_ones)
            to_flip = int(ones - N/2)
            h[idx_ones[:to_flip]] = 0.0
        elif ones < N/2:
            idx_zeros = np.where(h == 0.0)[0]
            np.random.shuffle(idx_zeros)
            to_flip = int(N/2 - ones)
            h[idx_zeros[:to_flip]] = 1.0
        return h

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_h = None
    best_score = np.inf
    
    # 1. Evaluate Uniform Solution (Baseline)
    h_uniform = np.full(N, 0.5)
    score_uniform = _max_overlap(h_uniform)
    best_h = h_uniform.copy()
    best_score = score_uniform
    
    # 2. Generate Structural Seeds
    seeds = []
    for i in range(5):
        seed = _generate_structural_seed(i, N)
        # Ensure exact sum
        s = np.sum(seed)
        if abs(s - N/2.0) > 1e-9:
            # Adjust one element to fix sum exactly (should be close for binary)
            diff = N/2.0 - s
            # Find an element to adjust
            if diff > 0:
                # Add to a 0 or low value
                idx = np.argmin(seed)
                seed[idx] += diff
            else:
                # Subtract from a 1 or high value
                idx = np.argmax(seed)
                seed[idx] += diff
            seed = np.clip(seed, 0.0, 1.0)
        seeds.append(seed)
        
        # Evaluate raw score of seed
        sc = _max_overlap(seed)
        if sc < best_score:
            best_score = sc
            best_h = seed.copy()
            
    # 3. Refine best seeds using Soft-Min Surrogate + L-BFGS-B
    # We will refine the top 3 distinct seeds (by score) and the uniform seed.
    # Sort seeds by score
    seed_scores = [( _max_overlap(s), i, s) for i, s in enumerate(seeds)]
    seed_scores.sort(key=lambda x: x[0])
    
    candidates_to_refine = [s for _, _, s in seed_scores[:3]]
    if not any(np.allclose(c, best_h) for c in candidates_to_refine):
         candidates_to_refine.append(best_h)
    
    for seed in candidates_to_refine:
        if time.time() > deadline - 10.0:
            break
            
        h_curr = seed.copy()
        # Ensure bounds
        h_curr = np.clip(h_curr, 0.0, 1.0)
        # Fix sum exactly before optimization
        s = np.sum(h_curr)
        diff = N/2.0 - s
        if abs(diff) > 1e-9:
             # Distribute difference evenly, then clip and fix again
             h_curr = h_curr + diff/N
             h_curr = np.clip(h_curr, 0.0, 1.0)
             s = np.sum(h_curr)
             diff = N/2.0 - s
             if abs(diff) > 1e-9:
                 idx = np.argmin(np.abs(h_curr - 0.5))
                 h_curr[idx] += diff
                 h_curr = np.clip(h_curr, 0.0, 1.0)
        
        # Soft-min optimization with annealing
        # Start with low beta (smooth), increase to high beta (sharp)
        betas = [1.0, 5.0, 20.0, 100.0]
        
        for beta in betas:
            if time.time() > deadline - 5.0:
                break
                
            def objective(x):
                # Enforce sum constraint? L-BFGS-B doesn't handle equality constraints well.
                # We will use a penalty term or project back?
                # Better: Optimize in a subspace or just use penalty.
                # Penalty for sum deviation
                sum_dev = np.sum(x) - N/2.0
                penalty = 1000.0 * sum_dev**2
                # Bounds are handled by L-BFGS-B
                val = _soft_max_overlap(x, beta) + penalty
                return val
            
            bounds = [(0.0, 1.0)] * N
            
            try:
                res = minimize(
                    objective, 
                    h_curr, 
                    method='L-BFGS-B', 
                    bounds=bounds, 
                    options={'maxiter': 50, 'ftol': 1e-8, 'disp': False}
                )
                h_opt = res.x
                # Project back to exact sum
                s = np.sum(h_opt)
                diff = N/2.0 - s
                h_opt = h_opt + diff/N
                h_opt = np.clip(h_opt, 0.0, 1.0)
                # Final sum fix
                s = np.sum(h_opt)
                diff = N/2.0 - s
                if abs(diff) > 1e-9:
                    idx = np.argmin(np.abs(h_opt - 0.5))
                    h_opt[idx] += diff
                    h_opt = np.clip(h_opt, 0.0, 1.0)
                
                # Evaluate true max overlap
                sc = _max_overlap(h_opt)
                if sc < best_score:
                    best_score = sc
                    best_h = h_opt.copy()
                    h_curr = h_opt.copy()
            except Exception:
                pass

    # Final check and return
    # Ensure best_h has exact sum
    s = np.sum(best_h)
    diff = N/2.0 - s
    if abs(diff) > 1e-12:
        best_h = best_h + diff/N
        best_h = np.clip(best_h, 0.0, 1.0)
        s = np.sum(best_h)
        diff = N/2.0 - s
        if abs(diff) > 1e-12:
            idx = np.argmin(np.abs(best_h - 0.5))
            best_h[idx] += diff
            best_h = np.clip(best_h, 0.0, 1.0)
            
    return best_h
# EVOLVE-BLOCK-END
