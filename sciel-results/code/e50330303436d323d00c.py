import numpy as np
import time
import random

N = 600
# Safety margin: The evaluator kills at 150s. We aim to finish well before.
TIME_LIMIT = 145

def compute_score(f):
    """
    Compute the normalized maximum of the autoconvolution.
    f: array of shape (N,)
    Returns: max(conv) / (sum(f))^2
    """
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    
    # Optimization: Use FFT for convolution if N is large enough, 
    # though for N=600, direct np.convolve is often fast enough.
    # However, we will compute conv once here for the score.
    # Note: For the coordinate descent, we will update conv incrementally.
    
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
        
    # np.convolve(f, f, 'full') has length 2N-1
    conv = np.convolve(f, f, mode='full')
    return np.max(conv) / (s * s)

def update_conv_incremental(conv, f, i, old_val, new_val):
    """
    Update the autoconvolution array when f[i] changes from old_val to new_val.
    conv[k] = sum_j f[j] f[k-j]
    The terms involving f[i] are:
    f[i] * f[k-i] and f[k-i] * f[i]
    So the change in conv[k] for all k is:
    delta_conv[k] = (new_val - old_val) * (f[k-i] + f[k-i]) 
                   = 2 * (new_val - old_val) * f[k-i]
    Wait, let's look at the indices carefully.
    conv = f * f.
    conv[k] = sum_{j=0}^{N-1} f[j] * f[k-j]
    When f[i] changes by delta, the terms where j=i or k-j=i change.
    Term 1: j=i -> f[i] * f[k-i]. Change: delta * f[k-i].
    Term 2: k-j=i -> j=k-i -> f[k-i] * f[i]. Change: f[k-i] * delta.
    Total change at index k: 2 * delta * f[k-i].
    
    Valid range for k:
    f indices are 0..N-1.
    k-i must be in 0..N-1 => i <= k <= i+N-1.
    Also k must be in 0..2N-2.
    So k ranges from max(0, i) to min(2N-2, i+N-1).
    
    Note: If i == k-i (i.e., 2i=k), the term is f[i]^2.
    The change is (new^2 - old^2) = (new-old)(new+old).
    The formula 2*delta*f[k-i] gives 2*delta*old_val.
    Difference: (new^2 - old^2) - 2*delta*old = (new-old)(new+old) - 2(new-old)old = (new-old)(new-old) = delta^2.
    So we need to add delta^2 to conv[2i] specifically.
    
    However, for small steps, delta^2 is negligible. 
    But for accuracy, let's handle it.
    Actually, simpler: 
    Old contribution to conv[k] from f[i]:
    If k-i in [0, N-1]:
      If k-i != i: 2 * old_val * f[k-i]
      If k-i == i: old_val^2
    New contribution:
      If k-i != i: 2 * new_val * f[k-i]
      If k-i == i: new_val^2
    
    Change:
      If k-i != i: 2 * (new_val - old_val) * f[k-i]
      If k-i == i: new_val^2 - old_val^2
    
    Let's implement this.
    """
    delta = new_val - old_val
    if abs(delta) < 1e-12:
        return conv
        
    # Range of k affected
    k_min = max(0, i)
    k_max = min(2*N - 2, i + N - 1)
    
    # Indices of f corresponding to k-i
    # k goes from k_min to k_max
    # f_idx = k - i
    f_indices = np.arange(k_min - i, k_max - i + 1)
    
    # The part of f involved
    f_vals = f[f_indices]
    
    # Base update: 2 * delta * f_vals
    conv[k_min:k_max+1] += 2 * delta * f_vals
    
    # Correction for the diagonal term (where k-i == i => k == 2i)
    if 2*i >= k_min and 2*i <= k_max:
        # We added 2 * delta * old_val.
        # We want new_val^2 - old_val^2.
        # Difference is (new^2 - old^2) - 2*delta*old = delta^2.
        conv[2*i] += delta * delta
        
    return conv

def coordinate_descent(f_init, max_iters=500, time_limit=10.0):
    """
    Perform coordinate descent on the step heights.
    """
    start_time = time.time()
    f = f_init.copy()
    f = np.maximum(f, 0.0)
    
    # Initialize convolution
    conv = np.convolve(f, f, mode='full')
    s = np.sum(f)
    
    if s < 1e-15:
        return f, np.inf
        
    current_score = np.max(conv) / (s * s)
    best_f = f.copy()
    best_score = current_score
    
    # To speed up, we can randomly permute the order of coordinates or just iterate.
    # Iterating 0..N-1 is fine.
    
    # Pre-compute valid indices for each i to avoid recalculating bounds?
    # For N=600, it's cheap enough.
    
    for iter in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        # Try to improve each coordinate
        improved = False
        for i in range(N):
            if time.time() - start_time > time_limit:
                break
                
            old_val = f[i]
            
            # Heuristic: Try a few candidate values for f[i]
            # 1. Keep as is
            # 2. Slightly increase
            # 3. Slightly decrease
            # 4. Set to 0
            # 5. Set to mean
            # 6. Optimize via 1D search? 
            # Since the objective is max(conv)/s^2, and conv is linear in f[i] (mostly),
            # the function of f[i] is roughly (A * f[i] + B) / (C + f[i])^2.
            # This is a unimodal function for f[i] > 0.
            
            # Let's do a simple ternary search or golden section search on f[i] in [0, max_f*1.5]
            # Or just try a few steps.
            # Given the time budget, a few steps is safer.
            
            # Current peak of conv
            peak_idx = np.argmax(conv)
            
            # Candidates
            # If old_val is 0, try small positive.
            # If old_val is large, try smaller.
            
            # Let's estimate the optimal f[i] analytically?
            # Let conv_new = conv + delta * (2 * f_shifted) + delta^2 * e_{2i}
            # s_new = s + delta
            # Score(delta) = max(conv_new) / (s+delta)^2
            # This is complex because max() is non-smooth.
            
            # Simpler: Try local perturbations.
            # Scale factor based on current value
            step_size = max(0.1, old_val * 0.5)
            
            candidates = [
                old_val,
                old_val + step_size,
                old_val - step_size,
                0.0,
                np.mean(f) if np.sum(f)>0 else 1.0
            ]
            
            # Ensure non-negative
            candidates = [max(0.0, c) for c in candidates]
            
            best_cand_val = old_val
            best_cand_score = current_score
            
            # We need to evaluate score for each candidate.
            # To avoid full conv recalculation, we use incremental update.
            # But we need to restore conv after each try.
            # So we copy conv? No, we can revert.
            
            conv_backup = conv.copy() # This is expensive (1200 floats). 
            # Actually, copying 1200 floats is very fast (~microseconds).
            # N=600, conv len 1199.
            
            for cand_val in candidates:
                if abs(cand_val - old_val) < 1e-9:
                    continue
                    
                # Update conv
                conv_temp = update_conv_incremental(conv_backup, f, i, old_val, cand_val)
                
                s_temp = s + (cand_val - old_val)
                if s_temp < 1e-15:
                    continue
                    
                score_temp = np.max(conv_temp) / (s_temp * s_temp)
                
                if score_temp < best_cand_score:
                    best_cand_score = score_temp
                    best_cand_val = cand_val
            
            # Apply best candidate
            if best_cand_val != old_val:
                f[i] = best_cand_val
                s = s + (best_cand_val - old_val)
                conv = update_conv_incremental(conv, f, i, old_val, best_cand_val)
                current_score = best_cand_score
                improved = True
                
                if current_score < best_score:
                    best_score = current_score
                    best_f = f.copy()
                    
        if not improved:
            # No improvement in a full sweep, might be stuck in local min.
            # Break early to save time for restarts?
            # Or do a random restart of coordinates?
            # Let's just break.
            break
            
    return best_f, best_score

def construct():
    start_time = time.time()
    
    # Initial best: Constant function
    f_const = np.ones(N)
    best_f, best_score = coordinate_descent(f_const, max_iters=20, time_limit=2.0)
    
    # List of initializations to try
    # 1. Gaussian centered at 0
    # 2. Gaussian centered at 0.1
    # 3. Box in middle
    # 4. Triangle
    # 5. Random smooth
    # 6. Random sparse
    
    x = np.linspace(-0.25, 0.25, N)
    
    inits = []
    
    # Gaussian 0
    f_g1 = np.exp(-((x - 0.0) / 0.1) ** 2)
    inits.append(f_g1)
    
    # Gaussian 0.1
    f_g2 = np.exp(-((x - 0.1) / 0.1) ** 2)
    inits.append(f_g2)
    
    # Box
    f_box = np.zeros(N)
    f_box[N//4 : 3*N//4] = 1.0
    inits.append(f_box)
    
    # Triangle
    f_tri = np.maximum(0, 1 - 4*np.abs(x))
    inits.append(f_tri)
    
    # Random smooth (sum of a few gaussians)
    f_rand1 = np.zeros(N)
    for _ in range(3):
        c = random.uniform(-0.2, 0.2)
        w = random.uniform(0.05, 0.15)
        f_rand1 += np.exp(-((x - c) / w) ** 2)
    inits.append(f_rand1)
    
    # Random noise (smoothed)
    f_rand2 = np.random.rand(N)
    # Simple smoothing
    kernel = np.ones(5)/5
    f_rand2 = np.convolve(f_rand2, kernel, mode='same')
    inits.append(f_rand2)
    
    # Process each initialization
    # Budget allocation:
    # Total time ~ 140s.
    # 6 inits. ~20s each?
    # Coordinate descent is fast per iteration.
    # Let's give each init 15s.
    
    time_per_init = 15.0
    
    for idx, f_init in enumerate(inits):
        if time.time() - start_time > TIME_LIMIT - 10:
            break
            
        # Normalize f_init to have sum 1? 
        # The score is scale invariant: max(c*conv)/(c*s)^2 = max(conv)/s^2.
        # So scaling doesn't matter. But numerical stability might.
        # Let's normalize to sum=1.
        s_init = np.sum(f_init)
        if s_init > 1e-6:
            f_init = f_init / s_init
        
        f_res, score_res = coordinate_descent(f_init, max_iters=1000, time_limit=time_per_init)
        
        if score_res < best_score:
            best_score = score_res
            best_f = f_res
            
        # Optional: Random restart from current best with noise
        if time.time() - start_time < TIME_LIMIT - 15:
            f_noise = best_f + np.random.randn(N) * 0.01
            f_noise = np.maximum(f_noise, 0)
            # Short run
            f_res2, score_res2 = coordinate_descent(f_noise, max_iters=200, time_limit=5.0)
            if score_res2 < best_score:
                best_score = score_res2
                best_f = f_res2

    # Final check
    # Ensure non-negative
    best_f = np.maximum(best_f, 0.0)
    
    # Return list of floats
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
