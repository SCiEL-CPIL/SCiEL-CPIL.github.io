import numpy as np
from scipy.optimize import minimize
import time

N = 200

# EVOLVE-BLOCK-START

def _linear_overlaps(h):
    """
    Compute linear overlap O_k = sum_i h[i] * (1 - h[i+k]) for k=0..N-1.
    """
    # np.correlate(h, 1-h, 'full')
    # The result length is 2N-1.
    # Index N-1 corresponds to lag 0.
    # Index N-1+k corresponds to lag k.
    # We want lags 0 to N-1.
    comp = 1.0 - h
    corr = np.correlate(h, comp, mode='full')
    # corr[N-1 : N-1+N]
    return corr[N-1 : N-1+N]

def _objective_and_grad(h, beta, mu, target_sum):
    """
    Objective: log-sum-exp(beta * overlaps) + mu * (sum(h) - target_sum)^2
    """
    overlaps = _linear_overlaps(h)
    
    max_ov = np.max(overlaps)
    scaled = beta * (overlaps - max_ov)
    exp_scaled = np.exp(scaled)
    sum_exp = np.sum(exp_scaled)
    
    obj_lse = max_ov * beta + np.log(sum_exp)
    
    weights = exp_scaled / sum_exp
    
    # Gradient of overlaps w.r.t h
    # O_k = sum_i h_i (1 - h_{i+k})
    # dO_k/dh_j = (1 - h_{j+k}) - h_{j-k}  (with bounds)
    
    # We need sum_k weights[k] * dO_k/dh_j
    # This is a convolution-like operation.
    # Let's compute it directly or via FFT.
    # Given N=200, direct computation is O(N^2) = 40k ops.
    # Gradient computation in Python loop is slow.
    # Use vectorized numpy.
    
    # dO_k/dh_j is non-zero only if j < N and j+k < N (for first term) and j-k >= 0 (for second term).
    # Actually, O_k = sum_{i=0}^{N-1-k} h_i (1 - h_{i+k}).
    # Term h_j appears in:
    # 1. As h_i: i=j. Requires j <= N-1-k => k <= N-1-j.
    #    Contribution: 1 * (1 - h_{j+k}).
    # 2. As h_{i+k}: i+k=j => i=j-k. Requires j-k >= 0 => k <= j.
    #    Contribution: h_{j-k} * (-1).
    
    # So dO_k/dh_j = (1 - h_{j+k}) * [k <= N-1-j] - h_{j-k} * [k <= j]
    
    # We want G_j = sum_k w_k dO_k/dh_j
    # G_j = sum_{k=0}^{N-1-j} w_k (1 - h_{j+k}) - sum_{k=0}^{j} w_k h_{j-k}
    
    # Term 1: T1_j = sum_{k=0}^{N-1-j} w_k (1 - h_{j+k})
    # Term 2: T2_j = sum_{k=0}^{j} w_k h_{j-k}
    
    # T2_j is a truncated convolution of w and h.
    # T1_j is a truncated correlation.
    
    # For N=200, we can just loop over j?
    # 200 iterations, each O(N). Total O(N^2).
    # In Python, 200 * 200 = 40k operations.
    # This is fast enough (< 0.1s).
    
    grad_ov = np.zeros(N)
    for j in range(N):
        # T1
        # k from 0 to N-1-j
        k_max1 = N - 1 - j
        if k_max1 >= 0:
            # w[0..k_max1]
            # h[j..j+k_max1]
            t1 = np.sum(weights[:k_max1+1] * (1.0 - h[j:j+k_max1+1]))
        else:
            t1 = 0.0
            
        # T2
        # k from 0 to j
        k_max2 = j
        if k_max2 >= 0:
            # w[0..k_max2]
            # h[j..j-k_max2] reversed?
            # h_{j-k}. k=0 -> h_j. k=1 -> h_{j-1}.
            # h[j-k] for k in 0..j is h[j], h[j-1], ..., h[0].
            # This is h[0..j] reversed.
            h_slice = h[:j+1][::-1]
            t2 = np.sum(weights[:k_max2+1] * h_slice)
        else:
            t2 = 0.0
            
        grad_ov[j] = t1 - t2
        
    sum_h = np.sum(h)
    grad_pen = 2.0 * mu * (sum_h - target_sum) * np.ones(N)
    
    total_grad = beta * grad_ov + grad_pen
    total_obj = obj_lse + mu * (sum_h - target_sum)**2
    
    return total_obj, total_grad

def _project_to_simplex(h, target_sum):
    n = len(h)
    h = np.clip(h, 0.0, 1.0)
    for _ in range(50):
        diff = target_sum - np.sum(h)
        if abs(diff) < 1e-12:
            break
        if diff > 0:
            free = h < 1.0 - 1e-10
        else:
            free = h > 1e-10
        if not np.any(free):
            break
        h[free] += diff / np.sum(free)
        h = np.clip(h, 0.0, 1.0)
    return h

def _snap_to_binary(h, target_sum):
    """
    Try to snap h to 0 or 1 to potentially improve discrete score.
    """
    # Sort by distance to 0.5
    h_copy = h.copy()
    # Simple approach: threshold at 0.5
    h_bin = (h_copy > 0.5).astype(float)
    # Adjust sum
    # If sum is too high, set some 1s to 0s (those closest to 0.5)
    # If sum is too low, set some 0s to 1s (those closest to 0.5)
    
    ones = np.where(h_bin == 1)[0]
    zeros = np.where(h_bin == 0)[0]
    
    # Sort ones by h value ascending (closest to 0 first)
    ones_sorted = ones[np.argsort(h_copy[ones])]
    # Sort zeros by h value descending (closest to 1 first)
    zeros_sorted = zeros[np.argsort(-h_copy[zeros])]
    
    current_sum = np.sum(h_bin)
    
    # Remove 1s if sum too high
    while current_sum > target_sum + 1e-6 and len(ones_sorted) > 0:
        idx = ones_sorted.pop(0)
        h_bin[idx] = 0.0
        current_sum -= 1.0
        
    # Add 1s if sum too low
    while current_sum < target_sum - 1e-6 and len(zeros_sorted) > 0:
        idx = zeros_sorted.pop(0)
        h_bin[idx] = 1.0
        current_sum += 1.0
        
    return h_bin

def construct():
    deadline = time.time() + 140.0
    target_sum = N / 2.0
    
    best_h = np.full(N, 0.5)
    best_score = np.max(_linear_overlaps(best_h))
    
    rng = np.random.RandomState(42)
    
    inits = []
    # 1. Uniform
    inits.append(np.full(N, 0.5))
    # 2. Random continuous
    for _ in range(5):
        h_rand = rng.uniform(0, 1, N)
        h_rand = _project_to_simplex(h_rand, target_sum)
        inits.append(h_rand)
    # 3. Random binary
    for _ in range(5):
        h_bin = (rng.rand(N) > 0.5).astype(float)
        # Adjust sum
        # Simple projection
        h_bin = _project_to_simplex(h_bin, target_sum)
        inits.append(h_bin)
    # 4. Block
    h_block = np.zeros(N)
    h_block[:N//2] = 1.0
    inits.append(h_block)
    # 5. Ramp
    h_ramp = np.linspace(0, 1, N)
    h_ramp = _project_to_simplex(h_ramp, target_sum)
    inits.append(h_ramp)
    
    betas = [5.0, 20.0, 50.0, 100.0]
    mus = [10.0, 100.0]
    
    for h0 in inits:
        if time.time() > deadline - 30.0:
            break
            
        current_h = h0.copy()
        
        for beta in betas:
            for mu in mus:
                if time.time() > deadline - 10.0:
                    break
                
                def obj_func(h):
                    return _objective_and_grad(h, beta, mu, target_sum)[0]
                
                def grad_func(h):
                    return _objective_and_grad(h, beta, mu, target_sum)[1]
                
                try:
                    res = minimize(
                        obj_func,
                        current_h,
                        jac=grad_func,
                        method='L-BFGS-B',
                        bounds=[(0.0, 1.0)] * N,
                        options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6}
                    )
                    
                    if res.success:
                        h_opt = res.x
                        h_opt = _project_to_simplex(h_opt, target_sum)
                        
                        # Try binary snap
                        h_snap = _snap_to_binary(h_opt, target_sum)
                        
                        score_opt = np.max(_linear_overlaps(h_opt))
                        score_snap = np.max(_linear_overlaps(h_snap))
                        
                        if score_opt < best_score:
                            best_score = score_opt
                            best_h = h_opt.copy()
                            current_h = h_opt.copy()
                            
                        if score_snap < best_score:
                            best_score = score_snap
                            best_h = h_snap.copy()
                            current_h = h_snap.copy()
                            
                except Exception:
                    pass

    return best_h

# EVOLVE-BLOCK-END
