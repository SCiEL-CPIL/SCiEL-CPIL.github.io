"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_TIME_LIMIT = 145.0  # Safety margin before 150s kill

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _compute_overlaps(h):
    """
    Compute O_k = sum_{j=0}^{N-k-1} h[j] * (1 - h[j+k]) for k=1..N-1.
    """
    n = len(h)
    overlaps = np.zeros(n - 1)
    for k in range(1, n):
        # h[:n-k] * (1 - h[k:])
        overlaps[k-1] = np.sum(h[:n-k] * (1.0 - h[k:]))
    return overlaps

def _logsumexp_overlaps(h, beta):
    """
    Compute (1/beta) * log(sum(exp(beta * O_k)))
    and its gradient.
    """
    n = len(h)
    overlaps = _compute_overlaps(h)
    
    # Numerical stability
    max_overlap = np.max(overlaps)
    if max_overlap > 500 / beta:
        # Scale to prevent overflow
        # log(sum(exp(beta * O))) = beta * max_O + log(sum(exp(beta * (O - max_O))))
        # (1/beta) * log(...) = max_O + (1/beta) * log(sum(exp(beta * (O - max_O))))
        shifted = beta * (overlaps - max_overlap)
        log_sum = np.log(np.sum(np.exp(shifted)))
        obj = max_overlap + log_sum / beta
        # Gradient w.r.t h
        # d/dh (1/beta log sum exp(beta O)) = sum (exp(beta O_k) / sum exp(beta O_k)) * dO_k/dh
        # Weights are softmax(beta * overlaps)
        weights = np.exp(shifted)
        weights /= np.sum(weights)
    else:
        shifted = beta * overlaps
        log_sum = np.log(np.sum(np.exp(shifted)))
        obj = log_sum / beta
        weights = np.exp(shifted)
        weights /= np.sum(weights)
        
    # Calculate gradient of the objective w.r.t h
    # Grad = sum_k weights[k] * dO_k/dh
    grad = np.zeros(n)
    
    # Precompute weights
    # w_k corresponds to overlap index k-1
    for k in range(1, n):
        w = weights[k-1]
        if w == 0:
            continue
            
        # dO_k/dh_i
        # Term 1: I(i < n-k)
        # Term 2: -I(i < n-k) * h[i+k]
        # Term 3: -I(i >= k) * h[i-k]
        
        # Efficient calculation:
        # Sum over k of w_k * I(i < n-k) = Sum_{k=1, k < n-i} w_k
        # Sum over k of w_k * I(i < n-k) * h[i+k] = Sum_{k=1, k < n-i} w_k * h[i+k]
        # Sum over k of w_k * I(i >= k) * h[i-k] = Sum_{k=1, k <= i} w_k * h[i-k]
        
        # This loop is O(N^2) total if done naively for each i.
        # N=100, N^2=10000. This is fine.
        pass 

    # Let's compute grad[i] directly for each i
    for i in range(n):
        g_i = 0.0
        # k ranges from 1 to n-1
        # Condition 1: i < n-k  => k < n-i
        # Condition 2: i >= k   => k <= i
        
        # We can iterate k, but that's O(N^2) per i -> O(N^3). Too slow?
        # N=100. 100^3 = 1e6. Python loop might be slow.
        # Let's vectorize or optimize.
        
        # Actually, let's just compute the full gradient matrix contribution?
        # Or use convolution?
        
        # dO_k/dh = e_{0..n-k-1} - h_{k..n-1} (shifted) - h_{0..n-k-1} (shifted)
        # Let A_k be the vector dO_k/dh.
        # A_k[i] = 1 if i < n-k else 0
        #         - h[i+k] if i < n-k else 0
        #         - h[i-k] if i >= k else 0
        
        # We can precompute the sum of weights for each condition?
        # No, h[i+k] depends on i and k.
        
        # Let's stick to the O(N^2) loop over k inside the objective/grad function.
        # The grad function is called many times.
        # Optimization:
        # grad = sum_k w_k * A_k
        
        # A_k = ones(0..n-k-1) - h[k:n] (padded) - h[0:n-k] (padded reversed?)
        
        # Let's do it efficiently.
        # Term 1: S1[i] = sum_{k=1}^{n-1-i} w_k  (if n-1-i >= 1)
        # Term 2: S2[i] = sum_{k=1}^{n-1-i} w_k * h[i+k]
        # Term 3: S3[i] = sum_{k=1}^{i} w_k * h[i-k]
        
        # We can compute S1, S2, S3 for all i using cumulative sums or convolutions.
        # S1 is just a cumulative sum of weights from the right.
        # S2 is a cross-correlation of weights and h?
        # S2[i] = sum_{j=i+1}^{n-1} w_{j-i} * h[j]  (let k=j-i)
        # This is a convolution of w (reversed?) and h.
        # S3[i] = sum_{j=0}^{i-1} w_{i-j} * h[j]
        # This is a convolution of w and h.
        
        # Using FFT for convolution might be overkill for N=100, but simple loops are O(N^2)=10k ops.
        # 10k ops * 1000 iterations = 10M ops. Python can do ~10M simple ops in 0.1s?
        # Maybe 1s. With 10 restarts, 10s. Acceptable.
        
        # Let's implement the O(N^2) grad calculation carefully.
        
        pass

    # Re-implementing grad calculation with vectorization in mind
    # grad = sum_k w_k * A_k
    
    # A_k vector:
    # indices 0 to n-k-1: 1 - h[k:]
    # indices k to n-1:   -h[:n-k]  (wait, h[i-k] for i>=k)
    # indices n-k to n-1: 0 (from term 1 and 2)
    # indices 0 to k-1:   0 (from term 3)
    
    # Let's build A_k for each k and add w_k * A_k to grad.
    # This is O(N^2) per gradient call.
    
    for k in range(1, n):
        w = weights[k-1]
        if w == 0:
            continue
        # A_k
        # Part 1: i < n-k
        # A_k[0 : n-k] += 1 - h[k : n]
        # Part 2: i >= k
        # A_k[k : n]   -= h[0 : n-k]
        
        # Note: Overlap in indices?
        # Part 1 covers [0, n-k-1]
        # Part 2 covers [k, n-1]
        # If k < n-k, i.e., k < n/2, there is an overlap region [k, n-k-1].
        # In this region, A_k[i] = (1 - h[i+k]) - h[i-k]
        
        # Vectorized addition
        grad[:n-k] += w * (1.0 - h[k:])
        grad[k:]   -= w * h[:n-k]
        
    return obj, grad

def construct():
    n = N
    target_sum = n / 2.0
    
    best_h = np.full(n, 0.5)
    best_score = float('inf')
    
    def _raw_score(h):
        overlaps = _compute_overlaps(h)
        return np.max(overlaps)
    
    best_score = _raw_score(best_h)
    
    np.random.seed(42)
    
    restarts = []
    
    # 1. Uniform
    restarts.append(np.full(n, 0.5))
    
    # 2. Binary patterns
    # Random binary with sum N/2
    for _ in range(10):
        h_bin = np.random.choice([0, 1], size=n)
        # Adjust sum to be exactly N/2
        ones = np.sum(h_bin == 1)
        if ones > n/2:
            # Flip some 1s to 0
            indices_1 = np.where(h_bin == 1)[0]
            to_flip = np.random.choice(indices_1, size=int(ones - n/2), replace=False)
            h_bin[to_flip] = 0
        elif ones < n/2:
            indices_0 = np.where(h_bin == 0)[0]
            to_flip = np.random.choice(indices_0, size=int(n/2 - ones), replace=False)
            h_bin[to_flip] = 1
        restarts.append(h_bin.astype(float))
        
    # 3. Sinusoidal
    x = np.linspace(0, 2*np.pi, n)
    for freq in [1, 2, 3]:
        for amp in [0.3, 0.5]:
            restarts.append(np.clip(0.5 + amp*np.sin(freq*x), 0, 1))
            
    # 4. Block patterns
    for block_size in [2, 4, 10]:
        pattern = np.tile([0.0, 1.0], n // block_size + 1)[:n]
        restarts.append(pattern * 0.5 + 0.25)

    # Optimization loop
    beta = 10.0
    penalty_lambda = 100.0
    
    for x0 in restarts:
        if _time_remaining() < 5.0:
            break
            
        # Ensure sum is close to target for L-BFGS-B start
        # L-BFGS-B doesn't handle equality constraints, so we use penalty.
        # But we should start close.
        # Simple projection:
        diff = target_sum - np.sum(x0)
        x0 = x0 + diff / n
        x0 = np.clip(x0, 0, 1)
        
        # Objective with penalty
        def obj_and_grad(params):
            h = np.clip(params, 0, 1)
            obj_lse, grad_lse = _logsumexp_overlaps(h, beta)
            
            # Penalty for sum constraint
            sum_err = np.sum(h) - target_sum
            pen = penalty_lambda * sum_err**2
            grad_pen = 2 * penalty_lambda * sum_err * np.ones(n)
            
            # Penalty for box constraints is handled by L-BFGS-B bounds,
            # but the gradient of the objective should ideally be 0 at bounds if not active.
            # Since we clip h inside, the gradient w.r.t params is 0 where clipped?
            # No, we pass params to L-BFGS-B, it handles bounds.
            # We should compute gradient w.r.t params, not h_clipped.
            # If we clip, the gradient is discontinuous.
            # Better to not clip in the objective, but rely on bounds.
            # However, if params go out of bounds, L-BFGS-B projects.
            # So we compute grad w.r.t params assuming h=params.
            
            # Recompute with h=params (no clip) for gradient correctness?
            # If params are slightly out of bounds, the overlap calculation might be weird.
            # But L-BFGS-B keeps them in bounds mostly.
            
            h_raw = params
            obj_lse, grad_lse = _logsumexp_overlaps(h_raw, beta)
            sum_err = np.sum(h_raw) - target_sum
            pen = penalty_lambda * sum_err**2
            grad_pen = 2 * penalty_lambda * sum_err * np.ones(n)
            
            total_obj = obj_lse + pen
            total_grad = grad_lse + grad_pen
            
            return total_obj, total_grad

        try:
            res = minimize(
                obj_and_grad, 
                x0, 
                method='L-BFGS-B', 
                bounds=[(0, 1)] * n,
                options={'maxiter': 500, 'maxfun': 1000, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < best_score: # Note: res.fun is penalized obj, not raw score
                # Evaluate raw score of the result
                h_res = np.clip(res.x, 0, 1)
                # Enforce sum constraint exactly for final score?
                # The evaluator checks sum is exactly N/2.
                # We should project the result to satisfy sum exactly.
                diff = target_sum - np.sum(h_res)
                # Distribute diff to elements that are not at bounds
                mask = (h_res > 1e-5) & (h_res < 1 - 1e-5)
                if np.sum(mask) > 0:
                    h_res[mask] += diff / np.sum(mask)
                else:
                    # If all at bounds, just add to all (might violate bounds slightly)
                    h_res += diff / n
                h_res = np.clip(h_res, 0, 1)
                
                # Final check on sum
                # If still off, adjust one element
                diff = target_sum - np.sum(h_res)
                if abs(diff) > 1e-9:
                    # Find an element not at bound
                    for i in range(n):
                        if 1e-5 < h_res[i] < 1 - 1e-5:
                            h_res[i] += diff
                            break
                    else:
                        h_res[0] += diff
                    h_res = np.clip(h_res, 0, 1)

                raw_score = _raw_score(h_res)
                if raw_score < best_score:
                    best_score = raw_score
                    best_h = h_res.copy()
                    
        except Exception as e:
            pass

    return best_h

# EVOLVE-BLOCK-END
