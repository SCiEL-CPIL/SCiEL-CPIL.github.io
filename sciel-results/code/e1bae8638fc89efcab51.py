import numpy as np
import time
from scipy.optimize import minimize

N = 400

def construct():
    start_time = time.time()
    DEADLINE = 145
    
    def evaluate(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        conv = np.convolve(h, h)
        max_conv = np.max(conv)
        val = abs(2.0 * N * max_conv / (s * s))
        return val

    best_value = np.inf
    best_h = np.ones(N)
    val_seed = evaluate(best_h)
    best_value = val_seed
    
    def smooth_objective(x, beta=100.0):
        h = x
        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e10
        conv = np.convolve(h, h)
        # Smooth max
        max_c = np.max(conv)
        shifted = conv - max_c
        # Avoid overflow
        if np.max(shifted) * beta > 500:
             # Scale down
             pass 
        log_sum_exp = max_c + np.log(np.sum(np.exp(beta * shifted))) / beta
        val = abs(2.0 * N * log_sum_exp / (s * s))
        if not np.isfinite(val):
            return 1e10
        return val

    def smooth_grad(x, beta=100.0):
        h = x
        s = np.sum(h)
        if abs(s) < 1e-10:
            return np.zeros_like(h)
        
        conv = np.convolve(h, h)
        max_c = np.max(conv)
        shifted = conv - max_c
        exp_shifted = np.exp(beta * shifted)
        sum_exp = np.sum(exp_shifted)
        weights = exp_shifted / sum_exp
        M = max_c + np.log(sum_exp) / beta
        
        # Gradient of M w.r.t h
        # M = logsumexp(beta * C) / beta
        # dM/dh = (1/beta) * sum_k weights[k] * beta * dC_k/dh = sum_k weights[k] * dC_k/dh
        grad_M = np.zeros(N)
        # dC_k/dh_j = 2 * h_{k-j} (with appropriate bounds)
        # This is equivalent to convolving h with weights, but flipped?
        # C_k = sum_i h_i h_{k-i}
        # dC_k/dh_j = h_{k-j} + h_{j} (if k-j is valid index for second term? No.)
        # Let's use the property that gradient of convolution is convolution with gradient.
        # grad_C = 2 * np.convolve(weights, h) ?
        # Let's check.
        # C_k = <h, h_{k-:}>
        # dC_k/dh_j = h_{k-j} + h_{k-j} = 2 h_{k-j} ?
        # Actually, if we view C as matrix H h, where H is correlation matrix.
        # H is symmetric.
        # grad_C = 2 H h ? No, C is vector.
        # grad_C_j = sum_k weights[k] * 2 h_{k-j}
        # This is 2 * (weights * h)_j (cross correlation).
        # np.correlate(weights, h, 'full') ?
        # Let's just compute it via convolution for safety, or use FFT.
        # weights has length 2N-1. h has length N.
        # grad_M = 2 * np.correlate(weights, h, mode='same')?
        # Let's try np.convolve(weights, h, mode='valid')? No.
        
        # Efficient way:
        # grad_M = 2 * np.convolve(weights, h, mode='full')
        # But we need to align it to N.
        # The term dC_k/dh_j is non-zero if 0 <= k-j < N.
        # So j must be in [k-N+1, k].
        
        # Let's use FFT for speed if N is large, but N=400 is small.
        # Simple loop is 400 * 400 = 160k ops. Fast enough.
        # But we can do better.
        # grad_M = 2 * np.correlate(weights, h, mode='full')
        # The result of correlate(weights, h) has length 2N-1.
        # We want the part corresponding to h.
        
        # Let's stick to the loop for correctness, it's fast enough for 145s.
        for k in range(len(weights)):
            if weights[k] == 0:
                continue
            # h_{k-j} is non-zero for j in [max(0, k-N+1), min(N-1, k)]
            j_start = max(0, k - N + 1)
            j_end = min(N - 1, k)
            if j_start > j_end:
                continue
            idx = np.arange(j_start, j_end + 1)
            # h_{k-j}
            h_vals = h[k - idx]
            # Check if k-j == j (diagonal)
            # If k-j == j, then 2j=k.
            # The derivative is 2*h_j.
            # My formula 2*h_{k-j} gives 2*h_j. Correct.
            grad_M[idx] += 2.0 * weights[k] * h_vals
            
        # Now objective is f(h) = | 2 N M / S^2 |
        # Let Q = 2 N M / S^2
        # f = |Q|
        # grad f = sign(Q) * grad Q
        # Q = A * M * S^{-2} where A = 2N
        # dQ/dh = A * (dM/dh * S^{-2} + M * d(S^{-2})/dh)
        # d(S^{-2})/dh = -2 S^{-3} * 1
        # dQ/dh = A * (grad_M / S^2 - 2 M / S^3)
        
        Q = 2.0 * N * M / (s * s)
        if abs(Q) < 1e-12:
            return np.zeros_like(h)
            
        grad_Q = (2.0 * N / (s * s)) * grad_M - (4.0 * N * M / (s * s * s)) * np.ones(N)
        
        return np.sign(Q) * grad_Q

    rng = np.random.RandomState(42)
    
    # Initial points
    inits = [np.ones(N)]
    # Add random inits
    for _ in range(10):
        inits.append(rng.normal(0, 1, N) + 0.5) # Ensure non-zero sum
    
    # Optimize
    for h0 in inits:
        if time.time() - start_time > DEADLINE:
            break
        try:
            res = minimize(
                smooth_objective,
                h0,
                jac=smooth_grad,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            h_opt = res.x
            val_opt = evaluate(h_opt)
            if val_opt < best_value:
                best_value = val_opt
                best_h = h_opt
        except Exception:
            pass
            
    return best_h
