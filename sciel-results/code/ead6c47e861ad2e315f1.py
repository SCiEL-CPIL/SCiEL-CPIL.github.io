import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = 140.0
    
    # We optimize h directly.
    # Objective: Minimize max_k O_k where O_k = sum_i h_i (1 - h_{i-k})
    # O_k = S - C_k. S is sum(h).
    # If we enforce S = N/2 via penalty, we effectively maximize min C_k.
    
    # Initial guess
    h0 = np.full(N, 0.5)
    
    def compute_overlaps(h):
        # Compute C_k = sum_i h_i h_{i-k} for k=1..N-1
        # Using FFT
        H = np.fft.fft(h)
        C = np.real(np.fft.ifft(H * np.conj(H)))
        # C[0] is sum h^2. C[k] is autocorr at lag k.
        # Overlaps O_k = S - C_k.
        S = np.sum(h)
        O = S - C[1:]  # shifts 1 to N-1
        # Note: C[N-k] = C[k] for real signals? Yes, autocorrelation is symmetric.
        # So we only need to check k=1..N//2? 
        # The max overlap is over all shifts. 
        # O_k = O_{N-k}. So max is over k=1..N//2.
        return O
    
    def objective_and_grad(h):
        O = compute_overlaps(h)
        
        # Smooth max of O
        # We want to minimize max(O).
        # Use log-sum-exp: max(O) ~ (1/beta) * log(sum(exp(beta * O)))
        beta = 10.0 # Sharpness
        # Shift for stability
        max_O = np.max(O)
        log_sum_exp = max_O + np.log(np.sum(np.exp(beta * (O - max_O))))
        obj_smooth = log_sum_exp / beta
        
        # Penalty for sum constraint
        target_sum = N / 2.0
        S = np.sum(h)
        penalty = 100.0 * (S - target_sum)**2
        
        obj = obj_smooth + penalty
        
        # Gradient
        # d obj_smooth / d h_j
        # d/dh_j [ (1/beta) log sum_k exp(beta O_k) ]
        # = (1/beta) * (sum_k exp(beta O_k) * beta dO_k/dh_j) / sum_k exp(beta O_k)
        # = sum_k p_k dO_k/dh_j, where p_k = exp(beta O_k) / sum exp(beta O_k)
        
        exp_terms = np.exp(beta * (O - max_O))
        p = exp_terms / np.sum(exp_terms)
        
        # dO_k/dh_j
        # O_k = S - C_k
        # dS/dh_j = 1
        # dC_k/dh_j = h_{j-k} + h_{j+k} (indices mod N)
        # So dO_k/dh_j = 1 - h_{j-k} - h_{j+k}
        
        # We can compute this vectorized.
        # For each k, we need the vector v_k where v_k[j] = 1 - h[(j-k)%N] - h[(j+k)%N]
        # grad_smooth = sum_k p_k * v_k
        # This is a weighted sum of shifted versions of h.
        # v_k = 1 - shift(h, k) - shift(h, -k)
        # grad_smooth = sum_k p_k * (1 - shift(h, k) - shift(h, -k))
        # grad_smooth = sum_k p_k - sum_k p_k shift(h, k) - sum_k p_k shift(h, -k)
        # sum_k p_k = 1.
        # So grad_smooth = 1 - (sum_k p_k shift(h, k)) - (sum_k p_k shift(h, -k))
        
        # Compute weighted shifts
        # We can use FFT for this too? 
        # sum_k p_k h_{j-k} is a convolution of h with p?
        # p is defined on k=1..N-1. p_0 is 0.
        # Let's just do it in spatial domain, N=100 is small.
        
        grad_smooth = np.zeros(N)
        for k in range(1, N):
            if p[k-1] == 0: continue # p is indexed 0..N-2 corresponding to k=1..N-1
            # shift h by k
            h_shifted_pos = np.roll(h, k) # h_{j-k} -> index j has value h[(j-k)%N]. np.roll(h, k) puts h[0] at k. 
            # Let's verify np.roll.
            # np.roll([1,2,3], 1) -> [3,1,2].
            # We want h_{j-k}. If j=0, we want h[-k] = h[N-k].
            # np.roll(h, k)[j] = h[(j-k)%N]. Correct.
            
            h_shifted_neg = np.roll(h, -k) # h_{j+k}. np.roll(h, -k)[j] = h[(j+k)%N]. Correct.
            
            grad_smooth += p[k-1] * (1.0 - h_shifted_pos - h_shifted_neg)
            
        # Gradient of penalty
        grad_penalty = 2.0 * 100.0 * (S - target_sum) * np.ones(N)
        
        grad = grad_smooth + grad_penalty
        
        return obj, grad

    # Run optimization
    # L-BFGS-B
    from scipy.optimize import minimize
    
    bounds = [(0.0, 1.0)] * N
    
    try:
        res = minimize(objective_and_grad, h0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 200, 'maxfun': 2000, 'ftol': 1e-10, 'gtol': 1e-8})
    except Exception:
        res = None
        
    if res is not None and res.success:
        h_best = res.x
    else:
        h_best = h0
        
    # Final exact projection to sum N/2 and [0,1]
    # 1. Clip
    h_best = np.clip(h_best, 0.0, 1.0)
    
    # 2. Adjust sum
    target = N / 2.0
    current_sum = np.sum(h_best)
    diff = target - current_sum
    
    if abs(diff) > 1e-9:
        # Water filling
        # If diff > 0, we need to add value. We can only add to h_i < 1.
        # If diff < 0, we need to subtract. We can only subtract from h_i > 0.
        
        # Iterative water filling to handle bounds
        # Sort indices by how much "room" they have
        # Simple approach: distribute evenly among those not at bound, iterate if clipping occurs.
        
        # To be fast and robust, we can do a few iterations of:
        # 1. Add diff/N to all.
        # 2. Clip.
        # 3. Redistribute remaining error among unclipped.
        
        # Given N=100, even a simple loop is fine.
        # But let's do a more direct method.
        
        # Identify free variables
        if diff > 0:
            # Need to increase
            # Candidates: h_i < 1
            # We want to increase them as much as needed.
            # To minimize distortion, increase uniformly?
            # Let's just do:
            h_temp = h_best.copy()
            # Add diff/N
            h_temp += diff / N
            h_temp = np.clip(h_temp, 0.0, 1.0)
            
            # Check if sum is correct. If clipping occurred, sum is wrong.
            # Iterate a few times
            for _ in range(10):
                s = np.sum(h_temp)
                d = target - s
                if abs(d) < 1e-9:
                    break
                # Distribute d among those not at bound in the direction of d
                if d > 0:
                    mask = h_temp < 1.0 - 1e-12
                else:
                    mask = h_temp > 1e-12
                
                if not np.any(mask):
                    break # Should not happen if target is in [0, N]
                    
                count = np.sum(mask)
                h_temp[mask] += d / count
                h_temp = np.clip(h_temp, 0.0, 1.0)
            
            h_best = h_temp
        else:
            # diff < 0, need to decrease
            h_temp = h_best.copy()
            h_temp += diff / N
            h_temp = np.clip(h_temp, 0.0, 1.0)
            
            for _ in range(10):
                s = np.sum(h_temp)
                d = target - s
                if abs(d) < 1e-9:
                    break
                if d > 0:
                    mask = h_temp < 1.0 - 1e-12
                else:
                    mask = h_temp > 1e-12
                
                if not np.any(mask):
                    break
                    
                count = np.sum(mask)
                h_temp[mask] += d / count
                h_temp = np.clip(h_temp, 0.0, 1.0)
            
            h_best = h_temp
            
    return h_best
