import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    best_seq = None
    best_score = float('inf')
    
    # Helper to compute max sidelobe from correlation array
    def get_score_from_corr(c):
        # c is length 2N-1
        # Sidelobes are c[0:N-1] and c[N:2N-1]
        # c[N-1] is C(0)
        left = c[:N-1]
        right = c[N:]
        max_side = max(np.max(np.abs(left)), np.max(np.abs(right)))
        return max_side

    # Helper to initialize correlation
    def init_corr(x):
        return np.correlate(x, x, mode='full')

    # Local search function with incremental updates
    def local_search(x, time_budget):
        x = x.copy()
        c = init_corr(x)
        current_score = get_score_from_corr(c)
        
        # Precompute indices for sidelobes
        side_indices = np.concatenate([np.arange(0, N-1), np.arange(N, 2*N-1)])
        
        while time.time() < time_budget:
            # Find the worst lag index in the correlation array
            # We want to minimize max |c[k]| for k != N-1
            # Create a masked view or just check the side indices
            side_vals = c[side_indices]
            abs_side_vals = np.abs(side_vals)
            max_idx_in_side = np.argmax(abs_side_vals)
            target_corr_idx = side_indices[max_idx_in_side]
            target_val = c[target_corr_idx]
            
            # Determine lag k
            # Correlation index j corresponds to lag:
            # if j < N-1, lag = N-1 - j (negative lag in some conventions, but autocorr is symmetric)
            # if j > N-1, lag = j - (N-1)
            # For autocorrelation of real signals, C(k) = C(-k).
            # Let's map target_corr_idx to a positive lag k (1 to N-1)
            if target_corr_idx < N-1:
                k = N-1 - target_corr_idx
            else:
                k = target_corr_idx - (N-1)
                
            # Try to flip a bit to reduce |target_val|
            # Flipping x[i] changes c[j] by delta_j
            # We want to find i such that |target_val + delta_{target_corr_idx}| is minimized
            # delta_j for flip at i:
            # c[j] = sum_{t} x[t] x[t + (j - (N-1))]
            # This mapping is tricky. Let's use the property:
            # C(k) = sum_{i=0}^{N-1-k} x[i] x[i+k]
            # If we flip x[i], the terms involving x[i] change sign.
            # Terms in C(k) involving x[i]:
            # 1. x[i] * x[i+k] (if i+k < N)
            # 2. x[i-k] * x[i] (if i-k >= 0)
            # So delta_C(k) = -2 * x[i] * (x[i+k] + x[i-k]) (with bounds)
            
            # However, we need to update the ENTIRE correlation array c.
            # Flipping x[i] affects all lags.
            # For each lag l (1 to N-1):
            # delta_C(l) = -2 * x[i] * (x[i+l] + x[i-l]) (with bounds)
            # This is O(N) per candidate i.
            # We have N candidates. So O(N^2) per iteration if we check all i.
            # To speed up, we can just check a subset or use a heuristic.
            # Given N=400, O(N^2) = 160k ops per iteration.
            # If we do 1000 iterations, that's 160M ops. Python might be slow.
            # Let's try to optimize the inner loop.
            
            best_flip = -1
            best_new_target_val = target_val
            best_delta_vec = None # Store the delta vector for the best flip to update c
            
            # Heuristic: Only check i where the potential change is large
            # Or just iterate all i, but vectorize if possible?
            # Vectorizing the delta calculation for all i for a fixed k is easy.
            # But we need to evaluate |target_val + delta_k(i)| for all i.
            
            # Let's compute delta_k(i) for all i for the current target lag k.
            # delta_k(i) = -2 * x[i] * (x[i+k] + x[i-k])
            
            # Create arrays for x[i+k] and x[i-k] with padding
            # x is 1D
            # x_plus = np.roll(x, -k) ? No, bounds matter.
            # x[i+k] is valid for i < N-k.
            # x[i-k] is valid for i >= k.
            
            x_plus = np.zeros(N)
            x_minus = np.zeros(N)
            
            # Fill x_plus[i] = x[i+k] if i+k < N else 0
            x_plus[:N-k] = x[k:]
            # Fill x_minus[i] = x[i-k] if i-k >= 0 else 0
            x_minus[k:] = x[:N-k]
            
            # delta_k(i) = -2 * x[i] * (x_plus[i] + x_minus[i])
            deltas = -2 * x * (x_plus + x_minus)
            
            # We want to minimize |target_val + deltas[i]|
            new_vals = target_val + deltas
            abs_new_vals = np.abs(new_vals)
            best_i = np.argmin(abs_new_vals)
            
            # Check if this flip actually improves the global max?
            # The heuristic is: if it reduces the absolute value of the current peak, do it.
            # It might increase other peaks, but local search usually works.
            if abs_new_vals[best_i] < abs(target_val):
                # Apply the flip
                x[best_i] *= -1
                
                # Update correlation array c incrementally
                # We need delta_c(l) for all lags l.
                # This is O(N) to compute the full delta vector.
                # delta_c(l) = -2 * x_new[best_i] * (x_new[best_i+l] + x_new[best_i-l])?
                # Wait, x changed. We should use the OLD x to calculate the change?
                # Let x_old be the state before flip.
                # x_new[i] = -x_old[i] for i=best_i, else same.
                # Change in C(l) = sum_{t} (x_new[t]x_new[t+l] - x_old[t]x_old[t+l])
                # Only terms involving index best_i change.
                # Term 1: x[best_i] * x[best_i+l] -> -x[best_i] * x[best_i+l]. Diff: -2 x[best_i] x[best_i+l]
                # Term 2: x[best_i-l] * x[best_i] -> x[best_i-l] * (-x[best_i]). Diff: -2 x[best_i-l] x[best_i]
                # So delta_C(l) = -2 * x_old[best_i] * (x_old[best_i+l] + x_old[best_i-l])
                
                # We have x_old in `x` before we flipped it? No, we just flipped it.
                # Let's restore x[best_i] to calculate delta using old values, or just use the fact that x[best_i] was -x_new[best_i].
                # Let's use x_old = x.copy() before flip? That's expensive.
                # We can calculate delta using x_new and x_old relation.
                # x_old[best_i] = -x_new[best_i]
                # x_old[best_i+l] = x_new[best_i+l] (since l!=0)
                # x_old[best_i-l] = x_new[best_i-l] (since l!=0)
                # So delta_C(l) = -2 * (-x_new[best_i]) * (x_new[best_i+l] + x_new[best_i-l])
                #               =  2 * x_new[best_i] * (x_new[best_i+l] + x_new[best_i-l])
                
                # Let's compute delta for all lags 1..N-1
                # We need to update c[0:N-1] and c[N:2N-1]
                
                # Construct delta vector for lags 1 to N-1
                # lags = 1..N-1
                # For each lag l, delta_l = 2 * x[best_i] * (x[best_i+l] + x[best_i-l]) (with bounds)
                
                # We can vectorize this for all l?
                # It's a convolution-like operation.
                # Let's just loop l from 1 to N-1. O(N).
                for l in range(1, N):
                    term1 = 0
                    if 0 <= best_i + l < N:
                        term1 = x[best_i + l]
                    term2 = 0
                    if 0 <= best_i - l < N:
                        term2 = x[best_i - l]
                    delta_l = 2 * x[best_i] * (term1 + term2)
                    
                    # Update c
                    # c[N-1+l] corresponds to lag l
                    c[N-1+l] += delta_l
                    # c[N-1-l] corresponds to lag -l
                    c[N-1-l] += delta_l
            else:
                break

            current_score = get_score_from_corr(c)
            if current_score < best_score:
                best_score = current_score
                best_seq = x.copy()

        return x, current_score

    # Run multiple restarts
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize
        x = np.random.choice([-1, 1], size=N)
        
        # Allocate time budget
        time_left = time_limit - (time.time() - start_time)
        if time_left <= 0: break
        budget = time_left / (num_restarts - r)
        
        # Ensure minimum budget
        budget = max(budget, 0.5)
        
        x, score = local_search(x, start_time + budget)
        
        if score < best_score:
            best_score = score
            best_seq = x

    if best_seq is None:
        # Fallback
        best_seq = np.ones(N)
        
    return best_seq
