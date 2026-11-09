import numpy as np
import math
import time
import random

def construct():
    # Start with a simple seed
    best_U = [0, 1, 3]
    best_score = -np.inf
    
    # Time limit
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer
    
    # Initial parameters for Simulated Annealing
    # We will search for sets of varying sizes and max values
    # Let's try to optimize a set of fixed size N? Or variable?
    # Variable size is harder to manage in SA.
    # Let's fix N and M? Or just let M grow?
    # Let's try a fixed N first, say N=10, 20, 50.
    # Or better, start with a small set and grow/shrink.
    
    # Let's use a representation that allows fast evaluation.
    # If M is large, FFT is expensive.
    # Let's restrict M to a reasonable range, e.g., < 10000.
    
    def evaluate(U):
        if not U:
            return -np.inf
        if 0 not in U:
            return -np.inf
        U = sorted(set(U))
        M = max(U)
        if M == 0:
            return 1.0 # |U-U|=1, |U+U|=1, log(1)=0, score=1
        
        # Create boolean arrays
        # Sumset: indices 0 to 2M
        # Diffset: indices -M to M -> shift by M -> 0 to 2M
        
        # Using FFT for convolution
        # Size of FFT needs to be power of 2 >= 2*max_size
        max_sum = 2 * M
        max_diff = 2 * M # Range -M to M
        
        # We can do both in one go or separately.
        # Sum: convolve v with v
        # Diff: convolve v with reversed v
        
        n_fft = 1
        while n_fft < 2 * max_sum + 1:
            n_fft *= 2
        
        v = np.zeros(n_fft, dtype=np.float64)
        for u in U:
            v[u] = 1.0
            
        # Sumset
        v_fft = np.fft.fft(v)
        sum_conv = np.fft.ifft(v_fft * v_fft).real
        # Sumset values are at indices 0..2M
        # Count non-zeros
        sum_counts = sum_conv[:2*M+1]
        sumset_size = np.sum(sum_counts > 1e-9)
        
        # Difference set
        # We want v[i] * v[j] where i-j = d.
        # This is correlation.
        # Correlation can be computed by convolving v with reversed v.
        # reversed v: v_rev[k] = v[M-k]? No, v is defined on 0..M.
        # Let's just use the property that |U-U| is the support of the autocorrelation.
        # Autocorrelation R[d] = sum u_i u_{i+d}.
        # Using FFT: ifft(fft(v) * conj(fft(v))) gives circular autocorrelation.
        # Since we zero-padded to n_fft > 2M, linear autocorrelation is obtained.
        # The output is centered.
        # For real input, ifft(fft(v) * conj(fft(v))) gives the autocorrelation sequence.
        # The peak is at index 0 (lag 0).
        # Positive lags are 1..M, negative lags are n_fft-1..n_fft-M.
        
        auto_corr = np.fft.ifft(v_fft * np.conj(v_fft)).real
        # The autocorrelation at lag 0 is |U|.
        # We need to check which lags have non-zero correlation.
        # Lags 0 to M are in auto_corr[0:M+1]
        # Lags -1 to -M are in auto_corr[n_fft-1:n_fft-M-1:-1]
        
        # Check positive lags
        pos_lags = auto_corr[1:M+1]
        # Check negative lags
        neg_lags = auto_corr[n_fft-1:n_fft-M-1:-1]
        
        # Count non-zeros in pos and neg
        pos_count = np.sum(pos_lags > 1e-9)
        neg_count = np.sum(neg_lags > 1e-9)
        
        # Lag 0 is always non-zero (since U is not empty)
        diffset_size = pos_count + neg_count + 1
        
        if sumset_size == 0:
            return -np.inf
            
        ratio = diffset_size / sumset_size
        if ratio <= 1:
            # log(ratio) <= 0, score < 1. We want > 1.
            # But we still calculate.
            pass
            
        denom = math.log(2 * M + 1)
        if denom == 0:
            return 1.0
            
        score = 1 + math.log(ratio) / denom
        return score

    # Simulated Annealing
    # State: List of integers U
    # We will try different sizes N.
    # Let's run SA for a few different N values.
    
    best_overall = -np.inf
    best_U_overall = [0, 1, 3]
    
    target_sizes = [5, 10, 20, 50, 100]
    
    for N in target_sizes:
        # Initialize U
        # Start with a geometric-like set or random
        U = [0]
        current_max = 1
        for i in range(1, N):
            # Add random element > current_max
            # Keep it small initially
            new_val = random.randint(current_max + 1, current_max + 10)
            U.append(new_val)
            current_max = new_val
        U = sorted(set(U))
        # Ensure length N
        while len(U) < N:
            U.append(random.randint(1, max(U)+10))
            U = sorted(set(U))
        if len(U) > N:
            # Remove largest?
            U = U[:N]
            
        current_score = evaluate(U)
        T = 1.0
        T_min = 0.01
        cooling_rate = 0.99
        
        while T > T_min and (time.time() - start_time) < time_limit:
            # Perturbation
            move_type = random.random()
            U_temp = U.copy()
            
            if move_type < 0.3:
                # Add element
                max_val = max(U_temp) if U_temp else 0
                new_val = random.randint(1, max_val + 5)
                if new_val not in U_temp:
                    U_temp.append(new_val)
            elif move_type < 0.6:
                # Remove element (keep 0)
                if len(U_temp) > 1:
                    idx = random.randint(1, len(U_temp) - 1)
                    U_temp.pop(idx)
            else:
                # Shift element
                if len(U_temp) > 1:
                    idx = random.randint(1, len(U_temp) - 1)
                    shift = random.randint(-2, 2)
                    new_val = U_temp[idx] + shift
                    if new_val >= 0 and new_val not in U_temp:
                        U_temp[idx] = new_val
                        U_temp = sorted(U_temp)
                    else:
                        continue # Invalid move, retry or skip
                        # To keep loop tight, just skip
            
            if len(U_temp) != len(U):
                # If size changed, we might want to allow it, but let's stick to fixed N for stability?
                # Actually, allowing size change is good.
                pass
            
            new_score = evaluate(U_temp)
            
            if new_score > current_score:
                U = U_temp
                current_score = new_score
                if current_score > best_overall:
                    best_overall = current_score
                    best_U_overall = U.copy()
            else:
                # Metropolis
                delta = current_score - new_score
                if delta < 0:
                    prob = 1.0
                else:
                    prob = math.exp(-delta / T)
                if random.random() < prob:
                    U = U_temp
                    current_score = new_score
            
            T *= cooling_rate
            
            # Check time
            if time.time() - start_time > time_limit:
                break
        
        if time.time() - start_time > time_limit:
            break

    if best_overall > -np.inf:
        return best_U_overall
    return [0, 1, 3]
