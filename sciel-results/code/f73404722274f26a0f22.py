import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Pre-compute indices for autocorrelation updates
    # k ranges from 1 to N-1
    k_indices = np.arange(1, N)
    idx_plus = (np.arange(N)[:, None] + k_indices[None, :]) % N
    idx_minus = (np.arange(N)[:, None] - k_indices[None, :]) % N
    
    rng = np.random.default_rng(42)
    
    best_h = None
    best_score = float('inf')
    
    # We might want to run multiple restarts if we get stuck
    # But let's try one long run with perturbation first.
    
    h = rng.choice([-1, 1], size=N).astype(np.float64)
    
    # Compute initial R
    # R[k] = sum_j h[j] h[j+k]
    # Using FFT for initial computation is faster than O(N^2) loop
    n_fft = 1 << (2 * N).bit_length()
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    R_full = np.fft.irfft(power, n=n_fft)
    R = R_full[1:N].copy()
    
    current_score = np.max(np.abs(R))
    best_h = h.copy()
    best_score = current_score
    
    stagnation_counter = 0
    perturbation_size = 5
    # Temperature for simulated annealing-like acceptance
    temperature = 1.0
    min_temperature = 0.01
    
    while time.time() < deadline - 10.0:  # Leave 10s for final polish
        # 1. Compute delta_R for all possible single-bit flips
        # delta_R[i, k] = -2 * h[i] * (h[(i+k)%N] + h[(i-k)%N])
        
        # h[idx_plus] is shape (N, N-1)
        # h[idx_minus] is shape (N, N-1)
        # sum_h is shape (N, N-1)
        sum_h = h[idx_plus] + h[idx_minus]
        
        # delta_R[i, k] = -2 * h[i] * sum_h[i, k]
        # h is shape (N,), so h[:, None] is (N, 1)
        delta_R = -2.0 * h[:, None] * sum_h
        
        # 2. Evaluate score for each flip
        # new_R = R[None, :] + delta_R  (shape N, N-1)
        # scores = max(abs(new_R), axis=1) (shape N,)
        
        # To save memory/time, we can compute max directly?
        # No, we need the full vector to compare.
        # But we can do:
        new_R = R[None, :] + delta_R
        scores = np.max(np.abs(new_R), axis=1)
        
        # 3. Find best flip
        best_idx = np.argmin(scores)
        best_candidate_score = scores[best_idx]
        
        if best_candidate_score < current_score:
            # Accept
            h[best_idx] = -h[best_idx]
            R = R + delta_R[best_idx]
            current_score = best_candidate_score
            
            if current_score < best_score:
                best_score = current_score
                best_h = h.copy()
            
            stagnation_counter = 0
        else:
            # Consider accepting worse moves with probability based on temperature
            # This helps escape local minima
            delta = best_candidate_score - current_score
            if delta > 0:
                # Compute acceptance probability
                # Use a simplified Metropolis criterion
                prob = np.exp(-delta / max(temperature, 1e-10))
                
                if rng.random() < prob:
                    # Accept worse move
                    h[best_idx] = -h[best_idx]
                    R = R + delta_R[best_idx]
                    current_score = best_candidate_score
                    stagnation_counter = 0
                else:
                    # Stagnation
                    stagnation_counter += 1
            else:
                # This shouldn't happen since we checked < above
                stagnation_counter += 1
            
            if stagnation_counter > 50:
                # Perturb: flip a few random bits
                num_flips = rng.integers(1, perturbation_size + 1)
                flip_indices = rng.choice(N, size=num_flips, replace=False)
                h[flip_indices] *= -1
                
                # Recompute R from scratch (or update, but from scratch is safer/simple)
                n_fft = 1 << (2 * N).bit_length()
                F = np.fft.rfft(h, n=n_fft)
                power = F * np.conj(F)
                R_full = np.fft.irfft(power, n=n_fft)
                R = R_full[1:N].copy()
                
                current_score = np.max(np.abs(R))
                
                if current_score < best_score:
                    best_score = current_score
                    best_h = h.copy()
                
                stagnation_counter = 0
                
                # Adapt perturbation size and temperature
                # If we're stuck with a bad score, perturb more and cool down
                if best_score > 15.0:
                    perturbation_size = min(20, perturbation_size + 2)
                    temperature = max(min_temperature, temperature * 0.9)
                elif best_score > 10.0:
                    perturbation_size = min(15, perturbation_size + 1)
                    temperature = max(min_temperature, temperature * 0.95)
                else:
                    perturbation_size = 5
                    temperature = max(min_temperature, temperature * 0.8)
    
    # Final polish pass: try all single-bit flips and accept any improvement
    # This is a high-precision polish with tighter criteria
    polish_deadline = start_time + deadline
    polish_improvements = 0
    
    while time.time() < polish_deadline:
        # Compute delta_R for all possible single-bit flips
        sum_h = h[idx_plus] + h[idx_minus]
        delta_R = -2.0 * h[:, None] * sum_h
        
        # Evaluate score for each flip
        new_R = R[None, :] + delta_R
        scores = np.max(np.abs(new_R), axis=1)
        
        # Find best flip
        best_idx = np.argmin(scores)
        best_candidate_score = scores[best_idx]
        
        if best_candidate_score < current_score:
            # Accept
            h[best_idx] = -h[best_idx]
            R = R + delta_R[best_idx]
            current_score = best_candidate_score
            polish_improvements += 1
            
            if current_score < best_score:
                best_score = current_score
                best_h = h.copy()
        else:
            # No improvement possible with single-bit flips
            break
    
    return best_h
