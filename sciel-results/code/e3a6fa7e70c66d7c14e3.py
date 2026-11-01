import numpy as np
import time
import sys

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin before 120s limit
    
    best_h = None
    best_val = float('inf')
    
    # Helper to compute PSL from precomputed R_pos
    def calc_psl_from_R(R_pos):
        peak = R_pos[0]
        if peak == 0:
            return 0.0
        sidelobes = R_pos[1:]
        max_side = np.max(np.abs(sidelobes))
        return max_side / peak

    # Multi-start Simulated Annealing
    while time.time() < deadline - 5.0:
        # Initialize with a random binary sequence
        h = np.random.choice([-1, 1], size=N)
        
        # Initialize R_pos (positive lags 0 to N-1) using FFT
        n_fft = 1 << (2 * N - 1).bit_length()
        F = np.fft.rfft(h, n=n_fft)
        R_full = np.fft.irfft(F * np.conj(F), n=n_fft)
        R_pos = R_full[:N].copy()
        
        current_val = calc_psl_from_R(R_pos)
        
        if current_val < best_val:
            best_val = current_val
            best_h = h.copy()
            
        T = 0.1
        T_min = 0.001
        alpha = 0.9995
        steps_since_improvement = 0
        max_steps_no_improve = 5000
        
        k = np.arange(N)
        
        while time.time() < deadline - 2.0:
            # Pick a random index to flip
            i = np.random.randint(N)
            old_val = h[i]
            
            # Calculate delta_h
            delta_h = -2.0 * old_val
            
            # Update R_pos incrementally
            idx_plus = i + k
            idx_minus = i - k
            
            mask_plus = (idx_plus >= 0) & (idx_plus < N)
            mask_minus = (idx_minus >= 0) & (idx_minus < N)
            
            term_plus = np.zeros(N)
            term_plus[mask_plus] = h[idx_plus[mask_plus]]
            
            term_minus = np.zeros(N)
            term_minus[mask_minus] = h[idx_minus[mask_minus]]
            
            delta_R = delta_h * (term_plus + term_minus)
            R_pos += delta_R
            
            # Flip h
            h[i] = -old_val
            
            new_val = calc_psl_from_R(R_pos)
            
            delta = new_val - current_val
            
            if delta < 0:
                # Accept improvement
                current_val = new_val
                steps_since_improvement = 0
                if new_val < best_val:
                    best_val = new_val
                    best_h = h.copy()
            else:
                # Accept worse with probability
                if T > T_min:
                    p = np.exp(-delta / T)
                    if np.random.random() < p:
                        current_val = new_val
                    else:
                        # Revert
                        R_pos -= delta_R
                        h[i] = old_val
                else:
                    # Revert
                    R_pos -= delta_R
                    h[i] = old_val
            
            T *= alpha
            if T < T_min:
                # Reheat
                T = 0.05
            
            steps_since_improvement += 1
            if steps_since_improvement > max_steps_no_improve:
                break

    if best_h is None:
        return np.random.choice([-1, 1], size=N)
        
    return best_h
