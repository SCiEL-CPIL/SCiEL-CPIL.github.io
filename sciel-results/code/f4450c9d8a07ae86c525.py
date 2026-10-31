"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initialize with a sinusoidal sign pattern to distribute autoconvolution energy
    # Use a low frequency to create smooth sign transitions
    x = np.ones(N)
    best_score = np.inf
    best_x = x.copy()
    
    # Try multiple frequency initializations
    for freq in [1, 2, 3, 5, 7, 11]:
        if time.time() > deadline:
            break
        
        # Create sign pattern from sinusoid
        t = np.arange(N)
        x = np.sign(np.sin(2 * np.pi * freq * t / N))
        # Ensure non-zero: replace zeros with +1
        x[x == 0] = 1.0
        
        # Local optimization: flip signs to reduce peak autoconvolution
        for iteration in range(500):
            if time.time() > deadline:
                break
            
            # Compute autoconvolution via FFT
            n_fft = 1 << (2 * N - 1).bit_length()
            F = np.fft.rfft(x, n=n_fft)
            autoconv = np.fft.irfft(F * np.conj(F), n=n_fft)[:2*N-1]
            
            # Find max absolute autoconvolution
            max_abs_idx = np.argmax(np.abs(autoconv))
            max_abs_val = np.abs(autoconv[max_abs_idx])
            
            # Compute integral
            integral = np.sum(x)
            if abs(integral) < 1e-10:
                # Avoid division by zero; perturb
                x[np.random.randint(N)] *= -1
                continue
            
            score = max_abs_val / (integral * integral)
            
            if score < best_score:
                best_score = score
                best_x = x.copy()
            
            # Try flipping a random position to see if it reduces the peak
            # Focus on positions that contribute to the peak lag
            pos = np.random.randint(N)
            old_val = x[pos]
            x[pos] = -old_val
            
            # Quick check: recompute autoconvolution
            F2 = np.fft.rfft(x, n=n_fft)
            autoconv2 = np.fft.irfft(F2 * np.conj(F2), n=n_fft)[:2*N-1]
            max_abs_val2 = np.max(np.abs(autoconv2))
            integral2 = np.sum(x)
            
            if abs(integral2) > 1e-10:
                score2 = max_abs_val2 / (integral2 * integral2)
                if score2 < score:
                    # Keep the flip
                    score = score2
                    max_abs_val = max_abs_val2
                    integral = integral2
                else:
                    # Revert
                    x[pos] = old_val
            else:
                # Revert if integral too small
                x[pos] = old_val
        
        if time.time() > deadline:
            break
    
    # Final check
    if time.time() > deadline:
        return best_x
    
    return best_x
# EVOLVE-BLOCK-END
