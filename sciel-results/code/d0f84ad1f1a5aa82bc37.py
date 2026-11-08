"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400
DEADLINE = 120.0

# EVOLVE-BLOCK-START
import numpy as np

def _compute_c3(signed_step):
    """
    Compute the C3 upper bound for a signed step function.
    The signed step function has N equal intervals on [-1/4, 1/4].
    C3 = max_t |autoconv(t)| where autoconv is the convolution of f with itself.
    """
    M = 4 * N  # fine grid
    dx = (0.5) / M  # domain length is 0.5
    
    # Build f on fine grid using vectorized repeat
    f_fine = np.repeat(signed_step, M // N)
    
    # Autoconvolution via FFT
    fft_f = np.fft.fft(f_fine, n=2*M)
    autoconv = np.fft.ifft(fft_f * fft_f)
    
    # The relevant part of the autocorrelation is for shifts 0 to M//2
    # We take the real part and look at the first M//2 + 1 elements
    # But we need to be careful about the definition of the bound.
    # The "third autocorrelation inequality" often refers to the L-infinity norm
    # of the autocorrelation function.
    # For a step function, the autocorrelation is piecewise constant.
    # The max absolute value will occur at one of the step boundaries or in the middle of a step.
    # Since we are using a fine grid, we can just take the max abs of the computed autocorrelation.
    
    # The autocorrelation is real, so we take the real part.
    # Due to numerical errors, there might be tiny imaginary parts.
    autoconv = np.real(autoconv)
    
    # The shifts range from 0 to M-1.
    # But due to symmetry, autocorr[k] = autocorr[M-k].
    # So we only need to check 0 to M//2.
    # However, the convolution computed by FFT is circular.
    # We need linear convolution.
    # To get linear convolution, we should zero-pad to at least 2*M - 1.
    # Let's redo with proper zero-padding.
    
    n_fft = 1 << (2 * M - 1).bit_length()
    fft_f = np.fft.fft(f_fine, n=n_fft)
    autoconv = np.fft.ifft(fft_f * fft_f, n=n_fft)
    autoconv = np.real(autoconv)
    
    # The linear autocorrelation for shifts 0 to M-1 is in autoconv[0:M]
    # We want the max absolute value.
    return np.max(np.abs(autoconv[:M]))

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initial guess: random signs
    np.random.seed(42)
    x0 = np.random.choice([-1.0, 1.0], size=N)
    
    best_x = x0.copy()
    best_score = _compute_c3(best_x)
    
    # Strategy:
    # 1. Local search with multi-flip moves
    # 2. Simulated Annealing for better global exploration
    
    # We will use a combination of hill climbing and simulated annealing.
    
    for restart in range(20):
        if time.time() > deadline:
            break
        
        if restart == 0:
            x = x0.copy()
        else:
            x = np.random.choice([-1.0, 1.0], size=N)
        
        current_score = _compute_c3(x)
        
        # Simulated Annealing
        T = 0.1
        T_min = 0.001
        cooling_rate = 0.99
        
        while T > T_min and time.time() < deadline:
            # Pick a random flip
            i = np.random.randint(N)
            x[i] = -x[i]
            new_score = _compute_c3(x)
            
            delta = new_score - current_score
            
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_x = x.copy()
            else:
                x[i] = -x[i]  # revert
            
            T *= cooling_rate
        
        # Local refinement after annealing
        improved = True
        while improved and time.time() < deadline:
            improved = False
            best_flip_idx = -1
            best_flip_score = current_score
            
            # Try all single flips
            for i in range(N):
                x[i] = -x[i]
                s = _compute_c3(x)
                x[i] = -x[i]
                
                if s < best_flip_score:
                    best_flip_score = s
                    best_flip_idx = i
            
            if best_flip_idx != -1:
                x[best_flip_idx] = -x[best_flip_idx]
                current_score = best_flip_score
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    best_x = x.copy()
        
    return best_x

# EVOLVE-BLOCK-END
