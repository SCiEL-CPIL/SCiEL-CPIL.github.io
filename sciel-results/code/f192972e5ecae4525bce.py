import numpy as np
import time

N = 600

def construct():
    start_time = time.time()
    timeout = 110  # Safety margin for 120s limit
    
    # Initialize with random binary sequence
    # Non-negative constraint
    f = np.random.rand(N)
    # Threshold to make it sparse/binary-ish initially? 
    # Random continuous might be better for smooth optimization
    f = np.clip(f, 0, 1)
    
    best_f = f.copy()
    best_score = float('inf')
    
    # Precompute indices for sidelobes
    # Autoconvolution length is 2N-1
    # Center is N-1
    # Sidelobes are indices 0 to N-2 and N to 2N-2
    
    for step in range(5000):
        if time.time() - start_time > timeout:
            break
            
        # Compute autoconvolution
        # Use FFT for speed
        f_fft = np.fft.fft(f, n=2*N)
        h_fft = f_fft * f_fft
        h = np.fft.ifft(h_fft).real[:2*N-1]
        
        # Normalize by main lobe (h[N-1])
        main_lobe = h[N-1]
        if main_lobe < 1e-12:
            # Avoid division by zero, reset
            f = np.random.rand(N)
            continue
            
        h_norm = h / main_lobe
        
        # Find max sidelobe
        # Mask out the main lobe
        mask = np.ones(2*N-1, dtype=bool)
        mask[N-1] = False
        
        h_sidelobes = h_norm[mask]
        max_val = np.max(h_sidelobes)
        
        # Update best
        if max_val < best_score:
            best_score = max_val
            best_f = f.copy()
            
        # Find the index of the max sidelobe
        # We need the actual index in the full h array
        # Get indices of True in mask
        valid_indices = np.where(mask)[0]
        max_idx_in_valid = np.argmax(h_sidelobes)
        k_max = valid_indices[max_idx_in_valid]
        
        # Calculate gradient of h[k_max] with respect to f
        # h[k] = sum_i f[i] f[i+k] (with appropriate bounds)
        # Gradient wrt f[j] is f[j+k] + f[j-k] (if in bounds)
        
        # Efficient gradient calculation using shifts
        # grad[j] = f[j+k_max] + f[j-k_max]
        
        grad = np.zeros(N)
        
        # Term 1: f[j+k_max]
        if k_max > 0:
            # f[j] contributes to h[k] via f[j] * f[j+k]
            # If h index is k_max, then f[j] * f[j+k_max]
            # So grad[j] += f[j+k_max]
            # j+k_max < N => j < N - k_max
            grad[:N-k_max] += f[k_max:]
            
            # Also f[j] * f[j-k_max] is not part of h[k_max] directly in this form?
            # h[k] = sum_i f[i] f[i+k].
            # If k > 0, term is f[i] f[i+k].
            # Derivative wrt f[j]:
            # 1. If j = i, term is f[j] f[j+k]. Deriv is f[j+k].
            # 2. If j = i+k, term is f[j-k] f[j]. Deriv is f[j-k].
            
            # So for k_max > 0:
            # grad[j] += f[j+k_max] for j < N-k_max
            # grad[j] += f[j-k_max] for j >= k_max
            
            grad[k_max:] += f[:N-k_max]
            
        elif k_max < 0:
            # Symmetric
            k_abs = -k_max
            grad[:N-k_abs] += f[k_abs:]
            grad[k_abs:] += f[:N-k_abs]
            
        # Normalize gradient to step size
        # We want to decrease the max sidelobe.
        # Move f in direction -grad.
        # But we must keep f >= 0.
        
        # Adaptive step size
        # Estimate Lipschitz constant or just use small step
        step_size = 0.1 / (np.max(np.abs(grad)) + 1e-9)
        
        f_new = f - step_size * grad
        f_new = np.clip(f_new, 0, None)
        
        # Normalize f to prevent drift in scale (metric is scale invariant)
        norm = np.linalg.norm(f_new)
        if norm > 1e-9:
            f_new = f_new / norm
        else:
            f_new = np.random.rand(N)
            
        f = f_new

    return list(best_f)

# EVOLVE-BLOCK-END
