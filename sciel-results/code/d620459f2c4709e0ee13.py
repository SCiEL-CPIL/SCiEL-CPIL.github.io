import numpy as np
import time

N = 400
t = np.arange(N)
t_centered = t - (N - 1) / 2.0

def _h_from_params(params):
    """
    Reconstruct h from parameters:
    params[0] = bias (DC component)
    params[1::2] = amplitudes
    params[2::2] = frequencies
    """
    h = np.full(N, params[0])
    n_sinusoids = (len(params) - 1) // 2
    for i in range(n_sinusoids):
        amp = params[1 + 2 * i]
        freq = params[2 + 2 * i]
        h += amp * np.sin(2 * np.pi * freq * t_centered / N)
    return h

def _objective(params):
    """
    Compute the value to minimize.
    value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h) ** 2)
    """
    h = _h_from_params(params)
    
    # Guard against zero sum
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e6
        
    # Compute autoconvolution
    # np.convolve(h, h) is full convolution, length 2N-1
    # We can use FFT for speed if N is large, but N=400 is small enough for direct or FFT.
    # Let's use FFT for robustness and speed in the loop.
    # FFT length needs to be at least 2N-1. Next power of 2 is 1024.
    f_len = 2 * N - 1
    # Pad to next power of 2 for efficiency
    fft_len = 1 << (f_len - 1).bit_length()
    
    H = np.fft.rfft(h, n=fft_len)
    conv_fft = H * H
    conv = np.fft.irfft(conv_fft, n=fft_len)[:f_len]
    
    max_conv = np.max(conv)
    
    val = abs(2 * N * max_conv / (s * s))
    return val

def construct():
    """
    Returns the optimal step heights.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initialize with a small number of sinusoids
    # Start with 5 sinusoids + 1 bias = 11 parameters
    n_sinusoids = 5
    n_params = 1 + 2 * n_sinusoids
    
    # Initial parameters
    # Bias = 1.0 (to ensure sum is not zero and positive)
    # Amplitudes = 0.1
    # Frequencies = small non-integer values
    best_params = np.zeros(n_params)
    best_params[0] = 1.0
    for i in range(n_sinusoids):
        best_params[1 + 2 * i] = 0.1
        best_params[2 + 2 * i] = 0.5 + i * 0.7
        
    best_val = _objective(best_params)
    
    # Simple Gradient Descent with numerical gradients
    # Since the objective involves a max operation, it's non-smooth, but we can approximate.
    # Alternatively, use a soft-max approximation for smoothness?
    # For now, let's try a simple random restart + local gradient descent (finite diff).
    
    lr = 0.05
    step_size_fd = 1e-4
    
    while time.time() < deadline:
        # Compute gradient numerically
        grad = np.zeros(n_params)
        for i in range(n_params):
            params_plus = best_params.copy()
            params_plus[i] += step_size_fd
            val_plus = _objective(params_plus)
            
            params_minus = best_params.copy()
            params_minus[i] -= step_size_fd
            val_minus = _objective(params_minus)
            
            grad[i] = (val_plus - val_minus) / (2 * step_size_fd)
            
        # Update
        new_params = best_params - lr * grad
        
        # Simple step size adjustment if it fails
        new_val = _objective(new_params)
        
        if new_val < best_val:
            best_val = new_val
            best_params = new_params
            # Maybe increase lr slightly if we made good progress?
            # lr = min(lr * 1.1, 0.5)
        else:
            # Reduce learning rate
            lr *= 0.8
            if lr < 1e-6:
                # Restart with different initial frequencies
                best_params[0] = 1.0
                for i in range(n_sinusoids):
                    best_params[1 + 2 * i] = np.random.uniform(-0.5, 0.5)
                    best_params[2 + 2 * i] = np.random.uniform(0.1, 2.0)
                lr = 0.05
                best_val = _objective(best_params)
                
        # Check time
        if time.time() > deadline - 5.0:
            break
            
    h_final = _h_from_params(best_params)
    
    # Final check
    if not np.all(np.isfinite(h_final)):
        return np.ones(N)
        
    return h_final

if __name__ == "__main__":
    h = construct()
    print(h[:10])
    val = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h) ** 2)
    print(f"Value: {val}")
