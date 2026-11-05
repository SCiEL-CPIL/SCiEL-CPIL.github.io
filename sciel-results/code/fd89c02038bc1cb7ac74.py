import numpy as np
import time
import math

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Precompute FFT size
    fft_size = 1 << (2 * N - 1).bit_length()
    
    def compute_bound(h):
        """
        Compute the upper bound on C3: dx * max(|autocorr|)
        """
        h_pad = np.zeros(fft_size)
        h_pad[:N] = h
        f_fft = np.fft.fft(h_pad)
        power = f_fft * np.conj(f_fft)
        conv = np.fft.ifft(power).real
        # The autocorrelation is centered. For linear convolution of length 2N-1,
        # the peak is at index N-1 in the unpadded version, or N-1 in the padded version if we align correctly.
        # Actually, if we pad to 2N-1, the autocorrelation lags are -N+1 ... N-1.
        # np.fft.ifft gives circular convolution. With zero padding to >= 2N-1, it is linear.
        # The zero-lag is at index 0 of the result if we use the standard definition?
        # No, ifft of |FFT|^2 gives the circular autocorrelation.
        # If we pad h to length L >= 2N-1, the result is the linear autocorrelation.
        # The zero lag is at index 0? No, usually the DC component is at 0.
        # Let's check: if h = [1, 0, 0], fft = [1, 1, 1], ifft(|fft|^2) = [1, 0, 0] + wrap?
        # Standard practice:
        # corr = np.fft.ifft(np.fft.fft(h, n=fft_size) * np.conj(np.fft.fft(h, n=fft_size))).real
        # The peak is at index 0. The lags are 0, 1, ..., N/2, -N/2, ..., -1.
        # We want the max absolute value over all lags.
        return np.max(np.abs(conv))

    def thue_morse(n):
        seq = np.array([1.0])
        while len(seq) < n:
            seq = np.concatenate([seq, -seq])
        return seq[:n]

    def matern_kernel(n, nu, length_scale, seed=0):
        """
        Generate a binary sequence from a Matérn GP.
        """
        rng = np.random.RandomState(seed)
        # Distances
        indices = np.arange(n)
        # Assuming unit spacing for kernel calculation, scaled by length_scale
        # The kernel is K(x_i, x_j) = exp(-sqrt(2*nu)/length_scale * |x_i - x_j|) * ...
        # Simplified Matérn for 1D:
        # K(r) = (2**(1-nu)/gamma(nu)) * (sqrt(2*nu)*r/length_scale)**nu * K_nu(sqrt(2*nu)*r/length_scale)
        # For simplicity and speed, we can use a Gaussian kernel approximation or a simpler exponential kernel
        # which is Matérn with nu=0.5.
        # Let's use nu=1.5 (more realistic smoothness)
        
        # To avoid scipy.special.kv dependency issues or slowness, we can use the property that
        # sampling from a GP is equivalent to solving a system or using Cholesky.
        # For n=400, Cholesky is fine.
        
        # Let's stick to a simpler "smooth" random field:
        # Low-pass filter random noise.
        # This is effectively a GP with a Gaussian kernel (nu -> infinity).
        
        # Generate white noise
        noise = rng.randn(n)
        # Apply low-pass filter
        # FFT, zero out high frequencies
        noise_fft = np.fft.fft(noise)
        # Keep only low frequencies
        cutoff = int(n * 0.2) # Keep 20% of frequencies
        noise_fft[cutoff:-cutoff] = 0
        smooth_noise = np.fft.ifft(noise_fft).real
        
        # Normalize
        smooth_noise = (smooth_noise - np.mean(smooth_noise)) / (np.std(smooth_noise) + 1e-9)
        
        # Threshold to binary
        # To make it more "step-like" but smooth, we can just sign it
        # Or use a soft threshold? The problem asks for signed step function, usually implies +/-1.
        # "signed step heights" -> likely +/-1.
        binary = np.sign(smooth_noise)
        # Handle zeros
        binary[binary == 0] = 1.0
        
        return binary

    def simulated_annealing(h_initial, time_budget):
        """
        Run Simulated Annealing to minimize the autocorrelation bound.
        """
        h = h_initial.copy()
        current_bound = compute_bound(h)
        best_h = h.copy()
        best_bound = current_bound
        
        # Cooling schedule
        # Start high, end low
        T_start = 5.0
        T_end = 0.01
        # Number of iterations depends on time. Let's estimate.
        # One iter takes ~10-20 us (FFT 800 size).
        # 140s / 20us = 7,000,000 iters.
        # We don't need that many. 10,000-50,000 is usually enough for N=400.
        
        # We will run until time_budget is reached.
        start_sa = time.time()
        
        # Adaptive cooling:
        # Instead of fixed steps, we check time.
        
        # We'll do a fixed number of steps per "phase" and adjust T.
        # Or just a simple geometric cooling with a large number of steps.
        
        # Let's use a simple loop with time check.
        T = T_start
        cooling_rate = 0.9999 # Tune this
        
        # To be safe, we run for a fixed number of iterations or until time out.
        # Let's try 50,000 iterations.
        
        for _ in range(50000):
            if time.time() - start_sa > time_budget:
                break
            
            # Pick a random index to flip
            idx = np.random.randint(N)
            
            old_val = h[idx]
            h[idx] = -old_val
            
            new_bound = compute_bound(h)
            delta = new_bound - current_bound
            
            if delta < 0 or np.random.random() < math.exp(-delta / T):
                # Accept move
                current_bound = new_bound
                if new_bound < best_bound:
                    best_bound = new_bound
                    best_h = h.copy()
            else:
                # Reject move
                h[idx] = old_val
            
            # Cool down
            T *= cooling_rate
            if T < T_end:
                T = T_end
                
        return best_h, best_bound

    # 1. Generate Initializations
    inits = []
    
    # Random
    np.random.seed(42)
    inits.append(np.random.choice([-1.0, 1.0], size=N))
    
    # Thue-Morse
    inits.append(thue_morse(N))
    
    # Matérn-like (Smooth Noise) with different seeds and "smoothness" (cutoffs)
    # We'll use the matern_kernel function which uses a cutoff.
    # Let's try different cutoffs (smoothness)
    for seed in range(5):
        h_smooth = matern_kernel(N, 1.5, 1.0, seed=seed)
        inits.append(h_smooth)
        
    # 2. Run SA on each
    best_global_h = inits[0]
    best_global_bound = compute_bound(best_global_h)
    
    # Time budget per SA run
    # Total time 140s.
    # Let's allocate 10s per run, 5 runs = 50s.
    # Or dynamically: run SA for X seconds, then next.
    
    # Let's do 4 runs of 20s each.
    time_per_run = 20.0
    
    for i, h_init in enumerate(inits):
        if time.time() > deadline - 5:
            break
            
        # Run SA
        h_opt, bound_opt = simulated_annealing(h_init, time_per_run)
        
        if bound_opt < best_global_bound:
            best_global_bound = bound_opt
            best_global_h = h_opt
            
        # If we found a very good solution, we might want to polish it with more time?
        # For now, just keep the best.
        
    # 3. Final Polish
    # If we have time left, run SA again on the best found with a longer budget?
    # Or just return.
    
    # Let's do one final long run on the best candidate if time permits
    remaining = deadline - (time.time() - start_time) - 2.0
    if remaining > 5.0:
        h_polished, bound_polished = simulated_annealing(best_global_h, remaining)
        if bound_polished < best_global_bound:
            best_global_bound = bound_polished
            best_global_h = h_polished
            
    return best_global_h

# Dummy function to satisfy "keep function names" if any were expected, 
# but construct is the main one.
# The prompt says "Keep the function names and signatures of the seed program."
# The seed program only exposes `construct`.

if __name__ == "__main__":
    # Self-test
    h = construct()
    print("Constructed sequence length:", len(h))
    print("Unique values:", np.unique(h))
    # Print bound
    fft_size = 1 << (2 * N - 1).bit_length()
    h_pad = np.zeros(fft_size)
    h_pad[:N] = h
    f_fft = np.fft.fft(h_pad)
    power = f_fft * np.conj(f_fft)
    conv = np.fft.ifft(power).real
    print("Max Autocorr:", np.max(np.abs(conv)))
