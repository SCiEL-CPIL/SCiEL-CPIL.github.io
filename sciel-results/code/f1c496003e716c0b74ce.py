import numpy as np
import time

N = 400
DEADLINE = 130.0  # Safety margin before 150s kill

def _compute_c3_full(signed_step):
    """Compute C3 upper bound using FFT autoconvolution."""
    h = 1.0 / N
    size = 1 << (2 * N - 1).bit_length()  # Next power of 2 >= 2*N
    
    f = np.zeros(size)
    # Map each interval to a contiguous block in the padded array
    # The function is defined on [-1/4, 1/4] with N intervals of width 1/N
    # We place it centered in the FFT array
    offset = size // 2 - N // 2
    for i in range(N):
        f[offset + i] = signed_step[i]
    
    fft_f = np.fft.fft(f)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real
    
    # Scale: each point in f represents a width of 1/size in the normalized domain
    # The actual integral scale: dx = 0.5 / size (since domain is [-1/4, 1/4], total width 0.5)
    # But wait, we have N intervals of width 1/N on [-1/4, 1/4], total width = 1
    # Actually [-1/4, 1/4] has width 1/2. So each interval has width (1/2)/N.
    # Let's re-examine. The problem says "400 equal intervals on [-1/4, 1/4]".
    # Width of interval = (1/4 - (-1/4)) / 400 = (1/2)/400 = 1/800.
    # In the FFT, we are approximating the integral.
    # f is the value at discrete points. The integral is sum(f_i * dx).
    # If we sample the step function at N points, and we want to convolve it with itself.
    # The autoconvolution at lag k is sum_i f(i) f(i-k).
    # The continuous integral is ∫ f(t) f(t-x) dt.
    # Approximation: sum_i f(i) f(i-k) * (width of interval).
    # Width of interval = 0.5 / N.
    
    dx = 0.5 / N
    # The FFT computes circular convolution. We need linear.
    # By zero-padding to size >= 2N, we get linear convolution.
    # The result autoconv is the sum of products.
    # We need to multiply by dx to get the integral.
    # Also, the FFT result is scaled by 1/size in ifft.
    # np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)) gives a * b (circular conv) scaled?
    # Actually, ifft(fft(a) * fft(b)) is the circular convolution of a and b.
    # The values are correct sums.
    # So autoconv[k] = sum_i f[i] f[i-k] (mod size).
    # We need to multiply by dx to approximate the integral.
    
    autoconv *= dx
    c3 = np.max(np.abs(autoconv))
    return c3

def construct():
    start_time = time.time()
    
    # Initialize with random signs (better than all ones for exploring)
    # Or start with all ones and let SA explore?
    # Let's start with a random configuration to avoid the all-ones local minimum trap immediately.
    # Actually, the seed program starts with all ones.
    # The "seed program" score is 1.0.
    # Let's try starting with a random configuration, but ensure we return the best found.
    # The task says "Keep the function names and signatures of the seed program".
    # It doesn't say I must start with the same initialization, but usually "improve" implies beating the baseline.
    # The baseline is likely the all-ones vector or a specific known configuration.
    # Let's initialize with all ones to be safe, but SA will move away if beneficial.
    
    best_step = np.ones(N, dtype=np.int8)
    current_step = best_step.copy()
    
    # Precompute FFT of the initial step to allow incremental updates?
    # Incremental FFT update:
    # If we flip s[i] from 1 to -1, the change in f is -2 at position i.
    # FFT(f_new) = FFT(f_old) + FFT(delta)
    # FFT(delta) where delta is -2 at pos i, 0 elsewhere.
    # FFT(delta)[k] = -2 * exp(-2j * pi * k * i / size)
    # This is O(size) per update, which is O(N).
    # Computing C3 requires ifft and max, which is O(size) = O(N).
    # So each step is O(N).
    # N=400, size=1024.
    # 1024 operations per step.
    # We can do ~10^6 steps in 1 second? No, Python overhead is high.
    # Let's benchmark: 100k steps might take 10-20s.
    # We have 130s. So ~1M steps is possible if optimized.
    
    # Let's implement the incremental update for speed.
    size = 1 << (2 * N - 1).bit_length()
    offset = size // 2 - N // 2
    
    # Build initial f
    f = np.zeros(size)
    for i in range(N):
        f[offset + i] = current_step[i]
    
    fft_f = np.fft.fft(f)
    
    # Precompute the complex exponentials for each position to speed up delta FFT
    # pos_in_fft = offset + i
    # delta_fft[k] = -2 * exp(-2j * pi * k * (offset+i) / size)
    # We can precompute exp(-2j * pi * k * offset / size) and exp(-2j * pi * k * i / size)
    # Or just compute the delta FFT on the fly using np.fft.fft on a sparse vector? No, that's slow.
    # Better: precompute a matrix E[i, k] = exp(-2j * pi * k * (offset+i) / size)?
    # That's N x size = 400 x 1024 = 400k complex numbers. 3.2MB. Acceptable.
    # But we only need one row at a time.
    # Let's precompute the twiddle factors.
    # twiddle[i, k] = exp(-2j * pi * k * i / size)
    # Then delta_fft = -2 * twiddle[i, :] * base_twiddle
    # where base_twiddle = exp(-2j * pi * k * offset / size)
    
    k_vals = np.arange(size)
    base_twiddle = np.exp(-2j * np.pi * k_vals * offset / size)
    # twiddle[i, k] = exp(-2j * pi * k * i / size)
    # We can compute this on the fly: exp(-2j * pi * i * k_vals / size)
    # This is O(size) per step.
    
    # Let's try to optimize the inner loop.
    # We need to update fft_f, then compute autoconv = ifft(fft_f * conj(fft_f)).real
    # Then find max abs.
    
    # To speed up max abs, we can use np.max(np.abs(...)).
    
    # Simulated Annealing Parameters
    T_start = 1.0
    T_end = 0.01
    # Cooling schedule: T = T_start * (T_end/T_start)^t
    # Or T = T_start / (1 + alpha * t)
    # Let's use geometric cooling.
    # We need to estimate the number of steps we can take.
    # Let's run for a fixed number of steps or until time limit.
    
    # Let's estimate step time.
    # One step:
    # 1. Flip bit
    # 2. Update fft_f: fft_f += -2 * delta_fft
    # 3. Compute autoconv: ifft(fft_f * conj(fft_f)).real
    # 4. Compute c3: max(abs(autoconv))
    # 5. Accept/Reject
    
    # Let's benchmark one step.
    # Actually, let's just run the loop and check time.
    
    # Initialize best
    best_c3 = _compute_c3_full(current_step)
    
    # SA Loop
    no_improve_count = 0
    patience = 5000
    
    # We need to know how many steps to do.
    # Let's just loop until deadline.
    
    step_count = 0
    # Precompute twiddle factors for all i?
    # twiddles[i] = exp(-2j * pi * i * k_vals / size)
    # This is an N x size matrix.
    # Let's create it.
    i_vals = np.arange(N)
    # twiddles[i, k] = exp(-2j * pi * i * k / size)
    # This is separable.
    # We can compute it as exp(-2j * pi * i_vals[:, None] * k_vals[None, :] / size)
    # But this is 400x1024.
    # Let's do it.
    twiddles = np.exp(-2j * np.pi * np.outer(i_vals, k_vals) / size)
    
    # Current fft_f
    # We already have fft_f from initial f.
    
    # For each step:
    # idx = random int
    # old_val = current_step[idx]
    # new_val = -old_val
    # delta_val = new_val - old_val = -2 * old_val
    # pos = offset + idx
    # delta_fft = delta_val * twiddles[idx, :] * base_twiddle
    # fft_f += delta_fft
    # autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real
    # c3 = np.max(np.abs(autoconv))
    
    # Note: ifft is O(N log N).
    # max is O(N).
    # Total O(N log N).
    
    # Let's run.
    while time.time() - start_time < DEADLINE - 5.0:
        step_count += 1
        
        # Cooling
        # Let's use a simple linear or geometric cooling based on step count.
        # Since we don't know total steps, let's use a time-based cooling?
        # Or just a long tail.
        # Let's use T = T_start * (T_end/T_start) ** (step_count / 100000)
        # This cools slowly.
        T = T_start * (T_end / T_start) ** (step_count / 50000.0)
        if T < T_end:
            T = T_end
            
        idx = np.random.randint(N)
        old_val = current_step[idx]
        new_val = -old_val
        current_step[idx] = new_val
        
        delta_val = new_val - old_val  # -2 or 2
        
        # Update FFT
        # delta_fft = delta_val * twiddles[idx, :] * base_twiddle
        fft_f += delta_val * (twiddles[idx, :] * base_twiddle)
        
        # Compute C3
        autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real
        c3 = np.max(np.abs(autoconv))
        
        # Acceptance
        if c3 < best_c3:
            best_step = current_step.copy()
            best_c3 = c3
            no_improve_count = 0
        else:
            # Probabilistic acceptance
            delta_c3 = c3 - best_c3
            if np.random.rand() < np.exp(-delta_c3 / T):
                # Accept
                pass
            else:
                # Reject
                current_step[idx] = old_val
                fft_f -= delta_val * (twiddles[idx, :] * base_twiddle)
                no_improve_count += 1
                if no_improve_count > patience:
                    # Restart with a new random configuration?
                    # Or just break?
                    # Let's try a random restart.
                    current_step = np.random.choice([-1, 1], size=N).astype(np.int8)
                    f = np.zeros(size)
                    for i in range(N):
                        f[offset + i] = current_step[i]
                    fft_f = np.fft.fft(f)
                    c3 = _compute_c3_full(current_step)
                    if c3 < best_c3:
                        best_step = current_step.copy()
                        best_c3 = c3
                    no_improve_count = 0
                    step_count = 0 # Reset cooling?
                    
    return best_step

# Keep the original _compute_c3 for compatibility if needed, though we use full version above
def _compute_c3(signed_step):
    return _compute_c3_full(signed_step)
