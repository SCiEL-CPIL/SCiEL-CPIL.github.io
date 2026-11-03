import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin
    
    # Objective function: max absolute autoconvolution sidelobe / N
    # We want to minimize this.
    
    def get_score(s):
        # s is array of -1 and 1
        fft_s = np.fft.fft(s)
        power = np.abs(fft_s) ** 2
        acf = np.fft.ifft(power).real
        # The DC component is at index 0, but ifft puts it at 0.
        # fftshift moves the zero frequency to the center.
        acf_shifted = np.fft.fftshift(acf)
        # The center is at N//2
        # We exclude the center peak (lag 0)
        # Note: acf_shifted is real but might have tiny imaginary parts due to numerical errors, we take real
        # The lags are -N//2 to N//2-1
        # The center index is N//2
        # We want max abs of all except center
        # Since acf is symmetric, we can just look at the positive lags or all
        # Let's just take the max of the absolute values of the shifted acf, 
        # but we must exclude the index corresponding to lag 0.
        # With fftshift, lag 0 is at index N//2.
        
        # To be safe, let's reconstruct the lags properly.
        # np.fft.ifft returns acf[0] = sum s_i^2 = N.
        # acf[k] = sum s_i s_{i+k} for k>0? No, ifft of power gives circular correlation.
        # For linear correlation, we need to zero-pad or just use the fact that N=400 is small enough 
        # that circular and linear might differ at the edges?
        # Actually, autoconvolution of a finite sequence is linear.
        # FFT computes circular correlation.
        # To get linear correlation via FFT, we should zero-pad to at least 2N.
        # Let's zero-pad to 2N to be correct.
        
        # Re-evaluating: 
        # Linear autoconvolution R(k) = sum_{i} s_i s_{i+k}
        # Using FFT: 
        # pad s to length 2N
        # fft_s = fft(s_padded)
        # power = abs(fft_s)^2
        # acf_circular = ifft(power)
        # The linear autoconvolution is contained in the first N elements of acf_circular?
        # Yes, if we pad to 2N, the circular correlation matches the linear correlation for lags -N+1 to N-1.
        
        s_padded = np.zeros(2 * N)
        s_padded[:N] = s
        fft_s = np.fft.fft(s_padded)
        power = np.abs(fft_s) ** 2
        acf = np.fft.ifft(power).real
        # The linear autoconvolution is in acf[0:N] and acf[N:2N]?
        # acf[0] is the sum of squares = N.
        # acf[k] for k=1..N-1 corresponds to lag k.
        # acf[N+k] corresponds to lag -k.
        # So we can just take acf[1:N] and acf[N+1:2N] (which is symmetric to acf[1:N])
        # So max abs is max(abs(acf[1:N]))
        
        peak = np.max(np.abs(acf[1:N]))
        return peak / N

    # Initial random sequence
    s = np.random.choice([-1, 1], N)
    current_score = get_score(s)
    best_s = s.copy()
    best_score = current_score
    
    # Simulated Annealing Parameters
    T = 0.05  # Initial temperature
    T_min = 0.001
    alpha = 0.999  # Cooling rate
    block_sizes = [4, 8, 16, 32]
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Choose a block size
        b_size = np.random.choice(block_sizes)
        
        # Choose start index
        start_idx = np.random.randint(0, N - b_size + 1)
        
        # Create a new sequence
        s_new = s.copy()
        s_new[start_idx:start_idx + b_size] *= -1
        
        new_score = get_score(s_new)
        delta = new_score - current_score
        
        if delta < 0:
            s = s_new
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_s = s.copy()
        else:
            # Accept with probability exp(-delta / T)
            if np.random.random() < np.exp(-delta / T):
                s = s_new
                current_score = new_score
        
        T *= alpha
        
        # If we have found a very good score, we might want to cool faster or stop
        # But let's keep going until deadline
        
    return best_s

# The interface requires construct() -> sequence of signed step heights
# The seed returned np.ones(N). The prompt says "signed step heights", implying -1 and 1.
# The score is the upper bound on C3.
# We return the best sequence found.
