import numpy as np
import time

N = 400

def _autocorr_peak_sidelobe(h):
    """
    Compute the peak sidelobe level of the aperiodic autocorrelation.
    h is a 1D numpy array of +/- 1.
    """
    # Zero-pad to avoid circular correlation
    n = len(h)
    # Next power of 2 >= 2*n - 1
    nfft = 1 << (2 * n - 1).bit_length()
    
    # FFT of h
    H = np.fft.rfft(h, n=nfft)
    # Power spectral density
    P = H * np.conj(H)
    # Inverse FFT to get autocorrelation
    acf = np.fft.irfft(P, n=nfft)
    
    # The aperiodic autocorrelation R(k) for k=0..n-1 is in acf[n-1-k] ... acf[n-1] ... 
    # Actually, standard irfft of |H|^2 gives circular correlation.
    # For aperiodic:
    # R(k) = sum_{i=0}^{n-1-k} h[i] h[i+k]
    # The linear convolution of h and reversed h gives the full correlation.
    # Let's use the direct slicing from the linear correlation result.
    # np.fft.irfft(H * conj(H)) gives circular correlation of length nfft.
    # The linear autocorrelation R(k) for k=0 to n-1 corresponds to indices:
    # R(0) is at index n-1 (if we consider the standard centering) or we can just extract it properly.
    
    # Easier: Use np.correlate? It's O(N^2) which is 160k ops. Very fast.
    # But FFT is O(N log N) ~ 400 * 9 = 3600 ops. FFT is faster.
    
    # Let's ensure correct extraction.
    # The linear autocorrelation r[k] for k=0..N-1 is:
    # r[k] = sum_{i=0}^{N-1-k} h[i] h[i+k]
    # In the FFT domain, if we compute ifft(fft(h, 2N) * conj(fft(h, 2N))), 
    # we get the circular correlation of length 2N.
    # The linear correlation is embedded in this.
    # Specifically, r[k] for k=0..N-1 is found at indices N-1-k to N-1? No.
    
    # Let's just use the property:
    # acf_full = ifft(fft(h, 2*N) * conj(fft(h, 2*N)))
    # The aperiodic autocorrelation R(k) for k >= 0 is acf_full[N-1 : 2*N-1][::-1]? 
    # Or simply:
    # R(k) = acf_full[N-1+k] for k=0..N-1?
    # Let's verify with small example.
    # h = [1, -1]
    # R(0) = 1+1=2. R(1) = -1.
    # fft([1, -1, 0, 0]) -> [0, 2, 0, -2] (approx)
    # |H|^2 -> [0, 4, 0, 4]
    # ifft -> [2, 0, -2, 0] / 4? No, ifft scales by 1/N.
    # ifft([0,4,0,4]) = [2, 0, -2, 0].
    # R(0) should be 2. R(1) should be -1.
    # The values are at indices 2 and 3?
    # Index 2: -2. Index 3: 0.
    # This is getting confusing. Let's stick to a reliable method.
    
    # Reliable Method:
    # r = np.correlate(h, h, mode='full')
    # R(k) is r[N-1+k] for k=0..N-1.
    # For N=400, N^2 = 160,000 operations. This is extremely fast in C (numpy).
    # 160k ops is negligible. We can do 100,000 such calculations in 120s easily.
    # So we can afford to compute the full correlation for every candidate flip if we are clever, 
    # or just use it for the evaluation step.
    
    r = np.correlate(h, h, mode='full')
    # r has length 2*N - 1.
    # r[N-1] is R(0).
    # r[N-1+k] is R(k).
    # r[N-1-k] is R(-k) = R(k).
    # We want max |R(k)| for k != 0.
    # Sidelobes are r[0:N-1] and r[N:2*N-1].
    # Due to symmetry, we can just look at r[N:2*N-1] (positive lags).
    sidelobes = r[N:2*N-1]
    return np.max(np.abs(sidelobes))

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_f = None
    best_val = float('inf')
    
    # Try multiple random initializations
    # 1. Random
    # 2. Random Walk (fBm-like)
    # 3. Chirp
    
    initializers = []
    
    # Random
    np.random.seed(42)
    initializers.append(np.sign(np.random.randn(N)) * 1.0)
    initializers[-1][initializers[-1] == 0] = 1.0
    
    # Random Walk
    np.random.seed(123)
    walk = np.cumsum(np.random.randn(N))
    initializers.append(np.sign(walk) * 1.0)
    initializers[-1][initializers[-1] == 0] = 1.0
    
    # Chirp
    np.random.seed(456)
    t = np.arange(N)
    initializers.append(np.sign(np.cos(2 * np.pi * t**2 / (N * 4))) * 1.0)
    initializers[-1][initializers[-1] == 0] = 1.0

    for init_f in initializers:
        if time.time() > deadline - 10.0:
            break
            
        f = init_f.copy()
        current_val = _autocorr_peak_sidelobe(f)
        
        # Simulated Annealing
        T = 5.0
        T_min = 0.05
        cooling = 0.999
        
        while T > T_min:
            if time.time() > deadline - 5.0:
                break
            
            # Pick a random index to flip
            idx = np.random.randint(N)
            f[idx] *= -1.0
            
            new_val = _autocorr_peak_sidelobe(f)
            delta = new_val - current_val
            
            # Accept if better or with probability exp(-delta/T)
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                current_val = new_val
                if new_val < best_val:
                    best_val = new_val
                    best_f = f.copy()
            else:
                # Revert
                f[idx] *= -1.0
            
            T *= cooling
        
        # If we found a better one, we might want to continue SA on this best one?
        # Or just let the next initializer try.
        # Since we update best_f, we are keeping track.
        
    return best_f
