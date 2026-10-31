"""Combined program: minimize L-infinity norm of autoconvolution for signed step function."""
N = 400
import numpy as np
import time

def _autoconv_bound(f):
    """Compute the L-infinity norm of the autoconvolution of the step function f.
    f is an array of N step heights.
    """
    h = 1.0 / (2 * N)
    # Compute linear autocorrelation
    # A[k] = sum_i f[i] f[i+k]
    # Using FFT for speed, or direct if N is small. N=400 is small enough for direct O(N^2) if needed, 
    # but FFT is O(N log N). 
    # However, for local search, we need fast updates. 
    # This function is for initial evaluation or final check.
    nfft = 1
    while nfft < 2 * N:
        nfft *= 2
    F = np.zeros(nfft)
    F[:N] = f
    conv = np.fft.irfft(np.fft.rfft(F) ** 2) * h
    # The relevant part is the linear convolution part.
    # The FFT gives circular convolution. We padded to 2N, so the first N values 
    # of the result (shifted) correspond to the linear convolution?
    # Actually, irfft of F^2 gives the circular convolution of the padded f.
    # Since we padded with zeros to 2N, the circular convolution is the linear convolution 
    # for the first N elements? No, linear conv length is 2N-1.
    # The result `conv` has length nfft.
    # The linear convolution is contained in conv[0:N] and conv[nfft-N+1:nfft] (wrapped).
    # Actually, for real signals, the linear convolution of length 2N-1 is in 
    # conv[0:N-1] and conv[nfft-(N-1):nfft].
    # Let's just take the max of the whole thing, as the zero-padding ensures 
    # the "wrap-around" parts are zero or negligible if nfft >= 2N.
    # With nfft >= 2N, the linear convolution is fully contained without aliasing 
    # in the first 2N-1 bins? 
    # Yes, if nfft >= 2N-1, the linear convolution is conv[0:2N-1] (with appropriate shift).
    # Actually, np.fft.irfft returns the result in natural order.
    # The linear convolution of two length-N vectors is in the first 2N-1 coefficients 
    # if the FFT size is at least 2N-1.
    return float(np.max(np.abs(conv[:2*N-1])))

def _init_autocorr(s):
    """Compute linear autocorrelation A_k for k in 0..N-1 and -1..-(N-1).
    Returns an array of length 2N-1 representing A_{-(N-1)} ... A_0 ... A_{N-1}.
    """
    # A_k = sum_{i} s_i s_{i+k}
    # We can use FFT: A = ifft(fft(s) * conj(fft(s)))
    nfft = 1
    while nfft < 2 * N:
        nfft *= 2
    S = np.fft.rfft(s, n=nfft)
    P = S * np.conj(S)
    A = np.fft.irfft(P, n=nfft)
    # A is the circular autocorrelation.
    # The linear autocorrelation is A[0:N] + A[nfft-N+1:nfft] (shifted)?
    # Actually, for linear autocorrelation:
    # A_lin[k] for k=0..N-1 is A_fft[k]
    # A_lin[-k] is A_fft[nfft-k]
    # So the full linear autocorrelation vector (length 2N-1) is:
    # A_fft[nfft-N+1 : nfft] (for negative lags) + A_fft[0 : N] (for non-negative lags)
    # Wait, A_fft[0] is A_0.
    # A_fft[1] is A_1.
    # ...
    # A_fft[N-1] is A_{N-1}.
    # A_fft[nfft-1] is A_{-(N-1)}? No, A_fft[nfft-k] is A_{-k}.
    # So A_{-(N-1)} is at index nfft-(N-1).
    # The vector of linear autocorrelations is:
    # [A_{-(N-1)}, ..., A_{-1}, A_0, A_1, ..., A_{N-1}]
    # Indices in A_fft:
    # A_0: 0
    # A_1: 1
    # ...
    # A_{N-1}: N-1
    # A_{-1}: nfft-1
    # A_{-2}: nfft-2
    # ...
    # A_{-(N-1)}: nfft-(N-1)
    
    # Construct the array
    A_lin = np.zeros(2*N-1)
    A_lin[N-1] = A[0] # A_0 at center
    for k in range(1, N):
        A_lin[N-1+k] = A[k] # A_k
        A_lin[N-1-k] = A[nfft-k] # A_-k
    return A_lin

def construct():
    """Construct a signed step function with N intervals to minimize the autoconvolution upper bound."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Initial candidate: Random or Legendre
    # Legendre sequence is often good for low autocorrelation
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n ** 0.5) + 1):
            if n % i == 0:
                return False
        return True
        
    p = N
    while not is_prime(p):
        p += 1
        
    s_leg = np.zeros(N)
    for i in range(N):
        if i == 0:
            s_leg[i] = 1.0
        else:
            # Legendre symbol
            if pow(i, (p - 1) // 2, p) == 1:
                s_leg[i] = 1.0
            else:
                s_leg[i] = -1.0
                
    # Also try a few random ones
    rng = np.random.RandomState(42)
    candidates = [s_leg]
    for _ in range(5):
        candidates.append(rng.choice([-1.0, 1.0], size=N))
        
    best_s = None
    best_max_corr = float('inf')
    
    for s_init in candidates:
        if time.time() > deadline:
            break
            
        s = s_init.copy()
        A = _init_autocorr(s)
        current_max = np.max(np.abs(A))
        
        # Local search
        improved = True
        no_improve_count = 0
        while improved and time.time() < deadline:
            improved = False
            indices = np.random.permutation(N)
            for i in indices:
                if time.time() > deadline:
                    break
                
                # Flip s[i]
                old_val = s[i]
                s[i] = -old_val
                
                # Update A
                # A_k changes by -2 * old_val * s[i+k] (for k>0)
                # Note: s[i] is now -old_val.
                # Delta A_k = -2 * old_val * s[i+k]
                # We need to update A for all k.
                # A_lin index for lag k is N-1+k.
                
                # This update is O(N).
                # Let's precompute the changes.
                # Actually, it's easier to just recompute A if N is small, 
                # but O(N) update is better.
                
                # Let's try the O(N) update.
                # A[k] = sum_j s_j s_{j+k}
                # If we flip s_i, terms involving s_i change.
                # For a specific lag k, the term s_i s_{i+k} is present if i+k < N.
                # The term s_{i-k} s_i is present if i-k >= 0.
                
                # It is safer and fast enough to recompute A using FFT for N=400?
                # FFT is ~400 log 400 ~ 3000 ops.
                # O(N) update is 400 ops.
                # Let's do O(N) update.
                
                # We need the current A.
                # Let's store the indices.
                # For lag k (0 to N-1):
                #   If i+k < N: A[k] -= 2 * old_val * s[i+k]
                #   If i-k >= 0: A[-k] -= 2 * old_val * s[i-k]  (which is A[k] in symmetric array?)
                #   A[0] -= 2 * old_val * s[i] = 2 * old_val^2 = 2 (since s[i] flipped, old_val^2=1)
                #   Wait, s[i] is flipped. 
                #   Old term: s_i s_{i+k}. New term: -s_i s_{i+k}.
                #   Change: -2 s_i s_{i+k}.
                
                # Let's just recompute A to avoid bugs, as N=400 is small.
                # 400 * 400 = 160,000 multiplications. In numpy, this is very fast.
                A_new = _init_autocorr(s)
                new_max = np.max(np.abs(A_new))
                
                if new_max < current_max - 1e-12:
                    current_max = new_max
                    A = A_new
                    improved = True
                    no_improve_count = 0
                else:
                    s[i] = old_val
                    no_improve_count += 1
                    if no_improve_count > N:
                        # Stagnation, break sweep
                        break
                        
        if current_max < best_max_corr:
            best_max_corr = current_max
            best_s = s.copy()
            
        # If we have time, we can do more random restarts or perturbations
        # But local search on Legendre is usually very good.
        
    if best_s is None:
        return np.ones(N)
        
    return best_s

