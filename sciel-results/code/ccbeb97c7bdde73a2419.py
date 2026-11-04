import numpy as np
import time

N = 400

def construct():
    """
    Construct a signed step function (400 intervals) to minimize the upper bound on C3.
    The metric is the peak of the absolute autoconvolution scaled by the interval width.
    We use a combination of deterministic chirp-based initialization and local search 
    (bit-flip hill climbing) to minimize this bound.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # --- Metric Definition ---
    # The step function has N intervals on [-1/4, 1/4], so total width is 0.5.
    # Interval width dx = 0.5 / N.
    # The autoconvolution (f*f)(t) for a step function is proportional to the 
    # discrete linear autoconvolution of the heights.
    # Specifically, if f(x) = h_i on [x_i, x_i+dx], then (f*f)(t_k) approx dx * sum_i h_i h_{i+k}.
    # The "upper bound on C3" is interpreted as dx * max_k |sum_i h_i h_{i+k}|.
    
    dx = 0.5 / N
    
    def compute_bound(h):
        """
        Compute the C3 upper bound: dx * max(|linear_autoconv(h)|)
        """
        # Linear autoconvolution via FFT
        # Pad h to length 2*N-1 for linear convolution
        h_pad = np.zeros(2 * N - 1)
        h_pad[:N] = h
        
        # FFT size needs to be at least 2*(2N-1) to avoid circular convolution artifacts
        # 2N-1 is the length of the linear convolution. 
        # We want the autoconvolution, which is convolution of h with reversed h? 
        # No, autoconvolution is h * h. 
        # Linear convolution of h (length N) with h (length N) has length 2N-1.
        # We use FFT of size >= 2N-1. Let's use next power of 2 for speed.
        fft_size = 1
        while fft_size < 2 * N - 1:
            fft_size *= 2
            
        H = np.fft.fft(h_pad, n=fft_size)
        # Autoconvolution is H * conj(H) for real signals? 
        # Convolution theorem: IFFT(FFT(h) * FFT(h)) gives h * h.
        # Since h is real, FFT(h) * conj(FFT(h)) = |FFT(h)|^2 gives the energy spectrum, 
        # and IFFT of that is the autocorrelation (circular).
        # For linear autoconvolution, we need IFFT(FFT(h_pad) * FFT(h_pad)).
        conv = np.fft.ifft(H * H).real
        
        # The linear autoconvolution is in conv[0 : 2*N-1]
        # We take the max absolute value
        max_abs = np.max(np.abs(conv[:2*N-1]))
        return dx * max_abs

    def local_search(h, time_budget=5.0):
        """
        Local search by flipping signs to reduce the bound.
        """
        h = np.array(h, dtype=np.float64)
        current_bound = compute_bound(h)
        start_ls = time.time()
        
        # Precompute FFT size
        fft_size = 1
        while fft_size < 2 * N - 1:
            fft_size *= 2
            
        improved = True
        while improved and (time.time() - start_ls) < time_budget and (time.time() < deadline - 2.0):
            improved = False
            best_gain = 0
            best_idx = -1
            
            # Try flipping each bit
            for i in range(N):
                # Flip
                h[i] = -h[i]
                
                # Compute new bound efficiently?
                # Recomputing FFT is O(N log N). For N=400, this is ~2048*log(2048) ~ 22k ops.
                # 400 flips * 22k ops = 8.8M ops per full sweep.
                # This is fast enough in Python/NumPy if vectorized or C-optimized.
                # Let's just recompute.
                h_pad = np.zeros(2 * N - 1)
                h_pad[:N] = h
                H = np.fft.fft(h_pad, n=fft_size)
                conv = np.fft.ifft(H * H).real
                new_bound = dx * np.max(np.abs(conv[:2*N-1]))
                
                gain = current_bound - new_bound
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                
                # Unflip
                h[i] = -h[i]
            
            if best_idx >= 0:
                h[best_idx] = -h[best_idx]
                current_bound -= best_gain
                improved = True
                
        return h, current_bound

    # --- Initialization ---
    
    # 1. Thue-Morse Sequence
    def thue_morse(n):
        seq = np.array([1.0])
        while len(seq) < n:
            seq = np.concatenate([seq, -seq])
        return seq[:n]
    
    # 2. Chirp-based Sequence (Zadoff-Chu like)
    # Product of two chirps with distinct prime moduli
    def chirp_sequence(p1, p2):
        # Ensure p1, p2 are primes > N
        # Primes: 401, 409, 419, 421, 431...
        k = np.arange(N)
        # Quadratic phase: exp(j * pi * k^2 / p)
        phase1 = np.exp(1j * np.pi * k**2 / p1)
        phase2 = np.exp(1j * np.pi * k**2 / p2)
        prod = phase1 * phase2
        # Take sign of imaginary part (as per directive)
        return np.sign(np.imag(prod))
        
    # Filter out zeros (if any, though sign of imag of complex exp is rarely exactly 0 for integer k, prime p)
    # If 0, map to 1
    def clean_signs(h):
        h = np.where(h == 0, 1.0, h)
        return h

    candidates = []
    
    # Thue-Morse
    h_tm = clean_signs(thue_morse(N))
    candidates.append(h_tm)
    
    # Chirps with various primes
    primes = [401, 409, 419, 421, 431, 433, 439, 443, 449, 457]
    for i in range(len(primes)):
        for j in range(i+1, len(primes)):
            h_chirp = clean_signs(chirp_sequence(primes[i], primes[j]))
            candidates.append(h_chirp)
            if len(candidates) > 10:
                break
        if len(candidates) > 10:
            break
            
    # Random sequences
    rng = np.random.RandomState(42)
    for _ in range(5):
        h_rand = rng.choice([-1.0, 1.0], size=N)
        candidates.append(h_rand)

    # --- Optimization Loop ---
    
    best_h = None
    best_bound = float('inf')
    
    # Evaluate initial candidates
    bounds = []
    for h in candidates:
        b = compute_bound(h)
        bounds.append(b)
        
    # Sort by bound
    sorted_indices = np.argsort(bounds)
    
    # Run local search on the top candidates
    # We have ~115 seconds. Local search sweeps are fast.
    # Let's allocate time dynamically.
    
    for idx in sorted_indices:
        if time.time() > deadline - 5.0:
            break
            
        h_init = candidates[idx]
        b_init = bounds[idx]
        
        # If this is already better than current best, we must search it
        # If not, we might still search it if it's close, as local search can improve it significantly
        # Strategy: Search top 3-5 candidates thoroughly.
        
        # Allocate time based on rank
        rank = list(sorted_indices).index(idx)
        if rank < 3:
            time_budget = 20.0
        elif rank < 5:
            time_budget = 10.0
        else:
            time_budget = 5.0
            
        # Check if we have enough time
        remaining = deadline - time.time() - 2.0
        if remaining < 1.0:
            break
            
        time_budget = min(time_budget, remaining)
        
        h_opt, b_opt = local_search(h_init, time_budget)
        
        if b_opt < best_bound:
            best_bound = b_opt
            best_h = h_opt
            
        # If we found a very good solution, we can stop early or continue to refine
        # The theoretical minimum for autoconvolution peak of +/-1 sequence is related to sqrt(N).
        # For N=400, sqrt(400)=20. dx=0.5/400=0.00125.
        # Ideal bound ~ 0.00125 * 20 = 0.025.
        # Current best is likely around 0.025-0.03.
        
    # If no local search was run (e.g. time out), pick best initial
    if best_h is None:
        best_idx = sorted_indices[0]
        best_h = candidates[best_idx]
        best_bound = bounds[best_idx]

    return best_h

# Test block (not executed by evaluator, but good for sanity check)
if __name__ == "__main__":
    start = time.time()
    h = construct()
    end = time.time()
    print(f"Time taken: {end - start:.2f}s")
    print(f"Length: {len(h)}")
    print(f"First 20: {h[:20]}")
    
    # Compute final score
    dx = 0.5 / N
    h_pad = np.zeros(2 * N - 1)
    h_pad[:N] = h
    fft_size = 1
    while fft_size < 2 * N - 1:
        fft_size *= 2
    H = np.fft.fft(h_pad, n=fft_size)
    conv = np.fft.ifft(H * H).real
    bound = dx * np.max(np.abs(conv[:2*N-1]))
    print(f"C3 Bound: {bound}")
    print(f"Score (1/bound): {1.0/bound}")
