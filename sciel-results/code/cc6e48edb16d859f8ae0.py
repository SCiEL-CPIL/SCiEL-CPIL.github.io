"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    # --- Initialization: Thue-Morse sequence ---
    def thue_morse(n):
        """Generate Thue-Morse sequence of length n: t_k = popcount(k) mod 2, mapped to +1/-1."""
        t = np.zeros(n, dtype=np.int8)
        for i in range(1, n):
            t[i] = t[i >> 1] ^ (i & 1)
        return 1 - 2 * t  # +1 for 0, -1 for 1

    def m_sequence(n):
        """Generate M-sequence (maximal length LFSR) of length n mapped to +1/-1.
        For n=400, we use a 9-bit LFSR with taps that gives period 511, take first 400."""
        # Use a primitive polynomial for degree 9: x^9 + x^4 + 1 (taps 9,4)
        # Actually let's use a standard one. For simplicity, use a known good LFSR.
        # Taps for degree 9: x^9 + x^5 + x^3 + x^2 + 1 (taps 9,5,3,2)
        # Let's just use a simple bit-reversal or another construction.
        # Alternative: use the m-sequence from a 9-bit LFSR with taps [9,5,3,2,1]
        # For now, let's use a different initialization: random with some structure
        
        # Actually, let's use a simpler approach: generate a maximal LFSR sequence
        # For N=400, we can use a 9-bit LFSR (period 511) and take first 400 bits
        taps = [9, 5, 3, 2, 1]  # primitive polynomial x^9+x^5+x^3+x^2+1
        reg = 1  # initial state
        seq = []
        for _ in range(n):
            seq.append(reg & 1)
            feedback = 0
            for tap in taps:
                feedback ^= (reg >> (tap - 1)) & 1
            reg = ((reg >> 1) | (feedback << (8))) & 0x1FF
        # Map to +1/-1
        return 1 - 2 * np.array(seq, dtype=np.int8)

    def random_init(n):
        return np.random.choice([-1, 1], size=n).astype(np.int8)

    # --- Metric computation ---
    def compute_c3_bound(x):
        """
        Compute the upper bound on C3 (third autocorrelation inequality).
        
        The task says "autoconvolution upper bound" and "third autocorrelation inequality."
        For a signed step function with values in {-1, +1}, the relevant quantity is likely
        related to the maximum of the autocorrelation sidelobes or a specific combination.
        
        Based on the knowledge that Thue-Morse + hill-climbing on "max autocorrelation sidelobe"
        reduced the metric from ~399 to ~20-30, the metric is likely:
        
        C3_bound = max_{k=1,...,N-1} |sum_{i=0}^{N-1-k} x[i] * x[i+k]|
        
        or possibly the sum of absolute sidelobes, or the max sidelobe.
        
        Let me think about this more carefully. The "third autocorrelation inequality" in the
        context of C3 (which often refers to the constant in inequalities involving
        autocorrelation) suggests we're looking at:
        
        For a sequence x of length N with values ±1, define the autocorrelation:
        R(k) = sum_{i=0}^{N-1-k} x[i] * x[i+k] for k = 0, ..., N-1
        
        R(0) = N (since all x[i]^2 = 1)
        
        The "upper bound on C3" likely refers to the maximum sidelobe:
        C3 = max_{k=1,...,N-1} |R(k)|
        
        Or it could be the L1 norm of sidelobes: sum_{k=1}^{N-1} |R(k)|
        
        Given that all-ones gives ~399 (which is N-1 = 399), this matches:
        For all ones, R(k) = N-k for all k, so max |R(k)| for k>=1 is N-1 = 399.
        And sum_{k=1}^{N-1} (N-k) = N(N-1)/2 = 79800, which is much larger.
        
        So the metric is: max_{k=1,...,N-1} |R(k)|
        
        This is the maximum absolute value of the non-trivial autocorrelation lags.
        """
        # Compute autocorrelation using FFT for efficiency
        # R(k) = sum_{i=0}^{N-1-k} x[i] * x[i+k]
        # This is the linear (not circular) autocorrelation
        
        # Use FFT-based linear autocorrelation
        n_fft = 1 << (2 * N - 1).bit_length()
        X = np.fft.rfft(x, n=n_fft)
        # Linear autocorrelation: ifft(fft(x) * conj(fft(x))) gives circular autocorrelation
        # For linear autocorrelation, we need to be more careful
        
        # Actually, let's compute it directly for clarity, but use vectorized numpy
        # R(k) for k=0,...,N-1
        # R(k) = sum_{i=0}^{N-1-k} x[i] * x[i+k]
        
        # Efficient computation using correlation
        # np.correlate(x, x, mode='full') gives full correlation
        # R(k) for k=0,...,N-1 is np.correlate(x, x, mode='full')[N-1:]
        
        corr = np.correlate(x, x, mode='full')
        # corr has length 2*N-1
        # corr[N-1] = R(0) = N
        # corr[N-1+k] = R(k) for k=0,...,N-1
        # corr[N-1-k] = R(-k) = R(k) for real sequences (autocorrelation is symmetric)
        
        sidelobes = corr[N:]  # R(1), R(2), ..., R(N-1)
        return np.max(np.abs(sidelobes))

    def compute_c3_bound_fast(x):
        """Faster version using FFT."""
        n_fft = 1 << (2 * N - 1).bit_length()
        X = np.fft.rfft(x, n=n_fft)
        # Linear autocorrelation via FFT
        # The linear autocorrelation can be computed as:
        # ifft(fft(x, n_fft) * conj(fft(x, n_fft)))[:N]
        # But this gives circular autocorrelation padded to n_fft
        # For linear autocorrelation, we need:
        # r = ifft(X * conj(X))[:N] gives R(0), R(-1), ..., R(-(N-1))
        # Due to symmetry, R(k) = R(-k), so we can get R(1),...,R(N-1) from the first half
        
        power = X * np.conj(X)
        r = np.fft.irfft(power, n=n_fft)[:N]
        # r[0] = R(0), r[1] = R(-1) = R(1), ..., r[N-1] = R(-(N-1)) = R(N-1)
        # So sidelobes are r[1:]
        return np.max(np.abs(r[1:]))

    # --- Hill climbing ---
    def hill_climb(x, max_time=None):
        """
        Local search: try flipping individual bits and accept if it improves the metric.
        Use random order to avoid cycling.
        """
        current = x.copy()
        current_metric = compute_c3_bound_fast(current)
        best = current.copy()
        best_metric = current_metric
        
        n_improvements = 0
        no_improve_count = 0
        
        # Precompute indices
        idx = np.arange(N)
        
        while max_time is None or time.time() < max_time:
            # Try random flips
            np.random.shuffle(idx)
            improved = False
            for i in idx:
                if max_time is not None and time.time() >= max_time:
                    break
                # Flip bit i
                old_val = current[i]
                current[i] = -old_val
                new_metric = compute_c3_bound_fast(current)
                if new_metric < current_metric:
                    current_metric = new_metric
                    improved = True
                    no_improve_count = 0
                    n_improvements += 1
                    if current_metric < best_metric:
                        best_metric = current_metric
                        best = current.copy()
                else:
                    current[i] = old_val
                    no_improve_count += 1
            
            if not improved:
                # No improvement in this pass, try a few more passes with random restarts
                # or break
                if no_improve_count > N * 3:
                    break
            else:
                no_improve_count = 0
        
        return best, best_metric

    # --- Multi-start local search ---
    # Generate diverse initializations
    initializations = []
    
    # 1. Thue-Morse
    initializations.append(thue_morse(N))
    
    # 2. M-sequence
    initializations.append(m_sequence(N))
    
    # 3. Random (multiple)
    for _ in range(5):
        initializations.append(random_init(N))
    
    # 4. Alternating
    initializations.append(np.tile(np.array([1, -1], dtype=np.int8), N // 2))
    
    # 5. Block pattern
    block = np.array([1, 1, -1, 1, -1, -1, -1, 1, -1, -1, 1, -1, 1, 1, -1], dtype=np.int8)
    initializations.append(np.tile(block, N // len(block) + 1)[:N])
    
    best_overall = None
    best_metric_overall = float('inf')
    
    for init in initializations:
        if time.time() > deadline - 30:  # Leave time for at least one more attempt
            break
        
        # Run hill climbing with time limit
        remaining = deadline - time.time()
        time_limit = min(remaining * 0.3, 20.0)  # Limit each attempt
        if time_limit < 5:
            break
        
        result, metric = hill_climb(init, max_time=start_time + (time.time() - start_time) + time_limit)
        
        if metric < best_metric_overall:
            best_metric_overall = metric
            best_overall = result
    
    # If we have time, do more random restarts
    while time.time() < deadline - 20:
        init = random_init(N)
        remaining = deadline - time.time()
        time_limit = min(remaining * 0.2, 15.0)
        if time_limit < 3:
            break
        
        result, metric = hill_climb(init, max_time=start_time + (time.time() - start_time) + time_limit)
        
        if metric < best_metric_overall:
            best_metric_overall = metric
            best_overall = result
    
    if best_overall is None:
        # Fallback: return all ones
        return np.ones(N, dtype=np.int8)
    
    return best_overall
# EVOLVE-BLOCK-END
