import numpy as np
import time

N = 400

def construct():
    """
    Returns a sequence of signed step heights (length N) that minimizes
    the upper bound on C3 (interpreted as peak third-order autocorrelation).
    """
    start_time = time.time()
    deadline = start_time + 130  # Safety margin (150s limit)

    def compute_triple_autocorr_peak(h):
        """
        Computes the maximum absolute value of the triple autocorrelation
        R3(t1, t2) = sum_n h[n] h[n+t1] h[n+t2]
        over all non-zero (t1, t2) pairs.
        """
        n = len(h)
        # Use FFT to compute efficiently
        # R3(t1, t2) can be computed via:
        # For each t1, compute A(t1) = sum_n h[n] h[n+t1] (autocorrelation at lag t1)
        # Then R3(t1, t2) = sum_n A(t1)[n] h[n+t2]
        # This is O(n^3) naive, but we can use FFT tricks
        
        # Simpler approach: for small N, use direct computation with vectorization
        # But N=400, so n^3 = 64M operations which is too slow per evaluation
        
        # Use the fact that for binary sequences, we can compute via:
        # R3(t1,t2) = sum_n h[n]h[n+t1]h[n+t2]
        # We can fix t1 and compute for all t2 using convolution
        
        # Better: use the spectral method
        # H(k) = FFT(h)
        # The triple autocorrelation's 2D FFT is related to |H|^2 * H or similar
        
        # Actually, let's use a different approach:
        # R3(t1, t2) = sum_n h[n] h[n+t1] h[n+t2]
        # = sum_n h[n] h[n+t1] h[n+t2]
        # For binary h, this is related to the count of patterns
        
        # For efficiency, let's compute the peak by sampling
        # But we need the true peak...
        
        # Alternative: use the identity that for real sequences:
        # The 3D autocorrelation can be computed as:
        # R3(t1,t2) = IFFT_2D( |FFT(h)|^2 * FFT(h) ) at appropriate indices
        
        # Let me try a direct but vectorized approach for the peak
        # Since we only need the peak, and N=400, let's be smart
        
        # Compute autocorrelation first: R(t) = sum_n h[n] h[n+t]
        nfft = 1 << (2 * n - 1).bit_length()
        H = np.fft.rfft(h, n=nfft)
        R = np.fft.irfft(H * np.conj(H), n=nfft)
        # R[0..n-1] are positive lags of autocorrelation
        
        # Now for each lag t1, we have A_t1[n] = h[n] * h[n+t1]
        # R3(t1, t2) = sum_n A_t1[n] * h[n+t2]
        # This is the cross-correlation of A_t1 and h at lag t2
        
        # To find the peak over all (t1, t2), we can:
        # For each t1, compute the cross-correlation of A_t1 with h, then take max abs
        # But this is O(n^2 * n) = O(n^3) which is 64M for n=400
        
        # Let's use a sampling approach for speed, but that won't give exact peak
        # Instead, let's compute it properly but efficiently
        
        # Actually, for the purpose of this task, let me reconsider what C3 means.
        # Given the task says "autoconvolution upper bound", maybe C3 is:
        # C3 = max_t | (h * h * h)(t) | where * is convolution
        # (h*h*h)(t) = sum_x sum_y h(x) h(y) h(t-x-y)
        
        # This is the triple convolution, and we want its peak absolute value.
        
        # Let me compute the triple convolution peak instead
        nfft3 = 1 << (3 * n - 2).bit_length()
        H = np.fft.rfft(h, n=nfft3)
        triple = np.fft.irfft(H ** 3, n=nfft3)
        # The triple convolution is in triple
        # For linear convolution, we need to extract the valid part
        # The valid part of triple convolution of length-n sequences is at indices [n-1, 3n-3]
        valid = triple[n-1:3*n-2]
        return np.max(np.abs(valid))

    def compute_triple_conv_peak(h):
        """Compute peak of triple convolution (h*h*h)."""
        n = len(h)
        nfft = 1 << (3 * n - 2).bit_length()
        H = np.fft.rfft(h, n=nfft)
        triple = np.fft.irfft(H ** 3, n=nfft)
        # Valid part: indices n-1 to 3n-3 (inclusive), so length 2n-2
        valid = triple[n-1:3*n-2]
        return np.max(np.abs(valid))

    def greedy_local_search(h, max_iterations=2000, deadline_check=True):
        """Greedy local search with single bit flips."""
        current_score = compute_triple_conv_peak(h)
        improved = True
        iterations = 0

        while improved and iterations < max_iterations:
            improved = False
            iterations += 1

            if deadline_check and time.time() > deadline - 5:
                break

            # Try random single bit flips
            for _ in range(20):
                if time.time() > deadline - 5:
                    break
                i = np.random.randint(N)
                h[i] *= -1
                new_score = compute_triple_conv_peak(h)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    h[i] *= -1

        return h, current_score

    def simulated_annealing(h, iterations=1500, T_start=1.0, T_end=0.01):
        """Simulated annealing with single-bit flips."""
        current_score = compute_triple_conv_peak(h)
        T = T_start

        for iteration in range(iterations):
            if time.time() > deadline - 10:
                break

            i = np.random.randint(N)
            h[i] *= -1
            new_score = compute_triple_conv_peak(h)
            delta = new_score - current_score
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                current_score = new_score
            else:
                h[i] *= -1

            T = T_start * (T_end / T_start) ** (iteration / iterations)

        return h, current_score

    best_h = None
    best_score = float('inf')

    # Generate initial sequences
    np.random.seed(42)
    initial_sequences = []
    # Random
    for _ in range(10):
        initial_sequences.append(np.random.choice([-1, 1], N))
    # Thue-Morse
    tm = np.zeros(N, dtype=np.int8)
    for i in range(N):
        x = i
        cnt = 0
        while x:
            cnt += x & 1
            x >>= 1
        tm[i] = 1 if cnt % 2 == 0 else -1
    initial_sequences.append(tm)
    # Alternating
    initial_sequences.append(np.array([1, -1] * (N // 2)))
    # Block patterns
    initial_sequences.append(np.tile([1, 1, -1, -1], N // 4))

    # Main search loop
    for init_idx, init_h in enumerate(initial_sequences):
        if time.time() > deadline - 20:
            break

        h = init_h.copy()

        # Phase 1: Simulated annealing
        h, sa_score = simulated_annealing(h, iterations=1000, T_start=1.0, T_end=0.01)

        # Phase 2: Greedy local search
        h, local_score = greedy_local_search(h, max_iterations=1000)

        # Update best
        final_score = compute_triple_conv_peak(h)
        if final_score < best_score:
            best_score = final_score
            best_h = h.copy()

        # Early termination
        if best_score < 0.5:
            break

    if best_h is None:
        best_h = np.random.choice([-1, 1], N)

    return best_h
