"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    rng = np.random.RandomState(42)
    
    h = rng.choice([1, -1], N).astype(np.float64)
    
    def compute_score(seq):
        """Compute the score: maximum absolute autocorrelation at non-zero lags, normalized."""
        n_fft = 1 << (2 * N - 1).bit_length()
        F = np.fft.fft(seq, n=n_fft)
        power = np.abs(F) ** 2
        acf = np.fft.ifft(power).real
        # For real signals, acf is symmetric. Lags 1 to N-1 cover all non-zero lags.
        max_val = np.max(np.abs(acf[1:N]))
        return max_val / N  # Normalize by N so all-ones gives ~0.9975, close to 1.0
    
    best_score = compute_score(h)
    best_h = h.copy()
    
    # Local search
    for iteration in range(500):  # Limit iterations to stay within time
        if time.time() > deadline - 10.0:
            break
        improved = False
        for i in range(N):
            h[i] *= -1
            score = compute_score(h)
            if score < best_score:
                best_score = score
                best_h = h.copy()
                improved = True
            else:
                h[i] *= -1
        if not improved:
            # Random restart
            h = rng.choice([1, -1], N).astype(np.float64)
            score = compute_score(h)
            if score < best_score:
                best_score = score
                best_h = h.copy()
            else:
                h = best_h.copy()
    
    return best_h
# EVOLVE-BLOCK-END
