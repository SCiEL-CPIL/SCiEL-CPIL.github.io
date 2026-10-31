"""Signed autocorrelation maximum absolute-value variant."""
import numpy as np
import time

N = 400


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    # We need to minimize: max_k |sum_i x_i x_{i+k}| / (sum_i x_i)^2
    # where the sum is over valid indices for the lag k.

    # Strategy: 
    # 1. Start with random sparse signed sequences
    # 2. Use local search (flip signs, zero-out, or introduce non-zeros)
    # 3. Track best by the score metric

    def score(x):
        """Compute max |autocorr| / (sum)^2. Lower is better."""
        S = np.sum(x)
        if abs(S) < 1e-12:
            return float('inf')
        # Compute autocorrelation for all lags
        # autocorr[k] = sum_{i} x[i] * x[i+k] for k=0..N-1
        # Using FFT-based approach for efficiency
        n = len(x)
        # Pad to next power of 2 for FFT
        size = 1
        while size < 2 * n - 1:
            size *= 2
        X = np.fft.rfft(x, size)
        acf = np.fft.irfft(X * np.conj(X), size)[:n]
        # acf[k] for k=0..n-1 corresponds to sum_i x[i]x[i+k]
        # But we need max absolute value
        max_abs = np.max(np.abs(acf))
        return max_abs / (S * S)

    def random_sparse_seq(density, seed=None):
        """Generate a random sparse signed sequence."""
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random
        mask = rng.random(N) < density
        signs = rng.choice([-1.0, 1.0], size=N)
        x = np.zeros(N)
        x[mask] = signs[mask]
        return x

    best_x = np.ones(N)
    best_score = score(best_x)

    # Try various densities and seeds
    densities = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    
    for density in densities:
        if time.time() > deadline:
            break
        for trial in range(50):
            if time.time() > deadline:
                break
            seed = trial * 1000 + int(density * 10000)
            x = random_sparse_seq(density, seed=seed)
            s = score(x)
            if s < best_score:
                best_score = s
                best_x = x.copy()
    
    # Local search: flip signs of individual elements
    if time.time() < deadline - 10:
        # Sort candidates by potential improvement (elements contributing to peak)
        for iteration in range(20000):
            if time.time() > deadline:
                break
            if iteration % 1000 == 0:
                # Occasionally try a new random restart
                if iteration > 0 and iteration % 5000 == 0:
                    density = 0.1 + 0.3 * np.random.random()
                    new_x = random_sparse_seq(density)
                    if score(new_x) < best_score:
                        best_score = score(new_x)
                        best_x = new_x.copy()
                        continue
            
            # Pick a random index to flip
            idx = np.random.randint(N)
            old_val = best_x[idx]
            best_x[idx] = -old_val
            new_score = score(best_x)
            if new_score < best_score:
                best_score = new_score
            else:
                best_x[idx] = old_val
            
            # Also try zeroing out
            if np.random.random() < 0.2:
                old_val = best_x[idx]
                best_x[idx] = 0.0
                new_score = score(best_x)
                if new_score < best_score:
                    best_score = new_score
                else:
                    best_x[idx] = old_val
            
            # Try introducing a non-zero if currently zero
            if best_x[idx] == 0.0 and np.random.random() < 0.1:
                sign = np.random.choice([-1.0, 1.0])
                best_x[idx] = sign
                new_score = score(best_x)
                if new_score < best_score:
                    best_score = new_score
                else:
                    best_x[idx] = 0.0

    return best_x
