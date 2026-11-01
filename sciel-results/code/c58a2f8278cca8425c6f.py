import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s safety margin for 150s limit
    
    # 1. Generate initial sequence using a deterministic spectral method
    # Zadoff-Chu sequence is a good starting point for low autocorrelation sidelobes
    # Length must be prime or of form 4k+1, 4k+3. 400 is not, so we use a subset or modify.
    # Actually, let's use a simpler approach: random initialization with a fixed seed for reproducibility,
    # but structured to avoid all-ones.
    # Better: Use a Barker sequence if available, or a pseudo-random sign sequence.
    # Let's start with a random sign sequence with a fixed seed.
    rng = np.random.default_rng(42)
    x = rng.choice([-1, 1], size=N)
    
    # 2. Define objective function: max absolute value of autocorrelation sidelobes
    def max_sidelobe(x):
        # Compute autocorrelation via FFT
        n = len(x)
        n_fft = 1 << (2 * n - 1).bit_length()
        X = np.fft.rfft(x, n=n_fft)
        acf = np.fft.irfft(X * np.conj(X), n=n_fft)[:n]
        # Sidelobes are indices 1..n-1 (and -1..-n+1, but irfft gives 0..n-1)
        # The main lobe is at index 0
        return np.max(np.abs(acf[1:]))
    
    # 3. Local search: flip signs to reduce max sidelobe
    best_x = x.copy()
    best_val = max_sidelobe(best_x)
    
    # Precompute FFT of x for faster updates? 
    # Actually, flipping one sign changes the entire autocorrelation.
    # We can use the property that autocorrelation is convolution.
    # But for simplicity and speed, we'll just recompute if the search is fast enough.
    # With N=400, FFT is very fast.
    
    # To speed up, we can try flipping each index and see if it improves.
    # This is O(N^2 log N) in worst case if we do it naively, but N=400 is small.
    # Let's do a simple greedy local search: iterate through indices, flip if it improves.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            old_val = best_x[i]
            best_x[i] = -old_val
            new_val = max_sidelobe(best_x)
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                best_x[i] = old_val
    
    # 4. If time remains, try a few random restarts or perturbations
    if time.time() < deadline:
        for _ in range(10):
            if time.time() > deadline:
                break
            # Perturb the current best
            x_perturbed = best_x.copy()
            # Flip a few random signs
            flip_indices = rng.choice(N, size=10, replace=False)
            x_perturbed[flip_indices] *= -1
            # Do a few iterations of local search on the perturbed version
            for _ in range(5):
                if time.time() > deadline:
                    break
                for i in range(N):
                    if time.time() > deadline:
                        break
                    old_val = x_perturbed[i]
                    x_perturbed[i] = -old_val
                    new_val = max_sidelobe(x_perturbed)
                    if new_val < best_val:
                        best_val = new_val
                        best_x = x_perturbed.copy()
                    else:
                        x_perturbed[i] = old_val
    
    return best_x
