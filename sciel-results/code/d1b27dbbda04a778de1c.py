import numpy as np

N = 400

def construct():
    """
    Simulated Annealing to minimize the sum of squares of non-zero autocorrelations.
    """
    
    def compute_autocorr_sum_sq(x):
        """
        Compute sum of squares of non-zero lags of linear autocorrelation using FFT.
        x is a vector of length N with values in {-1, 1}.
        """
        # Zero pad to 2*N to compute linear autocorrelation via circular autocorrelation
        x_pad = np.zeros(2 * N)
        x_pad[:N] = x
        X = np.fft.fft(x_pad)
        # Power spectrum
        power = X * np.conj(X)
        # Inverse FFT to get circular autocorrelation
        R_circ = np.fft.ifft(power).real
        # Linear autocorrelation is the first N elements of R_circ (lag 0 to N-1)
        # R_circ[0] is lag 0 (energy), R_circ[1] is lag 1, ..., R_circ[N-1] is lag N-1
        # We want to minimize sum of squares of lags 1 to N-1.
        # Note: The linear autocorrelation for lags 1..N-1 corresponds to R_circ[1:N].
        # The lags -1..-(N-1) are symmetric, so their squares are the same.
        # The task says "sum of squares of non-zero autocorrelations".
        # Usually, this implies the full autocorrelation sequence excluding lag 0.
        # R[k] for k=1..N-1 and R[k] for k=1-(N-1).
        # Since R is even, sum_{k=1}^{N-1} R[k]^2 + sum_{k=-(N-1)}^{-1} R[k]^2 = 2 * sum_{k=1}^{N-1} R[k]^2.
        # Minimizing one is equivalent to minimizing the other.
        non_central = R_circ[1:N]
        return np.sum(non_central**2)

    np.random.seed(42)
    
    # Initialize with random sequence
    x = np.random.choice([-1, 1], size=N)
    current_energy = compute_autocorr_sum_sq(x)
    
    best_x = x.copy()
    best_energy = current_energy
    
    # Simulated Annealing Parameters
    # We have ~120s. Let's aim for ~100s of computation.
    # Estimate cost per iteration: FFT of size 800 is fast. 
    # Let's run a large number of iterations.
    # Initial temperature high enough to accept bad moves, cool down to find local minima.
    T_start = 10.0
    T_end = 0.01
    alpha = 0.9999 # Cooling factor
    
    # Estimate number of iterations possible in 100s.
    # A single FFT+IFFT+sum on N=400 (pad 800) is very fast in numpy.
    # Let's try 50,000 iterations first to be safe, or dynamically adjust if we have time.
    # Actually, let's use a fixed number of iterations that is known to be safe.
    # 50,000 iterations should take < 10 seconds in numpy.
    # We can do multiple restarts or a longer single run.
    # Let's do a single long run with many restarts if stuck? 
    # Better: One long SA run.
    
    iterations = 200000 # This should be well within 120s.
    
    for i in range(iterations):
        T = T_start * (T_end / T_start) ** (i / iterations)
        
        # Pick a random index to flip
        idx = np.random.randint(N)
        old_val = x[idx]
        x[idx] *= -1
        
        new_energy = compute_autocorr_sum_sq(x)
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or np.random.random() < np.exp(-delta_e / T):
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_x = x.copy()
        else:
            # Revert
            x[idx] = old_val

    return best_x

# EVOLVE-BLOCK-END
