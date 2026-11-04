"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 600

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a nonnegative step function to minimize the normalized maximum
    of its autoconvolution.
    
    We interpret the goal as minimizing the Peak Sidelobe Level (PSL) of the 
    autocorrelation, which is a common metric for 'good' autocorrelation 
    properties. The uniform distribution has PSL ~ 1, while random or 
    sparse distributions can have lower PSL.
    
    We represent the step function f as |g|^2 where g is a complex signal.
    We optimize the amplitudes of g (keeping phases random) to minimize the PSL.
    """
    rng = np.random.RandomState(42)
    
    # Initialize amplitudes and phases
    # Start with a random amplitude profile to avoid the uniform local minimum
    amplitudes = rng.exponential(1.0, N)
    phases = rng.uniform(0, 2 * np.pi, N)
    
    # Construct g
    g = amplitudes * np.exp(1j * phases)
    f = np.abs(g)**2
    
    # Normalize f to sum to 1 (standard for such comparisons)
    f = f / np.sum(f)
    
    def calculate_psl(f_vec):
        # Compute autocorrelation
        # Using FFT for efficiency
        n_fft = 1 << (2 * N - 1).bit_length()
        F = np.fft.fft(f_vec, n=n_fft)
        power = np.abs(F)**2
        acf = np.fft.ifft(power).real
        # The autocorrelation is circular, but for N << n_fft, it's linear
        # acf[0] is the peak (lag 0)
        # acf[1:N] and acf[n_fft-N+1:n_fft] are the sidelobes
        peak = acf[0]
        if peak == 0:
            return 0
        # Extract sidelobes
        # The linear autocorrelation of length N is embedded in the circular one
        # Lags 1 to N-1 are at indices 1..N-1 and n_fft-(N-1)..n_fft-1
        side1 = acf[1:N]
        side2 = acf[n_fft-N+1:n_fft]
        max_side = max(np.max(side1), np.max(side2))
        return max_side / peak

    # Simulated Annealing
    current_f = f
    current_psl = calculate_psl(current_f)
    best_f = current_f.copy()
    best_psl = current_psl
    
    T = 1.0
    cooling = 0.999
    steps = 2000 # Keep it fast
    
    for i in range(steps):
        # Perturb amplitudes
        # Pick a random index and change its amplitude
        idx = rng.randint(N)
        old_amp = amplitudes[idx]
        # Multiplicative perturbation
        factor = rng.uniform(0.5, 2.0)
        amplitudes[idx] = old_amp * factor
        
        # Reconstruct f
        g = amplitudes * np.exp(1j * phases)
        new_f = np.abs(g)**2
        new_f = new_f / np.sum(new_f)
        
        new_psl = calculate_psl(new_f)
        
        delta = new_psl - current_psl
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_f = new_f
            current_psl = new_psl
            if new_psl < best_psl:
                best_psl = new_psl
                best_f = new_f.copy()
        else:
            amplitudes[idx] = old_amp
            
        T *= cooling
        
        if T < 1e-6:
            break
            
    return list(best_f)
# EVOLVE-BLOCK-END
