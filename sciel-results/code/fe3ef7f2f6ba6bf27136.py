"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Construct a signed step function with N equal intervals on [-1/4, 1/4].
    Uses Logistic Map at r=4 to generate initial candidates.
    Applies Simulated Annealing to minimize the L-infinity norm of the triple autocorrelation.
    """
    
    def compute_c3_bound(s):
        """
        Compute the upper bound on C3, defined as the maximum absolute value
        of the third-order autocorrelation coefficients.
        """
        # Pad to power of 2 for FFT efficiency
        N_pad = 1 << (3 * N - 1).bit_length()
        s_pad = np.zeros(N_pad)
        s_pad[:N] = s
        S = np.fft.fft(s_pad)
        C3 = np.fft.ifft(S ** 3).real
        return np.max(np.abs(C3))

    # Generate candidates using Logistic Map x_{n+1} = 4 x_n (1.0 - x_n)
    # We use several irrational seeds to explore different chaotic trajectories.
    seeds = [
        np.sqrt(2) % 1.0,
        np.sqrt(3) % 1.0,
        np.pi % 1.0,
        (1 + np.sqrt(5)) / 2 % 1.0,
        np.e % 1.0,
        0.123456789,
        0.987654321,
        0.5 + (1/np.sqrt(2)) % 1.0,
        np.log(2) % 1.0,
        np.arctan(1) % 1.0
    ]
    
    best_signs = None
    best_bound = float('inf')
    
    for seed in seeds:
        x = seed
        signs = np.zeros(N)
        for i in range(N):
            x = 4.0 * x * (1.0 - x)
            signs[i] = 1.0 if x > 0.5 else -1.0
            
        bound = compute_c3_bound(signs)
        
        if bound < best_bound:
            best_bound = bound
            best_signs = signs.copy()
            
    # Simulated Annealing to further minimize the bound
    signs = best_signs.copy()
    current_bound = best_bound
    best_signs_sa = signs.copy()
    best_bound_sa = best_bound
    
    # SA Parameters
    T = 1.0  # Initial temperature
    cooling_rate = 0.999
    min_T = 1e-4
    max_iterations = 5000  # Limit to ensure speed
    
    np.random.seed(42)  # For reproducibility in SA moves
    
    for i in range(max_iterations):
        if T < min_T:
            break
            
        # Pick a random index to flip
        idx = np.random.randint(N)
        
        # Flip the sign
        signs[idx] *= -1.0
        
        # Compute new bound
        new_bound = compute_c3_bound(signs)
        
        # Acceptance probability
        delta = new_bound - current_bound
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            # Accept move
            current_bound = new_bound
            if new_bound < best_bound_sa:
                best_bound_sa = new_bound
                best_signs_sa = signs.copy()
        else:
            # Revert move
            signs[idx] *= -1.0
            
        # Cool down
        T *= cooling_rate

    return best_signs_sa
# EVOLVE-BLOCK-END
