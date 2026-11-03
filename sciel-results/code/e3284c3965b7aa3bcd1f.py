"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Construct a signed step function with N equal intervals on [-1/4, 1/4].
    We use the Logistic Map at r=4 to generate sign sequences.
    We employ multiple chaotic initial conditions and a robust local search
    (multiple sweeps) to minimize the C3 bound (max abs triple autocorrelation).
    """
    
    def compute_c3_bound(s):
        """
        Compute the upper bound on C3, defined as the maximum absolute value
        of the third-order autocorrelation coefficients.
        """
        # Pad to power of 2 for FFT efficiency
        # Convolution of 3 sequences of length N has length 3N-2.
        N_pad = 1 << (3 * N - 1).bit_length()
        s_pad = np.zeros(N_pad)
        s_pad[:N] = s
        S = np.fft.fft(s_pad)
        C3 = np.fft.ifft(S ** 3).real
        return np.max(np.abs(C3))

    def local_search(s, max_sweeps=10):
        """
        Perform local search by flipping bits if it reduces the bound.
        Multiple sweeps are performed to escape shallow local minima.
        """
        current_bound = compute_c3_bound(s)
        s = s.copy()
        
        for _ in range(max_sweeps):
            improved = False
            for i in range(N):
                # Flip bit i
                s[i] *= -1.0
                new_bound = compute_c3_bound(s)
                
                if new_bound < current_bound:
                    current_bound = new_bound
                    improved = True
                else:
                    # Revert if no improvement
                    s[i] *= -1.0
            
            if not improved:
                break
                
        return s, current_bound

    # Generate diverse chaotic seeds
    # Use irrational-like numbers to ensure long periods
    seeds = [
        (np.sqrt(2) * 0.1) % 1.0,
        (np.sqrt(3) * 0.2) % 1.0,
        (np.pi * 0.3) % 1.0,
        ((1 + np.sqrt(5)) / 2 * 0.4) % 1.0,
        (np.e * 0.5) % 1.0,
        0.123456789,
        0.987654321,
        0.5 + (1/np.sqrt(2)) % 1.0,
        (np.sqrt(7) * 0.1) % 1.0,
        (np.sqrt(13) * 0.2) % 1.0
    ]
    
    best_signs = None
    best_bound = float('inf')
    
    for seed in seeds:
        x = seed
        signs = np.zeros(N)
        
        # Burn-in period to ensure we are in the chaotic regime
        # and not just the transient phase
        for _ in range(10):
            x = 4.0 * x * (1.0 - x)
            
        for i in range(N):
            x = 4.0 * x * (1.0 - x)
            signs[i] = 1.0 if x > 0.5 else -1.0
            
        # Run local search on this candidate
        searched_signs, bound = local_search(signs, max_sweeps=5)
        
        if bound < best_bound:
            best_bound = bound
            best_signs = searched_signs.copy()
            
    return best_signs
# EVOLVE-BLOCK-END
