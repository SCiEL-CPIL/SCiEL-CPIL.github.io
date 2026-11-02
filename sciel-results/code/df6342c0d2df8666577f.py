"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Construct a signed equal-width step function that minimizes the maximum
    absolute autoconvolution divided by the squared integral.
    
    Strategy: Use a random sign pattern with a specific structure to reduce
    autoconvolution peaks. We try several candidate patterns and return the one
    with the best score (lowest max_abs_autoconv / (integral^2)).
    
    The score reported by the evaluator is seed/value where value is the raw metric.
    Lower raw metric is better.
    """
    candidates = []
    
    # Candidate 1: Alternating signs (simplest anti-correlation)
    x1 = np.ones(N)
    x1[1::2] = -1
    candidates.append(x1)
    
    # Candidate 2: Block-based alternating signs
    block_size = 2
    x2 = np.ones(N)
    for i in range(0, N, 2 * block_size):
        x2[i:i+block_size] = 1
        x2[i+block_size:i+2*block_size] = -1
    candidates.append(x2)
    
    # Candidate 3: Random signs
    np.random.seed(42)
    x3 = np.random.choice([-1, 1], size=N)
    candidates.append(x3)
    
    # Candidate 4: Random signs with a different seed
    np.random.seed(123)
    x4 = np.random.choice([-1, 1], size=N)
    candidates.append(x4)
    
    # Candidate 5: Structured pattern based on quadratic residue
    x5 = np.ones(N)
    for i in range(N):
        if (i * i) % 2 == 0:
            x5[i] = 1
        else:
            x5[i] = -1
    candidates.append(x5)
    
    # Candidate 6: Another structured pattern
    x6 = np.ones(N)
    for i in range(N):
        if (i // 2) % 2 == 0:
            x6[i] = 1
        else:
            x6[i] = -1
    candidates.append(x6)
    
    # Evaluate each candidate
    best_x = None
    best_score = float('inf')
    
    for x in candidates:
        # Compute autoconvolution
        # For step functions, autoconvolution at lag k is sum_{i=0}^{N-1-k} x[i]*x[i+k]
        # We need the maximum absolute value of this
        # Use FFT for efficiency
        # Pad to avoid circular convolution
        n_fft = 1 << (2 * N - 1).bit_length()
        X = np.fft.fft(x, n=n_fft)
        autoconv = np.fft.ifft(X * np.conj(X)).real[:N]
        
        # Integral of x is sum(x)
        integral = np.sum(x)
        if abs(integral) < 1e-10:
            # If integral is zero, the denominator is zero, which is problematic.
            # In this case, the ratio is undefined. We should avoid such patterns.
            # But if we must, we can consider it as having a very large score (bad).
            score = float('inf')
        else:
            max_abs_autoconv = np.max(np.abs(autoconv))
            score = max_abs_autoconv / (integral * integral)
        
        if score < best_score:
            best_score = score
            best_x = x
    
    return best_x
# EVOLVE-BLOCK-END
