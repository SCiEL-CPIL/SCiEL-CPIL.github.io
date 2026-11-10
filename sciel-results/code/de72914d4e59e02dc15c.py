# EVOLVE-BLOCK-START
import numpy as np

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    # Strategy: Use f(k) = alpha * mu(k) for k >= 2, and f(1) determined by normalization.
    # Choose K such that sum_{k=1}^K mu(k)/k < 0.
    # This ensures M(x) < 0 for all x, satisfying M(x) <= 1.
    # Set alpha = 10 to maximize objective.
    
    # Mobius function for k=1..K
    # K=5 gives Sigma_5 = 1 - 1/2 - 1/3 - 1/5 = -0.0333
    # f(1) = -10 * Sigma_5 = 0.333
    # f(2) = -10, f(3) = -10, f(4) = 0, f(5) = -10
    
    K = 5
    domain = np.arange(1, K + 1, dtype=np.int64)
    
    # Calculate mu(k) for k=1..K
    # mu(1)=1, mu(2)=-1, mu(3)=-1, mu(4)=0, mu(5)=-1
    mu = np.array([1, -1, -1, 0, -1], dtype=np.float64)
    
    alpha = 10.0
    
    # Calculate Sigma_K
    Sigma_K = np.sum(mu / np.arange(1, K + 1))
    
    # f(1) = -alpha * Sigma_K
    # f(k) = alpha * mu(k) for k > 1
    
    values = np.empty(K, dtype=np.float64)
    values[0] = -alpha * Sigma_K
    values[1:] = alpha * mu[1:]
    
    # Clip to be safe (though should be within bounds)
    values = np.clip(values, -10, 10)
    
    return domain, values

# EVOLVE-BLOCK-END
