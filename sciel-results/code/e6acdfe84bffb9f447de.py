"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    # Objective: minimize R = max_k |A_k| / (sum x_i)^2
    # where A_k is the linear autoconvolution of x.
    # 
    # Key insight: A_0 = sum x_i^2. By Cauchy-Schwarz,
    # (sum x_i)^2 <= N * sum x_i^2, so R >= 1/N.
    # Equality iff all x_i are equal (constant vector).
    #
    # However, we explore whether signed sequences can achieve
    # a lower ratio by reducing the max absolute autoconvolution
    # while maintaining a large sum.
    #
    # For a constant vector x_i = c, sum = N*c, A_k = (N-|k|)*c^2 for |k| < N.
    # max |A_k| = N*c^2 (at k=0), so R = N*c^2 / (N*c)^2 = 1/N.
    #
    # Try: x_i = (-1)^i. Then sum = 0 (for even N), so R is undefined/infinite.
    # Try: x_i = 1 for i < N/2, -1 for i >= N/2.
    #   sum = 0, bad.
    # Try: x_i = 1 for i < N/2, -1 for i >= N/2, plus a small constant offset.
    #
    # Let's try a different approach: minimize the max absolute value of the
    # autoconvolution for a given sum. Use gradient-based optimization with
    # a smooth surrogate for the max absolute value.
    #
    # Due to time constraints, we first check if a simple structured sequence
    # beats the constant vector.
    
    deadline = time.time() + 100  # Safety margin before 120s limit
    best_val = 1.0 / N  # Value for constant vector
    best_x = np.ones(N, dtype=np.float64)
    
    # Try a few structured sequences quickly
    # 1. Alternating signs with a bias
    for bias in [0.1, 0.2, 0.3, 0.5]:
        x = np.ones(N)
        x[::2] = -bias
        s = np.sum(x)
        if abs(s) < 1e-10:
            continue
        A = np.convolve(x, x, mode='full')
        R = np.max(np.abs(A)) / (s ** 2)
        if R < best_val:
            best_val = R
            best_x = x.copy()
    
    # 2. Try a smooth gradient descent with log-sum-exp surrogate
    if time.time() < deadline - 20:
        # Use a smooth approximation: logsumexp(beta * |A_k|) / beta approximates max|A_k|
        def compute_R(x, beta=10.0):
            s = np.sum(x)
            if abs(s) < 1e-10:
                return 1e10
            A = np.convolve(x, x, mode='full')
            # Smooth max of |A_k|
            logsumexp_val = np.log(np.sum(np.exp(beta * np.abs(A)))) / beta
            return logsumexp_val / (s ** 2)
        
        # Start from a random initialization
        rng = np.random.default_rng(42)
        x = rng.standard_normal(N)
        x -= np.mean(x)  # Center to avoid sum issues
        x += 1.0  # Shift to positive
        
        # Simple gradient descent
        lr = 0.01
        for i in range(200):
            if time.time() > deadline - 10:
                break
            R = compute_R(x)
            if R < best_val:
                best_val = R
                best_x = x.copy()
            # Numerical gradient
            grad = np.zeros(N)
            eps = 1e-4
            for j in range(min(50, N)):  # Sample a few dimensions
                x_plus = x.copy()
                x_plus[j] += eps
                x_minus = x.copy()
                x_minus[j] -= eps
                grad[j] = (compute_R(x_plus) - compute_R(x_minus)) / (2 * eps)
            x -= lr * grad
        
        # Final exact evaluation of best candidate
        if time.time() < deadline - 5:
            A = np.convolve(best_x, best_x, mode='full')
            s = np.sum(best_x)
            if abs(s) > 1e-10:
                exact_R = np.max(np.abs(A)) / (s ** 2)
                if exact_R < best_val:
                    best_val = exact_R
    
    return best_x
# EVOLVE-BLOCK-END
