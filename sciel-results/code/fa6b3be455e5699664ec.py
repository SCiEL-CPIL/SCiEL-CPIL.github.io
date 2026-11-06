"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 60.0  # 60 seconds limit
    
    N = 8
    
    # 1. Generate Chebyshev nodes on [0, 1]
    # Roots of T_N(x) on [-1, 1] are cos((2k-1)pi / (2N)) for k=1..N
    k = np.arange(1, N + 1)
    x_cheb = np.cos((2 * k - 1) * np.pi / (2 * N))
    
    # Map to [0, 1]
    # z = (x + 1) / 2
    z_nodes = (x_cheb + 1.0) / 2.0
    
    # Sort nodes in increasing order
    # cos is decreasing from 0 to pi, so x_cheb is decreasing.
    # z_nodes will be decreasing. We want increasing.
    z_nodes = z_nodes[::-1]
    
    # 2. Calculate weights
    # We want weights w_i such that sum(w_i * z_i^p) = 1/(p+1) for p=0..N-1
    # This makes the rule exact for polynomials of degree up to N-1.
    
    # Construct Vandermonde matrix V where V[i, j] = z_nodes[i]^j
    # We solve V^T w = b, where b[j] = 1/(j+1)
    # Note: V is N x N. V[i, j] = z_i^j.
    # System: sum_i w_i z_i^j = b_j for j=0..N-1.
    # This is (V^T w)_j = b_j.
    
    V = np.vander(z_nodes, N, increasing=True)
    b = 1.0 / (np.arange(N) + 1.0)
    
    # Solve for weights
    try:
        w = np.linalg.solve(V.T, b)
    except np.linalg.LinAlgError:
        # Fallback to uniform weights if singular (unlikely)
        w = np.ones(N) / N
        z_nodes = (np.arange(N) + 0.5) / N
        
    # Ensure weights are positive
    # If any weight is negative due to numerical issues, we might need to adjust.
    # However, for Chebyshev nodes, weights should be positive.
    if np.any(w < 0):
        # If negative weights appear, fall back to Gauss-Legendre which guarantees positive weights
        # Or just clip and renormalize, but that breaks exactness.
        # Let's try Gauss-Legendre as a robust fallback if Chebyshev weights fail positivity.
        try:
            # Gauss-Legendre nodes and weights on [-1, 1]
            x_leg, w_leg = np.polynomial.legendre.leggauss(N)
            z_nodes = (x_leg + 1.0) / 2.0
            w = w_leg / 2.0 # Jacobian of mapping is 1/2
            z_nodes = np.sort(z_nodes)
            w = w[np.argsort(z_nodes)] # Ensure alignment
        except Exception:
            # Final fallback: Midpoint
            z_nodes = (np.arange(N) + 0.5) / N
            w = np.ones(N) / N

    # Normalize weights to sum to 1
    w_sum = np.sum(w)
    if w_sum > 0:
        w = w / w_sum
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 5.0:
        # Return best found (current) immediately
        pass
        
    return {"nodes": z_nodes, "weights": w}
# EVOLVE-BLOCK-END
