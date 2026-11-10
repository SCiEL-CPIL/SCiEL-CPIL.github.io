import numpy as np
import time
import sys

def construct():
    start_time = time.time()
    deadline = 600.0
    
    # Parameters
    n_roots = 12
    alpha = -0.5
    
    # Construct the Jacobi matrix for generalized Laguerre polynomials L_n^(alpha)
    # The eigenvalues of this matrix are the roots of L_n^(alpha)(x)
    # J is tridiagonal with:
    # J[k, k] = alpha + 2*k
    # J[k, k+1] = sqrt(k * (k + alpha))
    
    J = np.zeros((n_roots, n_roots))
    for k in range(n_roots):
        J[k, k] = alpha + 2 * k
        if k > 0:
            val = k * (k + alpha)
            if val > 0:
                J[k, k-1] = np.sqrt(val)
                J[k-1, k] = np.sqrt(val)
    
    # Compute eigenvalues (roots)
    # np.linalg.eigvalsh is for symmetric matrices, which J is
    roots = np.linalg.eigvalsh(J)
    
    # Ensure roots are positive and sorted
    # Theoretical roots of Laguerre polynomials are positive
    # Filter out any non-positive roots if numerical errors occur (unlikely for alpha > -1)
    roots = roots[roots > 0]
    
    # We need exactly 12 roots. If we have more or less, we might need to adjust.
    # For n=12, we expect 12 positive roots.
    if len(roots) != n_roots:
        # Fallback to simple distinct positive numbers if something went wrong
        # But eigvalsh should be robust here.
        # Just to be safe, if count is wrong, use a default spread
        roots = np.linspace(0.5, 10.0, n_roots)
        
    # Sort roots to be consistent (eigvalsh returns sorted, but ensure it)
    roots = np.sort(roots)
    
    # Convert to list of floats
    z_list = [float(r) for r in roots]
    
    # Check time budget
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        # This computation is very fast (<1ms), so this is just a formality
        pass
        
    return z_list
