"""
Refined Cohn-Goncalves Laguerre construction using exact zeros of generalized Laguerre polynomials.

Strategy:
Instead of using a heuristic surrogate objective and stochastic optimization (like Differential Evolution),
we leverage the known orthogonality and stability properties of classical orthogonal polynomials.
Specifically, we use the zeros of the generalized Laguerre polynomial L_{12}^{(1)}(x).
These zeros are known to provide optimal or near-optimal configurations for such uncertainty bounds
due to the minimization properties of orthogonal polynomials in weighted L2 spaces.

We compute these zeros exactly using the eigenvalues of the symmetric tridiagonal Jacobi matrix
associated with the generalized Laguerre polynomials. This is deterministic, fast, and numerically stable.
"""

import numpy as np
import time

# EVOLVE-BLOCK-START

def _get_laguerre_zeros(n, alpha):
    """
    Compute the zeros of the generalized Laguerre polynomial L_n^{(alpha)}(x)
    using the eigenvalues of the associated Jacobi matrix.
    
    The recurrence relation for generalized Laguerre polynomials is:
    x * L_k^{(alpha)}(x) = (2k + alpha + 1) L_{k+1}^{(alpha)}(x) - (k + alpha) L_k^{(alpha)}(x) - k L_{k-1}^{(alpha)}(x)
    
    This corresponds to a symmetric tridiagonal matrix J with:
    J[k, k] = 2k + alpha + 1  (for k=0..n-1)
    J[k, k+1] = J[k+1, k] = sqrt((k+1)(k+1+alpha))  (for k=0..n-2)
    
    The eigenvalues of J are the zeros of L_n^{(alpha)}(x).
    """
    # Construct the Jacobi matrix
    n_int = int(n)
    alpha = float(alpha)
    
    # Diagonal elements
    diag = np.array([2*k + alpha + 1 for k in range(n_int)], dtype=float)
    
    # Off-diagonal elements
    off_diag = np.array([np.sqrt((k+1)*(k+1+alpha)) for k in range(n_int-1)], dtype=float)
    
    # Build the full matrix (or use eigh_tridiagonal if available, but standard eigh is fine for n=12)
    J = np.diag(diag)
    for i in range(n_int - 1):
        J[i, i+1] = off_diag[i]
        J[i+1, i] = off_diag[i]
        
    # Compute eigenvalues
    # Since J is symmetric, eigenvalues are real
    eigenvalues = np.linalg.eigvalsh(J)
    
    # Sort eigenvalues (they should already be sorted, but just in case)
    eigenvalues = np.sort(eigenvalues)
    
    return eigenvalues

def construct():
    """
    Returns a list of 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    
    The task specifies m=12 and n=1. In the context of Cohn-Goncalves constructions for Fourier uncertainty,
    this often relates to the Laguerre polynomial of degree m with a specific parameter.
    The inspiration suggests using L_{12}^{(1)}(x).
    
    The zeros of L_{12}^{(1)}(x) are 12 distinct positive numbers.
    We return these zeros as the root locations.
    """
    
    # Deadline guard (though this computation is millisecond-fast)
    start_time = time.time()
    deadline = start_time + 570.0  # Safety margin
    
    n_roots = 12
    alpha = 1.0  # Based on the strategy hint: L_{12}^{(1)}
    
    # Compute the zeros
    roots = _get_laguerre_zeros(n_roots, alpha)
    
    # Ensure roots are positive (they should be for Laguerre polynomials with alpha > -1)
    if np.any(roots <= 0):
        # Fallback to standard Laguerre L_12^{(0)} if something goes wrong
        roots = _get_laguerre_zeros(n_roots, 0.0)
        if np.any(roots <= 0):
            # Final fallback: simple arithmetic progression
            roots = np.linspace(1.0, 12.0, n_roots)
            
    # Sort roots to be consistent
    roots = np.sort(roots)
    
    # Check time (just in case, though unnecessary for this fast op)
    if time.time() > deadline:
        # This should never happen, but return best effort
        pass
        
    return [float(r) for r in roots]

# EVOLVE-BLOCK-END
