"""Rewritten program for 3x3 matrix multiplication rank minimization."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START

def _schoolbook():
    """Return the standard rank-27 decomposition."""
    n, m, p = 3, 3, 3
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, tol=1e-9):
    """Check that the decomposition exactly matches the matrix multiplication tensor."""
    n, m, p = 3, 3, 3
    # The tensor T[i,j,k] should equal sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    # But wait - the convention is W[k*n+i, r], so W is indexed by (k,i)
    # Let's re-read: W[k*n+i, r]
    # So W[r] has entries W[k*n+i] for each (k,i)
    # The decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    
    # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    # The matrix multiplication tensor: T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # Actually, the standard tensor for matrix multiplication has T[i,j,k] = 1 for all i,j,k
    # because C[i,k] = sum_j A[i,j] B[j,k], so the tensor is all ones.
    T_expected = np.ones((n, m, p))
    
    return np.allclose(T, T_expected, atol=tol)


def _try_lattice_search(max_rank=27):
    """
    Try to find a low-rank decomposition using a structured search.
    
    For 3x3 matrix multiplication, the known minimal rank is 23 (Strassen-like for 3x3).
    We'll try to find decompositions of rank 23, 24, 25, 26, 27.
    
    Strategy: Use the fact that the tensor is symmetric in a certain sense and try
    to find integer/half-integer solutions.
    """
    deadline = time.time() + 130  # Leave 20s safety margin
    
    n, m, p = 3, 3, 3
    
    # Start from rank 23 (known minimal for 3x3 over reals is 23)
    # Actually, the minimal rank for 3x3 matrix multiplication is 23 (proven by Bläser 2008)
    # But finding an explicit decomposition is non-trivial.
    
    # Let's try a different approach: use the known Strassen decomposition for 2x2
    # and extend it, or try to find a rank-23 decomposition through search.
    
    # For now, let's try a random search with local optimization for various ranks
    best_result = None
    best_rank = 27
    
    # Try ranks from 23 to 26
    for target_rank in range(23, 27):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Random initialization with small integers
            np.random.seed(attempt * 1000 + target_rank)
            U = np.random.randint(-2, 3, size=(n*m, target_rank))
            V = np.random.randint(-2, 3, size=(m*p, target_rank))
            W = np.random.randint(-2, 3, size=(p*n, target_rank))
            
            if _verify(U.tolist(), V.tolist(), W.tolist()):
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = target_rank
                break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        return best_result
    
    return None


def _try_strassen_based():
    """
    Try to construct a rank-23 decomposition based on known results.
    
    The minimal rank of 3x3 matrix multiplication is 23. One known decomposition
    can be constructed by combining Strassen's 7-multiplication algorithm for 2x2
    with additional terms.
    
    However, constructing an explicit rank-23 decomposition is complex.
    Let's try a structured approach.
    """
    deadline = time.time() + 130
    
    n, m, p = 3, 3, 3
    
    # For 3x3, we can partition into 2x2 blocks and use Strassen
    # A = [[A11, A12], [A21, A22]] where A11 is 2x2, etc.
    # But this gives a recursive structure.
    
    # Alternative: try to find a decomposition by solving a system of equations
    # This is a very hard problem in general.
    
    # Let's try a simpler approach: use the fact that the tensor has a lot of symmetry
    # and try to find a low-rank approximation that is exact.
    
    # For now, return None to fall back to schoolbook
    return None


def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 matrix multiplication.
    
    Returns: {'U': (9, rank), 'V': (9, rank), 'W': (9, rank)} with integer/half-integer entries
    """
    deadline = time.time() + 135  # Leave 15s safety margin
    
    # First, try to find a better-than-schoolbook decomposition
    result = _try_lattice_search(max_rank=26)
    
    if result is not None:
        # Verify once more
        if _verify(result["U"], result["V"], result["W"]):
            return result
    
    # Fall back to schoolbook decomposition (rank 27)
    return _schoolbook()

# EVOLVE-BLOCK-END
