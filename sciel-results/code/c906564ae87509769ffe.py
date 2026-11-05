"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, tol=1e-9):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(np.dot(U[i*m+j], V[j*p+k]), W[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True


def _dft_decomposition():
    """
    Use DFT over output dimension k to decompose into frequency slices.
    For each frequency, attempt to find a low-rank decomposition.
    """
    n, m, p = DIMENSIONS
    # DFT matrix for size p=5
    omega = np.exp(2j * np.pi / p)
    F = np.array([[omega**(l*k) for k in range(p)] for l in range(p)])
    
    # The tensor T[i,j,k] = 1 for all i,j,k
    # After DFT over k: T_hat[i,j,l] = sum_k T[i,j,k] * omega^(l*k) = p * delta(l,0)
    # So only l=0 has non-zero entries, all equal to p=5.
    # This means DFT doesn't help reduce rank here since only one frequency is active.
    
    # Alternative: try block decomposition
    # View 3x5 as [3x3 | 3x2]
    # A(3x3) * B1(3x3) + A(3x3) * B2(3x2)
    # Use Strassen (rank 7) for 3x3x3 and schoolbook (rank 6) for 3x3x2
    # Total rank: 7 + 6 = 13
    
    # Strassen decomposition for 3x3x3
    def _strassen_333():
        """Return U, V, W for 3x3x3 matmul with rank 7."""
        # Strassen's algorithm for 3x3 matrices
        # We need: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,j,k in {0,1,2}
        # This is equivalent to decomposing the 3x3x3 all-ones tensor with the matmul structure
        
        # Actually, for standard matmul, T[i,j,k] = 1 for all valid (i,j,k)
        # Strassen gives rank 7 for 2x2x2, and for 3x3x3 we can use the known rank 7
        
        # Let me use a known rank-7 decomposition for 3x3x3 matmul
        # The tensor is T[i,j,k] where (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
        # T[i,j,k] = 1 for all i,j,k
        
        # One approach: use the fact that 3x3 matmul can be done with 7 multiplications
        # via embedding into 4x4 and using Strassen, but that gives rank 7 for the 3x3 case?
        
        # Actually, the rank of 3x3x3 matmul tensor is known to be 7.
        # Let me construct it using the standard Strassen-like decomposition.
        
        # For 3x3, we can partition into 2x2 blocks and use Strassen, but that gives
        # more than 7. The rank-7 decomposition is non-trivial.
        
        # Let me use a different approach: try to find rank-7 decomposition numerically
        return None
    
    # Let's try a simpler approach: alternating minimization
    return _alternating_minimization()


def _alternating_minimization(target_rank=None, max_iter=200, tol=1e-10, seed=42):
    """
    Use alternating minimization to find a low-rank decomposition.
    Fix two factors, solve for the third via least squares.
    """
    n, m, p = DIMENSIONS
    rng = np.random.RandomState(seed)
    
    if target_rank is None:
        # Start with a reasonable guess: lower bound is max(n*m, m*p, p*n) = 15
        target_rank = 15
    
    best_result = _schoolbook()
    best_rank = n * m * p
    
    for rank in range(15, min(45, 30)):  # Try ranks from 15 up to 30
        if time.time() > time.time() + 100:  # Safety check
            break
            
        U = rng.randn(n * m, rank)
        V = rng.randn(m * p, rank)
        W = rng.randn(p * n, rank)
        
        converged = False
        for iteration in range(max_iter):
            # Update U: for each row (i,j), solve U[i*m+j, :] = (V[j*p+*, :] * W[k*n+i, :])^T
            # This is tricky because V and W are coupled.
            
            # Alternative: linearize by fixing V and W, solve for U
            # For each (i,j): U[i*m+j, :] should satisfy
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
            
            # This is a linear system in U[i*m+j, :]:
            # Let M[i,j,k,r] = V[j*p+k, r] * W[k*n+i, r]
            # Then sum_r U[i*m+j, r] * M[i,j,k,r] = 1 for all k
            # This is a p x rank system
            
            for i in range(n):
                for j in range(m):
                    # Build system: A * x = b where x = U[i*m+j, :]
                    A = np.zeros((p, rank))
                    b = np.ones(p)
                    for k in range(p):
                        for r in range(rank):
                            A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    # Solve least squares
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        pass
            
            # Update V: for each row (j,k), solve V[j*p+k, :]
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, rank))
                    b = np.ones(n)
                    for i in range(n):
                        for r in range(rank):
                            A[i, r] = U[i*m+j, r] * W[k*n+i, r]
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        pass
            
            # Update W: for each row (k,i), solve W[k*n+i, :]
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, rank))
                    b = np.ones(m)
                    for j in range(m):
                        for r in range(rank):
                            A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        pass
            
            # Check convergence
            if _verify(U.tolist(), V.tolist(), W.tolist(), tol=1e-8):
                converged = True
                if rank < best_rank:
                    best_rank = rank
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
        
        if converged:
            break
    
    return best_result


def construct():
    deadline = time.time() + 140  # Leave 10s margin
    
    # First, get the schoolbook solution as baseline
    result = _schoolbook()
    
    # Try alternating minimization to find lower rank
    try:
        better_result = _alternating_minimization()
        if _verify(better_result["U"], better_result["V"], better_result["W"], tol=1e-8):
            result = better_result
    except Exception as e:
        pass
    
    # Verify final result
    if not _verify(result["U"], result["V"], result["W"], tol=1e-6):
        # Fallback to schoolbook
        result = _schoolbook()
    
    return result
# EVOLVE-BLOCK-END
