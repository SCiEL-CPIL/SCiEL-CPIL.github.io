"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_school = n * m * p  # 24
    
    # Try to find a better decomposition
    # The tensor is T[i,j,k] for i in [0,n), j in [0,m), k in [0,p)
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The tensor entry T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Let's try to find a rank-1 decomposition using random search
    # But first, let's try a structured approach
    
    # For matrix multiplication (2,2,6), we can try various rank values
    # The theoretical lower bound is at least max(n*m, m*p, p*n) = max(4, 12, 12) = 12
    
    best_decomp = None
    best_rank = rank_school
    
    # Schoolbook as fallback
    U_sb = np.zeros((n*m, rank_school))
    V_sb = np.zeros((m*p, rank_school))
    W_sb = np.zeros((p*n, rank_school))
    for c in range(rank_school):
        i = c // (m*p)
        j = (c // p) % m
        k = c % p
        U_sb[i*m+j, c] = 1
        V_sb[j*p+k, c] = 1
        W_sb[k*n+i, c] = 1
    best_decomp = (U_sb, V_sb, W_sb)
    
    start_time = time.time()
    timeout = 100  # Keep safety margin
    
    # Try structured decompositions for various ranks
    # We can try to exploit the structure of the problem
    
    # Let's try rank values from the theoretical lower bound up
    for rank in range(12, rank_school):
        if time.time() - start_time > timeout:
            break
        decomp = _try_rank(n, m, p, rank)
        if decomp is not None:
            best_decomp = decomp
            best_rank = rank
            break
    
    U, V, W = best_decomp
    
    # Convert to lists for the return interface
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

def _try_rank(n, m, p, rank):
    """Try to find a rank decomposition using structured patterns."""
    
    # First, try a simple pattern-based approach
    # For small dimensions, we can try to find exact integer/half-integer solutions
    
    # Let's try random search with numpy
    np.random.seed(42)
    
    for attempt in range(1000):
        if time.time() - 0 > 100:  # Placeholder for actual timing
            break
        
        # Initialize with random small values
        U = np.random.randint(0, 2, size=(n*m, rank)).astype(float)
        V = np.random.randint(0, 2, size=(m*p, rank)).astype(float)
        W = np.random.randint(0, 2, size=(p*n, rank)).astype(float)
        
        # Add some structure: try to make the decomposition work
        # This is a very basic approach - in practice, we'd need something smarter
        
        # Check if this decomposes the tensor
        if _check_decomposition(U, V, W, n, m, p):
            return (U, V, W)
    
    return None

def _check_decomposition(U, V, W, n, m, p):
    """Check if U, V, W decompose the matrix multiplication tensor."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # T[i,j,k] = 1 if this corresponds to a matrix multiplication term
                # For standard matrix multiplication, T[i,j,k] = delta_{j,k} (the j-th row of A times k-th column of B)
                # Actually, for (n x m) * (m x p), the tensor is T[i,j,k] where j is the summation index
                # T[i,j,k] = delta_{j,k} is not right...
                # Let me reconsider: for C = A * B where A is n x m and B is m x p
                # C[i,k] = sum_j A[i,j] * B[j,k]
                # So the tensor T[i,j,k] = 1 (it's the standard multiplication tensor)
                target = 1.0
                computed = np.dot(U[i*m+j, :], V[j*p+k, :])
                computed = np.dot(computed, W[k*n+i, :])
                # Wait, that's not right either. Let me re-read the spec.
                # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] summed over c
                computed = 0.0
                for c in range(U.shape[1]):
                    computed += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(computed - target) > 1e-9:
                    return False
    return True

# EVOLVE-BLOCK-END
