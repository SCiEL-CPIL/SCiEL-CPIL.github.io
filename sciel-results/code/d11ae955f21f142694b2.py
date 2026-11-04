"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
_START_TIME = time.time()
_DEADLINE = 145.0  # Safety margin for 150s limit

def _time_exceeded():
    return (time.time() - _START_TIME) > _DEADLINE

def _verify(U, V, W, n=4, m=5, p=5):
    """
    Verify that U, V, W correctly decompose the matrix multiplication tensor.
    For all i in [0,n), j in [0,m), k in [0,p):
        sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    """
    if not U or not V or not W:
        return False
    
    rank = len(U[0]) if U else 0
    if len(V[0]) != rank or len(W[0]) != rank:
        return False
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_arr[i * m + j]
                v_val = V_arr[j * p + k]
                w_val = W_arr[k * n + i]
                total = np.dot(u_val, v_val * w_val)
                if abs(total - 1.0) > 1e-9:
                    return False
    
    return True

def _verify_full(U, V, W, n=4, m=5, p=5):
    """
    Full verification: check that the decomposition correctly computes matrix multiplication
    for random test matrices.
    """
    if not U or not V or not W:
        return False
    
    rank = len(U[0]) if U else 0
    if len(V[0]) != rank or len(W[0]) != rank:
        return False
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for trial in range(10):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for c in range(rank):
            u_flat = U_arr[:, c]
            v_flat = V_arr[:, c]
            
            u_scalar = np.dot(u_flat, A.flatten())
            v_scalar = np.dot(v_flat, B.flatten())
            
            w_mat = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    w_mat[i, k] = W_arr[k * n + i, c]
            
            C_computed += w_mat * u_scalar * v_scalar
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    
    return True

def _try_rank_reduction(U, V, W, n=4, m=5, p=5, target_rank=99):
    """
    Try to reduce the rank by finding linear dependencies.
    Start from a rank-r decomposition and try to eliminate one term.
    """
    if _time_exceeded():
        return None
    
    r = len(U[0])
    if r <= target_rank:
        return {"U": U, "V": V, "W": W}
    
    # Try to find a linear combination that allows removing one term
    # Strategy: For each pair of terms, try to see if their contributions can be merged
    # or if one can be expressed in terms of others
    
    # A simpler approach: use the fact that the system is underdetermined for r > 100/65
    # and try to find a solution with rank = target_rank using optimization
    
    # For now, try a simple perturbation approach
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Try to merge the last two terms
    # We want to find alpha, beta such that we can replace terms r-2 and r-1 with a single term
    # This is complex, so let's try a different approach
    
    # Approach: Use the fact that we can add a small perturbation and then round
    # to find an exact solution with lower rank
    
    # Actually, let's try a more systematic approach:
    # The tensor T[i,j,k] = 1 can be decomposed as:
    # T[i,j,k] = sum_c U_c[i,j] * V_c[j,k] * W_c[i,k]
    
    # For the schoolbook, U_c[i,j] = 1 if c corresponds to (i,j,k), else 0
    # Similarly for V and W.
    
    # Can we find a decomposition with fewer terms?
    # One idea: use the fact that for fixed i and k, sum_j A[i,j]B[j,k] = C[i,k]
    # This suggests that we might be able to combine terms along the j-dimension.
    
    # Let's try: for each (i,k), we need sum_j U[i,j,c] * V[j,k,c] * W[i,k,c] = 1
    # If we set W[i,k,c] = 1 for all c, then we need sum_j U[i,j,c] * V[j,k,c] = 1
    # This is a constraint on U and V.
    
    # Let's try to construct a decomposition where W is simple.
    # Set W[k*n+i, c] = delta_{c, f(i,k)} for some function f
    # Then we need sum_j U[i*m+j, f(i,k)] * V[j*p+k, f(i,k)] = 1
    # This means for each (i,k), there's a specific term c = f(i,k) that handles it.
    # But this is just the schoolbook decomposition.
    
    # Alternative: allow W to have multiple nonzeros.
    # Let's try to use a block structure.
    
    # For now, return None to indicate no improvement found.
    return None

def construct():
    """
    Constructs a tensor decomposition for multiplying a 4x5 matrix by a 5x5 matrix.
    """
    if _time_exceeded():
        return {"U": [], "V": [], "W": []}
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook decomposition
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank_schoolbook
    
    # Try to find lower-rank decompositions
    # Strategy: Use structured ansatz to find rank < 100
    
    # Approach 1: Use the fact that m=5 and try to exploit the middle dimension
    # We can write: C[i,k] = sum_j A[i,j] B[j,k]
    # = sum_j (A[i,j] - A[i,0]) B[j,k] + A[i,0] sum_j B[j,k]
    # This doesn't directly help, but suggests we might be able to reduce terms.
    
    # Approach 2: Use a block decomposition
    # Split the 5x5 matrix into blocks and use known low-rank decompositions for smaller blocks.
    
    # For now, try a numerical search for rank 99
    # This is computationally expensive, so we'll only try if we have time
    
    if not _time_exceeded():
        # Try to find a rank-99 decomposition
        # Method: Start with schoolbook and try to eliminate one term
        # by adjusting the others.
        
        # The condition is: for all (i,j,k), sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1
        # If we remove term r-1, we need to adjust the remaining terms to compensate.
        
        # This is a nonlinear system, so we'll use a simple iterative approach.
        
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Try to merge the last two terms (r-2 and r-1)
        # Let's see if their contributions can be combined.
        # Term r-2: U[:,r-2], V[:,r-2], W[:,r-2]
        # Term r-1: U[:,r-1], V[:,r-1], W[:,r-1]
        
        # In the schoolbook, each term is nonzero at exactly one position in each matrix.
        # Term r-2 corresponds to some (i2,j2,k2) and term r-1 to (i1,j1,k1).
        # If (i2,j2,k2) != (i1,j1,k1), then they affect different tensor entries,
        # so we can't simply merge them.
        
        # However, if we allow the terms to have more structure, we might be able to
        # create a decomposition with fewer terms.
        
        # Let's try a different approach: use the fact that the tensor is symmetric
        # in some sense and try to find a more efficient decomposition.
        
        # For now, return the schoolbook decomposition as it's the only one we can
        # guarantee is correct.
        pass
    
    return best_result
# EVOLVE-BLOCK-END
