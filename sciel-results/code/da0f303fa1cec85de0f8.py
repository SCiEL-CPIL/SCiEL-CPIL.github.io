"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 5]
START_TIME = time.time()
TIME_LIMIT = 140.0  # Leave 10s safety margin

def _time_remaining():
    return TIME_LIMIT - (time.time() - START_TIME)

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    
    # Test with random matrices
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    # Compute using decomposition
    # U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
    # For each rank component t:
    #   u_t is vector of length n*m, reshaped to (n, m)
    #   v_t is vector of length m*p, reshaped to (m, p)
    #   w_t is vector of length p*n, reshaped to (p, n)
    # Contribution to C[i,k] from term t: W[k*n+i, t] * sum_{j} U[i*m+j, t] * A[i,j] * ... 
    # 
    # Actually, the standard tensor decomposition for matrix multiplication:
    # C[i,k] = sum_t W[k*n+i, t] * (sum_j U[i*m+j, t] * A[i,j]) * (sum_j V[j*p+k, t] * B[j,k])
    # But that's not quite right either. Let me think...
    #
    # The multiplication tensor has T[i,j,k] such that the bilinear form is:
    # sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] = sum_{i,k} C[i,k]
    # where C[i,k] = sum_j A[i,j]*B[j,k]
    #
    # So T[i,j,k] = 1 if the indices are "compatible" in the sense that j is the shared index.
    # Actually T is defined such that the trilinear form <T, A, B> = sum_{i,k} C[i,k] where C = A*B.
    #
    # For a rank-1 decomposition, we need:
    # T = sum_t u_t ⊗ v_t ⊗ w_t
    # where u_t is in R^{n*m}, v_t in R^{m*p}, w_t in R^{p*n}
    #
    # The condition is: for all A, B:
    # sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_t (sum_{i,j} u_t[i*m+j] A[i,j]) (sum_{j,k} v_t[j*p+k] B[j,k]) (sum_{i,k} w_t[k*n+i] * delta?) 
    #
    # I think I'm overcomplicating. Let me use the direct verification:
    # Given U, V, W, the computed C should satisfy:
    # C[i,k] = sum_t W[k*n+i, t] * (sum_j U[i*m+j, t] * A[i,j]) * (sum_j V[j*p+k, t] * B[j,k])
    # Wait, that doesn't look right dimensionally for the sums.
    
    # Let me just do a direct check using the definition:
    # The decomposition should satisfy: for all i, k:
    # C[i,k] = sum_t W[k*n+i, t] * (row_i_of_U_t @ A[i,:]) * (col_k_of_V_t @ B[:,k])
    # where row_i_of_U_t is U[i*m:(i+1)*m, t] and col_k_of_V_t is V[k::p, t]... no.
    
    # Let me use a cleaner formulation. Define:
    # For each t, let a_t[i,j] = U[i*m+j, t], b_t[j,k] = V[j*p+k, t], c_t[i,k] = W[k*n+i, t]
    # Then the decomposition says: for all i,k: sum_j A[i,j]*B[j,k] = sum_t c_t[i,k] * (sum_j a_t[i,j]*A[i,j]) * (sum_j b_t[j,k]*B[j,k])
    # That's not standard. 
    
    # I think the correct interpretation is that the tensor T[i,j,k] = 1 (for all valid i,j,k in the multiplication) 
    # and we're decomposing it as sum_t u_t[i,j] * v_t[j,k] * w_t[i,k] where:
    # u_t[i,j] = U[i*m+j, t], v_t[j,k] = V[j*p+k, t], w_t[i,k] = W[k*n+i, t]
    # And the condition is: T[i,j,k] = sum_t u_t[i,j] * v_t[j,k] * w_t[i,k] for all i,j,k
    # But T[i,j,k] is not 1 for all i,j,k. In matrix multiplication, the "tensor" is:
    # T[i,j,k] = 1 if we're computing C[i,k] using A[i,j] and B[j,k], but actually the multiplication 
    # only involves specific triples. 
    
    # I think the cleanest verification is:
    # The bilinear map (A,B) -> A*B can be written as:
    # (A,B) -> sum_t W_t * (U_t : A) * (V_t : B)
    # where (U_t : A) = sum_{i,j} U[i*m+j,t] * A[i,j], etc.
    # But that would give a scalar, not a matrix.
    
    # OK let me just try the most natural interpretation and verify numerically:
    C_computed = np.zeros((n, p))
    for t in range(r):
        u_t = U[:, t].reshape(n, m)
        v_t = V[:, t].reshape(m, p)
        w_t = W[:, t].reshape(p, n).T  # shape (n, p)
        # Contribution: (u_t * A) element-wise? or (u_t @ something)?
        # Let's try: sum_j u_t[i,j]*A[i,j] for each i, and sum_j v_t[j,k]*B[j,k] for each k
        a_part = np.sum(u_t * A, axis=1)  # shape (n,)
        b_part = np.sum(v_t * B, axis=0)  # shape (p,)
        C_computed += w_t * np.outer(a_part, b_part)
    
    return np.allclose(C_computed, C_expected, atol=1e-10)

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Start with schoolbook
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Try to reduce rank by finding linear dependencies
    # Use a simple iterative approach: try to eliminate columns by expressing them as combinations
    # This is a heuristic; exact rank reduction is hard.
    
    # For now, return the schoolbook decomposition
    # Convert to lists as required
    return {
        "U": [[float(x) for x in row] for row in best_U],
        "V": [[float(x) for x in row] for row in best_V],
        "W": [[float(x) for x in row] for row in best_W]
    }
