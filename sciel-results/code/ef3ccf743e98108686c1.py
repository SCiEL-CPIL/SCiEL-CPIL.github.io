import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_school = n * m * p
    
    def _verify(U, V, W):
        """Verify if U, V, W decompose the tensor."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # The tensor is T[i, j, k] = 1 for all i, j, k? No.
        # The multiplication is C[i, k] = sum_j A[i, j] B[j, k].
        # The tensor is sum_{i, j, k} e_{i, j} \otimes e_{j, k} \otimes e_{i, k}.
        # So T[i, j, k] = 1 for all i, j, k.
        # We need sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 for all i, j, k.
        
        # Efficient check:
        # C_decomp = A @ B
        # C_decomp[i, k] = sum_j A[i, j] B[j, k]
        # Using decomposition:
        # C_decomp[i, k] = sum_c (sum_j U[i*m+j, c] A[i, j]) * (sum_j V[j*p+k, c] B[j, k]) ? No.
        # The term is U[i*m+j, c] A[i, j] * V[j*p+k, c] B[j, k] * W[k*n+i, c].
        # This is not a simple matrix product.
        
        # Let's use random verification.
        np.random.seed(42)
        A = np.random.randint(-10, 10, (n, m)).astype(float)
        B = np.random.randint(-10, 10, (m, p)).astype(float)
        C = A @ B
        
        C_decomp = np.zeros((n, p))
        for c in range(U.shape[1]):
            u_c = U[:, c].reshape(n, m)
            v_c = V[:, c].reshape(m, p)
            w_c = W[:, c].reshape(p, n)
            
            # The contribution of term c to C[i, k] is:
            # sum_j u_c[i, j] * A[i, j] * v_c[j, k] * B[j, k] * w_c[k, i]
            
            # This can be written as:
            # Let X_c[i, j] = u_c[i, j] * A[i, j]
            # Let Y_c[j, k] = v_c[j, k] * B[j, k]
            # Then term is sum_j X_c[i, j] Y_c[j, k] w_c[k, i]
            
            # This is (X_c @ Y_c) * w_c^T? No.
            # (X_c @ Y_c)[i, k] = sum_j X_c[i, j] Y_c[j, k]
            # Then multiply by w_c[k, i].
            # This is an element-wise multiplication of (X_c @ Y_c) with w_c.T (since w_c[k, i] is w_c.T[i, k]).
            
            XY = u_c * A @ (v_c * B)
            C_decomp += XY * w_c.T
        
        return np.allclose(C, C_decomp, atol=1e-6)
    
    # Construct a rank-72 schoolbook decomposition
    rank = rank_school
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    # Try to find a lower rank decomposition using a simple search
    # We will try to find a decomposition with rank < 72
    # This is a hard problem, so we will use a simple heuristic.
    
    # Heuristic: Try to find a decomposition with rank 71, 70, etc.
    # We will use a random search.
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Try to find a lower rank decomposition
    for target_rank in range(71, 50, -1):
        # Try a few random initializations
        for attempt in range(10):
            U_try = np.zeros((n * m, target_rank))
            V_try = np.zeros((m * p, target_rank))
            W_try = np.zeros((p * n, target_rank))
            
            # Initialize with random values
            np.random.seed(attempt)
            U_try = np.random.randn(n * m, target_rank)
            V_try = np.random.randn(m * p, target_rank)
            W_try = np.random.randn(p * n, target_rank)
            
            # Verify
            if _verify(U_try, V_try, W_try):
                best_rank = target_rank
                best_U, best_V, best_W = U_try, V_try, W_try
                break
        
        if best_rank < rank:
            break
    
    # Convert to lists for the return interface
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
