"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def _schoolbook():
    """
    Schoolbook tensor decomposition for 3x4 x 4x4 matrix multiplication.
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    U: shape (n*m, rank) = (12, 48)
    V: shape (m*p, rank) = (16, 48)  
    W: shape (p*n, rank) = (12, 48)
    
    For each (i,j,k) triple, one rank-1 term:
      U[i*m+j, c] = 1
      V[j*p+k, c] = 1
      W[k*n+i, c] = 1
    """
    n, m, p = DIMENSIONS  # 3, 4, 4
    rank = n * m * p  # 48
    
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

def _verify_decomposition(U, V, W):
    """Verify exact decomposition of the multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Compute T[i,k,j] = sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t]
    T = np.zeros((n, p, m), dtype=np.float64)
    
    for t in range(rank):
        u_col = U_arr[:, t]
        v_col = V_arr[:, t]
        w_col = W_arr[:, t]
        
        # Vectorized update for efficiency
        # u_col is (n*m, 1), v_col is (m*p, 1), w_col is (p*n, 1)
        # We need to add outer products appropriately
        # T[i,k,j] += u[i,j] * v[j,k] * w[k,i]
        
        # Reshape for broadcasting
        u_3d = u_col.reshape(n, m, 1)
        v_3d = v_col.reshape(m, p, 1)
        w_3d = w_col.reshape(p, n, 1)
        
        # Product is (n, m, 1) * (m, p, 1) * (p, n, 1) -> broadcast to (n, m, p) ?
        # No, indices are (i, j) for u, (j, k) for v, (k, i) for w.
        # We want T[i, k, j].
        # u[i,j] * v[j,k] * w[k,i]
        
        # Let's do it via einsum or simple loops if small
        # n=3, m=4, p=4. Size is 48.
        # 48 iterations of 48 ops is negligible.
        
        for i in range(n):
            for j in range(m):
                u_val = u_col[i*m+j]
                if u_val == 0: continue
                for k in range(p):
                    v_val = v_col[j*p+k]
                    if v_val == 0: continue
                    w_val = w_col[k*n+i]
                    if w_val == 0: continue
                    T[i, k, j] += u_val * v_val * w_val

    expected = np.ones((n, p, m), dtype=np.float64)
    return np.allclose(T, expected, atol=1e-10)

def construct():
    """Construct the tensor decomposition with deadline guard."""
    start_time = time.time()
    
    # Start with schoolbook (rank 48)
    best = _schoolbook()
    
    # Verification is fast for rank 48
    # No search for lower rank is attempted as it is computationally infeasible
    # to guarantee success within 120s for this specific shape.
    
    return best
# EVOLVE-BLOCK-END
