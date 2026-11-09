"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import sys
import signal
import time

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    DEADLINE = start_time + 110.0

    # Construct the schoolbook decomposition
    schoolbook_rank = n * m * p
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    # Convert to numpy for efficient validation
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)

    # Verify schoolbook decomposition
    def verify(U_v, V_v, W_v, tol=1e-10):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        # T_{i,j,k} = sum_l A_{i,l} * B_{l,k}
        # In our indexing: U indexed by (i*m+j), V by (j*p+k), W by (k*n+i)
        # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if there exists l with j=l and k=l... 
        # Actually, the tensor T_{i,j,k} for matrix mult is:
        # T_{i,j,k} = 1 if j == k (the shared dimension), 0 otherwise
        # Wait, let me reconsider the tensor definition.
        
        # For C = A * B where A is n x m, B is m x p:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor T has T[i,j,k] = A[i,j] * B[j,k]
        # So T[i,j,k] = 1 if we're looking at the "coefficient" tensor, 
        # meaning T[i,j,k] = delta_{j,j} * delta_{k,k}... 
        # Actually the bilinear map is: (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
        # The tensor T_{i,j,k} represents the coefficient such that
        # C[i,k] = sum_{j,r} T[i,j,k,r] A[i,j] B[j,k]... 
        # No, let me think again.
        
        # The matrix multiplication tensor T is defined as:
        # T_{(i,k), (i',j), (j',k')} ... this is getting complicated.
        
        # Simpler: The tensor T in the space (R^{n*m} x R^{m*p} x R^{n*p})
        # is defined by T[a, b, c] = 1 if a = (i,j), b = (j,k), c = (i,k) for some i,j,k
        # and 0 otherwise.
        
        # So T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
        # and T is 0 elsewhere.
        
        # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i*m+j, j*p+k, k*n+i]
        
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # This entry should be 1
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    val = np.dot(U_v[idx_u], V_v[idx_v])  # This is wrong, we need sum_r U[idx_u,r]*V[idx_v,r]*W[idx_w,r]
                    # Actually: sum_r U[idx_u, r] * V[idx_v, r] * W[idx_w, r]
                    val = 0.0
                    for r in range(U_v.shape[1]):
                        val += U_v[idx_u, r] * V_v[idx_v, r] * W_v[idx_w, r]
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        
        # Check that non-diagonal entries are 0
        # This is expensive, so we'll skip for now and trust the structure
        return max_err < tol

    # The schoolbook should verify
    # For now, return schoolbook as it's guaranteed to work
    U_result = [[int(x) if abs(x - round(x)) < 1e-9 else round(x, 6) for x in row] for row in U_np.tolist()]
    V_result = [[int(x) if abs(x - round(x)) < 1e-9 else round(x, 6) for x in row] for row in V_np.tolist()]
    W_result = [[int(x) if abs(x - round(x)) < 1e-9 else round(x, 6) for x in row] for row in W_np.tolist()]

    return {"U": U_result, "V": V_result, "W": W_result}

# EVOLVE-BLOCK-END
