"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
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

def construct():
    deadline = time.time() + 130.0
    fallback = _schoolbook()
    n, m, p = DIMENSIONS
    rows_u = n * m
    rows_v = m * p
    rows_w = p * n

    def verify(U, V, W):
        """Check if U @ diag(1) @ V^T matches expected, using W for trace convention."""
        # The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # For matrix multiplication, T[i,j,k] should be 1 if the (i,j) entry of A times (j,k) entry of B
        # contributes to (i,k) of C, which is always the case for standard matmul.
        # Actually, for matmul: C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor is T[a,b,d] where a=i*m+j (row of A, col of A), b=j*p+k (row of B, col of B), d=k*n+i (row of C, col of C)
        # T[a,b,d] = 1 if the multiplication of A[i,j] and B[j,k] contributes to C[i,k], which is always true.
        # So we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all valid (i,j,k)
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Build expected tensor
        T_exp = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_exp[i*m+j, j*p+k, k*n+i] = 1.0
        
        # Compute from decomposition
        # T_comp[a,b,d] = sum_c U[a,c] * V[b,c] * W[d,c]
        # This is not a simple matrix product. Let's compute it efficiently.
        # T_comp = einsum('ac,bc,dc->abd', U, V, W)
        T_comp = np.einsum('ac,bc,dc->abd', U_np, V_np, W_np, optimize=True)
        
        return np.allclose(T_comp, T_exp, atol=1e-10)

    # Start from schoolbook rank and try to reduce
    best = fallback
    best_rank = len(fallback["U"][0])

    # Try greedy reduction: for each rank from 27 up to best_rank-1, try to find a valid decomposition
    # For small matrices, we can try systematic search over small integer/half-integer values
    
    vals = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    
    # Try ranks from 27 to 53
    for rank in range(27, best_rank):
        if time.time() > deadline:
            break
            
        # For small rank, try random restarts with limited search space
        max_attempts = 5000
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            U = np.array([[random.choice(vals) for _ in range(rank)] for _ in range(rows_u)])
            V = np.array([[random.choice(vals) for _ in range(rank)] for _ in range(rows_v)])
            W = np.array([[random.choice(vals) for _ in range(rank)] for _ in range(rows_w)])
            
            if verify(U.tolist(), V.tolist(), W.tolist()):
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = rank
                break
    
    return best
# EVOLVE-BLOCK-END
