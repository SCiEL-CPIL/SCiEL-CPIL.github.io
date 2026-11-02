import numpy as np
import time
import random

_START_TIME = time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 2.0

def _remaining():
    return _DEADLINE - _SAFETY_MARGIN - (time.time() - _START_TIME)

def construct():
    n, m, p = 2, 4, 5
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Build the multiplication tensor T[i,j,k] = 1 if i==k (for C[i,k] = sum_j A[i,j]*B[j,k])
    # T has shape (n*m, m*p, p*n) where T[i*m+j, j'*p+k, k'*n+i'] = 1 if j==j' and i==i'
    # Actually T[a, b, c] where a=i*m+j, b=j'*p+k, c=k'*n+i'
    # T[a,b,c] = 1 if j==j' and i==i'
    
    # Let's verify our current decomposition is correct and try to find a better one
    # by local search: start with rank-36 solution, try to merge/remove terms
    
    def _verify(U_mat, V_mat, W_mat):
        """Check if U, V, W correctly decompose the matrix multiplication tensor."""
        # For each output entry C[i,k] = sum_j A[i,j] * B[j,k]
        # The decomposition should give: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # for all i, j, k. This must hold for all i in [0,n), j in [0,m), k in [0,p)
        # i.e., sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(len(U_mat[0])):
                        s += U_mat[i*m+j][r] * V_mat[j*p+k][r] * W_mat[k*n+i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    def add_strassen(col_offset, j_offset, k_offset):
        U[0 * m + j_offset + 0][col_offset + 0] = 1
        U[1 * m + j_offset + 1][col_offset + 0] = 1
        V[(j_offset + 0) * p + k_offset + 0][col_offset + 0] = 1
        V[(j_offset + 1) * p + k_offset + 1][col_offset + 0] = 1
        W[(k_offset + 0) * n + 0][col_offset + 0] = 1
        W[(k_offset + 1) * n + 1][col_offset + 0] = 1

        U[1 * m + j_offset + 0][col_offset + 1] = 1
        U[1 * m + j_offset + 1][col_offset + 1] = 1
        V[(j_offset + 0) * p + k_offset + 0][col_offset + 1] = 1
        W[(k_offset + 0) * n + 1][col_offset + 1] = 1
        W[(k_offset + 1) * n + 1][col_offset + 1] = -1

        U[0 * m + j_offset + 0][col_offset + 2] = 1
        V[(j_offset + 0) * p + k_offset + 1][col_offset + 2] = 1
        V[(j_offset + 1) * p + k_offset + 1][col_offset + 2] = -1
        W[(k_offset + 1) * n + 0][col_offset + 2] = 1
        W[(k_offset + 1) * n + 1][col_offset + 2] = 1

        U[1 * m + j_offset + 1][col_offset + 3] = 1
        V[(j_offset + 0) * p + k_offset + 0][col_offset + 3] = -1
        V[(j_offset + 1) * p + k_offset + 0][col_offset + 3] = 1
        W[(k_offset + 0) * n + 0][col_offset + 3] = 1
        W[(k_offset + 0) * n + 1][col_offset + 3] = 1

        U[0 * m + j_offset + 0][col_offset + 4] = 1
        U[0 * m + j_offset + 1][col_offset + 4] = 1
        V[(j_offset + 1) * p + k_offset + 1][col_offset + 4] = 1
        W[(k_offset + 0) * n + 0][col_offset + 4] = -1
        W[(k_offset + 1) * n + 0][col_offset + 4] = 1

        U[0 * m + j_offset + 0][col_offset + 5] = -1
        U[1 * m + j_offset + 0][col_offset + 5] = 1
        V[(j_offset + 0) * p + k_offset + 0][col_offset + 5] = 1
        V[(j_offset + 0) * p + k_offset + 1][col_offset + 5] = 1
        W[(k_offset + 1) * n + 1][col_offset + 5] = 1

        U[0 * m + j_offset + 1][col_offset + 6] = 1
        U[1 * m + j_offset + 1][col_offset + 6] = -1
        V[(j_offset + 1) * p + k_offset + 0][col_offset + 6] = 1
        V[(j_offset + 1) * p + k_offset + 1][col_offset + 6] = 1
        W[(k_offset + 0) * n + 0][col_offset + 6] = 1

    def add_schoolbook_2x2x1(col_offset, j_offset, k_offset):
        U[0 * m + j_offset + 0][col_offset + 0] = 1
        V[(j_offset + 0) * p + k_offset][col_offset + 0] = 1
        W[k_offset * n + 0][col_offset + 0] = 1
        
        U[0 * m + j_offset + 1][col_offset + 1] = 1
        V[(j_offset + 1) * p + k_offset][col_offset + 1] = 1
        W[k_offset * n + 0][col_offset + 1] = 1
        
        U[1 * m + j_offset + 0][col_offset + 2] = 1
        V[(j_offset + 0) * p + k_offset][col_offset + 2] = 1
        W[k_offset * n + 1][col_offset + 2] = 1
        
        U[1 * m + j_offset + 1][col_offset + 3] = 1
        V[(j_offset + 1) * p + k_offset][col_offset + 3] = 1
        W[k_offset * n + 1][col_offset + 3] = 1

    col = 0
    add_strassen(col, 0, 0); col += 7
    add_strassen(col, 0, 2); col += 7
    add_schoolbook_2x2x1(col, 0, 4); col += 4
    
    add_strassen(col, 2, 0); col += 7
    add_strassen(col, 2, 2); col += 7
    add_schoolbook_2x2x1(col, 2, 4); col += 4
    
    # Verify correctness
    if not _verify(U, V, W):
        raise RuntimeError("Decomposition verification failed")
    
    # Try local search to reduce rank
    # Convert to numpy for efficient operations
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Try removing columns (terms) and see if we can compensate
    # First, check if any column is redundant
    for r in range(rank):
        if _remaining() < 5:
            break
        # Try removing column r
        U_test = np.delete(U_np, r, axis=1)
        V_test = np.delete(V_np, r, axis=1)
        W_test = np.delete(W_np, r, axis=1)
        
        # Check if the remaining terms still span the tensor
        # The tensor T[i*m+j, j'*p+k, k'*n+i'] = 1 if j==j' and i==i'
        # We need: sum_{r' != r} U[i*m+j, r'] * V[j*p+k, r'] * W[k*n+i, r'] = 1 for all i,j,k
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r2 in range(rank - 1):
                        s += U_test[i*m+j, r2] * V_test[j*p+k, r2] * W_test[k*n+i, r2]
                    if abs(s - 1) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            best_rank = rank - 1
            best_U = U_test.tolist()
            best_V = V_test.tolist()
            best_W = W_test.tolist()
            U_np, V_np, W_np = U_test, V_test, W_test
            rank = best_rank
            break  # Restart search with reduced rank
    
    if best_rank < 36:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    return {"U": U, "V": V, "W": W}
