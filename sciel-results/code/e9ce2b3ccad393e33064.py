"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    deadline = time.time() + 140.0
    
    def u_idx(i, j):
        return i * m + j
    
    def v_idx(j, k):
        return j * p + k
    
    def w_idx(k, i):
        return k * n + i
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        U_cols.append(np.array(u, dtype=float))
        V_cols.append(np.array(v, dtype=float))
        W_cols.append(np.array(w, dtype=float))
    
    def add_strassen_2x2_2(ai, aj, bj, bk):
        # M1 = (a+d)(e+h)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[u_idx(ai, aj)] = 1; u[u_idx(ai+1, aj+1)] = 1
        v[v_idx(bj, bk)] = 1; v[v_idx(bj+1, bk+1)] = 1
        w[w_idx(bk, ai)] = 1; w[w_idx(bk+1, ai+1)] = 1
        add_term(u, v, w)
        # M2 = (c+d)e
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[u_idx(ai+1, aj)] = 1; u[u_idx(ai+1, aj+1)] = 1
        v[v_idx(bj, bk)] = 1
        w[w_idx(bk, ai+1)] = 1; w[w_idx(bk+1, ai+1)] = -1
        add_term(u, v, w)
        # M3 = a(f-h)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[u_idx(ai, aj)] = 1
        v[v_idx(bj, bk+1)] = 1; v[v_idx(bj+1, bk+1)] = -1
        w[w_idx(bk+1, ai)] = 1; w[w_idx(bk+1, ai+1)] = 1
        add_term(u, v, w)
        # M4 = d(g-e)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[u_idx(ai+1, aj+1)] = 1
        v[v_idx(bj+1, bk)] = 1; v[v_idx(bj, bk)] = -1
        w[w_idx(bk, ai)] = 1; w[w_idx(bk, ai+1)] = 1
        add_term(u, v, w)
        # M5 = (a+b)h
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[u_idx(ai, aj)] = 1; u[u_idx(ai, aj+1)] = 1
        v[v_idx(bj+1, bk+1)] = 1
        w[w_idx(bk, ai)] = -1; w[w_idx(bk+1, ai)] = 1
        add_term(u, v, w)
        # M6 = (c-a)(e+f)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[u_idx(ai+1, aj)] = 1; u[u_idx(ai, aj)] = -1
        v[v_idx(bj, bk)] = 1; v[v_idx(bj, bk+1)] = 1
        w[w_idx(bk+1, ai+1)] = 1
        add_term(u, v, w)
        # M7 = (b-d)(g+h)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[u_idx(ai, aj+1)] = 1; u[u_idx(ai+1, aj+1)] = -1
        v[v_idx(bj+1, bk)] = 1; v[v_idx(bj+1, bk+1)] = 1
        w[w_idx(bk, ai)] = 1
        add_term(u, v, w)
    
    def add_2x2x3(ai, aj, bj, bk):
        add_strassen_2x2_2(ai, aj, bj, bk)
        # 4 additional terms for the third column
        for di in range(2):
            for dj in range(2):
                u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
                u[u_idx(ai+di, aj+dj)] = 1
                v[v_idx(bj+dj, bk+2)] = 1
                w[w_idx(bk+2, ai+di)] = 1
                add_term(u, v, w)
    
    # Build the decomposition using 2x2 blocks
    # A is 4x4: blocks at (0,0), (0,2), (2,0), (2,2)
    # B is 4x5: blocks at (0,0), (0,2), (2,0), (2,2) where col blocks are 2-wide
    # Block (ai, aj) of A multiplies block (bj, bk) of B
    # For 4x4 * 4x5:
    #   A blocks: 2x2 each, positions (0,0), (0,2), (2,0), (2,2)
    #   B blocks: 2x2 or 2x3, positions (0,0), (0,2), (2,0), (2,2)
    #   The "middle" dimension m=4 splits into two 2-wide blocks
    
    # Top-left: A[0:2, 0:2] * B[0:2, 0:5] -> 2x2 * 2x5 = 2x2x3 (split B into 2x2 + 2x3)
    add_strassen_2x2_2(0, 0, 0, 0)  # A[0:2,0:2] * B[0:2,0:2]
    add_2x2x3(0, 0, 0, 2)          # A[0:2,0:2] * B[0:2,2:5]
    
    # Top-right: A[0:2, 2:4] * B[0:2, 0:5]
    add_strassen_2x2_2(0, 2, 0, 0)  # A[0:2,2:4] * B[0:2,0:2]
    add_2x2x3(0, 2, 0, 2)          # A[0:2,2:4] * B[0:2,2:5]
    
    # Bottom-left: A[2:4, 0:2] * B[2:4, 0:5]
    add_strassen_2x2_2(2, 0, 2, 0)  # A[2:4,0:2] * B[2:4,0:2]
    add_2x2x3(2, 0, 2, 2)          # A[2:4,0:2] * B[2:4,2:5]
    
    # Bottom-right: A[2:4, 2:4] * B[2:4, 0:5]
    add_strassen_2x2_2(2, 2, 2, 0)  # A[2:4,2:4] * B[2:4,0:2]
    add_2x2x3(2, 2, 2, 2)          # A[2:4,2:4] * B[2:4,2:5]
    
    U_mat = np.column_stack(U_cols) if U_cols else np.zeros((n*m, 0))
    V_mat = np.column_stack(V_cols) if V_cols else np.zeros((m*p, 0))
    W_mat = np.column_stack(W_cols) if W_cols else np.zeros((p*n, 0))
    
    def verify(U_mat, V_mat, W_mat, tol=1e-10):
        rank = U_mat.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ui = i * m + j
                    vi = j * p + k
                    wi = k * n + i
                    val = np.dot(U_mat[ui], V_mat[vi] * W_mat[wi])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Verify initial decomposition
    if not verify(U_mat, V_mat, W_mat):
        # Fallback to schoolbook
        rank = n * m * p
        U_sb = np.zeros((n*m, rank))
        V_sb = np.zeros((m*p, rank))
        W_sb = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j, col] = 1
                    V_sb[j*p+k, col] = 1
                    W_sb[k*n+i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    best_rank = U_mat.shape[1]
    best_U, best_V, best_W = U_mat.copy(), V_mat.copy(), W_mat.copy()
    
    # Try to remove terms one by one
    current_U, current_V, current_W = U_mat.copy(), V_mat.copy(), W_mat.copy()
    current_rank = current_U.shape[1]
    
    max_iterations = 200
    iteration = 0
    while iteration < max_iterations and time.time() < deadline:
        iteration += 1
        removed = False
        for c in range(current_rank):
            if time.time() > deadline:
                break
            mask = np.ones(current_rank, dtype=bool)
            mask[c] = False
            U_test = current_U[:, mask]
            V_test = current_V[:, mask]
            W_test = current_W[:, mask]
            if verify(U_test, V_test, W_test):
                current_U, current_V, current_W = U_test, V_test, W_test
                current_rank -= 1
                removed = True
                break
        if not removed:
            break
    
    if current_rank < best_rank:
        best_rank = current_rank
        best_U, best_V, best_W = current_U, current_V, current_W
    
    # Convert to list format with half-integer check
    def to_halfint_list(mat):
        result = []
        for row in mat:
            result.append([float(round(x * 2) / 2) for x in row])
        return result
    
    return {
        "U": to_halfint_list(best_U),
        "V": to_halfint_list(best_V),
        "W": to_halfint_list(best_W)
    }
# EVOLVE-BLOCK-END
