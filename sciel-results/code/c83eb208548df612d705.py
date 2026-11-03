"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = start_time + 110
    
    n, m, p = DIMENSIONS  # A: n×m, B: m×p, C: n×p
    
    # Correct Strassen decomposition for 2×2×2 matrix multiplication
    # p1 = (a+d)(e+h), p2 = (c+d)e, p3 = a(f-h), p4 = d(g-e), p5 = (a+b)h, p6 = (c-a)(e+f), p7 = (b-d)(g+h)
    # C[0,0] = p1 + p4 - p5 + p7
    # C[0,1] = p3 + p5
    # C[1,0] = p2 + p4
    # C[1,1] = p1 - p2 + p3 + p6
    
    # Each term: (U_2x2, V_2x2, W_2x2) where U[i,j] is coeff of A[i,j], V[j,k] of B[j,k], W[i,k] of contribution to C[i,k]
    strassen_terms = [
        # p1: U=[[1,0],[0,1]], V=[[1,0],[0,1]], W=[[1,0],[0,1]]
        ([[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]]),
        # p2: U=[[0,0],[1,1]], V=[[1,0],[0,0]], W=[[0,0],[1,-1]]
        ([[0,0],[1,1]], [[1,0],[0,0]], [[0,0],[1,-1]]),
        # p3: U=[[1,0],[0,0]], V=[[0,1],[0,-1]], W=[[0,1],[0,1]]
        ([[1,0],[0,0]], [[0,1],[0,-1]], [[0,1],[0,1]]),
        # p4: U=[[0,0],[0,1]], V=[[-1,0],[0,0]], W=[[1,0],[1,0]]
        ([[0,0],[0,1]], [[-1,0],[0,0]], [[1,0],[1,0]]),
        # p5: U=[[1,1],[0,0]], V=[[0,0],[0,1]], W=[[-1,0],[0,1]]
        ([[1,1],[0,0]], [[0,0],[0,1]], [[-1,0],[0,1]]),
        # p6: U=[[-1,0],[1,0]], V=[[1,1],[0,0]], W=[[0,0],[0,1]]
        ([[-1,0],[1,0]], [[1,1],[0,0]], [[0,0],[0,1]]),
        # p7: U=[[0,1],[-1,0]], V=[[0,0],[1,1]], W=[[1,0],[0,0]]
        ([[0,1],[-1,0]], [[0,0],[1,1]], [[1,0],[0,0]]),
    ]
    
    # Verify Strassen numerically
    def verify_strassen():
        for u_pat, v_pat, w_pat in strassen_terms:
            pass  # Just checking structure
        # Full verification: build 2x2x2 tensor from Strassen and check it equals all-1s
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        for r, (u_pat, v_pat, w_pat) in enumerate(strassen_terms):
            for di in range(2):
                for dj in range(2):
                    U[di*2+dj, r] = u_pat[di][dj]
            for dj in range(2):
                for dk in range(2):
                    V[dj*2+dk, r] = v_pat[dj][dk]
            for dk in range(2):
                for di in range(2):
                    W[dk*2+di, r] = w_pat[di][dk]
        
        # Check: sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r] == 1 for all i,j,k
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = 0
                    for r in range(7):
                        s += U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r]
                    if abs(s - 1) > 1e-10:
                        return False
        return True
    
    strassen_valid = verify_strassen()
    
    # Try multiple partition strategies and pick the best valid one
    # Strategy 1: Fixed partition (rows: 2+1, mid: 2+2, cols: 2+3)
    # Strategy 2: Different partitions
    # Strategy 3: Pure schoolbook (fallback)
    
    def build_decomposition(row_parts, mid_parts, col_parts, use_strassen=True):
        """Build decomposition given partitions. Returns (U, V, W, rank) or None if invalid."""
        U_cols = []
        V_cols = []
        W_cols = []
        
        for ri, (r_start, r_end) in enumerate(row_parts):
            for mi, (m_start, m_end) in enumerate(mid_parts):
                for ci, (c_start, c_end) in enumerate(col_parts):
                    rc = r_end - r_start
                    mc = m_end - m_start
                    cc = c_end - c_start
                    
                    if use_strassen and strassen_valid and rc == 2 and mc == 2 and cc == 2:
                        # Use Strassen
                        for u_pat, v_pat, w_pat in strassen_terms:
                            u_col = [0] * (n * m)
                            v_col = [0] * (m * p)
                            w_col = [0] * (p * n)
                            for di in range(2):
                                for dj in range(2):
                                    u_col[(r_start + di) * m + (m_start + dj)] = u_pat[di][dj]
                            for dj in range(2):
                                for dk in range(2):
                                    v_col[(m_start + dj) * p + (c_start + dk)] = v_pat[dj][dk]
                            for dk in range(2):
                                for di in range(2):
                                    w_col[(c_start + dk) * n + (r_start + di)] = w_pat[di][dk]
                            U_cols.append(u_col)
                            V_cols.append(v_col)
                            W_cols.append(w_col)
                    else:
                        # Schoolbook
                        for di in range(rc):
                            for dj in range(mc):
                                for dk in range(cc):
                                    u_col = [0] * (n * m)
                                    v_col = [0] * (m * p)
                                    w_col = [0] * (p * n)
                                    i = r_start + di
                                    j = m_start + dj
                                    k = c_start + dk
                                    u_col[i * m + j] = 1
                                    v_col[j * p + k] = 1
                                    w_col[k * n + i] = 1
                                    U_cols.append(u_col)
                                    V_cols.append(v_col)
                                    W_cols.append(w_col)
        
        rank = len(U_cols)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for r in range(rank):
            for i in range(n * m):
                U[i][r] = U_cols[r][i]
            for i in range(m * p):
                V[i][r] = V_cols[r][i]
            for i in range(p * n):
                W[i][r] = W_cols[r][i]
        
        # Verify
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    max_err = max(max_err, abs(s - 1))
        
        if max_err > 1e-10:
            return None
        
        return (U, V, W, rank)
    
    # Define candidate partitions
    # Each partition is a list of (start, end) tuples
    candidates = []
    
    # Strategy 1: rows 2+1, mid 2+2, cols 2+3
    candidates.append(([(0,2),(2,3)], [(0,2),(2,4)], [(0,2),(2,5)]))
    
    # Strategy 2: rows 1+2, mid 2+2, cols 3+2
    candidates.append(([(0,1),(1,3)], [(0,2),(2,4)], [(0,3),(3,5)]))
    
    # Strategy 3: rows 2+1, mid 2+2, cols 3+2
    candidates.append(([(0,2),(2,3)], [(0,2),(2,4)], [(0,3),(3,5)]))
    
    # Strategy 4: rows 1+1+1, mid 2+2, cols 2+3
    candidates.append(([(0,1),(1,2),(2,3)], [(0,2),(2,4)], [(0,2),(2,5)]))
    
    # Strategy 5: rows 2+1, mid 1+1+2, cols 2+3
    candidates.append(([(0,2),(2,3)], [(0,1),(1,2),(2,4)], [(0,2),(2,5)]))
    
    # Strategy 6: rows 1+2, mid 1+3, cols 2+3
    candidates.append(([(0,1),(1,3)], [(0,1),(1,4)], [(0,2),(2,5)]))
    
    # Strategy 7: All schoolbook (no Strassen)
    candidates.append(([(0,3)], [(0,4)], [(0,5)]))
    
    best_result = None
    best_rank = float('inf')
    
    for row_parts, mid_parts, col_parts in candidates:
        if time.time() > DEADLINE - 5:
            break
        
        result = build_decomposition(row_parts, mid_parts, col_parts, use_strassen=True)
        if result is not None:
            U, V, W, rank = result
            if rank < best_rank:
                best_rank = rank
                best_result = (U, V, W)
    
    # If no valid result, fall back to pure schoolbook
    if best_result is None:
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
    
    U, V, W = best_result
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
