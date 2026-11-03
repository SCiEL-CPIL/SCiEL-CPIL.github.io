import numpy as np
import time

def construct():
    n, m, p = 3, 4, 7
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    def schoolbook_decomp(n, m, p):
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
        return U, V, W

    def strassen_decomp():
        # 2x2x2 rank 7
        rank = 7
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(4)]
        W = [[0]*rank for _ in range(4)]
        
        # P1: (a+d)(e+h)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # P2: (c+d)e
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1
        
        # P3: a(f-h)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        
        # P4: d(g-e)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # P5: (a+b)h
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        # P6: (c-d)(f+g)
        U[2][5] = 1; U[3][5] = -1
        V[1][5] = 1; V[2][5] = 1
        W[3][5] = 1
        
        # P7: (b-d)(e+g)
        U[1][6] = 1; U[3][6] = -1
        V[0][6] = 1; V[2][6] = 1
        W[0][6] = 1
        
        return U, V, W

    def combine_decomps(decomps, n, m, p, i_off, j_off, k_off):
        total_rank = sum(len(u[0]) for u, v, w, _, _, _ in decomps)
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        rank_offset = 0
        for u_sub, v_sub, w_sub, n_sub, m_sub, p_sub in decomps:
            r_sub = len(u_sub[0])
            
            for i_local in range(n_sub):
                for j_local in range(m_sub):
                    i_global = i_off + i_local
                    j_global = j_off + j_local
                    u_row = i_global * m + j_global
                    u_sub_row = i_local * m_sub + j_local
                    for r in range(r_sub):
                        U[u_row][rank_offset + r] = u_sub[u_sub_row][r]
                        
            for j_local in range(m_sub):
                for k_local in range(p_sub):
                    j_global = j_off + j_local
                    k_global = k_off + k_local
                    v_row = j_global * p + k_global
                    v_sub_row = j_local * p_sub + k_local
                    for r in range(r_sub):
                        V[v_row][rank_offset + r] = v_sub[v_sub_row][r]
                        
            for k_local in range(p_sub):
                for i_local in range(n_sub):
                    k_global = k_off + k_local
                    i_global = i_off + i_local
                    w_row = k_global * n + i_global
                    w_sub_row = k_local * n_sub + i_local
                    for r in range(r_sub):
                        W[w_row][rank_offset + r] = w_sub[w_sub_row][r]
                        
            rank_offset += r_sub
            
        return U, V, W

    def solve(n, m, p):
        if time.time() > deadline:
             return schoolbook_decomp(n, m, p)
             
        if n == 1 or m == 1 or p == 1:
            return schoolbook_decomp(n, m, p)
            
        if m >= 4:
            m1 = 2
            m2 = m - 2
            U1, V1, W1 = solve(n, m1, p)
            U2, V2, W2 = solve(n, m2, p)
            return combine_decomps([(U1, V1, W1, n, m1, p), (U2, V2, W2, n, m2, p)], n, m, p, 0, 0, 0)
        elif p >= 4:
            p1 = 2
            p2 = p - 2
            U1, V1, W1 = solve(n, m, p1)
            U2, V2, W2 = solve(n, m, p2)
            return combine_decomps([(U1, V1, W1, n, m, p1), (U2, V2, W2, n, m, p2)], n, m, p, 0, 0, 0)
        elif n >= 3:
            n1 = 2
            n2 = n - 2
            U1, V1, W1 = solve(n1, m, p)
            U2, V2, W2 = solve(n2, m, p)
            return combine_decomps([(U1, V1, W1, n1, m, p), (U2, V2, W2, n2, m, p)], n, m, p, 0, 0, 0)
        else:
            if n == 2 and m == 2 and p == 2:
                U, V, W = strassen_decomp()
                return U, V, W
            else:
                return schoolbook_decomp(n, m, p)

    # Execute the recursive solver
    U, V, W = solve(n, m, p)
    
    # Verify using numpy for speed
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    rank = U_np.shape[1]
    
    # Vectorized verification
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                actual = np.dot(U_np[u_idx], V_np[v_idx] * W_np[w_idx])
                if abs(actual - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
            
    if not valid:
        U, V, W = schoolbook_decomp(n, m, p)
        return {"U": U, "V": V, "W": W}
        
    return {"U": U, "V": V, "W": W}
