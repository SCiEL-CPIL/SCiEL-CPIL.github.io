"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # Strassen rank-7 for 2x2x2
    # Verified decomposition
    def strassen_2x2x2():
        u = [
            [1, 0, 1, 0],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 1, 0, -1],
        ]
        v = [
            [1, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 1, 0, -1],
            [-1, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ]
        w = [
            [1, 0, 0, 1],
            [0, 1, 0, 0],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
        ]
        return u, v, w
    
    # Verify Strassen decomposition for 2x2x2
    import numpy as np
    u_s, v_s, w_s = strassen_2x2x2()
    T_check = np.zeros((4, 4, 4))
    for r in range(7):
        T_check += np.outer(np.outer(u_s[r], v_s[r]), w_s[r]).reshape(4, 4, 4)
    
    # Expected: T[i*2+j, j*2+k, k*2+i] = 1
    valid = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                x = i * 2 + j
                y = j * 2 + k
                z = k * 2 + i
                if abs(T_check[x, y, z] - 1.0) > 1e-9:
                    valid = False
                # Also check other (x,y,z) with same i,j,k pattern shouldn't exist
    for x in range(4):
        for y in range(4):
            for z in range(4):
                i1, j1 = x // 2, x % 2
                j2, k1 = y // 2, y % 2
                k2, i2 = z // 2, z % 2
                if j1 != j2 or i1 != i2 or k1 != k2:
                    if abs(T_check[x, y, z]) > 1e-9:
                        valid = False
                        break
            else:
                continue
            break
        else:
            continue
        break
    
    if not valid:
        # Strassen decomposition is wrong, use schoolbook
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Build block decomposition
    # n=3: [2, 1], m=5: [2, 2, 1], p=6: [2, 2, 2]
    n_splits = [2, 1]
    m_splits = [2, 2, 1]
    p_splits = [2, 2, 2]
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    for ni, nb in enumerate(n_splits):
        for mj, mb in enumerate(m_splits):
            for pk, pb in enumerate(p_splits):
                i_start = sum(n_splits[:ni])
                j_start = sum(m_splits[:mj])
                k_start = sum(p_splits[:pk])
                
                if nb == 2 and mb == 2 and pb == 2:
                    for r in range(7):
                        u_row = [0.0] * (n * m)
                        v_row = [0.0] * (m * p)
                        w_row = [0.0] * (p * n)
                        
                        for a_idx in range(4):
                            i_local = a_idx // 2
                            j_local = a_idx % 2
                            u_row[i_start + i_local] = u_s[r][a_idx]
                        
                        for b_idx in range(4):
                            j_local = b_idx // 2
                            k_local = b_idx % 2
                            v_row[j_start + j_local] = v_s[r][b_idx]
                        
                        for c_idx in range(4):
                            k_local = c_idx // 2
                            i_local = c_idx % 2
                            w_row[k_start + i_local] = w_s[r][c_idx]
                        
                        U_rows.append(u_row)
                        V_rows.append(v_row)
                        W_rows.append(w_row)
                else:
                    for i_local in range(nb):
                        for j_local in range(mb):
                            for k_local in range(pb):
                                u_row = [0.0] * (n * m)
                                v_row = [0.0] * (m * p)
                                w_row = [0.0] * (p * n)
                                u_row[i_start + i_local] = 1.0
                                v_row[j_start + j_local] = 1.0
                                w_row[k_start + k_local] = 1.0
                                U_rows.append(u_row)
                                V_rows.append(v_row)
                                W_rows.append(w_row)
    
    U = U_rows
    V = V_rows
    W = W_rows
    
    # Final verification
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    ok = True
    np.random.seed(123)
    for _ in range(200):
        i = np.random.randint(n)
        j = np.random.randint(m)
        k = np.random.randint(p)
        x = i * m + j
        y = j * p + k
        z = k * n + i
        T_val = np.sum(U_np[x, :] * V_np[y, :] * W_np[z, :])
        if abs(T_val - 1.0) > 1e-9:
            ok = False
            break
    
    if not ok:
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
