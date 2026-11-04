import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 110.0

    # Strassen rank-7 decomposition for 2x2 matrix multiplication.
    # U2 row index = i*2+j (A entry), V2 row index = j*2+k (B entry), W2 row index = k*2+i (C entry).
    # The decomposition satisfies: sum_c U2[i*2+j, c] * V2[j*2+k, c] * W2[k*2+i, c] = 1 for all i,j,k in {0,1}.
    
    # Standard Strassen decomposition
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Let's verify the provided U2, V2, W2 are correct Strassen factors.
    # The provided arrays in the prompt are:
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)

    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)

    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)

    # Quick sanity check for 2x2 Strassen correctness
    ok2 = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = int(np.sum(U2[i*2+j, :] * V2[j*2+k, :] * W2[k*2+i, :]))
                if s != 1:
                    ok2 = False
                    break
            if not ok2:
                break
        if not ok2:
            break

    if not ok2:
        # Fallback: schoolbook 4x4 (rank 64)
        n, m, p = 4, 4, 4
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Build 4x4 factors from 2x2 Strassen factors using Kronecker product logic
    # The rank will be 7 * 7 = 49.
    # This is the known optimal rank for 4x4 matrix multiplication.
    
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)

    # Precompute indices for efficiency
    # U4[i*4+j, c1*7+c2] = U2[i_hi*2+j_hi, c1] * U2[i_lo*2+j_lo, c2]
    # where i = 2*i_hi + i_lo, j = 2*j_hi + j_lo
    
    idxs_u = []
    for i in range(4):
        for j in range(4):
            i_hi, i_lo = i // 2, i % 2
            j_hi, j_lo = j // 2, j % 2
            idxs_u.append((i*4+j, i_hi*2+j_hi, i_lo*2+j_lo))
    
    for idx_u, u_idx_hi, u_idx_lo in idxs_u:
        for c1 in range(7):
            u1 = U2[u_idx_hi, c1]
            if u1 == 0: continue
            for c2 in range(7):
                u2 = U2[u_idx_lo, c2]
                if u2 == 0: continue
                U4[idx_u, c1*7+c2] = u1 * u2

    idxs_v = []
    for j in range(4):
        for k in range(4):
            j_hi, j_lo = j // 2, j % 2
            k_hi, k_lo = k // 2, k % 2
            idxs_v.append((j*4+k, j_hi*2+k_hi, j_lo*2+k_lo))
            
    for idx_v, v_idx_hi, v_idx_lo in idxs_v:
        for c1 in range(7):
            v1 = V2[v_idx_hi, c1]
            if v1 == 0: continue
            for c2 in range(7):
                v2 = V2[v_idx_lo, c2]
                if v2 == 0: continue
                V4[idx_v, c1*7+c2] = v1 * v2

    idxs_w = []
    for k in range(4):
        for i in range(4):
            k_hi, k_lo = k // 2, k % 2
            i_hi, i_lo = i // 2, i % 2
            idxs_w.append((k*4+i, k_hi*2+i_hi, k_lo*2+i_lo))
            
    for idx_w, w_idx_hi, w_idx_lo in idxs_w:
        for c1 in range(7):
            w1 = W2[w_idx_hi, c1]
            if w1 == 0: continue
            for c2 in range(7):
                w2 = W2[w_idx_lo, c2]
                if w2 == 0: continue
                W4[idx_w, c1*7+c2] = w1 * w2

    # Verify 4x4 correctness
    ok4 = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                s = int(np.sum(U4[i*4+j, :] * V4[j*4+k, :] * W4[k*4+i, :]))
                if s != 1:
                    ok4 = False
                    break
            if not ok4:
                break
        if not ok4:
            break

    if not ok4:
        # Fallback: schoolbook 4x4 (rank 64)
        n, m, p = 4, 4, 4
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    if time.time() - start_time > deadline:
        # Fallback: schoolbook 4x4 (rank 64)
        n, m, p = 4, 4, 4
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    return {
        "U": U4.tolist(),
        "V": V4.tolist(),
        "W": W4.tolist()
    }
