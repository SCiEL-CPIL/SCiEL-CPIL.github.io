import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Partition indices
    # J groups: {0,1}, {2,3}, {4}
    # K groups: {0,1}, {2,3}, {4,5}
    
    j_groups = [
        [0, 1],
        [2, 3],
        [4]
    ]
    k_groups = [
        [0, 1],
        [2, 3],
        [4, 5]
    ]

    # Calculate total rank
    rank = 0
    for jg in j_groups:
        for kg in k_groups:
            if len(jg) == 2 and len(kg) == 2:
                rank += 7
            elif len(jg) == 1 and len(kg) == 2:
                rank += 4
            elif len(jg) == 2 and len(kg) == 1:
                rank += 4
            else:
                rank += 2

    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0

    for jg in j_groups:
        for kg in k_groups:
            if len(jg) == 2 and len(kg) == 2:
                j1, j2 = jg
                k1, k2 = kg
                
                # A indices
                u_00 = 0 * m + j1
                u_01 = 0 * m + j2
                u_10 = 1 * m + j1
                u_11 = 1 * m + j2
                
                # B indices
                v_00 = j1 * p + k1
                v_01 = j1 * p + k2
                v_10 = j2 * p + k1
                v_11 = j2 * p + k2
                
                # C indices (W)
                # C[k1, 0], C[k1, 1], C[k2, 0], C[k2, 1]
                w_00 = k1 * n + 0
                w_01 = k1 * n + 1
                w_10 = k2 * n + 0
                w_11 = k2 * n + 1
                
                # M1: (A00 + A11)(B00 + B11)
                U[u_00, col] = 1
                U[u_11, col] = 1
                V[v_00, col] = 1
                V[v_11, col] = 1
                W[w_00, col] = 1
                W[w_11, col] = 1
                col += 1
                
                # M2: (A10 + A11)B00
                U[u_10, col] = 1
                U[u_11, col] = 1
                V[v_00, col] = 1
                W[w_10, col] = 1
                W[w_11, col] = -1
                col += 1
                
                # M3: A00(B01 - B11)
                U[u_00, col] = 1
                V[v_01, col] = 1
                V[v_11, col] = -1
                W[w_01, col] = 1
                W[w_11, col] = 1
                col += 1
                
                # M4: A11(B10 - B00)
                U[u_11, col] = 1
                V[v_10, col] = 1
                V[v_00, col] = -1
                W[w_00, col] = 1
                W[w_10, col] = 1
                col += 1
                
                # M5: (A00 + A01)B11
                U[u_00, col] = 1
                U[u_01, col] = 1
                V[v_11, col] = 1
                W[w_00, col] = -1
                W[w_01, col] = 1
                col += 1
                
                # M6: (A10 - A00)(B00 + B01)
                U[u_10, col] = 1
                U[u_00, col] = -1
                V[v_00, col] = 1
                V[v_01, col] = 1
                W[w_11, col] = 1
                col += 1
                
                # M7: (A01 - A10)(B10 + B11)
                U[u_01, col] = 1
                U[u_10, col] = -1
                V[v_10, col] = 1
                V[v_11, col] = 1
                W[w_00, col] = 1
                col += 1

            elif len(jg) == 1 and len(kg) == 2:
                j1 = jg[0]
                k1, k2 = kg
                
                # C[k1, 0] = A[0, j1] B[j1, k1]
                U[0 * m + j1, col] = 1
                V[j1 * p + k1, col] = 1
                W[k1 * n + 0, col] = 1
                col += 1
                
                # C[k1, 1] = A[1, j1] B[j1, k1]
                U[1 * m + j1, col] = 1
                V[j1 * p + k1, col] = 1
                W[k1 * n + 1, col] = 1
                col += 1
                
                # C[k2, 0] = A[0, j1] B[j1, k2]
                U[0 * m + j1, col] = 1
                V[j1 * p + k2, col] = 1
                W[k2 * n + 0, col] = 1
                col += 1
                
                # C[k2, 1] = A[1, j1] B[j1, k2]
                U[1 * m + j1, col] = 1
                V[j1 * p + k2, col] = 1
                W[k2 * n + 1, col] = 1
                col += 1

            elif len(jg) == 2 and len(kg) == 1:
                j1, j2 = jg
                k1 = kg[0]
                
                # C[k1, 0] = A[0, j1] B[j1, k1] + A[0, j2] B[j2, k1]
                U[0 * m + j1, col] = 1
                V[j1 * p + k1, col] = 1
                W[k1 * n + 0, col] = 1
                col += 1
                
                U[0 * m + j2, col] = 1
                V[j2 * p + k1, col] = 1
                W[k1 * n + 0, col] = 1
                col += 1
                
                # C[k1, 1] = A[1, j1] B[j1, k1] + A[1, j2] B[j2, k1]
                U[1 * m + j1, col] = 1
                V[j1 * p + k1, col] = 1
                W[k1 * n + 1, col] = 1
                col += 1
                
                U[1 * m + j2, col] = 1
                V[j2 * p + k1, col] = 1
                W[k1 * n + 1, col] = 1
                col += 1

            else:
                # 1x1 block
                j1 = jg[0]
                k1 = kg[0]
                
                U[0 * m + j1, col] = 1
                V[j1 * p + k1, col] = 1
                W[k1 * n + 0, col] = 1
                col += 1
                
                U[1 * m + j1, col] = 1
                V[j1 * p + k1, col] = 1
                W[k1 * n + 1, col] = 1
                col += 1

    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Verify
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U[i*m+j, :]
                v_row = V[j*p+k, :]
                w_row = W[k*n+i, :]
                result = np.sum(u_row * v_row * w_row)
                if abs(result - 1) > 1e-10:
                    valid = False
                    break
            if not valid: break
        if not valid: break
        
    if valid:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        U_fallback = np.zeros((n * m, 60), dtype=float)
        V_fallback = np.zeros((m * p, 60), dtype=float)
        W_fallback = np.zeros((p * n, 60), dtype=float)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fallback[i*m+j, c] = 1
                    V_fallback[j*p+k, c] = 1
                    W_fallback[k*n+i, c] = 1
                    c += 1
        return {"U": U_fallback.tolist(), "V": V_fallback.tolist(), "W": W_fallback.tolist()}

# EVOLVE-BLOCK-END
