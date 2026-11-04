import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 115  # safety margin before 150s kill

    def schoolbook():
        n, m, p = 4, 4, 4
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    # 2x2 Strassen decomposition
    # Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    # T[i,j,k] = sum_c U[i*2+j,c] * V[j*2+k,c] * W[k*2+i,c] = 1
    
    # Standard Strassen 2x2
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need U, V, W such that U[i,j] V[j,k] W[k,i] sums to 1 for T[i,j,k]=1.
    # The matrices provided in the original code seem to be a valid Strassen decomposition.
    # I will use them directly.
    
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)
    
    # Verify 2x2 Strassen
    valid2 = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = np.dot(U2[i*2+j], V2[j*2+k] * W2[k*2+i])
                if abs(s - 1) > 1e-9:
                    valid2 = False
                    break
            if not valid2:
                break
        if not valid2:
            break
    
    if not valid2:
        return schoolbook()
    
    # Build 4x4 via Kronecker with proper permutation
    U4_kron = np.kron(U2, U2)
    V4_kron = np.kron(V2, V2)
    W4_kron = np.kron(W2, W2)
    
    perm_U = np.zeros(16, dtype=int)
    perm_V = np.zeros(16, dtype=int)
    perm_W = np.zeros(16, dtype=int)
    
    for S in range(16):
        i = S // 4
        j = S % 4
        i_a, i_b = i // 2, i % 2
        j_a, j_b = j // 2, j % 2
        perm_U[S] = (i_a*2 + j_a) * 4 + (i_b*2 + j_b)
        
    for S in range(16):
        j = S // 4
        k = S % 4
        j_a, j_b = j // 2, j % 2
        k_a, k_b = k // 2, k % 2
        perm_V[S] = (j_a*2 + k_a) * 4 + (j_b*2 + k_b)
        
    for S in range(16):
        k = S // 4
        i = S % 4
        k_a, k_b = k // 2, k % 2
        i_a, i_b = i // 2, i % 2
        perm_W[S] = (k_a*2 + i_a) * 4 + (k_b*2 + i_b)
        
    U4 = U4_kron[perm_U, :]
    V4 = V4_kron[perm_V, :]
    W4 = W4_kron[perm_W, :]
    
    # Verify 4x4
    # T[i,j,k] = sum_c U[i*4+j, c] * V[j*4+k, c] * W[k*4+i, c]
    
    # Vectorized check
    i_idx = np.repeat(np.arange(4), 16)
    j_idx = np.tile(np.repeat(np.arange(4), 4), 4)
    k_idx = np.tile(np.arange(4), 16)
    
    u_rows = i_idx * 4 + j_idx
    v_rows = j_idx * 4 + k_idx
    w_rows = k_idx * 4 + i_idx
    
    U_sel = U4[u_rows, :]
    V_sel = V4[v_rows, :]
    W_sel = W4[w_rows, :]
    
    T_flat = np.einsum('ij,ij->i', np.einsum('ij,ij->ij', U_sel, V_sel), W_sel)
    
    if not np.allclose(T_flat, 1.0, atol=1e-9):
        return schoolbook()
    
    def to_list(mat):
        result = []
        for row in mat:
            r = []
            for val in row:
                if abs(val - round(val)) < 1e-9:
                    r.append(int(round(val)))
                elif abs(val - (round(val*2)/2)) < 1e-9:
                    r.append(round(val*2)/2)
                else:
                    r.append(float(val))
            result.append(r)
        return result

    if time.time() - start_time > DEADLINE:
        return schoolbook()
        
    return {
        "U": to_list(U4),
        "V": to_list(V4),
        "W": to_list(W4)
    }
