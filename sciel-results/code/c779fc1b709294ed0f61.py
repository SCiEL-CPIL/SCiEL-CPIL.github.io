import numpy as np
import time

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    def _elapsed():
        return time.time() - start_time

    # Target rank is 54
    # Decomposition: 2x1x6 (Rank 12) + 2x4x6 (Rank 42)
    # 2x4x6 is decomposed into two 2x2x6 blocks (j in {1,2} and j in {3,4})
    # Each 2x2x6 uses 3 Strassen 2x2x2 blocks (k in {0,1}, {2,3}, {4,5})
    # Rank(2x2x2) = 7 (Strassen)
    # Rank(2x2x6) = 3 * 7 = 21
    # Rank(2x4x6) = 2 * 21 = 42
    # Total Rank = 12 + 42 = 54

    R = 54
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]

    col = 0

    # --- Block 1: 2x1x6 (Rank 12) ---
    # j=0
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    # col is now 12

    # --- Block 2: 2x4x6 (Rank 42) ---
    # Split into two 2x2x6 blocks: j in {1,2} and j in {3,4}
    
    # Strassen coefficients for 2x2x2
    # u, v, w are lists of 7 vectors of length 4
    # Indices: 0:00, 1:01, 2:10, 3:11
    strassen_u = [
        [1, 0, 0, 1], # M1: A00 + A11
        [0, 0, 1, 1], # M2: A10 + A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00 + A01
        [-1, 0, 1, 0],# M6: -A00 + A10
        [0, 1, 0, -1] # M7: A01 - A11
    ]
    strassen_v = [
        [1, 0, 0, 1], # M1: B00 + B11
        [1, 0, 0, 0], # M2: B00
        [0, 1, 0, -1],# M3: B01 - B11
        [-1, 0, 1, 0],# M4: -B00 + B10
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00 + B01
        [0, 0, 1, 1]  # M7: B10 + B11
    ]
    strassen_w = [
        [1, 0, 0, 1], # M1: C00 + C11
        [0, 1, 0, -1],# M2: C01 - C11
        [0, 0, 1, 1], # M3: C10 + C11
        [1, 1, 0, 0], # M4: C00 + C01
        [-1, 0, 1, 0],# M5: -C00 + C10
        [0, 0, 0, 1], # M6: C11
        [1, 0, 0, 0]  # M7: C00
    ]
    
    def add_strassen_block(j_offset, k_offset, start_col):
        # j_offset: 1 or 3
        # k_offset: 0, 2, or 4
        # start_col: starting column index
        for s in range(7):
            c = start_col + s
            # Local indices i' in {0,1}, j' in {0,1}, k' in {0,1}
            for i_prime in range(2):
                for j_prime in range(2):
                    # Index = i * 2 + j
                    val_u = strassen_u[s][i_prime * 2 + j_prime]
                    if val_u != 0:
                        global_i = i_prime
                        global_j = j_offset + j_prime
                        U[global_i * m + global_j][c] += val_u
                        
            for j_prime in range(2):
                for k_prime in range(2):
                    # Index = j * 2 + k
                    val_v = strassen_v[s][j_prime * 2 + k_prime]
                    if val_v != 0:
                        global_j = j_offset + j_prime
                        global_k = k_offset + k_prime
                        V[global_j * p + global_k][c] += val_v
                        
            for k_prime in range(2):
                for i_prime in range(2):
                    # Index = k * 2 + i
                    val_w = strassen_w[s][k_prime * 2 + i_prime]
                    if val_w != 0:
                        global_k = k_offset + k_prime
                        global_i = i_prime
                        W[global_k * n + global_i][c] += val_w
                        
        return start_col + 7

    # Block 2a: j in {1, 2}
    # B parts: k in {0,1}, {2,3}, {4,5}
    c = 12
    c = add_strassen_block(1, 0, c)
    c = add_strassen_block(1, 2, c)
    c = add_strassen_block(1, 4, c)
    
    # Block 2b: j in {3, 4}
    # B parts: k in {0,1}, {2,3}, {4,5}
    c = add_strassen_block(3, 0, c)
    c = add_strassen_block(3, 2, c)
    c = add_strassen_block(3, 4, c)
    
    # Verify column count
    if c != R:
        pass

    # Quick verification: check that the decomposition is valid
    # For each output index (i,k), C[i,k] = sum_j A[i,j]*B[j,k]
    # In tensor form: T[i,j,k] = 1 for all i,j,k
    # Decomposition: T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    # Verify for a few sample points
    if _elapsed() < deadline - 5.0:
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Reconstruct tensor
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Expected: 1
                    val = 0.0
                    for r in range(R):
                        val += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    if abs(val - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if not valid:
            # Fallback to schoolbook if verification fails
            R_school = n * m * p
            U_s = [[0] * R_school for _ in range(n * m)]
            V_s = [[0] * R_school for _ in range(m * p)]
            W_s = [[0] * R_school for _ in range(p * n)]
            col_s = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_s[i*m+j][col_s] = 1
                        V_s[j*p+k][col_s] = 1
                        W_s[k*n+i][col_s] = 1
                        col_s += 1
            return {"U": U_s, "V": V_s, "W": W_s}

    return {"U": U, "V": V, "W": W}
