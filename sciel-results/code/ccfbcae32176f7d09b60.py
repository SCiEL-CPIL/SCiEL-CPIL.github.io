import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_222_factors():
    """
    Returns U, V, W for rank-7 decomposition of 2x2x2 matrix multiplication.
    U, V, W are 4x7 matrices.
    Indices:
    U row: i*2 + j
    V row: j*2 + k
    W row: k*2 + i
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00 + A11)(B00 + B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1  # C00
    W[3][0] = 1  # C11
    
    # M2 = (A10 + A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1  # C10
    W[3][1] = -1 # C11
    
    # M3 = A00(B01 - B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1  # C01
    W[3][2] = 1  # C11
    
    # M4 = A11(B10 - B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1  # C00
    W[2][3] = 1  # C10
    
    # M5 = (A00 + A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1 # C00
    W[1][4] = 1  # C01
    
    # M6 = (A10 - A00)(B00 + B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1  # C11
    
    # M7 = (A01 - A11)(B10 + B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1  # C00
    
    return U, V, W

def _place_strassen_block(global_U, global_V, global_W, 
                          local_U, local_V, local_W,
                          i0, j0, k0, col_offset):
    """
    Places a 2x2x2 Strassen block into the global matrices.
    i0, j0, k0 are the starting indices for i, j, k.
    col_offset is the starting column in the global matrices.
    """
    # Local indices:
    # u_idx: li*2 + lj
    # v_idx: lj*2 + lk
    # w_idx: lk*2 + li
    
    for c in range(7):
        for li in range(2):
            for lj in range(2):
                local_u_idx = li * 2 + lj
                global_i = i0 + li
                global_j = j0 + lj
                global_u_row = global_i * 5 + global_j
                global_U[global_u_row][col_offset + c] = local_U[local_u_idx][c]
                
        for lj in range(2):
            for lk in range(2):
                local_v_idx = lj * 2 + lk
                global_j = j0 + lj
                global_k = k0 + lk
                global_v_row = global_j * 5 + global_k
                global_V[global_v_row][col_offset + c] = local_V[local_v_idx][c]
                
        for lk in range(2):
            for li in range(2):
                local_w_idx = lk * 2 + li
                global_k = k0 + lk
                global_i = i0 + li
                global_w_row = global_k * 3 + global_i
                global_W[global_w_row][col_offset + c] = local_W[local_w_idx][c]

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    deadline = time.time() + 100  # safety margin
    
    # Define the 4 Strassen blocks
    # (i0, j0, k0)
    blocks = [
        (0, 0, 0),
        (0, 0, 2),
        (0, 2, 0),
        (0, 2, 2)
    ]
    
    # Calculate total rank
    # 4 blocks * 7 rank = 28
    # Remaining tuples: 75 - 4*8 = 43
    # Total rank = 28 + 43 = 71
    rank = 71
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Get Strassen factors once
    U_s, V_s, W_s = _strassen_222_factors()
    
    # Place Strassen blocks
    for (i0, j0, k0) in blocks:
        _place_strassen_block(U, V, W, U_s, V_s, W_s, i0, j0, k0, col)
        col += 7
        
    # Place Schoolbook terms for the remaining indices
    # We need to identify which (i,j,k) tuples are NOT in the Strassen blocks.
    # Strassen blocks cover:
    # i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}
    # Specifically:
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    # Block 2: i in {0,1}, j in {0,1}, k in {2,3}
    # Block 3: i in {0,1}, j in {2,3}, k in {0,1}
    # Block 4: i in {0,1}, j in {2,3}, k in {2,3}
    
    # So, if i in {0,1} AND j in {0,1,2,3} AND k in {0,1,2,3}, it is covered.
    # Otherwise, it needs schoolbook.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 4 and k < 4:
                    # Covered by Strassen blocks
                    continue
                # Schoolbook term
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    if col != rank:
        # Fallback to pure schoolbook if logic error
        # This should not happen if the counting is correct
        # 4*8 = 32 covered. 75-32=43 remaining. 28+43=71.
        U = [[0] * 75 for _ in range(n * m)]
        V = [[0] * 75 for _ in range(m * p)]
        W = [[0] * 75 for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
        
    # Verification
    try:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(W_np[k*n + i, :], U_np[i*m + j, :] * V_np[j*p + k, :])
                    if abs(s - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        if not valid:
            # Fallback to schoolbook
            U = [[0] * 75 for _ in range(n * m)]
            V = [[0] * 75 for _ in range(m * p)]
            W = [[0] * 75 for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1
                        col += 1
                        if time.time() > deadline:
                            break
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
            return {"U": U, "V": V, "W": W}
    except Exception:
        # Fallback to schoolbook
        U = [[0] * 75 for _ in range(n * m)]
        V = [[0] * 75 for _ in range(m * p)]
        W = [[0] * 75 for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
