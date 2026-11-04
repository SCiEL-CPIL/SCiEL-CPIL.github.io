import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Return the classic Strassen 2x2x2 decomposition with 7 terms.
    U, V, W are 4x7 matrices (row-major indexing).
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Term 1: M1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Term 2: M2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # Term 3: M3 = a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # Term 4: M4 = d(g-e)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[2][3] = 1
    
    # Term 5: M5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # Term 6: M6 = (c-d)(f+g)
    U[2][5] = 1; U[3][5] = -1
    V[1][5] = 1; V[2][5] = 1
    W[3][5] = 1
    
    # Term 7: M7 = (b-d)(e+f)
    U[1][6] = 1; U[3][6] = -1
    V[0][6] = 1; V[1][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _construct_5x5x5():
    n, m, p = 5, 5, 5
    nm, mp, pn = n * m, m * p, p * n
    
    # Blocks:
    # Top-Left (TL): i in {0,1}, j in {0,1}, k in {0,1} -> Strassen (7)
    # Bottom-Right (BR): i in {2,3,4}, j in {2,3,4}, k in {2,3,4} -> Schoolbook (27)
    # Top-Right (TR): i in {0,1}, j in {0..4}, k in {2,3,4} -> Schoolbook (30)
    # Bottom-Left (BL): i in {2,3,4}, j in {0..4}, k in {0,1} -> Schoolbook (30)
    
    total_rank = 7 + 27 + 30 + 30
    U = [[0] * total_rank for _ in range(nm)]
    V = [[0] * total_rank for _ in range(mp)]
    W = [[0] * total_rank for _ in range(pn)]
    
    col = 0
    
    # 1. TL Block: Strassen
    U_s, V_s, W_s = _strassen_2x2()
    for c in range(7):
        for r in range(4):
            # Local indices 0,1 map to Global 0,1
            U[r][col + c] = U_s[r][c]
            V[r][col + c] = V_s[r][c]
            W[r][col + c] = W_s[r][c]
    col += 7
    
    # 2. BR Block: Schoolbook for 3x3
    # i, j, k in {2, 3, 4}
    for i in range(2, 5):
        for j in range(2, 5):
            for k in range(2, 5):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # 3. TR Block: Schoolbook for 2x5x3
    # i in {0,1}, j in {0..4}, k in {2,3,4}
    for i in range(0, 2):
        for j in range(0, 5):
            for k in range(2, 5):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # 4. BL Block: Schoolbook for 3x5x2
    # i in {2,3,4}, j in {0..4}, k in {0,1}
    for i in range(2, 5):
        for j in range(0, 5):
            for k in range(0, 2):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}, total_rank

def _verify_fast(n, m, p, U, V, W):
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    rank = U_arr.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U_arr[i * m + j]
                v_row = V_arr[j * p + k]
                w_row = W_arr[k * n + i]
                val = np.dot(u_row, v_row * w_row)
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    result, rank = _construct_5x5x5()
    
    if _verify_fast(n, m, p, result["U"], result["V"], result["W"]):
        return result
    
    # Fallback: Pure schoolbook (rank 125)
    U = [[0] * 125 for _ in range(25)]
    V = [[0] * 125 for _ in range(25)]
    W = [[0] * 125 for _ in range(25)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
