"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    # Rank 22 decomposition
    # 7 for A_L B_T1 (2x2x2 Strassen)
    # 7 for A_L B_T2 (2x2x2 Strassen)
    # 8 for A_R B_B  (2x1x4 Schoolbook)
    rank = 22
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val, col):
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val

    # Indices:
    # A: 2x3. A[i,j] index i*3+j
    # B: 3x4. B[j,k] index j*4+k
    # C: 2x4. C[i,k] index k*2+i (Wait, W[k*n+i, c] -> W index is k*n+i)
    # Note: The problem states W[k*n+i, r]. So W row index is k*n + i.
    # n=2. So W index is 2*k + i.
    
    # Block 1: A_L B_T1
    # A_L is A[:, 0:2] (2x2)
    # B_T1 is B[0:2, 0:2] (2x2)
    # Strassen for C1 (2x2) corresponding to C[:, 0:2]
    # C1[i,k] = C[i, k] for k in 0,1.
    
    # Strassen terms for 2x2x2:
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B01-B11)
    # M4: A11(B10-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    # Map to global indices
    # A_L indices:
    # A00 -> A[0,0] -> idx 0
    # A01 -> A[0,1] -> idx 1
    # A10 -> A[1,0] -> idx 3
    # A11 -> A[1,1] -> idx 4
    
    # B_T1 indices:
    # B00 -> B[0,0] -> idx 0
    # B01 -> B[0,1] -> idx 1
    # B10 -> B[1,0] -> idx 4
    # B11 -> B[1,1] -> idx 5
    
    # C1 indices (W):
    # C00 -> C[0,0] -> W idx 0*2+0 = 0
    # C01 -> C[0,1] -> W idx 1*2+0 = 2
    # C10 -> C[1,0] -> W idx 0*2+1 = 1
    # C11 -> C[1,1] -> W idx 1*2+1 = 3
    
    col = 0
    # M1
    add_term(0, 1, 0, 1, 0, 1, col) # A00, B00, C00
    add_term(4, 1, 0, 1, 0, 1, col) # A11, B00, C00 (Wait, M1 is (A00+A11)(B00+B11))
    # Correct Strassen:
    # M1 = (A00+A11)(B00+B11)
    # U: A00(1), A11(1). V: B00(1), B11(1).
    # W: C00(1), C01(1), C10(1), C11(-1) ? No.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's fill U, V, W for each M
    
    # M1
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[5][col] = 1
    W[0][col] = 1  # C00
    W[2][col] = 0  # C01
    W[1][col] = 0  # C10
    W[3][col] = 1  # C11
    col += 1
    
    # M2
    U[3][col] = 1; U[4][col] = 1 # A10+A11
    V[0][col] = 1                # B00
    W[0][col] = 0
    W[2][col] = 0
    W[1][col] = 1  # C10
    W[3][col] = -1 # C11
    col += 1
    
    # M3
    U[0][col] = 1                # A00
    V[1][col] = 1; V[5][col] = -1 # B01-B11
    W[0][col] = 0
    W[2][col] = 1  # C01
    W[1][col] = 0
    W[3][col] = 1  # C11
    col += 1
    
    # M4
    U[4][col] = 1                # A11
    V[4][col] = 1; V[0][col] = -1 # B10-B00
    W[0][col] = 1  # C00
    W[2][col] = 0
    W[1][col] = 1  # C10
    W[3][col] = 0
    col += 1
    
    # M5
    U[0][col] = 1; U[1][col] = 1 # A00+A01
    V[5][col] = 1                # B11
    W[0][col] = -1 # C00
    W[2][col] = 1  # C01
    W[1][col] = 0
    W[3][col] = 0
    col += 1
    
    # M6
    U[3][col] = 1; U[0][col] = -1 # A10-A00
    V[0][col] = 1; V[1][col] = 1  # B00+B01
    W[0][col] = 0
    W[2][col] = 0
    W[1][col] = 0
    W[3][col] = 1  # C11
    col += 1
    
    # M7
    U[1][col] = 1; U[4][col] = -1 # A01-A11
    V[4][col] = 1; V[5][col] = 1  # B10+B11
    W[0][col] = 1  # C00
    W[2][col] = 0
    W[1][col] = 0
    W[3][col] = 0
    col += 1
    
    # Block 2: A_L B_T2
    # B_T2 is B[0:2, 2:4] (2x2)
    # B indices:
    # B02 -> B[0,2] -> idx 2
    # B03 -> B[0,3] -> idx 3
    # B12 -> B[1,2] -> idx 6
    # B13 -> B[1,3] -> idx 7
    
    # C2 indices (W):
    # C02 -> C[0,2] -> W idx 2*2+0 = 4
    # C03 -> C[0,3] -> W idx 3*2+0 = 6
    # C12 -> C[1,2] -> W idx 2*2+1 = 5
    # C13 -> C[1,3] -> W idx 3*2+1 = 7
    
    # M1
    U[0][col] = 1; U[4][col] = 1
    V[2][col] = 1; V[7][col] = 1
    W[4][col] = 1
    W[6][col] = 0
    W[5][col] = 0
    W[7][col] = 1
    col += 1
    
    # M2
    U[3][col] = 1; U[4][col] = 1
    V[2][col] = 1
    W[4][col] = 0
    W[6][col] = 0
    W[5][col] = 1
    W[7][col] = -1
    col += 1
    
    # M3
    U[0][col] = 1
    V[3][col] = 1; V[7][col] = -1
    W[4][col] = 0
    W[6][col] = 1
    W[5][col] = 0
    W[7][col] = 1
    col += 1
    
    # M4
    U[4][col] = 1
    V[6][col] = 1; V[2][col] = -1
    W[4][col] = 1
    W[6][col] = 0
    W[5][col] = 1
    W[7][col] = 0
    col += 1
    
    # M5
    U[0][col] = 1; U[1][col] = 1
    V[7][col] = 1
    W[4][col] = -1
    W[6][col] = 1
    W[5][col] = 0
    W[7][col] = 0
    col += 1
    
    # M6
    U[3][col] = 1; U[0][col] = -1
    V[2][col] = 1; V[3][col] = 1
    W[4][col] = 0
    W[6][col] = 0
    W[5][col] = 0
    W[7][col] = 1
    col += 1
    
    # M7
    U[1][col] = 1; U[4][col] = -1
    V[6][col] = 1; V[7][col] = 1
    W[4][col] = 1
    W[6][col] = 0
    W[5][col] = 0
    W[7][col] = 0
    col += 1
    
    # Block 3: A_R B_B
    # A_R is A[:, 2] (2x1)
    # A02 -> idx 2
    # A12 -> idx 5
    
    # B_B is B[2, :] (1x4)
    # B20 -> idx 8
    # B21 -> idx 9
    # B22 -> idx 10
    # B23 -> idx 11
    
    # C indices:
    # C00 -> W 0
    # C01 -> W 2
    # C02 -> W 4
    # C03 -> W 6
    # C10 -> W 1
    # C11 -> W 3
    # C12 -> W 5
    # C13 -> W 7
    
    # Z1: A02 B20 -> C00
    U[2][col] = 1
    V[8][col] = 1
    W[0][col] = 1
    col += 1
    
    # Z2: A02 B21 -> C01
    U[2][col] = 1
    V[9][col] = 1
    W[2][col] = 1
    col += 1
    
    # Z3: A02 B22 -> C02
    U[2][col] = 1
    V[10][col] = 1
    W[4][col] = 1
    col += 1
    
    # Z4: A02 B23 -> C03
    U[2][col] = 1
    V[11][col] = 1
    W[6][col] = 1
    col += 1
    
    # Z5: A12 B20 -> C10
    U[5][col] = 1
    V[8][col] = 1
    W[1][col] = 1
    col += 1
    
    # Z6: A12 B21 -> C11
    U[5][col] = 1
    V[9][col] = 1
    W[3][col] = 1
    col += 1
    
    # Z7: A12 B22 -> C12
    U[5][col] = 1
    V[10][col] = 1
    W[5][col] = 1
    col += 1
    
    # Z8: A12 B23 -> C13
    U[5][col] = 1
    V[11][col] = 1
    W[7][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
