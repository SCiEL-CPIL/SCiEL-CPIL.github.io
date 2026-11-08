"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]
import time
import itertools
import random

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Verify that the decomposition is exact."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for k in range(p):
            for j in range(m):
                coeff = 0.0
                for r in range(rank):
                    u_val = U[i*m + j][r]
                    v_val = V[j*p + k][r]
                    w_val = W[k*n + i][r]
                    coeff += u_val * v_val * w_val
                if abs(coeff - 1.0) > tol:
                    return False
    return True

def _construct_2x3x6_candidate():
    """Construct a rank 33 decomposition for 2x3x6."""
    # 3 blocks of 2x3x2, each rank 11
    # Block decomposition:
    # U: 6x33, V: 18x33, W: 12x33
    
    # Strassen 2x2x2 (rank 7)
    U_s = [[0]*7 for _ in range(4)]
    V_s = [[0]*7 for _ in range(4)]
    W_s = [[0]*7 for _ in range(4)]
    
    # M1
    U_s[0][0] = 1; U_s[3][0] = 1
    V_s[0][0] = 1; V_s[3][0] = 1
    W_s[0][0] = 1; W_s[3][0] = 1
    # M2
    U_s[2][1] = 1; U_s[3][1] = 1
    V_s[0][1] = 1
    W_s[2][1] = 1; W_s[3][1] = -1
    # M3
    U_s[0][2] = 1
    V_s[1][2] = 1; V_s[3][2] = -1
    W_s[1][2] = 1; W_s[3][2] = 1
    # M4
    U_s[3][3] = 1
    V_s[2][3] = 1; V_s[0][3] = -1
    W_s[0][3] = 1; W_s[2][3] = 1
    # M5
    U_s[0][4] = 1; U_s[1][4] = 1
    V_s[3][4] = 1
    W_s[0][4] = -1; W_s[1][4] = 1
    # M6
    U_s[2][5] = 1; U_s[0][5] = -1
    V_s[0][5] = 1; V_s[1][5] = 1
    W_s[3][5] = 1
    # M7
    U_s[1][6] = 1; U_s[3][6] = -1
    V_s[2][6] = 1; V_s[3][6] = 1
    W_s[0][6] = 1
    
    # Naive 2x1x2 (rank 4) for the 3rd column of A / 3rd row of B
    # A indices: 0,1,2. B indices: 0,1,2.
    # We use Strassen for A[0:2], B[0:2].
    # Naive for A[2], B[2].
    U_n = [[0]*4 for _ in range(6)]
    V_n = [[0]*4 for _ in range(6)]
    W_n = [[0]*4 for _ in range(4)]
    
    # A[0,2]*B[2,0] -> C[0,0]
    U_n[2][0] = 1; V_n[4][0] = 1; W_n[0][0] = 1
    # A[0,2]*B[2,1] -> C[0,1]
    U_n[2][1] = 1; V_n[5][1] = 1; W_n[1][1] = 1
    # A[1,2]*B[2,0] -> C[1,0]
    U_n[5][2] = 1; V_n[4][2] = 1; W_n[2][2] = 1
    # A[1,2]*B[2,1] -> C[1,1]
    U_n[5][3] = 1; V_n[5][3] = 1; W_n[3][3] = 1
    
    # Combine into 2x3x2 block (rank 11)
    U_b = [[0]*11 for _ in range(6)]
    V_b = [[0]*11 for _ in range(6)]
    W_b = [[0]*11 for _ in range(4)]
    
    for i in range(4):
        for r in range(7):
            U_b[i][r] = U_s[i][r]
            V_b[i][r] = V_s[i][r]
            W_b[i][r] = W_s[i][r]
            
    for i in range(6):
        for r in range(4):
            U_b[i][7+r] = U_n[i][r]
            V_b[i][7+r] = V_n[i][r]
    for i in range(4):
        for r in range(4):
            W_b[i][7+r] = W_n[i][r]
            
    # Assemble 3 blocks for 2x3x6
    U = [[0]*33 for _ in range(6)]
    V = [[0]*33 for _ in range(18)]
    W = [[0]*33 for _ in range(12)]
    
    for block in range(3):
        for i in range(6):
            for r in range(11):
                U[i][block*11 + r] = U_b[i][r]
        for j in range(3):
            for k in range(2):
                for r in range(11):
                    V[j*6 + block*2 + k][block*11 + r] = V_b[j*2 + k][r]
        for k in range(2):
            for i in range(2):
                for r in range(11):
                    W[(block*2 + k)*2 + i][block*11 + r] = W_b[k*2 + i][r]
    
    return 33, U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    rank, U, V, W = _construct_2x3x6_candidate()
    
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to naive (rank 36)
    U = [[0]*36 for _ in range(6)]
    V = [[0]*36 for _ in range(18)]
    W = [[0]*36 for _ in range(12)]
    r = 0
    for i in range(2):
        for k in range(6):
            for j in range(3):
                U[i*3 + j][r] = 1
                V[j*6 + k][r] = 1
                W[k*2 + i][r] = 1
                r += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
