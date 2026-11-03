import time
import numpy as np
from scipy import linalg

DIMENSIONS = [3, 4, 4]
_START_TIME = time.time()
_TIMEOUT = 115  # safety margin before 120s limit

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """
    Returns the U, V, W factors for a 2x2x2 Strassen decomposition.
    Indices:
    U: (2*2, 7) = (4, 7)
    V: (2*2, 7) = (4, 7)
    W: (2*2, 7) = (4, 7)
    
    Input A indices: 00, 01, 10, 11 -> flat 0, 1, 2, 3
    Input B indices: 00, 01, 10, 11 -> flat 0, 1, 2, 3
    Output C indices: 00, 01, 10, 11 -> flat 0, 1, 2, 3
    
    Strassen products:
    M1 = (A00 + A11)(B00 + B11)
    M2 = (A10 + A11) B00
    M3 = A00 (B10 - B11)
    M4 = A11 (B01 + B11)
    M5 = (A00 - A10)(B10 + B11)
    M6 = (A01 + A10)(B00 - B10)
    M7 = (A01 - A11)(B01 + B11)
    
    C00 = M1 + M4 - M5 - M7
    C01 = M3 + M7
    C10 = M2 + M5
    C11 = M1 - M2 + M3 + M6
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[2, 0] = 1 # C00, C10? No.
    # W maps M to C.
    # C00 gets M1. C11 gets M1.
    # Wait, W is (p*n, rank). p=2, n=2.
    # C indices: 00 (0), 01 (1), 10 (2), 11 (3).
    # M1 contributes to C00 and C11.
    W[0, 0] = 1
    W[3, 0] = 1
    
    # M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1 # C10
    W[3, 1] = -1 # C11
    
    # M3
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1 # C01
    W[3, 2] = 1 # C11
    
    # M4
    U[3, 3] = 1
    V[1, 3] = 1; V[3, 3] = 1
    W[0, 3] = 1 # C00
    
    # M5
    U[0, 4] = 1; U[2, 4] = -1
    V[2, 4] = 1; V[3, 4] = 1
    W[0, 4] = -1 # C00
    W[2, 4] = 1 # C10
    
    # M6
    U[1, 5] = 1; U[2, 5] = 1
    V[0, 5] = 1; V[2, 5] = -1
    W[3, 5] = 1 # C11
    
    # M7
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1 # C00
    W[1, 6] = 1 # C01
    
    return U, V, W

def _naive_1x4x4():
    """
    Returns U, V, W for 1x4x4 naive decomposition.
    Rank 4.
    U: (1*4, 4) = (4, 4)
    V: (4*4, 4) = (16, 4)
    W: (4*1, 4) = (4, 4)
    
    C_k = sum_j A_j B_{j,k}
    Term r (0..3): A_r * (row r of B)
    U[r, r] = 1
    V[r*4 + k, r] = 1 for all k
    W[k, r] = 1
    """
    U = np.zeros((4, 4))
    V = np.zeros((16, 4))
    W = np.zeros((4, 4))
    
    for r in range(4):
        U[r, r] = 1
        for k in range(4):
            V[r * 4 + k, r] = 1
            W[k, r] = 1
            
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=4
    
    # We construct a rank 32 decomposition.
    # Top part: 2x4x4 using 4 Strassen blocks (rank 28)
    # Bottom part: 1x4x4 using naive (rank 4)
    
    rank = 32
    U_total = np.zeros((n * m, rank))
    V_total = np.zeros((m * p, rank))
    W_total = np.zeros((p * n, rank))
    
    # --- Top Part (i=0,1) ---
    # A_top is 2x4. Indices i in {0,1}, j in {0,1,2,3}.
    # B is 4x4. Indices j in {0,1,2,3}, k in {0,1,2,3}.
    # C_top is 2x4. Indices i in {0,1}, k in {0,1,2,3}.
    
    # We split A_top into A11 (2x2, cols 0,1) and A12 (2x2, cols 2,3).
    # We split B into B11 (rows 0,1, cols 0,1), B12 (rows 0,1, cols 2,3),
    #               B21 (rows 2,3, cols 0,1), B22 (rows 2,3, cols 2,3).
    
    # Products:
    # P1 = A11 * B11
    # P2 = A12 * B21
    # P3 = A11 * B12
    # P4 = A12 * B22
    
    # C_top_left (cols 0,1) = P1 + P2
    # C_top_right (cols 2,3) = P3 + P4
    
    # We need to map the local Strassen indices to global indices.
    
    def add_strassen_block(U_local, V_local, W_local, 
                           a_rows, a_cols, 
                           b_rows, b_cols, 
                           c_rows, c_cols,
                           offset):
        """
        Add a Strassen block to the global matrices.
        a_rows: list of global i indices
        a_cols: list of global j indices
        b_rows: list of global j indices
        b_cols: list of global k indices
        c_rows: list of global i indices
        c_cols: list of global k indices
        offset: rank offset
        """
        for r in range(7):
            # U
            for li, ui in enumerate(U_local[:, r]):
                if ui != 0:
                    gi = a_rows[li // 2]
                    gj = a_cols[li % 2]
                    U_total[gi * m + gj, offset + r] += ui
                    
            # V
            for lj, vj in enumerate(V_local[:, r]):
                if vj != 0:
                    gj = b_rows[lj // 2]
                    gk = b_cols[lj % 2]
                    V_total[gj * p + gk, offset + r] += vj
                    
            # W
            for lk, wk in enumerate(W_local[:, r]):
                if wk != 0:
                    gk = c_cols[lk % 2]
                    gi = c_rows[lk // 2]
                    W_total[gk * n + gi, offset + r] += wk

    # Strassen U, V, W for one block
    U_s, V_s, W_s = _strassen_2x2()
    
    # Block 1: A11 (rows 0,1; cols 0,1) * B11 (rows 0,1; cols 0,1)
    # Contributes to C rows 0,1; cols 0,1
    add_strassen_block(U_s, V_s, W_s,
                       [0, 1], [0, 1],
                       [0, 1], [0, 1],
                       [0, 1], [0, 1],
                       0)
                       
    # Block 2: A12 (rows 0,1; cols 2,3) * B21 (rows 2,3; cols 0,1)
    # Contributes to C rows 0,1; cols 0,1
    add_strassen_block(U_s, V_s, W_s,
                       [0, 1], [2, 3],
                       [2, 3], [0, 1],
                       [0, 1], [0, 1],
                       7)
                       
    # Block 3: A11 (rows 0,1; cols 0,1) * B12 (rows 0,1; cols 2,3)
    # Contributes to C rows 0,1; cols 2,3
    add_strassen_block(U_s, V_s, W_s,
                       [0, 1], [0, 1],
                       [0, 1], [2, 3],
                       [0, 1], [2, 3],
                       14)
                       
    # Block 4: A12 (rows 0,1; cols 2,3) * B22 (rows 2,3; cols 2,3)
    # Contributes to C rows 0,1; cols 2,3
    add_strassen_block(U_s, V_s, W_s,
                       [0, 1], [2, 3],
                       [2, 3], [2, 3],
                       [0, 1], [2, 3],
                       21)

    # --- Bottom Part (i=2) ---
    # A_bot is 1x4 (row 2, cols 0,1,2,3)
    # B is 4x4
    # C_bot is 1x4 (row 2, cols 0,1,2,3)
    
    U_n, V_n, W_n = _naive_1x4x4()
    
    offset = 28
    for r in range(4):
        # U: A_bot[j, r] = A[2, j]
        # U_local is (4, 4). Row j, Col r.
        # U_total[2*m + j, offset + r] += U_n[j, r]
        for j in range(4):
            if U_n[j, r] != 0:
                U_total[2 * m + j, offset + r] += U_n[j, r]
                
        # V: B[j, k]
        # V_local is (16, 4). Row j*4+k, Col r.
        # V_total[j*p + k, offset + r] += V_n[j*4+k, r]
        for j in range(4):
            for k in range(4):
                if V_n[j*4+k, r] != 0:
                    V_total[j * p + k, offset + r] += V_n[j*4+k, r]
                    
        # W: C[k]
        # W_local is (4, 4). Row k, Col r.
        # W_total[k*n + 2, offset + r] += W_n[k, r]
        for k in range(4):
            if W_n[k, r] != 0:
                W_total[k * n + 2, offset + r] += W_n[k, r]

    # Convert to lists
    U_list = [[float(val) for val in row] for row in U_total.tolist()]
    V_list = [[float(val) for val in row] for row in V_total.tolist()]
    W_list = [[float(val) for val in row] for row in W_total.tolist()]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
