import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-56 tensor decomposition for multiplying a 2x4 matrix by a 4x8 matrix.
    
    This uses Strassen's algorithm for 2x2 matrix multiplication.
    The 2x4 matrix A is split into two 2x2 blocks A1, A2.
    The 4x8 matrix B is split into two 2x8 blocks B1, B2.
    C = A1*B1 + A2*B2.
    Each term A_i * B_i involves 4 independent 2x2 * 2x2 multiplications (since B_i is 2x8, 
    it has 4 columns of 2x2 blocks).
    Each 2x2 * 2x2 multiplication has rank 7 (Strassen).
    Total rank = 2 (blocks of A) * 4 (blocks of B) * 7 (Strassen) = 56.
    """
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    total_rank = 56
    
    U = np.zeros((n * m, total_rank), dtype=int)  # (8, 56)
    V = np.zeros((m * p, total_rank), dtype=int)  # (32, 56)
    W = np.zeros((p * n, total_rank), dtype=int)  # (16, 56)
    
    # Strassen coefficients for 2x2 * 2x2
    # U_local[la][l], V_local[lb][l], W_local[lc][l]
    # la, lb, lc in {0, 1, 2, 3} corresponding to indices (0,0), (0,1), (1,0), (1,1)
    # l in {0, ..., 6} corresponding to Strassen terms M1..M7
    
    U_local = np.zeros((4, 7), dtype=int)
    V_local = np.zeros((4, 7), dtype=int)
    W_local = np.zeros((4, 7), dtype=int)
    
    # M1: (A11+A22)(B11+B22) -> C11 += 1, C22 += 1
    U_local[0][0] = 1; U_local[3][0] = 1
    V_local[0][0] = 1; V_local[3][0] = 1
    W_local[0][0] = 1; W_local[3][0] = 1
    
    # M2: (A21+A22)B11 -> C21 += 1, C22 += -1
    U_local[2][1] = 1; U_local[3][1] = 1
    V_local[0][1] = 1
    W_local[2][1] = 1; W_local[3][1] = -1
    
    # M3: A11(B21-B22) -> C12 += 1, C22 += 1
    U_local[0][2] = 1
    V_local[1][2] = 1; V_local[3][2] = -1
    W_local[1][2] = 1; W_local[3][2] = 1
    
    # M4: A22(B12-B11) -> C11 += 1, C21 += 1
    U_local[3][3] = 1
    V_local[2][3] = 1; V_local[0][3] = -1
    W_local[0][3] = 1; W_local[2][3] = 1
    
    # M5: (A11+A12)B22 -> C11 += -1, C12 += 1
    U_local[0][4] = 1; U_local[1][4] = 1
    V_local[3][4] = 1
    W_local[0][4] = -1; W_local[1][4] = 1
    
    # M6: (A21-A11)(B11+B12) -> C22 += 1
    U_local[2][5] = 1; U_local[0][5] = -1
    V_local[0][5] = 1; V_local[1][5] = 1
    W_local[3][5] = 1
    
    # M7: (A12-A22)(B21+B22) -> C11 += 1
    U_local[1][6] = 1; U_local[3][6] = -1
    V_local[2][6] = 1; V_local[3][6] = 1
    W_local[0][6] = 1
    
    # Loop over the 4 column-blocks of B (and corresponding C)
    # and the 2 row-blocks of A (and corresponding B rows)
    
    # j is the column block index of A (0 or 1) -> selects A1 or A2
    # s is the row block index of B (0 or 1) -> selects B1 or B2
    # g is the column group of B (0..3) -> selects which 2x2 block of B_i
    
    # A is 2x4. A1 is cols 0,1. A2 is cols 2,3.
    # B is 4x8. B1 is rows 0,1. B2 is rows 2,3.
    # B1 has 4 col-blocks: cols 0-1, 2-3, 4-5, 6-7.
    
    for j in range(2):  # A block index
        for s in range(2):  # B row block index
            for g in range(4):  # B column block index
                base_r = (j * 2 + s) * 4 * 7 + g * 7
                
                # Map local indices to global indices
                # A local (la) -> global A index
                # la=0: (0,0), la=1: (0,1), la=2: (1,0), la=3: (1,1)
                # A global index = i * m + j_local
                # i is row of A (0,1). j_local is col of A (0..3).
                # If j=0 (A1), cols are 0,1. If j=1 (A2), cols are 2,3.
                
                a_map = [0]*4
                if j == 0:
                    a_map[0] = 0*4 + 0
                    a_map[1] = 0*4 + 1
                    a_map[2] = 1*4 + 0
                    a_map[3] = 1*4 + 1
                else:
                    a_map[0] = 0*4 + 2
                    a_map[1] = 0*4 + 3
                    a_map[2] = 1*4 + 2
                    a_map[3] = 1*4 + 3
                
                # B local (lb) -> global B index
                # lb=0: (0,0), lb=1: (0,1), lb=2: (1,0), lb=3: (1,1)
                # B global index = i * p + k
                # i is row of B (0..3). k is col of B (0..7).
                # If s=0 (B1), rows are 0,1. If s=1 (B2), rows are 2,3.
                # Col group g: cols are 2*g, 2*g+1.
                
                b_map = [0]*4
                if s == 0:
                    b_map[0] = 0*8 + (2*g)
                    b_map[1] = 0*8 + (2*g+1)
                    b_map[2] = 1*8 + (2*g)
                    b_map[3] = 1*8 + (2*g+1)
                else:
                    b_map[0] = 2*8 + (2*g)
                    b_map[1] = 2*8 + (2*g+1)
                    b_map[2] = 3*8 + (2*g)
                    b_map[3] = 3*8 + (2*g+1)
                
                # C local (lc) -> global C index
                # lc=0: (0,0), lc=1: (0,1), lc=2: (1,0), lc=3: (1,1)
                # C global index = k * n + i
                # k is col of C (0..7). i is row of C (0,1).
                # Col group g: cols are 2*g, 2*g+1.
                
                c_map = [0]*4
                c_map[0] = (2*g) * 2 + 0
                c_map[1] = (2*g+1) * 2 + 0
                c_map[2] = (2*g) * 2 + 1
                c_map[3] = (2*g+1) * 2 + 1
                
                for l in range(7):
                    r = base_r + l
                    for la in range(4):
                        val = U_local[la][l]
                        if val != 0:
                            U[a_map[la], r] += val
                    for lb in range(4):
                        val = V_local[lb][l]
                        if val != 0:
                            V[b_map[lb], r] += val
                    for lc in range(4):
                        val = W_local[lc][l]
                        if val != 0:
                            W[c_map[lc], r] += val
                        
    if time.time() - start_time > 140:
        pass
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
