import time
import sys

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    # U: (n*m, rank) -> (8, rank)
    # V: (m*p, rank) -> (32, rank)
    # W: (p*n, rank) -> (16, rank)
    
    # Strassen 2x2x2 decomposition
    # Local indices for A, B, C are 0..3
    # A: i*2+j, B: j*2+k, C: i*2+k
    # W global index: k*n+i
    
    U_local = [[0]*7 for _ in range(4)]
    V_local = [[0]*7 for _ in range(4)]
    W_local = [[0]*7 for _ in range(4)]
    
    # r=0 (M1)
    U_local[0][0] = 1; U_local[3][0] = 1
    V_local[0][0] = 1; V_local[3][0] = 1
    W_local[0][0] = 1
    W_local[3][0] = 1
    
    # r=1 (M2)
    U_local[2][1] = 1; U_local[3][1] = 1
    V_local[0][1] = 1
    W_local[2][1] = 1
    W_local[3][1] = -1
    
    # r=2 (M3)
    U_local[0][2] = 1
    V_local[1][2] = 1; V_local[3][2] = -1
    W_local[1][2] = 1
    W_local[3][2] = 1
    
    # r=3 (M4)
    U_local[3][3] = 1
    V_local[2][3] = 1; V_local[0][3] = -1
    W_local[0][3] = 1
    W_local[2][3] = 1
    
    # r=4 (M5)
    U_local[0][4] = 1; U_local[1][4] = 1
    V_local[3][4] = 1
    W_local[0][4] = -1
    W_local[1][4] = 1
    
    # r=5 (M6)
    U_local[2][5] = 1; U_local[0][5] = -1
    V_local[0][5] = 1; V_local[1][5] = 1
    W_local[3][5] = 1
    
    # r=6 (M7)
    U_local[1][6] = 1; U_local[3][6] = -1
    V_local[2][6] = 1; V_local[3][6] = 1
    W_local[0][6] = 1

    total_rank = 56
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # s: 0 for A cols 0,1 / B rows 0,1
    # s: 1 for A cols 2,3 / B rows 2,3
    # j: 0..3 for C col pairs 2j, 2j+1
    
    for s in range(2):
        for j in range(4):
            base_r = (s * 4 + j) * 7
            
            # A map
            a_map = [0]*4
            if s == 0:
                a_map[0] = 0*4 + 0
                a_map[1] = 0*4 + 1
                a_map[2] = 1*4 + 0
                a_map[3] = 1*4 + 1
            else:
                a_map[0] = 0*4 + 2
                a_map[1] = 0*4 + 3
                a_map[2] = 1*4 + 2
                a_map[3] = 1*4 + 3
                
            # B map
            b_map = [0]*4
            row_offset = s * 2
            b_map[0] = (row_offset)*8 + (2*j)
            b_map[1] = (row_offset)*8 + (2*j+1)
            b_map[2] = (row_offset+1)*8 + (2*j)
            b_map[3] = (row_offset+1)*8 + (2*j+1)
            
            # C map (W index: k*n+i)
            c_map = [0]*4
            c_map[0] = (2*j) * 2 + 0
            c_map[1] = (2*j+1) * 2 + 0
            c_map[2] = (2*j) * 2 + 1
            c_map[3] = (2*j+1) * 2 + 1

            for l in range(7):
                r = base_r + l
                for la in range(4):
                    val = U_local[la][l]
                    if val != 0:
                        U[a_map[la]][r] += val
                for lb in range(4):
                    val = V_local[lb][l]
                    if val != 0:
                        V[b_map[lb]][r] += val
                for lc in range(4):
                    val = W_local[lc][l]
                    if val != 0:
                        W[c_map[lc]][r] += val
                        
    if time.time() - start_time > 140:
        pass
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
