import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140
    
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # Rank-56 decomposition for 2x4 * 4x8 = 2x8
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Strategy: Split into 8 independent 2x2 * 2x2 sub-problems
    # A is 2x4, split into two 2x2 blocks: A_left (cols 0-1), A_right (cols 2-3)
    # B is 4x8, split rows into top (0-1) and bottom (2-3), and cols into 4 groups of 2
    # For each (j_group, k_group):
    #   j_group in {0, 1}: selects which 2x2 block of A (and corresponding rows of B)
    #   k_group in {0, 1, 2, 3}: selects which 2 columns of B (and thus 2 columns of C)
    # Each sub-problem is 2x2 * 2x2, solved with Strassen (rank 7)
    # Total rank: 2 * 4 * 7 = 56
    
    total_rank = 56
    U = np.zeros((n * m, total_rank), dtype=int)  # shape (8, 56)
    V = np.zeros((m * p, total_rank), dtype=int)  # shape (32, 56)
    W = np.zeros((p * n, total_rank), dtype=int)  # shape (16, 56)
    
    # W index convention: W[k*n + i, r] where k is column index, i is row index
    # C[i, k] -> W[k*2 + i, r]
    
    # Strassen 2x2 rank-7 decomposition
    # Local indices for 2x2 matrices:
    # A: flat index i*2+j (i,j in {0,1}) -> 0,1,2,3
    # B: flat index j*2+k (j,k in {0,1}) -> 0,1,2,3
    # C: flat index i*2+k (i,k in {0,1}) -> 0,1,2,3 (but we map to W convention)
    
    # Strassen components:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U_local = np.zeros((4, 7), dtype=int)
    V_local = np.zeros((4, 7), dtype=int)
    W_local = np.zeros((4, 7), dtype=int)
    
    # r=0 (M1)
    U_local[0,0]=1; U_local[3,0]=1
    V_local[0,0]=1; V_local[3,0]=1
    W_local[0,0]=1; W_local[3,0]=1
    
    # r=1 (M2)
    U_local[2,1]=1; U_local[3,1]=1
    V_local[0,1]=1
    W_local[2,1]=1; W_local[3,1]=-1
    
    # r=2 (M3)
    U_local[0,2]=1
    V_local[1,2]=1; V_local[3,2]=-1
    W_local[1,2]=1; W_local[3,2]=1
    
    # r=3 (M4)
    U_local[3,3]=1
    V_local[2,3]=1; V_local[0,3]=-1
    W_local[0,3]=1; W_local[2,3]=1
    
    # r=4 (M5)
    U_local[0,4]=1; U_local[1,4]=1
    V_local[3,4]=1
    W_local[0,4]=-1; W_local[1,4]=1
    
    # r=5 (M6)
    U_local[2,5]=1; U_local[0,5]=-1
    V_local[0,5]=1; V_local[1,5]=1
    W_local[3,5]=1
    
    # r=6 (M7)
    U_local[1,6]=1; U_local[3,6]=-1
    V_local[2,6]=1; V_local[3,6]=1
    W_local[0,6]=1
    
    # Iterate over all 8 sub-problems
    for j_group in range(2):
        for k_group in range(4):
            base_r = (j_group * 4 + k_group) * 7
            
            # Map local A indices (0-3) to global U indices
            # A is 2x4, flat index = i*4 + j
            # j_group=0: cols 0,1; j_group=1: cols 2,3
            # Local: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
            a_map = [
                0 * 4 + j_group * 2,
                0 * 4 + j_group * 2 + 1,
                1 * 4 + j_group * 2,
                1 * 4 + j_group * 2 + 1
            ]
            
            # Map local B indices (0-3) to global V indices
            # B is 4x8, flat index = j*8 + k
            # j_group=0: rows 0,1; j_group=1: rows 2,3
            # k_group: cols 2*k_group, 2*k_group+1
            # Local: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1) where local j,k in {0,1}
            b_map = [
                (j_group * 2) * 8 + k_group * 2,
                (j_group * 2) * 8 + k_group * 2 + 1,
                (j_group * 2 + 1) * 8 + k_group * 2,
                (j_group * 2 + 1) * 8 + k_group * 2 + 1
            ]
            
            # Map local C indices (0-3) to global W indices
            # C is 2x8, W index = k*2 + i
            # Local: 0=(i=0,k=0), 1=(i=0,k=1), 2=(i=1,k=0), 3=(i=1,k=1)
            # Global k = 2*k_group + local_k
            # Global i = local_i
            c_map = [
                (k_group * 2) * 2 + 0,
                (k_group * 2 + 1) * 2 + 0,
                (k_group * 2) * 2 + 1,
                (k_group * 2 + 1) * 2 + 1
            ]
            
            for l in range(7):
                r = base_r + l
                for la in range(4):
                    val = U_local[la, l]
                    if val != 0:
                        U[a_map[la], r] += val
                for lb in range(4):
                    val = V_local[lb, l]
                    if val != 0:
                        V[b_map[lb], r] += val
                for lc in range(4):
                    val = W_local[lc, l]
                    if val != 0:
                        W[c_map[lc], r] += val
    
    if time.time() > DEADLINE:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
