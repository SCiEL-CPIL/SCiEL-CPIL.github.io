import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    
    # Rank 56 decomposition using 8 independent Strassen 2x2x2 blocks.
    # This is the best known simple decomposition for 2x4x8.
    # Lower bound is 32, but 56 is achieved by 8 * 7.
    
    # Strassen 2x2x2 components
    # U_s, V_s, W_s are 4x7 matrices
    U_s = np.zeros((4, 7), dtype=int)
    V_s = np.zeros((4, 7), dtype=int)
    W_s = np.zeros((4, 7), dtype=int)
    
    # M1
    U_s[0,0]=1; U_s[3,0]=1
    V_s[0,0]=1; V_s[3,0]=1
    W_s[0,0]=1; W_s[3,0]=1
    # M2
    U_s[2,1]=1; U_s[3,1]=1
    V_s[0,1]=1
    W_s[2,1]=1; W_s[3,1]=-1
    # M3
    U_s[0,2]=1
    V_s[1,2]=1; V_s[3,2]=-1
    W_s[1,2]=1; W_s[3,2]=1
    # M4
    U_s[3,3]=1
    V_s[2,3]=1; V_s[0,3]=-1
    W_s[0,3]=1; W_s[2,3]=1
    # M5
    U_s[0,4]=1; U_s[1,4]=1
    V_s[3,4]=1
    W_s[0,4]=-1; W_s[1,4]=1
    # M6
    U_s[2,5]=1; U_s[0,5]=-1
    V_s[0,5]=1; V_s[1,5]=1
    W_s[3,5]=1
    # M7
    U_s[1,6]=1; U_s[3,6]=-1
    V_s[2,6]=1; V_s[3,6]=1
    W_s[0,6]=1

    R = 56
    U = np.zeros((n*m, R), dtype=int)
    V = np.zeros((m*p, R), dtype=int)
    W = np.zeros((p*n, R), dtype=int)
    
    # Indices mapping
    # A is 2x4. Flattened: A[i, j] -> i*4 + j
    # B is 4x8. Flattened: B[j, k] -> j*8 + k
    # C is 2x8. Flattened: C[i, k] -> k*2 + i (Trace convention W[k*n+i, r])
    
    # We iterate over 4 column-pairs of A (j=0,1,2,3 in the original 2x4 split logic? No.)
    # The previous logic:
    # j in 0..3 corresponds to column pair of the 4-column dimension of A?
    # A is 2x4.
    # We split A into two 2x2 blocks: A_left (cols 0,1) and A_right (cols 2,3).
    # We split B into two 2x8 blocks: B_top (rows 0,1) and B_bottom (rows 2,3).
    # C = A_left B_top + A_right B_bottom.
    
    # For A_left B_top (2x2 * 2x8):
    # B_top is 2x8. We split B_top into 4 2x2 blocks: B_top[0:2, 0:2], B_top[0:2, 2:4], etc.
    # Let's call the column pair index of B_top 'k_pair' (0..3).
    # Each 2x2 * 2x2 uses 7 rank terms.
    # Total 4 * 7 = 28 terms for A_left B_top.
    
    # For A_right B_bottom (2x2 * 2x8):
    # Similarly, 28 terms.
    
    # Total 56.
    
    # Mapping details:
    # For block (s, k_pair) where s in {0,1} (0: A_left/B_top, 1: A_right/B_bottom)
    # and k_pair in {0,1,2,3} (column pair index in the 8-column dimension of B)
    
    for s in range(2):
        for k_pair in range(4):
            base_r = (s * 4 + k_pair) * 7
            
            # A indices
            # s=0: A[0, 0:2], A[1, 0:2] -> indices 0, 1, 4, 5
            # s=1: A[0, 2:4], A[1, 2:4] -> indices 2, 3, 6, 7
            if s == 0:
                a_idx = [0, 1, 4, 5]
            else:
                a_idx = [2, 3, 6, 7]
            
            # B indices
            # B is 4x8.
            # s=0: Rows 0,1. Cols 2*k_pair, 2*k_pair+1.
            #   B[0, 2*k_pair], B[0, 2*k_pair+1], B[1, 2*k_pair], B[1, 2*k_pair+1]
            #   Indices: 0*8 + 2*k_pair, 0*8 + 2*k_pair+1, 1*8 + 2*k_pair, 1*8 + 2*k_pair+1
            # s=1: Rows 2,3. Cols 2*k_pair, 2*k_pair+1.
            #   B[2, 2*k_pair], B[2, 2*k_pair+1], B[3, 2*k_pair], B[3, 2*k_pair+1]
            #   Indices: 2*8 + 2*k_pair, 2*8 + 2*k_pair+1, 3*8 + 2*k_pair, 3*8 + 2*k_pair+1
            if s == 0:
                b_idx = [0*8 + 2*k_pair, 0*8 + 2*k_pair + 1, 1*8 + 2*k_pair, 1*8 + 2*k_pair + 1]
            else:
                b_idx = [2*8 + 2*k_pair, 2*8 + 2*k_pair + 1, 3*8 + 2*k_pair, 3*8 + 2*k_pair + 1]
            
            # C indices
            # C is 2x8.
            # Output of this block is 2x2 submatrix of C.
            # Rows: 0, 1. Cols: 2*k_pair, 2*k_pair+1.
            # C[0, 2*k_pair], C[0, 2*k_pair+1], C[1, 2*k_pair], C[1, 2*k_pair+1]
            # Trace convention: W[k*n+i, r] -> index = k*2 + i.
            # C(0, 2*k_pair) -> 2*k_pair * 2 + 0 = 4*k_pair
            # C(0, 2*k_pair+1) -> (2*k_pair+1) * 2 + 0 = 4*k_pair + 2
            # C(1, 2*k_pair) -> 2*k_pair * 2 + 1 = 4*k_pair + 1
            # C(1, 2*k_pair+1) -> (2*k_pair+1) * 2 + 1 = 4*k_pair + 3
            # Strassen W_local order: 0:C00, 1:C01, 2:C10, 3:C11
            # So c_idx should map:
            # 0 -> C(0, 2*k_pair) = 4*k_pair
            # 1 -> C(0, 2*k_pair+1) = 4*k_pair + 2
            # 2 -> C(1, 2*k_pair) = 4*k_pair + 1
            # 3 -> C(1, 2*k_pair+1) = 4*k_pair + 3
            c_idx = [4*k_pair, 4*k_pair + 2, 4*k_pair + 1, 4*k_pair + 3]
            
            for l in range(7):
                r = base_r + l
                for la in range(4):
                    val = U_s[la, l]
                    if val != 0:
                        U[a_idx[la], r] += val
                for lb in range(4):
                    val = V_s[lb, l]
                    if val != 0:
                        V[b_idx[lb], r] += val
                for lc in range(4):
                    val = W_s[lc, l]
                    if val != 0:
                        W[c_idx[lc], r] += val

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
