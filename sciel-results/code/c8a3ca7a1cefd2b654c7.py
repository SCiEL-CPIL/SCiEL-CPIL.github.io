import time
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 2, 4, 8
    
    # Strassen decomposition for 2x2x2
    # U, V, W are 2x7 matrices
    U_local = np.zeros((2, 7), dtype=int)
    V_local = np.zeros((2, 7), dtype=int)
    W_local = np.zeros((2, 7), dtype=int)
    
    # r=0
    U_local[0,0] = 1; U_local[1,0] = 1
    V_local[0,0] = 1; V_local[1,0] = 1
    W_local[0,0] = 1; W_local[1,0] = 1
    
    # r=1
    U_local[1,1] = 1
    V_local[0,1] = 1
    W_local[1,1] = 1; W_local[0,1] = -1
    
    # r=2
    U_local[0,2] = 1
    V_local[1,2] = 1; V_local[0,2] = -1
    W_local[0,2] = 1; W_local[1,2] = 1
    
    # r=3
    U_local[1,3] = 1
    V_local[1,3] = 1; V_local[0,3] = -1
    W_local[1,3] = 1; W_local[0,3] = 1
    
    # r=4
    U_local[0,4] = 1; U_local[1,4] = 1
    V_local[1,4] = 1
    W_local[0,4] = -1; W_local[1,4] = 1
    
    # r=5
    U_local[1,5] = 1; U_local[0,5] = -1
    V_local[0,5] = 1; V_local[1,5] = 1
    W_local[1,5] = 1
    
    # r=6
    U_local[1,6] = 1; U_local[0,6] = -1
    V_local[1,6] = 1; V_local[0,6] = 1
    W_local[0,6] = 1
    
    total_rank = 56
    U = np.zeros((n*m, total_rank), dtype=int)
    V = np.zeros((m*p, total_rank), dtype=int)
    W = np.zeros((p*n, total_rank), dtype=int)
    
    # 8 blocks: 2 j-groups, 4 k-groups
    for gj in range(2):
        for gk in range(4):
            block_id = gj * 4 + gk
            for l in range(7):
                r = block_id * 7 + l
                
                # Map local indices to global indices
                for li in range(2):
                    gi = li
                    val = U_local[li, l]
                    if val != 0:
                        U[gi*m + (2*gj + 0), r] += val # Wait, U depends on i and j
                        # U index is i*m + j
                        # Here j is local j' which is 0 or 1
                        # So global j is 2*gj + j'
                        # But U_local is 2x7, so li is the local i
                        # We need to iterate over local j' as well?
                        # No, U_local[li, l] is the coefficient for U[i, j] in the l-th term.
                        # But U_local is defined for a 2x2x2 block, so it has indices i, j, k.
                        # My U_local definition above is incorrect. It should be 2x2x7 or similar.
                        # Let's redefine U_local as a 3D tensor or a flattened 2D matrix.
                        # U[i, j, l] for i,j in 0..1, l in 0..6.
                        pass
                
                # Correct approach:
                # U_local is a 2x2x7 tensor? No, U is (n*m, rank) = (4, rank) for the block.
                # So U_local should be 4x7.
                # Let's rebuild U_local, V_local, W_local as 4x7, 4x7, 4x7 for a 2x2x2 block.
                # But the block is 2x2x2, so U is 2*2=4 rows, V is 2*2=4 rows, W is 2*2=4 rows.
                # Wait, the dimensions of U, V, W for a 2x2x2 block are:
                # U: (2*2, 7) = (4, 7)
                # V: (2*2, 7) = (4, 7)
                # W: (2*2, 7) = (4, 7)
                
                # Let's define the local matrices properly.
                # For a 2x2x2 block, the indices are i, j, k in {0,1}.
                # U[i*2+j, l], V[j*2+k, l], W[k*2+i, l].
                
                pass
                
    # Let's restart the local matrix construction.
    U_local = np.zeros((4, 7), dtype=int)
    V_local = np.zeros((4, 7), dtype=int)
    W_local = np.zeros((4, 7), dtype=int)
    
    # r=0
    # U[i,j], V[j,k], W[k,i]
    # Term 0: (A00+A11)(B00+B11) -> C00 += M0, C11 += M0
    # M0 = (A00+A11)(B00+B11)
    # U: A00, A11
    # V: B00, B11
    # W: C00, C11
    U_local[0,0] = 1; U_local[3,0] = 1 # i=0,j=0 and i=1,j=1
    V_local[0,0] = 1; V_local[3,0] = 1 # j=0,k=0 and j=1,k=1
    W_local[0,0] = 1; W_local[3,0] = 1 # k=0,i=0 and k=1,i=1
    
    # r=1
    # M1 = (A10+A11)B00
    # U: A10, A11
    # V: B00
    # W: C10, C11
    U_local[2,1] = 1; U_local[3,1] = 1
    V_local[0,1] = 1
    W_local[2,1] = 1; W_local[3,1] = -1 # C10 += M1, C11 -= M1? No.
    # C10 = A10 B00 + A11 B10
    # C11 = A10 B01 + A11 B11
    # M1 = (A10+A11)B00 = A10 B00 + A11 B00
    # This doesn't directly give C10 or C11.
    # Strassen for 2x2x2:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B10-B00)
    # M3 = A11(B01-B10)
    # M4 = (A00+A10)B11
    # M5 = (A01-A11)(B10+B11)
    # M6 = (A10-A00)(B00+B01)
    # C00 = M0 + M3 - M5 + M6
    # C01 = M4 + M5
    # C10 = M1 + M2
    # C11 = M0 - M1 + M4 + M2
    
    # Let's use these formulas.
    # U_local[i*2+j, l], V_local[j*2+k, l], W_local[k*2+i, l]
    
    # r=0 (M0)
    U_local[0,0] = 1; U_local[3,0] = 1
    V_local[0,0] = 1; V_local[3,0] = 1
    W_local[0,0] = 1; W_local[3,0] = 1
    
    # r=1 (M1)
    U_local[2,1] = 1; U_local[3,1] = 1
    V_local[0,1] = 1
    W_local[2,1] = 1; W_local[3,1] = -1
    
    # r=2 (M2)
    U_local[0,2] = 1
    V_local[2,2] = 1; V_local[0,2] = -1
    W_local[2,2] = 1; W_local[3,2] = 1
    
    # r=3 (M3)
    U_local[3,3] = 1
    V_local[1,3] = 1; V_local[2,3] = -1
    W_local[0,3] = 1; W_local[2,3] = 1
    
    # r=4 (M4)
    U_local[0,4] = 1; U_local[1,4] = 1
    V_local[3,4] = 1
    W_local[0,4] = -1; W_local[1,4] = 1
    
    # r=5 (M5)
    U_local[1,5] = 1; U_local[0,5] = -1
    V_local[2,5] = 1; V_local[3,5] = 1
    W_local[3,5] = 1
    
    # r=6 (M6)
    U_local[1,6] = 1; U_local[0,6] = -1
    V_local[0,6] = 1; V_local[1,6] = 1
    W_local[0,6] = 1
    
    # Now map to global
    for gj in range(2):
        for gk in range(4):
            block_id = gj * 4 + gk
            for l in range(7):
                r = block_id * 7 + l
                for lu in range(4):
                    val = U_local[lu, l]
                    if val != 0:
                        # lu = i*2 + j'
                        li = lu // 2
                        lj = lu % 2
                        gi = li
                        gj_global = 2*gj + lj
                        U[gi*m + gj_global, r] += val
                for lv in range(4):
                    val = V_local[lv, l]
                    if val != 0:
                        # lv = j'*2 + k'
                        lj = lv // 2
                        lk = lv % 2
                        gj_global = 2*gj + lj
                        gk_global = 2*gk + lk
                        V[gj_global*p + gk_global, r] += val
                for lw in range(4):
                    val = W_local[lw, l]
                    if val != 0:
                        # lw = k'*2 + i
                        lk = lw // 2
                        li = lw % 2
                        gk_global = 2*gk + lk
                        gi = li
                        W[gk_global*n + gi, r] += val
                        
    # Verify
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(total_rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if s != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
            
    if valid:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to naive
        U_naive = np.zeros((n*m, n*m*p), dtype=int)
        V_naive = np.zeros((m*p, n*m*p), dtype=int)
        W_naive = np.zeros((p*n, n*m*p), dtype=int)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i*m*p + j*p + k
                    U_naive[i*m+j, r] = 1
                    V_naive[j*p+k, r] = 1
                    W_naive[k*n+i, r] = 1
        return {"U": U_naive.tolist(), "V": V_naive.tolist(), "W": W_naive.tolist()}
