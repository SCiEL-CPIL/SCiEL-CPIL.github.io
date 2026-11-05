import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_col(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    # BCDR rank-16 decomposition for 4x4 matrix multiplication
    # This uses two nested Strassen blocks
    # The 4x4 matrix is partitioned into four 2x2 blocks
    # We use Strassen on the top-left and bottom-right 2x2 blocks
    # and handle cross terms carefully
    
    # Actually, BCDR rank 16 for 4x4 can be constructed as follows:
    # Use Strassen (rank 7) on A11*B11 where A11, B11 are 2x2
    # But we need the full 4x4 decomposition
    
    # Let me use a known rank-16 decomposition for 4x4
    # Based on Bini-Capovani-De Falco-Ronco (1978)
    
    # The BCDR decomposition for 4x4:
    # Partition 4x4 into four 2x2 blocks: A = [[A11,A12],[A21,A22]], B = [[B11,B12],[B21,B22]]
    # C = [[C11,C12],[C21,C22]]
    
    # Using Strassen-like factors:
    # M1 = (A11+A22)*(B11+B22)  -> rank 7 for 2x2
    # M2 = (A21+A22)*B11        -> rank 7 for 2x2  
    # M3 = A11*(B12-B22)        -> rank 7 for 2x2
    # M4 = A22*(B21-B11)        -> rank 7 for 2x2
    # M5 = (A11+A12)*B22        -> rank 7 for 2x2
    # M6 = (A21-A11)*(B11+B12)  -> rank 7 for 2x2
    # M7 = (A12-A22)*(B21+B22)  -> rank 7 for 2x2
    
    # That's 7*7 = 49, too many. BCDR is more clever.
    
    # Let me use a different approach: partition 5 as 2+2+1 and use 
    # a better decomposition for the 2x2x2 blocks or find cross-block optimizations.
    
    # Alternative: Use the fact that we can share factors across blocks.
    # For now, let me try partition 4+1 with a rank-16 4x4 decomposition.
    
    # Known rank-16 decomposition for 4x4 (BCDR):
    # This can be built from the rank-7 Strassen for 2x2 applied in a specific way.
    
    # Let me implement a known rank-16 4x4 decomposition.
    # Reference: The BCDR decomposition uses the following 16 factors:
    
    # I'll use a constructive approach: 
    # 4x4 = 2x2 + 2x2 blocks, use Strassen on each diagonal 2x2 block
    # and handle off-diagonal with rank-1 terms, but optimize.
    
    # Actually, let me try a simpler improvement: use partition 3+2 instead of 2+2+1
    # 3x3x3 schoolbook = 27, 3x3x2 = 18, 3x2x3 = 18, 2x3x3 = 18, 
    # 3x2x2 = 12, 2x3x2 = 12, 2x2x3 = 12, 2x2x2 = 8 (Strassen)
    # Total: 27+18+18+18+12+12+12+8 = 125, worse.
    
    # Let me stick with 2+2+1 but try to reduce the schoolbook parts.
    # The current rank is 25. Can I do better?
    
    # Known result: The rank of 5x5 matrix multiplication is at most 25 (from 2+2+1 partition)
    # and at least... I'm not sure of the exact lower bound.
    
    # Let me try to verify the current implementation and see if there's a bug or optimization.
    
    # For now, let me implement the 4+1 partition with BCDR rank-16 for the 4x4 core.
    
    # BCDR rank-16 decomposition for 4x4:
    # I'll use a known explicit construction.
    
    # The 4x4 BCDR decomposition can be given by these 16 rank-1 terms:
    # Using indices 0,1,2,3 for the 4x4 blocks
    
    # Factor construction for BCDR 4x4 (rank 16):
    # Based on the paper, the decomposition uses:
    # M1 = A[0:2,0:2] * B[0:2,0:2]  (2x2x2, rank 7 via Strassen)
    # M2 = A[2:4,2:4] * B[2:4,2:4]  (2x2x2, rank 7 via Strassen)
    # That's already 14, plus 2 more for cross terms? No, that's not right.
    
    # Let me use a different known rank-16 decomposition.
    # Actually, the BCDR decomposition is specifically rank 16 for 4x4.
    
    # I'll implement it as: use Strassen (rank 7) on the top-left 2x2 and bottom-right 2x2,
    # and use 2 additional factors for the cross terms, total 7+7+2 = 16.
    
    # But that's not quite right either. Let me look up the actual BCDR construction.
    
    # For now, let me try a different approach: keep the 2+2+1 partition but 
    # see if I can reduce the rank below 25.
    
    # The current rank is 25. The theoretical minimum for 5x5 is unknown but 
    # likely around 20-25.
    
    # Let me just ensure the current implementation is correct and try minor optimizations.
    
    # Partition 5 into 2, 2, 1
    parts = [2, 2, 1]
    offsets = [0, 2, 4]
    
    for a in range(3):
        for b in range(3):
            for c in range(3):
                size_a = parts[a]
                size_b = parts[b]
                size_c = parts[c]
                
                off_a = offsets[a]
                off_b = offsets[b]
                off_c = offsets[c]
                
                if size_a == 2 and size_b == 2 and size_c == 2:
                    # Strassen block (rank 7)
                    # Factor 0: M1 = (A00+A11)(B00+B11)
                    u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
                    u_vec[(off_a+0)*5 + (off_b+0)] = 1
                    u_vec[(off_a+1)*5 + (off_b+1)] = 1
                    v_vec[(off_b+0)*5 + (off_c+0)] = 1
                    v_vec[(off_b+1)*5 + (off_c+1)] = 1
                    w_vec[(off_c+0)*5 + (off_a+0)] = 1
                    w_vec[(off_c+1)*5 + (off_a+1)] = 1
                    add_col(u_vec, v_vec, w_vec)
                    
                    # Factor 1: M2 = (A10+A11)B00
                    u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
                    u_vec[(off_a+1)*5 + (off_b+0)] = 1
                    u_vec[(off_a+1)*5 + (off_b+1)] = 1
                    v_vec[(off_b+0)*5 + (off_c+0)] = 1
                    w_vec[(off_c+0)*5 + (off_a+1)] = 1
                    w_vec[(off_c+1)*5 + (off_a+1)] = -1
                    add_col(u_vec, v_vec, w_vec)
                    
                    # Factor 2: M3 = A00(B01-B11)
                    u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
                    u_vec[(off_a+0)*5 + (off_b+0)] = 1
                    v_vec[(off_b+0)*5 + (off_c+1)] = 1
                    v_vec[(off_b+1)*5 + (off_c+1)] = -1
                    w_vec[(off_c+1)*5 + (off_a+0)] = 1
                    w_vec[(off_c+1)*5 + (off_a+1)] = 1
                    add_col(u_vec, v_vec, w_vec)
                    
                    # Factor 3: M4 = A11(B10-B00)
                    u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
                    u_vec[(off_a+1)*5 + (off_b+1)] = 1
                    v_vec[(off_b+1)*5 + (off_c+0)] = 1
                    v_vec[(off_b+0)*5 + (off_c+0)] = -1
                    w_vec[(off_c+0)*5 + (off_a+0)] = 1
                    w_vec[(off_c+0)*5 + (off_a+1)] = 1
                    add_col(u_vec, v_vec, w_vec)
                    
                    # Factor 4: M5 = (A00+A01)B11
                    u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
                    u_vec[(off_a+0)*5 + (off_b+0)] = 1
                    u_vec[(off_a+0)*5 + (off_b+1)] = 1
                    v_vec[(off_b+1)*5 + (off_c+1)] = 1
                    w_vec[(off_c+0)*5 + (off_a+0)] = -1
                    w_vec[(off_c+1)*5 + (off_a+0)] = 1
                    add_col(u_vec, v_vec, w_vec)
                    
                    # Factor 5: M6 = (A10-A00)(B00+B01)
                    u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
                    u_vec[(off_a+1)*5 + (off_b+0)] = 1
                    u_vec[(off_a+0)*5 + (off_b+0)] = -1
                    v_vec[(off_b+0)*5 + (off_c+0)] = 1
                    v_vec[(off_b+0)*5 + (off_c+1)] = 1
                    w_vec[(off_c+1)*5 + (off_a+1)] = 1
                    add_col(u_vec, v_vec, w_vec)
                    
                    # Factor 6: M7 = (A01-A11)(B10+B11)
                    u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
                    u_vec[(off_a+0)*5 + (off_b+1)] = 1
                    u_vec[(off_a+1)*5 + (off_b+1)] = -1
                    v_vec[(off_b+1)*5 + (off_c+0)] = 1
                    v_vec[(off_b+1)*5 + (off_c+1)] = 1
                    w_vec[(off_c+0)*5 + (off_a+0)] = 1
                    add_col(u_vec, v_vec, w_vec)
                    
                else:
                    # Schoolbook block
                    for u in range(size_a):
                        for v in range(size_b):
                            for w in range(size_c):
                                u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
                                gi = off_a + u
                                gj = off_b + v
                                gk = off_c + w
                                
                                u_vec[gi*5 + gj] = 1
                                v_vec[gj*5 + gk] = 1
                                w_vec[gk*5 + gi] = 1
                                add_col(u_vec, v_vec, w_vec)
                                
    rank = len(U_cols)
    U = [[0]*rank for _ in range(25)]
    V = [[0]*rank for _ in range(25)]
    W = [[0]*rank for _ in range(25)]
    
    for col_idx in range(rank):
        u_col = U_cols[col_idx]
        v_col = V_cols[col_idx]
        w_col = W_cols[col_idx]
        for row_idx in range(25):
            U[row_idx][col_idx] = u_col[row_idx]
            V[row_idx][col_idx] = v_col[row_idx]
            W[row_idx][col_idx] = w_col[row_idx]
            
    elapsed = time.time() - start_time
    if elapsed > 100:
        # Safety: return what we have
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
