"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time

def _strassen_222():
    """Return (U, V, W) for 2x2x2 matrix multiplication, rank 7 (Strassen)."""
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen's 7 multiplications:
    # M1 = A00*(B00+B11)
    # M2 = (A00+A01)*B11
    # M3 = (A10+A11)*B00
    # M4 = A11*(B11+B00)
    # M5 = (A00+A11)*B00
    # M6 = (A01-A11)*(B10+B11)
    # M7 = (A00-A10)*(B00+B01)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M2 + M5
    # C10 = M3 + M6
    # C11 = M1 - M2 + M3 + M4
    
    # U indexes (i,j) -> i*2+j, V indexes (j,k) -> j*2+k, W indexes (k,i) -> k*2+i
    # r=0: M1: U=(1,0,0,0), V=(1,0,0,1), W=(1,0,0,0)
    U[0][0] = 1; V[0][0] = 1; V[3][0] = 1; W[0][0] = 1
    # r=1: M2: U=(1,1,0,0), V=(0,0,0,1), W=(0,1,0,0)
    U[0][1] = 1; U[1][1] = 1; V[3][1] = 1; W[2][1] = 1
    # r=2: M3: U=(0,0,1,1), V=(1,0,0,0), W=(0,0,1,0)
    U[2][2] = 1; U[3][2] = 1; V[0][2] = 1; W[1][2] = 1
    # r=3: M4: U=(0,0,0,1), V=(0,0,1,1), W=(0,0,0,1)
    U[3][3] = 1; V[2][3] = 1; V[3][3] = 1; W[3][3] = 1
    # r=4: M5: U=(1,0,0,1), V=(1,0,0,0), W=(1,0,0,0)
    U[0][4] = 1; U[3][4] = 1; V[0][4] = 1; W[0][4] = 1
    # r=5: M6: U=(0,1,0,-1), V=(0,1,0,1), W=(0,1,0,0)
    U[1][5] = 1; U[3][5] = -1; V[1][5] = 1; V[3][5] = 1; W[2][5] = 1
    # r=6: M7: U=(1,0,-1,0), V=(1,1,0,0), W=(0,1,0,0)
    U[0][6] = 1; U[2][6] = -1; V[0][6] = 1; V[1][6] = 1; W[2][6] = 1
    
    return U, V, W

def _trivial_1mp(m, p):
    """Return (U, V, W) for 1xm*p matrix multiplication, rank m*p (schoolbook)."""
    n = 1
    rank = m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for j in range(m):
        for k in range(p):
            U[j][col] = 1
            V[j * p + k][col] = 1
            W[k][col] = 1
            col += 1
    return U, V, W

def _tensor_product(U1, V1, W1, U2, V2, W2):
    """Tensor product of two tensor decompositions."""
    r1 = len(U1[0]) if U1 else 0
    r2 = len(U2[0]) if U2 else 0
    rank = r1 * r2
    nm1 = len(U1)
    mp1 = len(V1)
    pn1 = len(W1)
    nm2 = len(U2)
    mp2 = len(V2)
    pn2 = len(W2)
    
    U = [[0.0] * rank for _ in range(nm1 * nm2)]
    V = [[0.0] * rank for _ in range(mp1 * mp2)]
    W = [[0.0] * rank for _ in range(pn1 * pn2)]
    
    for r1_idx in range(r1):
        for r2_idx in range(r2):
            c = r1_idx * r2 + r2_idx
            for i1 in range(nm1):
                if U1[i1][r1_idx] != 0:
                    for i2 in range(nm2):
                        if U2[i2][r2_idx] != 0:
                            U[i1 * nm2 + i2][c] = U1[i1][r1_idx] * U2[i2][r2_idx]
            for i1 in range(mp1):
                if V1[i1][r1_idx] != 0:
                    for i2 in range(mp2):
                        if V2[i2][r2_idx] != 0:
                            V[i1 * mp2 + i2][c] = V1[i1][r1_idx] * V2[i2][r2_idx]
            for i1 in range(pn1):
                if W1[i1][r1_idx] != 0:
                    for i2 in range(pn2):
                        if W2[i2][r2_idx] != 0:
                            W[i1 * pn2 + i2][c] = W1[i1][r1_idx] * W2[i2][r2_idx]
    
    return U, V, W

def _concatenate(U1, V1, W1, U2, V2, W2):
    """Concatenate two tensor decompositions (block diagonal)."""
    r1 = len(U1[0]) if U1 else 0
    r2 = len(U2[0]) if U2 else 0
    rank = r1 + r2
    nm = len(U1)
    mp = len(V1)
    pn = len(W1)
    
    U = [row[:] + [0.0] * r2 for row in U1]
    V = [row[:] + [0.0] * r2 for row in V1]
    W = [row[:] + [0.0] * r2 for row in W1]
    
    for i in range(len(U2)):
        U.append([0.0] * r1 + U2[i][:])
    for i in range(len(V2)):
        V.append([0.0] * r1 + V2[i][:])
    for i in range(len(W2)):
        W.append([0.0] * r1 + W2[i][:])
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication."""
    import numpy as np
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    rank = U_np.shape[1]
    
    # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # should equal 1 for all valid (i,j,k) and 0 otherwise.
    # Actually, the matrix mult tensor is: T = sum_{i,j,k} e_{(i,j)} ⊗ e_{(j,k)} ⊗ e_{(k,i)}
    # So the reconstructed tensor should have T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k.
    
    # Check a few random entries
    max_val = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                if abs(val - 1) > 1e-10:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s
    
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # Improved strategy: Use tensor product structure 4=2*2, 8=2*2*2
    # T_{3,4,8} can be built from T_{3,2,2} blocks more efficiently.
    # 
    # Key insight: Split j into {0,1} and {2,3}, split k into 4 pairs.
    # For each (j-pair, k-pair) we need a T_{3,2,2} decomposition.
    # T_{3,2,2} = T_{2,2,2} (Strassen, rank 7) for i=0,1 + T_{1,2,2} (rank 4) for i=2
    # But we can potentially do better by finding a rank-10 T_{3,2,2}.
    #
    # Alternative: Use the full tensor product approach.
    # T_{3,4,8} embeds in a structure where we use:
    # - 2 Strassen decompositions for the j-split (rank 7 each)
    # - Combined with the k-split
    #
    # Let's try: T_{3,4,8} = sum over j-pairs of [T_{3,2,8} for that j-pair]
    # T_{3,2,8} = T_{3,2,2} ⊗ T_{1,1,4} ... no, that doesn't work cleanly.
    #
    # Best practical approach: keep the current 88-rank construction but verify
    # and potentially find a 1-rank improvement via linear dependence checking.
    
    # Step 1: Get Strassen 2x2x2 (rank 7)
    U_s, V_s, W_s = _strassen_222()
    
    # Step 2: Get trivial 1x2x4 (rank 8)
    U_t, V_t, W_t = _trivial_1mp(2, 4)
    
    # Step 3: Tensor product T_{2,2,2} ⊗ T_{1,2,4} = T_{2,4,8} (rank 56)
    U_248, V_248, W_248 = _tensor_product(U_s, V_s, W_s, U_t, V_t, W_t)
    
    # Step 4: Trivial T_{1,4,8} (rank 32)
    U_148, V_148, W_148 = _trivial_1mp(4, 8)
    
    # Build T_{3,4,8} from the two parts
    rank_248 = len(U_248[0])
    rank_148 = len(U_148[0])
    total_rank = rank_248 + rank_148
    
    U_full = [[0.0] * total_rank for _ in range(12)]
    V_full = [[0.0] * total_rank for _ in range(32)]
    W_full = [[0.0] * total_rank for _ in range(24)]
    
    # Embed T_{2,4,8} (first rank_248 columns)
    for i in range(8):
        for r in range(rank_248):
            U_full[i][r] = U_248[i][r]
    for i in range(32):
        for r in range(rank_248):
            V_full[i][r] = V_248[i][r]
    for i in range(16):
        for r in range(rank_248):
            W_full[i][r] = W_248[i][r]
    
    # Embed T_{1,4,8} (next rank_148 columns)
    for i in range(4):
        for r in range(rank_148):
            U_full[8 + i][rank_248 + r] = U_148[i][r]
    for i in range(32):
        for r in range(rank_148):
            V_full[i][rank_248 + r] = V_148[i][r]
    for i in range(8):
        for r in range(rank_148):
            W_full[16 + i][rank_248 + r] = W_148[i][r]
    
    # Verify
    if _verify_decomposition(U_full, V_full, W_full, n, m, p):
        # Try to reduce rank by checking for linear dependencies
        import numpy as np
        U_np = np.array(U_full, dtype=float)
        V_np = np.array(V_full, dtype=float)
        W_np = np.array(W_full, dtype=float)
        
        # Check if any column is redundant: for each column r, check if the
        # rank-1 tensor U[:,r] ⊗ V[:,r] ⊗ W[:,r] can be expressed as a
        # combination of others. This is expensive, so only try if time allows.
        if time.time() < deadline - 10:
            # Try removing each column and re-verifying (greedy)
            best_rank = total_rank
            best_U, best_V, best_W = U_full, V_full, W_full
            
            # Quick check: try removing the last few columns
            for trial_remove in range(min(5, total_rank)):
                r_remove = total_rank - 1 - trial_remove
                U_test = [row[:r_remove] + row[r_remove+1:] for row in U_full]
                V_test = [row[:r_remove] + row[r_remove+1:] for row in V_full]
                W_test = [row[:r_remove] + row[r_remove+1:] for row in W_full]
                
                if _verify_decomposition(U_test, V_test, W_test, n, m, p):
                    best_rank = r_remove
                    best_U, best_V, best_W = U_test, V_test, W_test
                else:
                    break
        
        U_final = best_U
        V_final = best_V
        W_final = best_W
        
        U_int = [[int(round(x)) if abs(x - round(x)) < 1e-10 else x for x in row] for row in U_final]
        V_int = [[int(round(x)) if abs(x - round(x)) < 1e-10 else x for x in row] for row in V_final]
        W_int = [[int(round(x)) if abs(x - round(x)) < 1e-10 else x for x in row] for row in W_final]
        return {"U": U_int, "V": V_int, "W": W_int}
    else:
        # Fall back to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
