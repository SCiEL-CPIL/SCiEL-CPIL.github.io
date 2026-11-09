"""Matrix multiplication 4x4x6 tensor decomposition - Strassen-based rank reduction."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2():
    """Return (U, V, W) for 2x2x2 matrix mult, rank 7 (Strassen)."""
    # Strassen's 7 products for 2x2:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    #
    # U: 4x7, V: 4x7, W: 4x7
    # U rows: A11, A12, A21, A22 (index i*m+j, m=2)
    # V rows: B11, B12, B21, B22 (index j*p+k, p=2)
    # W rows: C11, C12, C21, C22 (index k*n+i, n=2)
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A11+A22)*(B11+B22)
    U[0][0] = 1; U[3][0] = 1  # A11, A22
    V[0][0] = 1; V[2][0] = 1  # B11, B21
    W[0][0] = 1; W[3][0] = 1  # C11, C22
    
    # M2 = (A21+A22)*B11
    U[2][1] = 1; U[3][1] = 1  # A21, A22
    V[0][1] = 1  # B11
    W[2][1] = 1; W[3][1] = -1  # C21, C22
    
    # M3 = A11*(B12-B22)
    U[0][2] = 1  # A11
    V[1][2] = 1; V[3][2] = -1  # B12, B22
    W[1][2] = 1; W[3][2] = 1  # C12, C22
    
    # M4 = A22*(B21-B11)
    U[3][3] = 1  # A22
    V[2][3] = 1; V[0][3] = -1  # B21, B11
    W[0][3] = 1; W[2][3] = 1  # C11, C21
    
    # M5 = (A11+A12)*B22
    U[0][4] = 1; U[1][4] = 1  # A11, A12
    V[3][4] = 1  # B22
    W[0][4] = -1; W[1][4] = 1  # C11, C12
    
    # M6 = (A21-A11)*(B11+B12)
    U[2][5] = 1; U[0][5] = -1  # A21, A11
    V[0][5] = 1; V[1][5] = 1  # B11, B12
    W[3][5] = 1  # C22
    
    # M7 = (A12-A22)*(B21+B22)
    U[1][6] = 1; U[3][6] = -1  # A12, A22
    V[2][6] = 1; V[3][6] = 1  # B21, B22
    W[0][6] = 1  # C11
    
    return U, V, W

def _compose_4x4x4():
    """Compose Strassen 2x2 to get 4x4x4 rank-49 decomposition."""
    U2, V2, W2 = _strassen_2x2()
    
    # 4x4 = 2x2 blocks of 2x2
    # A = [[A11, A12],[A21, A22]] where each is 2x2
    # B = [[B11, B12],[B21, B22]]
    # C = [[C11, C12],[C21, C22]]
    
    # For each Strassen product M_c (c=0..6), we need to compute
    # (sum of A blocks) * (sum of B blocks) which is another 2x2 mult
    # So each M_c requires 7 more products (Strassen again)
    # Total: 7 * 7 = 49
    
    U = [[0]*49 for _ in range(16)]  # 4*4 = 16 rows
    V = [[0]*49 for _ in range(16)]  # 4*4 = 16 rows  
    W = [[0]*49 for _ in range(16)]  # 4*4 = 16 rows
    
    # A block indices: A11=[0,1,4,5], A12=[0,1,6,7], A21=[8,9,12,13], A22=[8,9,14,15]
    # In flat index (i*m+j, m=4): A11 rows 0,1,4,5; A12 rows 0,1,6,7; A21 rows 8,9,12,13; A22 rows 8,9,14,15
    # Wait, for 4x4, i*m+j with m=4: 
    # A11: i=0,1 j=0,1 -> 0,1,4,5
    # A12: i=0,1 j=2,3 -> 2,3,6,7
    # A21: i=2,3 j=0,1 -> 8,9,12,13
    # A22: i=2,3 j=2,3 -> 10,11,14,15
    
    # B block indices (j*p+k, p=4):
    # B11: j=0,1 k=0,1 -> 0,1,4,5
    # B12: j=0,1 k=2,3 -> 2,3,6,7
    # B21: j=2,3 k=0,1 -> 8,9,12,13
    # B22: j=2,3 k=2,3 -> 10,11,14,15
    
    # C block indices (k*n+i, n=4):
    # C11: k=0,1 i=0,1 -> 0,1,4,5
    # C12: k=0,1 i=2,3 -> 2,3,6,7
    # C21: k=2,3 i=0,1 -> 8,9,12,13
    # C22: k=2,3 i=2,3 -> 10,11,14,15
    
    A_blocks = [[0,1,4,5], [2,3,6,7], [8,9,12,13], [10,11,14,15]]  # A11, A12, A21, A22
    B_blocks = [[0,1,4,5], [2,3,6,7], [8,9,12,13], [10,11,14,15]]  # B11, B12, B21, B22
    C_blocks = [[0,1,4,5], [2,3,6,7], [8,9,12,13], [10,11,14,15]]  # C11, C12, C21, C22
    
    for c in range(7):
        base = c * 7
        # U contribution: for each A block ab, for each 2x2 position (i2,j2) in block
        for ab in range(4):
            coeff_a = U2[ab][c]
            if coeff_a == 0:
                continue
            for idx, flat in enumerate(A_blocks[ab]):
                for i2 in range(2):
                    for j2 in range(2):
                        # 2x2 position (i2,j2) within block ab
                        # In the 2x2 Strassen, U2[i2*2+j2][c] gives the coefficient
                        # But we need to map: the 2x2 block ab has elements at positions
                        # within the 4x4. The 2x2 Strassen U has 4 rows (2x2).
                        # Row i2*2+j2 in U2 corresponds to position (i2,j2) in the 2x2 block.
                        u2_val = U2[i2*2+j2][c]
                        if u2_val == 0:
                            continue
                        U[flat][base + i2*2+j2] = u2_val
        
        # V contribution
        for bb in range(4):
            coeff_b = V2[bb][c]
            if coeff_b == 0:
                continue
            for idx, flat in enumerate(B_blocks[bb]):
                for j2 in range(2):
                    for k2 in range(2):
                        v2_val = V2[j2*2+k2][c]
                        if v2_val == 0:
                            continue
                        V[flat][base + j2*2+k2] = v2_val
        
        # W contribution
        for cb in range(4):
            coeff_w = W2[cb][c]
            if coeff_w == 0:
                continue
            for idx, flat in enumerate(C_blocks[cb]):
                for k2 in range(2):
                    for i2 in range(2):
                        w2_val = W2[k2*2+i2][c]
                        if w2_val == 0:
                            continue
                        W[flat][base + k2*2+i2] = w2_val
    
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify decomposition."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _compose_4x4x6():
    """Compose to get 4x4x6 decomposition using Strassen structure."""
    # Strategy: 4x4x6 = compute A(4x4) * B(4x6)
    # Split B into B1(4x3) and B2(4x3)
    # For each, pad to 4x4 and use rank-49 4x4x4 decomposition
    # Total rank: 2 * 49 = 98. Worse than 96.
    #
    # Alternative: Use the 4x4x4 rank-49 directly on the full 4x4x6 tensor.
    # The tensor T[i,j,k] = 1 for i in 0..3, j in 0..3, k in 0..5.
    # We can embed this into a 4x4x8 tensor (pad k to 8) and use two 4x4x4 decompositions.
    # But that's still 98.
    #
    # Better: Just use the rank-49 4x4x4 and note that for 4x4x6, we can do:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # This is the same tensor structure. The 4x4x4 rank-49 gives us a decomposition
    # where T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1 for k in 0..3.
    # For k=4,5, we need additional terms.
    #
    # Actually, the simplest approach that beats 96: use the known fact that
    # rank(4x4x6) <= rank(4x4x4) + rank(4x4x2) is not valid for tensor rank.
    #
    # Let me just use the Strassen 4x4x4 rank-49 and extend to 4x4x6 by
    # adding 4x4x2 schoolbook (rank 32) terms, total 81. But I need to verify
    # this actually works as a valid tensor decomposition.
    
    # Actually, the correct way: the 4x4x6 tensor T[i,j,k] = 1 for all i,j,k
    # (i in 0..3, j in 0..3, k in 0..5) can be decomposed as:
    # T = T_4x4x4 (for k=0..3) + T_4x4x2 (for k=4..5)
    # where T_4x4x4[i,j,k] = 1 for k in 0..3, 0 otherwise
    # and T_4x4x2[i,j,k] = 1 for k in 4..5, 0 otherwise
    #
    # rank(T) <= rank(T_4x4x4) + rank(T_4x4x2) = 49 + 32 = 81
    # This is valid because the supports are disjoint in the k dimension.
    
    U444, V444, W444 = _compose_4x4x4()
    
    # Schoolbook for 4x4x2 (k=4,5)
    U442 = [[0]*32 for _ in range(16)]
    V442 = [[0]*32 for _ in range(16)]
    W442 = [[0]*32 for _ in range(16)]
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(2):  # k=4,5 -> local k=0,1
                U442[i*4+j][col] = 1
                V442[j*4+k][col] = 1
                W442[k*4+i][col] = 1
                col += 1
    
    # Combine: total rank = 49 + 32 = 81
    total_rank = 81
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(16)]
    W = [[0]*total_rank for _ in range(16)]
    
    # Copy 4x4x4 part (columns 0..48)
    for r in range(16):
        for c in range(49):
            U[r][c] = U444[r][c]
            V[r][c] = V444[r][c]
            W[r][c] = W444[r][c]
    
    # Copy 4x4x2 part (columns 49..80)
    # But the W indices for 4x4x2 are k*4+i where k is 0,1 (local)
    # In the full 4x4x6, k=4,5 so W index should be k*4+i where k=4,5
    # The 4x4x2 decomposition has W[k_local*4+i] where k_local=0,1
    # We need to map: for k_local=0 -> k_full=4, k_local=1 -> k_full=5
    # So W442[k_local*4+i] should go to W[(k_local+4)*4+i]
    for r in range(16):
        for c in range(32):
            U[r][49+c] = U442[r][c]
            V[r][49+c] = V442[r][c]
    # Remap W indices
    for k_local in range(2):
        for i in range(4):
            src_row = k_local*4 + i
            dst_row = (k_local+4)*4 + i
            for c in range(32):
                W[dst_row][49+c] = W442[src_row][c]
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n, m, p = 4, 4, 6
    schoolbook_rank = n * m * p  # 96
    
    # Schoolbook as fallback
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    best = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank
    
    # Try Strassen-based 4x4x6 decomposition (rank 81)
    try:
        if time.time() < deadline - 10:
            U, V, W = _compose_4x4x6()
            if _verify(U, V, W, n, m, p):
                best = {"U": U, "V": V, "W": W}
                best_rank = 81
    except Exception:
        pass
    
    # Try to improve further with random search if time permits
    if time.time() < deadline - 20 and best_rank > 75:
        try:
            rng = np.random.RandomState(42)
            for target_rank in [80, 78, 76, 74, 72]:
                if time.time() > deadline - 15:
                    break
                # Random initialization
                U_r = rng.randn(16, target_rank) * 0.1
                V_r = rng.randn(16, target_rank) * 0.1
                W_r = rng.randn(16, target_rank) * 0.1
                
                def compute_error(Ur, Vr, Wr):
                    err = 0.0
                    for i in range(4):
                        for j in range(4):
                            for k in range(6):
                                s = np.dot(Ur[i*4+j], Vr[j*4+k]) * W_r[k*4+i][j]  # wrong
                                # Actually: sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
                                s = np.sum(Ur[i*4+j] * Vr[j*4+k] * Wr[k*4+i])
                                err += (1.0 - s) ** 2
                    return err
                
                err = compute_error(U_r, V_r, W_r)
                step = 0.01
                for it in range(500):
                    if time.time() > deadline - 12:
                        break
                    mat = rng.randint(3)
                    if mat == 0:
                        ri, ci = rng.randint(16), rng.randint(target_rank)
                        old = U_r[ri, ci]
                        U_r[ri, ci] = old + rng.randn() * step
                    elif mat == 1:
                        ri, ci = rng.randint(16), rng.randint(target_rank)
                        old = V_r[ri, ci]
                        V_r[ri, ci] = old + rng.randn() * step
                    else:
                        ri, ci = rng.randint(16), rng.randint(target_rank)
                        old = W_r[ri, ci]
                        W_r[ri, ci] = old + rng.randn() * step
                    
                    new_err = compute_error(U_r, V_r, W_r)
                    if new_err < err:
                        err = new_err
                        step = min(step * 1.05, 0.1)
                        if err < 1e-6:
                            break
                    else:
                        if mat == 0:
                            U_r[ri, ci] = old
                        elif mat == 1:
                            V_r[ri, ci] = old
                        else:
                            W_r[ri, ci] = old
                        step = max(step * 0.95, 0.001)
                
                if err < 1e-4:
                    U_list = [[round(x, 8) for x in row] for row in U_r.tolist()]
                    V_list = [[round(x, 8) for x in row] for row in V_r.tolist()]
                    W_list = [[round(x, 8) for x in row] for row in W_r.tolist()]
                    if _verify(U_list, V_list, W_list, n, m, p):
                        best = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = target_rank
                        break
        except Exception:
            pass
    
    return best
# EVOLVE-BLOCK-END
