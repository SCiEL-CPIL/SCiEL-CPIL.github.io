import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-9):
    """Verify using numpy einsum for speed."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64).reshape(n, m, rank)
    V_arr = np.array(V, dtype=np.float64).reshape(m, p, rank)
    W_arr = np.array(W, dtype=np.float64).reshape(p, n, rank)
    reconstructed = np.einsum('ijc,jkc,kic->ijk', U_arr, V_arr, W_arr)
    expected = np.ones((n, m, p), dtype=np.float64)
    return np.allclose(expected, reconstructed, atol=tol)

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _strassen_2x2():
    """Rank-7 decomposition for 2x2 x 2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) where indices are (row, col) for 2x2.
    """
    # 7 Strassen terms for C = A*B where A,B,C are 2x2
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Term 0: m0 = (a00+a11)(b00+b11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Term 1: m1 = (a10+a11)b00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # Term 2: m2 = a00(b01-b11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # Term 3: m3 = a11(b10-b00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[2][3] = 1; W[0][3] = 1
    
    # Term 4: m4 = (a00+a01)b11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[1][4] = -1; W[3][4] = 1
    
    # Term 5: m5 = (a10-a00)(b00+b01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # Term 6: m6 = (a01-a11)(b10+b11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _build_4x4x4_strassen():
    """Build rank-28 decomposition for 4x4 x 4x4 using 2x2 Strassen blocks.
    Split 4x4 into four 2x2 blocks. Use Strassen on the block level.
    Each Strassen term involves multiplying 2x2 submatrices, contributing 4 scalar terms.
    Total rank = 7 * 4 = 28.
    """
    U2, V2, W2 = _strassen_2x2()
    
    # 4x4 matrix has blocks at positions (0,0), (0,1), (1,0), (1,1) for 2x2 blocks
    # Block (br, bc) occupies rows 2*br..2*br+1, cols 2*bc..2*bc+1
    # For U (A matrix, 4x4): element (i,j) is in block (i//2, j//2), local pos (i%2, j%2)
    # For V (B matrix, 4x4): element (j,k) is in block (j//2, k//2), local pos (j%2, k%2)
    # For W (C matrix, 4x4): element (k,i) is in block (k//2, i//2), local pos (k%2, i%2)
    
    rank = 28
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    # For each of the 7 Strassen terms and each of the 4 local positions within 2x2:
    for s in range(7):  # Strassen term
        for lr in range(2):  # local row within 2x2 block
            for lc in range(2):  # local col within 2x2 block
                c = s * 4 + lr * 2 + lc
                
                # For U: the Strassen term s uses certain blocks of A
                # U2[i][s] gives coefficient for A element (i,j) in term s
                # We need to map: for each A element (i,j), which block and local pos
                # Block row = i//2, block col = j//2, local row = i%2, local col = j%2
                # The Strassen term s operates on blocks, so U2[br*2+lr][s] * ... 
                # Actually U2 is indexed by local position within 2x2, and s is the term
                # For a 4x4 matrix, element (i,j): block (i//2, j//2), local (i%2, j%2)
                # The coefficient for this element in term s is U2[i%2][s] if the block is used
                # But wait - Strassen operates on 2x2 blocks, so term s uses specific blocks
                # U2[local_row][s] gives the weight for that local position in term s
                # But all 2x2 blocks that participate get the same local pattern
                
                # For 4x4 x 4x4 with 2x2 blocks:
                # A has 4 blocks: A00, A01, A10, A11 (each 2x2)
                # B has 4 blocks: B00, B01, B10, B11
                # Strassen terms use combinations of these blocks
                # U2[i][s] where i is 0..3 represents the local position in the 2x2 A matrix
                # But for 4x4, we need to know which BLOCK is being referenced
                
                # Let me reconsider. The 2x2 Strassen decomposition works on 2x2 matrices.
                # For 4x4, we treat it as a 2x2 block matrix where each block is 2x2.
                # The Strassen algorithm then multiplies these 2x2 blocks.
                # Each "multiplication" of two 2x2 blocks is itself a 2x2x2 tensor (rank 4 schoolbook or rank 7 Strassen)
                # But for simplicity, use schoolbook for the inner 2x2 multiplications: rank 4 per Strassen term
                # Total: 7 * 4 = 28
                
                # For term s, the U part: for each A element (i,j):
                # block_row = i//2, block_col = j//2, local_r = i%2, local_c = j%2
                # The Strassen term s has a coefficient for each block position
                # Let's define which blocks each Strassen term uses:
                
                # m0 = (A00+A11)(B00+B11): A blocks 0,0 and 1,1; B blocks 0,0 and 1,1
                # m1 = (A10+A11)B00: A blocks 1,0 and 1,1; B block 0,0
                # m2 = A00(B01-B11): A block 0,0; B blocks 0,1 and 1,1
                # m3 = A11(B10-B00): A block 1,1; B blocks 1,0 and 0,0
                # m4 = (A00+A01)B11: A blocks 0,0 and 0,1; B block 1,1
                # m5 = (A10-A00)(B00+B01): A blocks 1,0 and 0,0; B blocks 0,0 and 0,1
                # m6 = (A01-A11)(B10+B11): A blocks 0,1 and 1,1; B blocks 1,0 and 1,1
                
                # For each term, define which (block_row, block_col) pairs are used and their coefficients
                block_coeffs_U = [
                    [(0,0,1), (1,1,1)],           # m0
                    [(1,0,1), (1,1,1)],           # m1
                    [(0,0,1)],                     # m2
                    [(1,1,1)],                     # m3
                    [(0,0,1), (0,1,1)],           # m4
                    [(1,0,1), (0,0,-1)],          # m5
                    [(0,1,1), (1,1,-1)],          # m6
                ]
                block_coeffs_V = [
                    [(0,0,1), (1,1,1)],           # m0
                    [(0,0,1)],                     # m1
                    [(0,1,1), (1,1,-1)],          # m2
                    [(1,0,1), (0,0,-1)],          # m3
                    [(1,1,1)],                     # m4
                    [(0,0,1), (0,1,1)],           # m5
                    [(1,0,1), (1,1,1)],           # m6
                ]
                block_coeffs_W = [
                    [(0,0,1), (1,1,1)],           # m0
                    [(1,0,1), (1,1,-1)],          # m1
                    [(0,1,1), (1,1,1)],           # m2
                    [(1,0,1), (0,0,1)],           # m3
                    [(0,1,-1), (1,1,1)],          # m4
                    [(1,1,1)],                     # m5
                    [(0,0,1)],                     # m6
                ]
                
                for (br, bc, coeff_u) in block_coeffs_U[s]:
                    for lr2 in range(2):
                        for lc2 in range(2):
                            i = br * 2 + lr2
                            j = bc * 2 + lc2
                            U[i * 4 + j][c] += coeff_u * 1  # local position weight is 1 for schoolbook inner
                            
                for (br, bc, coeff_v) in block_coeffs_V[s]:
                    for lr2 in range(2):
                        for lc2 in range(2):
                            j = br * 2 + lr2
                            k = bc * 2 + lc2
                            V[j * 4 + k][c] += coeff_v * 1
                            
                for (br, bc, coeff_w) in block_coeffs_W[s]:
                    for lr2 in range(2):
                        for lc2 in range(2):
                            k = br * 2 + lr2
                            i = bc * 2 + lc2
                            W[k * 4 + i][c] += coeff_w * 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Start with schoolbook
    U_school, V_school, W_school = _schoolbook(n, m, p)
    if not _verify_decomposition_fast(U_school, V_school, W_school, n, m, p):
        raise ValueError("Schoolbook failed")
    
    best_rank = n * m * p
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Try structured 4x4x4 Strassen + boundary terms for the 5th column
    # 4x4 x 4x4 part: rank 28 via Strassen
    # 4x4 x 4x1 part (5th column of B, or rather the extra output column): 
    # Actually for 4x4 * 4x5: C = A*B where A is 4x4, B is 4x5
    # Split B into B_left (4x4) and B_right (4x1)
    # C = A*B_left + A*B_right
    # A*B_left: 4x4 * 4x4 -> rank 28 via Strassen
    # A*B_right: 4x4 * 4x1 -> rank 16 via schoolbook (each of 16 output elements needs one term)
    # But we can do better: A*B_right where B_right is 4x1
    # C_right[i,k] = sum_j A[i,j] * B[j,4] for k=4 (only 5th column)
    # This is just matrix-vector multiply, rank = 4*1 = 4? No, it's 4*4 = 16 scalar products
    # But as a tensor: for each (i, j, k=4): U[i,j,c]*V[j,4,c]*W[4,i,c] = 1
    # We need 16 terms for this part (one per (i,j) pair)
    # Total: 28 + 16 = 44
    
    if time.time() < deadline:
        U44, V44, W44 = _build_4x4x4_strassen()
        # Verify the 4x4x4 part
        if _verify_decomposition_fast(U44, V44, W44, 4, 4, 4):
            # Now add the boundary: 4x4 * 4x1 part
            # For k=4 (5th column), C[i,4] = sum_j A[i,j]*B[j,4]
            # Need terms where W[k=4, i, c] is nonzero
            # Add 16 schoolbook terms for (i,j,k=4)
            rank_total = 28 + 16
            U_new = [[0]*rank_total for _ in range(16)]
            V_new = [[0]*rank_total for _ in range(20)]  # 4*5
            W_new = [[0]*rank_total for _ in range(20)]  # 5*4
            
            # Copy the 4x4x4 Strassen part (k=0..3)
            for i in range(4):
                for j in range(4):
                    for c in range(28):
                        U_new[i*4+j][c] = U44[i*4+j][c]
                        V_new[j*4+0][c] = V44[j*4+0][c] if j*4+0 < 16 else 0
                        # V is 4x5, so V[j, k] for k=0..4. For the 4x4 part, k=0..3
                        V_new[j*4 + 0][c] = 0  # reset, will set properly
                        W_new[0*4+i][c] = 0  # reset
                    
            # Let me redo this more carefully
            U_new = [[0]*rank_total for _ in range(16)]
            V_new = [[0]*rank_total for _ in range(20)]
            W_new = [[0]*rank_total for _ in range(20)]
            
            # 4x4x4 Strassen part: A(4x4) * B_left(4x4) -> C_left(4x4)
            # U[i,j,c] for i,j in 0..3, c in 0..27
            # V[j,k,c] for j in 0..3, k in 0..3, c in 0..27 (k=4 is 0)
            # W[k,i,c] for k in 0..3, i in 0..3, c in 0..27 (k=4 is 0)
            for c in range(28):
                for i in range(4):
                    for j in range(4):
                        U_new[i*4+j][c] = U44[i*4+j][c]
                for j in range(4):
                    for k in range(4):
                        V_new[j*5+k][c] = V44[j*4+k][c]
                for k in range(4):
                    for i in range(4):
                        W_new[k*4+i][c] = W44[k*4+i][c]
            
            # Boundary: A(4x4) * B_right(4x1) -> C_right(4x1)
            # C_right[i,4] = sum_j A[i,j]*B[j,4]
            # For each (i,j): term c = 28 + i*4 + j
            for i in range(4):
                for j in range(4):
                    c = 28 + i*4 + j
                    U_new[i*4+j][c] = 1
                    V_new[j*5+4][c] = 1
                    W_new[4*4+i][c] = 1
            
            # Verify
            if _verify_decomposition_fast(U_new, V_new, W_new, 4, 4, 5):
                best_rank = rank_total
                best_result = {"U": U_new, "V": V_new, "W": W_new}
            else:
                # Debug: check what went wrong
                pass
    
    # Try to search for even better decompositions using random perturbation
    if time.time() < deadline and best_rank > 40:
        import random
        rng = random.Random(12345)
        target_ranks = [42, 40, 38, 36, 32, 28, 24, 20, 16, 12, 8]
        
        for target_rank in target_ranks:
            if time.time() > deadline:
                break
            if target_rank >= best_rank:
                continue
            
            # Try a few random initializations
            attempts = min(50, int((deadline - time.time()) / 0.5))
            for _ in range(attempts):
                if time.time() > deadline:
                    break
                
                U = [[0]*target_rank for _ in range(16)]
                V = [[0]*target_rank for _ in range(20)]
                W = [[0]*target_rank for _ in range(20)]
                
                choices = [-1, -0.5, 0, 0.5, 1]
                for c in range(target_rank):
                    for i in range(16):
                        U[i][c] = rng.choice(choices)
                    for i in range(20):
                        V[i][c] = rng.choice(choices)
                        W[i][c] = rng.choice(choices)
                
                if _verify_decomposition_fast(U, V, W, 4, 4, 5):
                    best_rank = target_rank
                    best_result = {"U": U, "V": V, "W": W}
                    break
    
    if time.time() > deadline:
        return best_result
    
    return best_result
