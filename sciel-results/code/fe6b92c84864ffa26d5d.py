import numpy as np
import time
import itertools

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
    def schoolbook():
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
        return {"U": U, "V": V, "W": W}
    
    # The matrix multiplication tensor T[i,j,k] = delta[i,k]*delta[j,j] 
    # Actually: T[i, j, k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor is T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] = 1 for all valid i,j,k (it's the all-ones tensor in the appropriate sense)
    # Wait no. The matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k.
    # A rank-1 decomposition term is u_i * v_j * w_k where:
    #   u has shape (n*m,) indexed by (i,j)
    #   v has shape (m*p,) indexed by (j,k)  
    #   w has shape (p*n,) indexed by (k,i)
    # The product gives: sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i]
    # This should equal 1 for all i,j,k.
    
    # Strategy: Generate sparse rank-1 candidates and use greedy set cover.
    # A sparse rank-1 term has few non-zero entries in each factor.
    # For exact decomposition with integer/half-integer entries, we need
    # the sum of products to be exactly 1.
    
    # Let's try a different approach: use known sub-multiplications.
    # For 4x5x6, we can split into blocks.
    # 
    # Key insight: Use the fact that we can decompose using smaller known results.
    # 
    # Approach: Try to build a valid decomposition with rank < 120.
    # 
    # Let's use a systematic approach:
    # 1. Start with schoolbook (rank 120)
    # 2. Try to merge/simplify terms
    
    # Actually, let's try the greedy set-cover approach properly.
    # 
    # The tensor T[i,j,k] = 1 for all i in range(n), j in range(m), k in range(p)
    # 
    # A rank-1 term is specified by:
    #   U_part: vector of length n*m (indexed by i*m+j)
    #   V_part: vector of length m*p (indexed by j*p+k)
    #   W_part: vector of length p*n (indexed by k*n+i)
    # 
    # The contribution to entry (i,j,k) is:
    #   U_part[i*m+j] * V_part[j*p+k] * W_part[k*n+i]
    # 
    # We need: sum over all terms of the above = 1 for all (i,j,k)
    
    # For sparse terms, let's consider terms where each factor has limited support.
    # 
    # A very simple candidate: for a fixed (i0, j0, k0), the term with
    # U[i0*m+j0]=1, V[j0*p+k0]=1, W[k0*n+i0]=1, all else 0.
    # This gives 1 at (i0,j0,k0) and 0 elsewhere. That's schoolbook.
    
    # To do better, we need terms that cover multiple entries.
    # 
    # Let's try: for a fixed j, consider terms where U has support on {(i,j) for all i},
    # V has support on {(j,k) for all k}, W has support on {(k,i) for all k,i}.
    # 
    # This is getting complex. Let me try a more direct approach.
    
    # Try: use Strassen-like ideas on sub-blocks.
    # 
    # For 4x5x6:
    # Split into:
    #   Block 1: rows 0-1 of A, cols 0-3 of B -> 2x4x4 subproblem
    #   Block 2: rows 2-3 of A, cols 0-3 of B -> 2x4x4 subproblem  
    #   Block 3: rows 0-1 of A, col 4 of B -> 2x1x6 subproblem
    #   Block 4: rows 2-3 of A, col 4 of B -> 2x1x6 subproblem
    # 
    # This doesn't directly help because the column dimension is shared.
    
    # Let me try a completely different approach: 
    # Use the fact that rank(4x5x6) <= rank(4x4x4) + rank(4x1x6) * something
    
    # Actually, let's just try to find ANY valid decomposition with rank < 120
    # using a simple heuristic.
    
    # Simple idea: For each j, we can compute the j-th "slice" using
    # rank 4*6 = 24 operations (schoolbook for 4x6).
    # Total: 5 * 24 = 120. That's schoolbook.
    
    # To do better, we need to share computation across j values.
    
    # Let's try: for pairs of j values, see if we can save.
    
    # Given time constraints, let me try a specific construction:
    # 
    # Use the following: For 4x5x6, note that 5 = 4 + 1.
    # C = A1 * B1 + A2 * B2 where A1 is 4x4, B1 is 4x6, A2 is 4x1, B2 is 1x6.
    # rank(4x4x6) + rank(4x1x6)
    # rank(4x1x6) = 4*1*6 = 24 (it's just outer product, can't do better)
    # rank(4x4x6): we can use Strassen on 4x4x4 part + 4x4x2 part
    #   = rank(4x4x4) + rank(4x4x2)
    #   rank(4x4x4) <= 2*rank(2x2x2) = 2*7 = 14 (using Strassen on 2x2 blocks)
    #   Wait, 4x4x4 can be done with 2 Strassen 2x2x2 operations? No.
    #   4x4x4: split into 2x2 blocks. We have 2x2=4 sub-multiplications of 2x2x2.
    #   Each 2x2x2 needs rank 7, so total 4*7 = 28.
    #   But we can do better with higher-level Strassen.
    #   Actually rank(4x4x4) is known to be at most 49 (from 7x7x7 Strassen applied differently)
    #   or we can use the fact that 4x4x4 can be done in rank 49.
    #   
    # This is getting too theoretical. Let me just implement a valid improvement.
    
    # Practical approach: Use block decomposition with Strassen where possible.
    
    def get_strassen_2x2():
        """Returns U, V, W for 2x2x2 matrix multiplication, rank 7.
        U: shape (4, 7), indexed by [i*2+j] for i,j in {0,1}
        V: shape (4, 7), indexed by [j*2+k] for j,k in {0,1}
        W: shape (4, 7), indexed by [k*2+i] for k,i in {0,1}
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Strassen formulas for 2x2:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1 - M4 + M5 - M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # U indices: [0]=A11, [1]=A12, [2]=A21, [3]=A22
        # V indices: [0]=B11, [1]=B12, [2]=B21, [3]=B22
        # W indices: [0]=C11, [1]=C12, [2]=C21, [3]=C22
        
        # M1: (A11+A22)(B11+B22)
        U[0,0] = 1; U[3,0] = 1
        V[0,0] = 1; V[3,0] = 1
        W[0,0] = 1; W[3,0] = 1  # C11 += M1, C22 += M1
        
        # M2: (A21+A22)B11
        U[2,1] = 1; U[3,1] = 1
        V[0,1] = 1
        W[2,1] = 1  # C21 += M2
        W[3,1] = -1  # C22 -= M2
        
        # M3: A11(B12-B22)
        U[0,2] = 1
        V[1,2] = 1; V[3,2] = -1
        W[1,2] = 1  # C12 += M3
        W[3,2] = 1  # C22 += M3
        
        # M4: A22(B21-B11)
        U[3,3] = 1
        V[2,3] = 1; V[0,3] = -1
        W[0,3] = -1  # C11 -= M4
        W[2,3] = 1   # C21 += M4
        
        # M5: (A11+A12)B22
        U[0,4] = 1; U[1,4] = 1
        V[3,4] = 1
        W[0,4] = 1   # C11 += M5
        W[1,4] = 1   # C12 += M5
        
        # M6: (A21-A11)(B11+B12)
        U[2,5] = 1; U[0,5] = -1
        V[0,5] = 1; V[1,5] = 1
        W[3,5] = 1   # C22 += M6
        
        # M7: (A12-A22)(B21+B22)
        U[1,6] = 1; U[3,6] = -1
        V[2,6] = 1; V[3,6] = 1
        W[0,6] = -1  # C11 -= M7
        
        return U, V, W
    
    # Verify Strassen 2x2x2
    def verify_2x2():
        U, V, W = get_strassen_2x2()
        # Check: for all i,j,k in {0,1}, sum_r U[i*2+j,r]*V[j*2+k,r]*W[k*2+i,r] == 1
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    total = 0
                    for r in range(7):
                        total += U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r]
                    if total != 1:
                        return False
        return True
    
    # Let's verify
    if not verify_2x2():
        # Fall back to schoolbook
        return schoolbook()
    
    # Now build 4x4x4 using 4 Strassen 2x2x2 operations
    # A is 4x4, split into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B11, B12], [B21, B22]] where each is 2x2
    # C = [[C11, C12], [C21, C22]] where each is 2x2
    # 
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    #
    # Each 2x2x2 multiplication uses rank 7.
    # Total for 4x4x4: 4 * 7 = 28
    
    def build_4x4x4():
        """Build 4x4x4 decomposition using 4 Strassen 2x2x2 operations.
        U: shape (16, 28), indexed by [i*4+j]
        V: shape (16, 28), indexed by [j*4+k]
        W: shape (16, 28), indexed by [k*4+i]
        """
        U4 = np.zeros((16, 28))
        V4 = np.zeros((16, 28))
        W4 = np.zeros((16, 28))
        
        U2, V2, W2 = get_strassen_2x2()
        
        # 4 sub-multiplications:
        # (block_row, block_col) in {(0,0), (0,1), (1,0), (1,1)}
        # For block (br, bc):
        #   A block is rows br*2..br*2+1, cols bc*2..bc*2+1
        #   B block is rows bc*2..bc*2+1, cols bc*2..bc*2+1
        #   C block is rows br*2..br*2+1, cols bc*2..bc*2+1
        
        blocks = [(0,0), (0,1), (1,0), (1,1)]
        for idx, (br, bc) in enumerate(blocks):
            r_offset = idx * 7
            # U: A[i, j] where i in [br*2, br*2+2), j in [bc*2, bc*2+2)
            # U index: i*4 + j
            # Within block: local_i = i - br*2, local_j = j - bc*2
            # U2 index: local_i*2 + local_j
            for li in range(2):
                for lj in range(2):
                    global_i = br*2 + li
                    global_j = bc*2 + lj
                    U4[global_i*4 + global_j, r_offset:r_offset+7] = U2[li*2+lj, :]
            
            # V: B[j, k] where j in [bc*2, bc*2+2), k in [bc*2, bc*2+2)
            # V index: j*4 + k
            # Within block: local_j = j - bc*2, local_k = k - bc*2
            # V2 index: local_j*2 + local_k
            for lj in range(2):
                for lk in range(2):
                    global_j = bc*2 + lj
                    global_k = bc*2 + lk
                    V4[global_j*4 + global_k, r_offset:r_offset+7] = V2[lj*2+lk, :]
            
            # W: C[k, i] where k in [bc*2, bc*2+2), i in [br*2, br*2+2)
            # W index: k*4 + i
            # Within block: local_k = k - bc*2, local_i = i - br*2
            # W2 index: local_k*2 + local_i
            for lk in range(2):
                for li in range(2):
                    global_k = bc*2 + lk
                    global_i = br*2 + li
                    W4[global_k*4 + global_i, r_offset:r_offset+7] = W2[lk*2+li, :]
        
        return U4, V4, W4
    
    # Verify 4x4x4
    def verify_4x4x4():
        U4, V4, W4 = build_4x4x4()
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    total = 0
                    for r in range(28):
                        total += U4[i*4+j, r] * V4[j*4+k, r] * W4[k*4+i, r]
                    if total != 1:
                        return False
        return True
    
    if not verify_4x4x4():
        return schoolbook()
    
    # Now build 4x5x6:
    # A is 4x5, B is 5x6, C is 4x6
    # 
    # Split B's columns into 4 + 2:
    # C = A[:, :4] * B[:4, :] + A[:, 4:] * B[4:, :]
    # 
    # Part 1: 4x4x6
    # Part 2: 4x1x6
    #
    # For 4x4x6: split B's 6 columns into 4+2
    #   = 4x4x4 + 4x4x2
    #   4x4x4: rank 28 (as above)
    #   4x4x2: rank 4*4*2 = 32 (schoolbook) -- can we do better?
    #   Actually 4x4x2: for each of the 2 output columns, we do 4x4x1
    #   4x4x1: rank 16 (schoolbook), total 32.
    #
    # For 4x1x6: rank 4*1*6 = 24 (outer product, can't improve)
    #
    # Total: 28 + 32 + 24 = 84
    #
    # Can we do better on 4x4x2? 
    # 4x4x2: C is 4x2. C[:,0] = A*B0, C[:,1] = A*B1 where B0,B1 are 4x1.
    # This is two 4x4x1 problems. Each has rank 16. Total 32.
    # No obvious improvement.
    
    # Let's build 4x4x6:
    def build_4x4x2():
        """Build 4x4x2 decomposition using Strassen 4x4x4 (rank 28).
        We only need output columns k=0,1 (local), so W has shape (8, 28)
        covering W[k*4+i] for k in [0,2), i in [0,4).
        U: shape (16, 28), indexed by [i*4+j]
        V: shape (16, 28), indexed by [j*4+k] for j in [0,4), k in [0,2)
        W: shape (8, 28), indexed by [k*4+i] for k in [0,2), i in [0,4)
        """
        U444, V444, W444 = build_4x4x4()
        # U is the same
        # V: we need V[j*4+k] for k in [0,2) -> first 8 rows of V444
        V442 = V444[:8, :]  # rows 0-7: j*4+k for j=0,1,2,3 and k=0,1
        # W: we need W[k*4+i] for k in [0,2) -> first 8 rows of W444
        W442 = W444[:8, :]  # rows 0-7: k*4+i for k=0,1 and i=0,1,2,3
        return U444, V442, W442

    def build_4x4x6():
        """Build 4x4x6 decomposition.
        U: shape (16, rank), indexed by [i*4+j] for i in [0,4), j in [0,4)
        V: shape (24, rank), indexed by [j*6+k] for j in [0,4), k in [0,6)
        W: shape (24, rank), indexed by [k*4+i] for k in [0,6), i in [0,4)
        """
        # 4x4x4 part (columns 0-3): rank 28
        U444, V444, W444 = build_4x4x4()
        
        # 4x4x2 part (columns 4-5): rank 28 (using Strassen 4x4x4, truncated)
        U442, V442, W442 = build_4x4x2()
        
        rank = 28 + 28
        U = np.zeros((16, rank))
        V = np.zeros((24, rank))
        W = np.zeros((24, rank))
        
        # Part 1: 4x4x4 (output columns k=0..3)
        U[:, :28] = U444
        V[:16, :28] = V444  # V444 has shape (16, 28): j*4+k for j,k in [0,4)
        W[:16, :28] = W444  # W444 has shape (16, 28): k*4+i for k,i in [0,4)
        
        # Part 2: 4x4x2 (output columns k=4..5)
        U[:, 28:] = U442
        # V442 has shape (8, 28): local j*4+k for j in [0,4), k in [0,2)
        # Global: j*6 + (4+k) for j in [0,4), k in [0,2)
        for j in range(4):
            for lk in range(2):
                V[j*6 + 4 + lk, 28:] = V442[j*4 + lk, :]
        # W442 has shape (8, 28): local k*4+i for k in [0,2), i in [0,4)
        # Global: (4+k)*4 + i for k in [0,2), i in [0,4)
        for lk in range(2):
            for li in range(4):
                W[(4+lk)*4 + li, 28:] = W442[lk*4 + li, :]
        
        return U, V, W
    
    # Verify 4x4x6
    def verify_4x4x6():
        U, V, W = build_4x4x6()
        rank = U.shape[1]
        for i in range(4):
            for j in range(4):
                for k in range(6):
                    total = 0.0
                    for r in range(rank):
                        total += U[i*4+j, r] * V[j*6+k, r] * W[k*4+i, r]
                    if abs(total - 1.0) > 1e-10:
                        return False
        return True
    
    if not verify_4x4x6():
        return schoolbook()
    
    # Build 4x1x6:
    def build_4x1x6():
        """Build 4x1x6 decomposition. Rank 24.
        U: shape (4, 24), indexed by [i*1+j] = [i] for i in [0,4), j=0
        V: shape (6, 24), indexed by [j*6+k] for j=0, k in [0,6)
        W: shape (24, 24), indexed by [k*4+i] for k in [0,6), i in [0,4)
        """
        U = np.zeros((4, 24))
        V = np.zeros((6, 24))
        W = np.zeros((24, 24))
        col = 0
        for i in range(4):
            for k in range(6):
                U[i, col] = 1
                V[k, col] = 1
                W[k*4+i, col] = 1
                col += 1
        return U, V, W
    
    # Verify 4x1x6
    def verify_4x1x6():
        U, V, W = build_4x1x6()
        for i in range(4):
            for k in range(6):
                total = 0
                for r in range(24):
                    total += U[i, r] * V[k, r] * W[k*4+i, r]
                if total != 1:
                    return False
        return True
    
    if not verify_4x1x6():
        return schoolbook()
    
    # Now combine for 4x5x6:
    # A is 4x5, B is 5x6, C is 4x6
    # 
    # U: shape (20, rank), indexed by [i*5+j] for i in [0,4), j in [0,5)
    # V: shape (30, rank), indexed by [j*6+k] for j in [0,5), k in [0,6)
    # W: shape (24, rank), indexed by [k*4+i] for k in [0,6), i in [0,4)
    
    def build_4x5x6():
        U446, V446, W446 = build_4x4x6()
        U416, V416, W416 = build_4x1x6()
        
        rank = 56 + 24  # 80
        
        U = np.zeros((20, rank))
        V = np.zeros((30, rank))
        W = np.zeros((24, rank))
        
        # 4x4x6 part: A[:, 0:4] * B[0:4, :]
        # U446[i*4+j, r] -> U[i*5+j, r]
        for i in range(4):
            for j in range(4):
                U[i*5+j, :56] = U446[i*4+j, :]
        
        # V446[j*6+k, r] -> V[j*6+k, r] (j in [0,4), same indexing)
        V[:24, :56] = V446
        W[:, :56] = W446
        
        # 4x1x6 part: A[:, 4] * B[4, :]
        # U416[i, r] -> U[i*5+4, r]
        for i in range(4):
            U[i*5+4, 56:80] = U416[i, :]
        
        # V416[k, r] -> V[4*6+k, r] = V[24+k, r]
        V[24:, 56:80] = V416
        W[:, 56:80] = W416
        
        return U, V, W
    
    # Verify 4x5x6
    def verify_4x5x6():
        U, V, W = build_4x5x6()
        rank = U.shape[1]
        for i in range(4):
            for j in range(5):
                for k in range(6):
                    total = 0.0
                    for r in range(rank):
                        total += U[i*5+j, r] * V[j*6+k, r] * W[k*4+i, r]
                    if abs(total - 1.0) > 1e-10:
                        return False
        return True
    
    if time.time() - start_time > DEADLINE:
        return schoolbook()
    
    U, V, W = build_4x5x6()
    rank = U.shape[1]
    
    # Final verification with exact integer arithmetic
    U_int = np.round(U).astype(int)
    V_int = np.round(V).astype(int)
    W_int = np.round(W).astype(int)
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U_int[i*m+j, r] * V_int[j*p+k, r] * W_int[k*n+i, r]
                if total != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        return schoolbook()
    
    # Convert to lists of lists
    U_list = U_int.tolist()
    V_list = V_int.tolist()
    W_list = W_int.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
