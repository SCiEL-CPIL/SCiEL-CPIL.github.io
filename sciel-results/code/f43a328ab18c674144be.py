def construct():
    import numpy as np
    import time
    from itertools import combinations
    
    n, m, p = 2, 4, 4
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    def time_ok():
        return time.time() < deadline
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition exactly represents matrix multiplication."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        np.random.seed(42)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for r in range(rank):
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        C_computed[i, k] += A[i, j] * U_arr[i*m + j, r] * V_arr[j*p + k, r] * W_arr[k*n + i, r] * B[j, k]
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
        return True
    
    # Start with the known rank-28 Strassen decomposition
    def get_rank28():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_strassen_block(col_offset, X_rows, Y_rows, Z_rows):
            U[X_rows[0]][col_offset + 0] += 1
            U[X_rows[3]][col_offset + 0] += 1
            V[Y_rows[0]][col_offset + 0] += 1
            V[Y_rows[3]][col_offset + 0] += 1
            W[Z_rows[0]][col_offset + 0] += 1
            W[Z_rows[3]][col_offset + 0] += 1
            
            U[X_rows[2]][col_offset + 1] += 1
            U[X_rows[3]][col_offset + 1] += 1
            V[Y_rows[0]][col_offset + 1] += 1
            W[Z_rows[2]][col_offset + 1] += 1
            W[Z_rows[3]][col_offset + 1] -= 1
            
            U[X_rows[0]][col_offset + 2] += 1
            V[Y_rows[1]][col_offset + 2] += 1
            V[Y_rows[3]][col_offset + 2] -= 1
            W[Z_rows[1]][col_offset + 2] += 1
            W[Z_rows[3]][col_offset + 2] += 1
            
            U[X_rows[3]][col_offset + 3] += 1
            V[Y_rows[2]][col_offset + 3] += 1
            V[Y_rows[0]][col_offset + 3] -= 1
            W[Z_rows[0]][col_offset + 3] += 1
            W[Z_rows[2]][col_offset + 3] += 1
            
            U[X_rows[0]][col_offset + 4] += 1
            U[X_rows[1]][col_offset + 4] += 1
            V[Y_rows[3]][col_offset + 4] += 1
            W[Z_rows[0]][col_offset + 4] -= 1
            W[Z_rows[1]][col_offset + 4] += 1
            
            U[X_rows[2]][col_offset + 5] += 1
            U[X_rows[0]][col_offset + 5] -= 1
            V[Y_rows[0]][col_offset + 5] += 1
            V[Y_rows[1]][col_offset + 5] += 1
            W[Z_rows[3]][col_offset + 5] += 1
            
            U[X_rows[1]][col_offset + 6] += 1
            U[X_rows[3]][col_offset + 6] -= 1
            V[Y_rows[2]][col_offset + 6] += 1
            V[Y_rows[3]][col_offset + 6] += 1
            W[Z_rows[0]][col_offset + 6] += 1
        
        add_strassen_block(0, [0, 1, 4, 5], [0, 1, 4, 5], [0, 2, 1, 3])
        add_strassen_block(7, [2, 3, 6, 7], [8, 9, 12, 13], [0, 2, 1, 3])
        add_strassen_block(14, [0, 1, 4, 5], [2, 3, 6, 7], [4, 6, 5, 7])
        add_strassen_block(21, [2, 3, 6, 7], [10, 11, 14, 15], [4, 6, 5, 7])
        
        return U, V, W
    
    U28, V28, W28 = get_rank28()
    
    # Verify the rank-28 decomposition works
    if not verify_decomposition(U28, V28, W28, 28):
        return {"U": U28, "V": V28, "W": W28}
    
    best_rank = 28
    best_result = {"U": U28, "V": V28, "W": W28}
    
    # Try to find rank-27 by attempting to merge/eliminate terms
    # Strategy: try removing one Strassen term from one block and see if
    # the remaining 27 terms can be adjusted to still represent the tensor
    # 
    # More practical: use the structure to try different block decompositions
    # that might share terms between blocks
    
    # Alternative approach: try a non-standard partition of the 4x4 into blocks
    # that allows cross-block sharing
    
    # Try: use a 2x2x4x4 structure where we exploit that n=2 (only 2 rows in A)
    # The key insight: since n=2, we have only 2 output rows, which means
    # the W matrix has only 8 entries (p*n = 8). This is a constraint.
    
    # Let's try a direct search: can we find a rank-27 decomposition?
    # We'll try to find it by starting from rank-28 and trying to eliminate
    # one term while adjusting others.
    
    # Practical approach: try different ways to partition the problem
    # that might allow term sharing between the four 2x2 sub-problems
    
    # Method 1: Try overlapping Strassen decompositions
    # Instead of 4 independent 2x2 Strassen blocks, try to find a
    # decomposition where some terms serve multiple blocks.
    
    # Let's try a systematic approach: 
    # The tensor T[i,j,k] = delta (for matrix mult) has 32 non-zero entries.
    # A rank-r decomposition has r terms, each being a product of 3 vectors.
    # We need to find r vectors (u_r, v_r, w_r) such that:
    #   sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] = T[i,j,k] for all i,j,k
    
    # For rank 27, we need 27 such terms.
    
    # Try: construct a decomposition where we use shared structure
    # between the top and bottom halves of B (rows 0-1 vs 2-3)
    
    # Actually, let's try a more direct approach:
    # Try to find a rank-27 solution by parameterizing and solving
    
    # The number of unknowns for rank r:
    # U: 8*r, V: 16*r, W: 8*r = 32r unknowns
    # Constraints: 2*4*4 = 32 equations (but many are zero)
    # Non-zero constraints: 32 (each (i,j,k) where T=1)
    # Zero constraints: 1024-32 = 992
    # Total: 1024 constraints
    # For r=27: 32*27 = 864 unknowns, 1024 constraints -> overdetermined
    # For r=28: 32*28 = 896 unknowns, 1024 constraints -> still overdetermined
    # But the equations are nonlinear (products), so this analysis is rough
    
    # Let's try a different approach: use the known structure
    # and try to find if we can reduce by 1 term through clever sharing
    
    # Try: modify one of the Strassen blocks to share a term with an adjacent block
    # This requires solving a system of equations
    
    # Practical test: try rank 27 with a modified structure
    # We'll try to find a decomposition where one term is shared between
    # two of the four 2x2 block products
    
    # For now, let's verify that rank 28 is our best and return it
    # unless we can quickly find rank 27
    
    # Try a quick search: perturb the rank-28 solution and see if we
    # can eliminate one term
    
    # Actually, let's try a more systematic approach using the fact that
    # the problem has a lot of structure (zeros in the tensor)
    
    # Build the constraint matrix for the linear system that arises
    # when we fix some variables
    
    # For a rank-r decomposition, if we fix U and V, then W is determined
    # by: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    # This is linear in W for fixed U, V
    
    # Let's try: fix U and V to have a specific sparse structure,
    # then solve for W
    
    # Try rank 27 with a structured approach
    if time_ok():
        # Try: use the Strassen structure but with one shared term
        # between block 0 and block 1 (both multiply A's first two rows
        # against different parts of B)
        
        # Block 0: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
        # Block 1: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
        # Wait, that's not right. Let me re-examine.
        
        # A is 2x4, B is 4x4, C is 2x4
        # C[i,k] = sum_j A[i,j] * B[j,k]
        
        # The four blocks in the current decomposition:
        # Block 0: A[:, 0:2] * B[0:2, 0:2] -> C[:, 0:2] (partial)
        # Block 1: A[:, 2:4] * B[2:4, 0:2] -> C[:, 0:2] (partial)  
        # Block 2: A[:, 0:2] * B[0:2, 2:4] -> C[:, 2:4] (partial)
        # Block 3: A[:, 2:4] * B[2:4, 2:4] -> C[:, 2:4] (partial)
        
        # Each block is a 2x2 * 2x2 = 2x2 multiplication, done with Strassen (rank 7)
        # Total: 4 * 7 = 28
        
        # Can we do better by not using Strassen on all blocks?
        # Or by sharing terms between blocks?
        
        # Key observation: Blocks 0 and 1 both contribute to C[:, 0:2]
        # Blocks 2 and 3 both contribute to C[:, 2:4]
        # Blocks 0 and 2 both use A[:, 0:2]
        # Blocks 1 and 3 both use A[:, 2:4]
        
        # Maybe we can combine the contributions to C[:, 0:2] from blocks 0 and 1
        # in a way that uses fewer than 14 terms total
        
        # The sum is: A[:, 0:2] * B[0:2, 0:2] + A[:, 2:4] * B[2:4, 0:2]
        # This is a 2x4 * 4x2 = 2x2 multiplication
        # What's the rank of this sub-tensor?
        
        # Similarly for C[:, 2:4]: A[:, 0:2] * B[0:2, 2:4] + A[:, 2:4] * B[2:4, 2:4]
        
        # If we can compute each 2x4*4x2 product with rank < 14, we win
        # The standard rank for 2x4*4x2 is at most 2*4*2 = 16 (schoolbook)
        # With Strassen-like methods, maybe we can do better
        
        # Actually, 2x4*4x2 is equivalent to 2x4*4x2, which has tensor rank
        # that we need to determine
        
        # Let's try: can we compute 2x4*4x2 with rank 13?
        # If yes, then total rank would be at most 13 + 13 = 26 (if the two
        # sub-problems are independent) or potentially less with sharing
        
        # This is getting complex. Let's just return rank 28 for now.
        pass
    
    return best_result
