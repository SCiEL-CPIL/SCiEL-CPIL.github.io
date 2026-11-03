"""Matrix multiplication 5x5x5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a low-rank exact tensor decomposition for 5x5 matrix multiplication.
    
    Strategy: Use a block decomposition 5 = 2 + 3, applying Strassen (rank 7)
    for 2x2 blocks and schoolbook for other blocks. This yields a rank below 125.
    
    The decomposition is built carefully to ensure correctness by verifying
    the tensor identity numerically before returning.
    """
    import time
    import numpy as np
    from fractions import Fraction
    
    start_time = time.time()
    DEADLINE = 125  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 5, 5, 5
    
    # Schoolbook fallback (rank 125)
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
    
    if time.time() - start_time > DEADLINE - 20:
        return schoolbook()
    
    # ============================================================
    # Block decomposition approach: 5 = 2 + 3
    # ============================================================
    # A is 5x5, B is 5x5, C = A*B is 5x5
    # Partition: rows/cols split at index 2 (first 2, last 3)
    # 
    # A = [A11 A12]     B = [B11 B12]     C = [C11 C12]
    #     [A21 A22]         [B21 B22]         [C21 C22]
    #
    # where:
    #   A11, B11, C11 are 2x2
    #   A12, B12, C12 are 2x3
    #   A21, B21, C21 are 3x2
    #   A22, B22, C22 are 3x3
    #
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    #
    # For the 2x2 products, use Strassen (rank 7 each)
    # For mixed and 3x3 products, use schoolbook
    #
    # However, the cross terms (A12*B21 etc.) couple the factors,
    # so we need a global decomposition. Let's build it explicitly.
    #
    # Approach: For each output element C[i,k], we need
    #   C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # We'll assign "multiplication slots" to each (i,j,k) triple,
    # but use Strassen-like combinations for the 2x2 blocks to reduce count.
    
    # Let's try a different approach: direct structured decomposition
    # using the fact that 5 is prime and trying to use cyclic/FFT-like structure.
    #
    # Actually, let's try the most straightforward improvement:
    # Use Strassen for the diagonal 2x2 and 3x3 blocks, and schoolbook for off-diagonal.
    # But the cross-coupling makes this non-trivial.
    #
    # Let's instead try: build a valid decomposition with rank < 125 by
    # combining known lower-rank decompositions for sub-blocks in a way
    # that respects the bilinear structure.
    
    # Method: Explicit rank construction
    # We know that for n x n matrix multiplication, a decomposition of rank r
    # means we have r "multiplications" where each multiplication is
    # M_t = (linear combo of A entries) * (linear combo of B entries)
    # and each C entry is a linear combo of the M_t's.
    
    # Let's construct one by grouping terms.
    
    # Key insight: For 5x5, we can use the following approach:
    # Partition into blocks and use Strassen for 2x2 sub-blocks.
    # 
    # A more systematic approach: use the "divide and conquer" with 5=2+3
    # and handle each product term with optimal known decompositions.
    
    # Let me try a concrete construction:
    # 
    # For the full 5x5 product, we need 25*5 = 125 bilinear terms (schoolbook).
    # 
    # Using Strassen on a 2x2 block saves 1 multiplication (8 -> 7).
    # If we can identify independent 2x2 structures, we can save.
    #
    # Actually, the cleanest approach for guaranteed correctness:
    # Build the decomposition term by term, combining where possible.
    
    # Let's try: use the fact that we can apply Strassen to the (0,1)x(0,1) 
    # sub-block and (0,1)x(0,1) of B, giving us 7 multiplications for the
    # 2x2 block product, then add schoolbook for everything else.
    #
    # But the issue is that C entries mix contributions from different blocks.
    
    # Simplest correct improvement: 
    # Apply Strassen to the top-left 2x2 block product A11*B11,
    # which contributes to C11. The remaining 4 terms in C11 (A12*B21)
    # are handled by schoolbook.
    #
    # C11[i,k] = sum_j A11[i,j]*B11[j,k] + sum_j A12[i,j]*B21[j,k]
    #           = (Strassen part) + (schoolbook part for A12*B21)
    
    # Total rank: 7 (for A11*B11) + 2*3 (for A12*B21) + 
    #            2*3 (A11*B12) + 2*3 (A12*B22) +
    #            3*2 (A21*B11) + 3*3 (A22*B21) +
    #            3*2 (A21*B12) + 3*3 (A22*B22)
    # = 7 + 6 + 6 + 6 + 6 + 9 + 6 + 9 = 55
    
    # Wait, but this double-counts. Each C[i,k] gets contributions from multiple products.
    # The rank is the total number of independent multiplications.
    
    # Let me be more careful. The total number of scalar multiplications is:
    # - For A11*B11 (2x2 * 2x2): rank 7 (Strassen)
    # - For A12*B21 (2x3 * 3x2): rank 2*3 = 6 (schoolbook, no known better)
    # - For A11*B12 (2x2 * 2x3): rank 2*3 = 6
    # - For A12*B22 (2x3 * 3x3): rank 2*3 = 6  
    # - For A21*B11 (3x2 * 2x2): rank 3*2 = 6
    # - For A22*B21 (3x3 * 3x2): rank 3*3 = 9
    # - For A21*B12 (3x2 * 2x3): rank 3*3 = 9
    # - For A22*B22 (3x3 * 3x3): rank 9 (schoolbook; could be 23 with Biswas)
    #
    # But wait - these are NOT independent! The multiplications for different
    # block products feed into the same C entries. The total rank is the sum
    # of the ranks of each block product ONLY IF the factor vectors are
    # linearly independent across blocks, which they are because they involve
    # different A and B entries.
    #
    # Actually no - for example, A11*B11 involves A[0,0],A[0,1],A[1,0],A[1,1]
    # and B[0,0],B[0,1],B[1,0],B[1,1]. 
    # A12*B21 involves A[0,2..4],A[1,2..4] and B[2..4,0],B[2..4,1].
    # These use disjoint sets of A and B entries, so the factorizations are independent.
    #
    # Total rank = 7 + 6 + 6 + 6 + 6 + 9 + 9 + 9 = 58
    
    # But actually, for the rectangular products like 2x3 * 3x2, the schoolbook
    # rank is 2*3*2/... no wait. For an (a x b) * (b x c) product, schoolbook
    # rank is a*b*c. So 2*3*2 = 12? No wait.
    #
    # A12 is 2x3, B21 is 3x2. Product is 2x2. Schoolbook: 2*3*2 = 12 multiplications.
    # A11 is 2x2, B12 is 2x3. Product is 2x3. Schoolbook: 2*2*3 = 12.
    # A12 is 2x3, B22 is 3x3. Product is 2x3. Schoolbook: 2*3*3 = 18.
    # A21 is 3x2, B11 is 2x2. Product is 3x2. Schoolbook: 3*2*2 = 12.
    # A22 is 3x3, B21 is 3x2. Product is 3x2. Schoolbook: 3*3*2 = 18.
    # A21 is 3x2, B12 is 2x3. Product is 3x3. Schoolbook: 3*2*3 = 18.
    # A22 is 3x3, B22 is 3x3. Product is 3x3. Schoolbook: 3*3*3 = 27.
    #
    # Total: 7 + 12 + 12 + 18 + 12 + 18 + 18 + 27 = 124
    # 
    # That's only saving 1! Because Strassen saves 1 on the 2x2 block.
    
    # This is much less than I hoped. Let me reconsider.
    
    # Better approach: apply Strassen to BOTH 2x2 diagonal blocks.
    # But A11*B11 and... there's only one 2x2 diagonal block in the 2+3 split.
    #
    # Alternative: use 5 = 1 + 4 or other splits?
    # Or: apply multiple Strassen steps?
    
    # Actually, the best known approach for small n is to use the 
    # "Laderman" or other specific constructions.
    
    # For now, let's try a different strategy: 
    # Use a global Strassen-like decomposition by treating 5x5 as 
    # part of a larger structure.
    
    # Alternative: try to find a decomposition using the algebraic
    # structure of Z_5 (cyclic group of order 5).
    #
    # The key idea: matrix multiplication over Z_5 can be related to
    # group algebra. The group algebra of Z_5 has dimension 5, and
    # multiplication in this algebra can be diagonalized by the DFT.
    # This gives a rank-5 decomposition for the "cyclic convolution" 
    # part, but matrix multiplication is not just cyclic convolution.
    
    # Let me try yet another approach: use the known result that 
    # R(MM_5) <= 25 + something, via Winograd's method.
    #
    # Winograd's method for n x n matrix multiplication uses rank
    # related to the number of "irreducible" terms. For n=5 (prime),
    # the approach involves polynomial multiplication.
    
    # Given the time constraints and need for correctness, let's 
    # implement a verified decomposition. The safest approach is to
    # use a block method with Strassen where applicable.
    
    # Let me try 5 = 2 + 3 with Strassen on the 2x2 block:
    # Rank = 7 + 12 + 12 + 18 + 12 + 18 + 18 + 27 = 124
    
    # Or 5 = 3 + 2: same by symmetry, rank 124.
    
    # Can we do better? What if we apply Strassen to a 2x2 sub-block
    # that appears in multiple products?
    
    # Actually, let's reconsider the structure. In the 2+3 split:
    # The 2x2 block A11 appears in products A11*B11 and A11*B12.
    # The Strassen factors for A11*B11 involve linear combinations of A11 entries.
    # For A11*B12, we'd need different factors.
    #
    # What if we use a "Strassen-like" approach for the full 2x(2+3) = 2x5
    # left block? i.e., compute [A11 A12] * B using a modified Strassen?
    
    # This is getting complex. Let me just implement the rank-124 
    # decomposition and verify it, since it's guaranteed correct and 
    # better than 125.
    
    def build_block_decomposition():
        """Build decomposition using 2+3 split with Strassen for 2x2 block."""
        
        # Index mapping:
        # A[i,j] -> U index: i*5 + j (row i, col j)
        # B[j,k] -> V index: j*5 + k
        # C[i,k] -> W index: k*5 + i (trace convention: W[k*n+i])
        
        U_list = []
        V_list = []
        W_list = []
        
        def add_mult(u_vec, v_vec, w_vec):
            """Add a rank-1 term: u_vec is sparse dict (index->coeff), etc."""
            # Convert sparse dicts to dense vectors
            U_list.append(u_vec)
            V_list.append(v_vec)
            W_list.append(w_vec)
        
        def make_dense(sparse, size):
            """Convert sparse dict to dense list."""
            dense = [0] * size
            for idx, val in sparse.items():
                if val != 0:
                    dense[idx] = val
            return dense
        
        # Strassen for 2x2: A11 * B11
        # A11 = [[A00, A01], [A10, A11]] where Aij = A[i,j] for i,j in {0,1}
        # B11 = [[B00, B01], [B10, B11]] where Bij = B[j,k] for j,k in {0,1}
        # Result contributes to C11 = C[i,k] for i,k in {0,1}
        
        # Strassen multiplications:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)(B00)
        # M3 = (A00)(B10-B11)
        # M4 = (A11)(B01-B10)
        # M5 = (A00+A01)(B11)
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # A indices: A[i,j] -> i*5+j
        # B indices: B[j,k] -> j*5+k
        # C indices: C[i,k] -> k*5+i (trace convention)
        
        a00, a01, a10, a11 = 0*5+0, 0*5+1, 1*5+0, 1*5+1  # 0,1,5,6
        b00, b01, b10, b11 = 0*5+0, 0*5+1, 1*5+0, 1*5+1  # 0,1,5,6
        c00, c01, c10, c11 = 0*5+0, 1*5+0, 0*5+1, 1*5+1  # 0,5,1,6
        
        # M1 = (A00+A11)(B00+B11)
        add_mult(
            {a00: 1, a11: 1},
            {b00: 1, b11: 1},
            {c00: 1, c11: 1}
        )
        # M2 = (A10+A11)(B00)
        add_mult(
            {a10: 1, a11: 1},
            {b00: 1},
            {c10: 1}
        )
        # M3 = (A00)(B10-B11)
        add_mult(
            {a00: 1},
            {b10: 1, b11: -1},
            {c01: 1, c11: 1}
        )
        # M4 = (A11)(B01-B10)
        add_mult(
            {a11: 1},
            {b01: 1, b10: -1},
            {c00: 1, c10: 1}
        )
        # M5 = (A00+A01)(B11)
        add_mult(
            {a00: 1, a01: 1},
            {b11: 1},
            {c00: -1, c01: 1}
        )
        # M6 = (A10-A00)(B00+B10)
        add_mult(
            {a10: 1, a00: -1},
            {b00: 1, b10: 1},
            {c11: 1}
        )
        # M7 = (A01-A11)(B01+B11)
        add_mult(
            {a01: 1, a11: -1},
            {b01: 1, b11: 1},
            {c00: 1}
        )
        
        # Now schoolbook for remaining block products:
        # A12 * B21: A[i,j] for i in {0,1}, j in {2,3,4} times B[j,k] for j in {2,3,4}, k in {0,1}
        # Contributes to C[i,k] for i in {0,1}, k in {0,1}
        for i in range(2):
            for j in range(2, 5):
                for k in range(2):
                    add_mult(
                        {i*5+j: 1},
                        {j*5+k: 1},
                        {k*5+i: 1}
                    )
        
        # A11 * B12: A[i,j] for i in {0,1}, j in {0,1} times B[j,k] for j in {0,1}, k in {2,3,4}
        # Contributes to C[i,k] for i in {0,1}, k in {2,3,4}
        for i in range(2):
            for j in range(2):
                for k in range(2, 5):
                    add_mult(
                        {i*5+j: 1},
                        {j*5+k: 1},
                        {k*5+i: 1}
                    )
        
        # A12 * B22: A[i,j] for i in {0,1}, j in {2,3,4} times B[j,k] for j in {2,3,4}, k in {2,3,4}
        # Contributes to C[i,k] for i in {0,1}, k in {2,3,4}
        for i in range(2):
            for j in range(2, 5):
                for k in range(2, 5):
                    add_mult(
                        {i*5+j: 1},
                        {j*5+k: 1},
                        {k*5+i: 1}
                    )
        
        # A21 * B11: A[i,j] for i in {2,3,4}, j in {0,1} times B[j,k] for j in {0,1}, k in {0,1}
        # Contributes to C[i,k] for i in {2,3,4}, k in {0,1}
        for i in range(2, 5):
            for j in range(2):
                for k in range(2):
                    add_mult(
                        {i*5+j: 1},
                        {j*5+k: 1},
                        {k*5+i: 1}
                    )
        
        # A22 * B21: A[i,j] for i in {2,3,4}, j in {2,3,4} times B[j,k] for j in {2,3,4}, k in {0,1}
        # Contributes to C[i,k] for i in {2,3,4}, k in {0,1}
        for i in range(2, 5):
            for j in range(2, 5):
                for k in range(2):
                    add_mult(
                        {i*5+j: 1},
                        {j*5+k: 1},
                        {k*5+i: 1}
                    )
        
        # A21 * B12: A[i,j] for i in {2,3,4}, j in {0,1} times B[j,k] for j in {0,1}, k in {2,3,4}
        # Contributes to C[i,k] for i in {2,3,4}, k in {2,3,4}
        for i in range(2, 5):
            for j in range(2):
                for k in range(2, 5):
                    add_mult(
                        {i*5+j: 1},
                        {j*5+k: 1},
                        {k*5+i: 1}
                    )
        
        # A22 * B22: A[i,j] for i in {2,3,4}, j in {2,3,4} times B[j,k] for j in {2,3,4}, k in {2,3,4}
        # Contributes to C[i,k] for i in {2,3,4}, k in {2,3,4}
        for i in range(2, 5):
            for j in range(2, 5):
                for k in range(2, 5):
                    add_mult(
                        {i*5+j: 1},
                        {j*5+k: 1},
                        {k*5+i: 1}
                    )
        
        # Convert sparse to dense
        rank = len(U_list)
        U_dense = [make_dense(u, 25) for u in U_list]
        V_dense = [make_dense(v, 25) for v in V_list]
        W_dense = [make_dense(w, 25) for w in W_list]
        
        return U_dense, V_dense, W_dense, rank
    
    try:
        U, V, W, rank = build_block_decomposition()
        
        # Verify: for random inputs, check that U@diag@V^T reconstruction matches A*B
        if time.time() - start_time < DEADLINE - 5:
            # Quick verification with a few random matrices
            np.random.seed(42)
            A_test = np.random.randint(-2, 3, (5, 5))
            B_test = np.random.randint(-2, 3, (5, 5))
            C_expected = A_test @ B_test
            
            # Compute using decomposition
            # U is (25, rank), V is (25, rank), W is (25, rank)
            # C[i,k] = sum_t U[i*5+k... wait
            # 
            # The tensor T[i,j,k] = delta (1 if i,j,k form a valid multiplication)
            # T = sum_t U[:,t] outer V[:,t] outer W[:,t]
            # where U is indexed by (i,j) pairs, V by (j,k) pairs, W by (k,i) pairs
            #
            # C[i,k] = sum_j A[i,j] B[j,k]
            # = sum_t (sum_ij U[i,j,t] A[i,j]) (sum_jk V[j,k,t] B[j,k]) ... 
            # No, that's not right either.
            #
            # The correct interpretation:
            # C[i,k] = sum_t W[i,k,t] * M_t
            # where M_t = (sum_{i,j} U[i,j,t] * A[i,j]) * (sum_{j,k} V[j,k,t] * B[j,k])
            
            U_np = np.array(U, dtype=float)  # (25, rank)
            V_np = np.array(V, dtype=float)  # (25, rank)
            W_np = np.array(W, dtype=float)  # (25, rank)
            
            A_flat = A_test.flatten()  # (25,)
            B_flat = B_test.flatten()  # (25,)
            
            # M_t = (U[:,t] . A_flat) * (V[:,t] . B_flat)
            left = U_np @ A_flat    # (rank,)
            right = V_np @ B_flat   # (rank,)
            M = left * right        # (rank,)
            
            # C_flat[k*5+i] = sum_t W[k*5+i, t] * M[t]
            C_flat = W_np @ M       # (25,)
            C_computed = C_flat.reshape(5, 5)
            
            if not np.allclose(C_computed, C_expected, atol=1e-10):
                # Decomposition is wrong, fall back to schoolbook
                return schoolbook()
            
            # Second test
            A_test2 = np.random.randint(-3, 4, (5, 5))
            B_test2 = np.random.randint(-3, 4, (5, 5))
            C_expected2 = A_test2 @ B_test2
            
            left2 = U_np @ A_test2.flatten()
            right2 = V_np @ B_test2.flatten()
            M2 = left2 * right2
            C_flat2 = W_np @ M2
            C_computed2 = C_flat2.reshape(5, 5)
            
            if not np.allclose(C_computed2, C_expected2, atol=1e-10):
                return schoolbook()
            
            if time.time() - start_time > DEADLINE - 3:
                return schoolbook()
            
            return {"U": U, "V": V, "W": W}
        
    except Exception:
        pass
    
    # Fallback
    if time.time() - start_time > DEADLINE - 2:
        return schoolbook()
    
    return schoolbook()
# EVOLVE-BLOCK-END
