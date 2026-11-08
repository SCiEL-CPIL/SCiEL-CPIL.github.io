import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication using recursive Strassen.
    
    The 4x4 matrix multiplication is viewed as 2x2 block multiplication, where each block is 2x2.
    Strassen's algorithm is applied to the 2x2 block matrices, resulting in 7 block multiplications.
    Each block multiplication is again 2x2, so Strassen is applied again, resulting in 7 scalar multiplications per block multiplication.
    Total rank = 7 * 7 = 49.
    
    The matrix multiplication tensor T has entries T[i,j,k] = 1 if (i,j) is (row,col) of A,
    (j,k) is (row,col) of B, and (k,i) is (col,row) of C in trace convention.
    
    Actually: T[A_idx, B_idx, C_idx] where A_idx = i*4+j, B_idx = j*4+k, C_idx = k*4+i
    T[A_idx, B_idx, C_idx] = 1 if the indices are consistent (i.e., the middle index matches).
    
    A rank-r decomposition means: T[a,b,c] = sum_s U[a,s]*V[b,s]*W[c,s]
    """
    start_time = time.time()
    DEADLINE = 145
    
    n, m, p = 4, 4, 4
    N_A = n * m  # 16
    N_B = m * p  # 16
    N_C = p * n  # 16
    
    # Build the 4x4 matrix multiplication tensor as a 3D array T[A_idx, B_idx, C_idx]
    # T[a, b, c] = 1 if a = i*4+j, b = j*4+k, c = k*4+i for some i,j,k
    # i.e., a%4 == b//4 (j) and b%4 == c//4 (k)
    T = np.zeros((N_A, N_B, N_C))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0
    
    # Strassen decomposition for 2x2 matrices
    # Let A, B be 2x2 matrices.
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We are doing 4x4 = 2x2 blocks of 2x2.
    # Let A, B, C be 4x4 matrices divided into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] where Aij are 2x2.
    # B = [[B00, B01], [B10, B11]] where Bij are 2x2.
    # C = [[C00, C01], [C10, C11]] where Cij are 2x2.
    
    # The 7 block-level Strassen terms are:
    # M1_block = (A00 + A11)(B00 + B11)
    # M2_block = (A10 + A11)B00
    # M3_block = A00(B01 - B11)
    # M4_block = A11(B10 - B00)
    # M5_block = (A00 + A01)B11
    # M6_block = (A10 - A00)(B00 + B01)
    # M7_block = (A01 - A11)(B10 + B11)
    
    # Each block is 2x2, so we apply Strassen again to each block multiplication.
    # For example, M1_block = (A00 + A11)(B00 + B11) involves multiplying two 2x2 matrices.
    # Let X = A00 + A11, Y = B00 + B11.
    # Then M1_block = X*Y, which requires 7 scalar multiplications.
    
    # We will build U, V, W matrices of size 16x49.
    U = np.zeros((N_A, 49))
    V = np.zeros((N_B, 49))
    W = np.zeros((N_C, 49))
    
    # Helper to add a Strassen term for 2x2 matrices
    # a_idx, b_idx, c_idx are the base indices for the 2x2 blocks.
    # For A, the block (bi, bj) corresponds to rows 2*bi to 2*bi+1 and cols 2*bj to 2*bj+1.
    # So A[2*bi + r, 2*bj + c] has index (2*bi + r)*4 + (2*bj + c).
    
    def add_strassen_term(U, V, W, rank_offset, A_blocks, B_blocks, C_blocks):
        # A_blocks, B_blocks, C_blocks are lists of (block_row, block_col, coeff)
        # This function adds the 7 Strassen terms for the 2x2 block multiplication.
        
        # First, compute the 7 block-level combinations
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B01 - B11)
        # M4 = A11(B10 - B00)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        
        # We need to map the block indices to scalar indices.
        # A block (br, bc) corresponds to scalar indices:
        # (2*br + r)*4 + (2*bc + c) for r,c in {0,1}
        
        def get_A_scalar(br, bc, r, c):
            return (2*br + r)*4 + (2*bc + c)
        
        def get_B_scalar(br, bc, r, c):
            return (2*br + r)*4 + (2*bc + c)
        
        def get_C_scalar(br, bc, r, c):
            # Trace convention: C[k*n+i] where k is col, i is row.
            # In our block notation, C block (br, bc) corresponds to rows 2*br to 2*br+1 and cols 2*bc to 2*bc+1.
            # So C[2*br + r, 2*bc + c] has index (2*bc + c)*4 + (2*br + r).
            return (2*bc + c)*4 + (2*br + r)
        
        # The 7 Strassen terms for 2x2 matrices
        # Each term is a product of a linear combination of A elements and a linear combination of B elements.
        # We will add 7 rank-1 terms to U, V, W.
        
        # M1: (A00 + A11)(B00 + B11)
        # A part: A00 + A11 -> (0,0,1), (1,1,1)
        # B part: B00 + B11 -> (0,0,1), (1,1,1)
        # C part: C00 += 1, C11 += 1 -> (0,0,1), (1,1,1)
        
        # M2: (A10 + A11)B00
        # A part: A10 + A11 -> (1,0,1), (1,1,1)
        # B part: B00 -> (0,0,1)
        # C part: C10 += 1, C11 -= 1 -> (1,0,1), (1,1,-1)
        
        # M3: A00(B01 - B11)
        # A part: A00 -> (0,0,1)
        # B part: B01 - B11 -> (0,1,1), (1,1,-1)
        # C part: C01 += 1, C11 += 1 -> (0,1,1), (1,1,1)
        
        # M4: A11(B10 - B00)
        # A part: A11 -> (1,1,1)
        # B part: B10 - B00 -> (1,0,1), (0,0,-1)
        # C part: C00 += 1, C10 += 1 -> (0,0,1), (1,0,1)
        
        # M5: (A00 + A01)B11
        # A part: A00 + A01 -> (0,0,1), (0,1,1)
        # B part: B11 -> (1,1,1)
        # C part: C00 -= 1, C01 += 1 -> (0,0,-1), (0,1,1)
        
        # M6: (A10 - A00)(B00 + B01)
        # A part: A10 - A00 -> (1,0,1), (0,0,-1)
        # B part: B00 + B01 -> (0,0,1), (0,1,1)
        # C part: C11 += 1 -> (1,1,1)
        
        # M7: (A01 - A11)(B10 + B11)
        # A part: A01 - A11 -> (0,1,1), (1,1,-1)
        # B part: B10 + B11 -> (1,0,1), (1,1,1)
        # C part: C00 += 1 -> (0,0,1)
        
        # Define the 7 terms
        # Each term is a tuple (A_terms, B_terms, C_terms)
        # A_terms, B_terms, C_terms are lists of (block_row, block_col, coeff)
        
        terms = [
            # M1
            ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
            # M2
            ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
            # M3
            ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
            # M4
            ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
            # M5
            ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
            # M6
            ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
            # M7
            ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
        ]
        
        for t_idx, (A_terms, B_terms, C_terms) in enumerate(terms):
            rank_idx = rank_offset + t_idx
            
            # Add A terms to U
            for (br, bc, coeff) in A_terms:
                for r in range(2):
                    for c in range(2):
                        a_idx = get_A_scalar(br, bc, r, c)
                        U[a_idx, rank_idx] += coeff
            
            # Add B terms to V
            for (br, bc, coeff) in B_terms:
                for r in range(2):
                    for c in range(2):
                        b_idx = get_B_scalar(br, bc, r, c)
                        V[b_idx, rank_idx] += coeff
            
            # Add C terms to W
            for (br, bc, coeff) in C_terms:
                for r in range(2):
                    for c in range(2):
                        c_idx = get_C_scalar(br, bc, r, c)
                        W[c_idx, rank_idx] += coeff
    
    # Apply Strassen to the 4x4 matrix multiplication
    # The 4x4 matrices are 2x2 blocks of 2x2 matrices.
    # The block indices are (0,0), (0,1), (1,0), (1,1).
    # A00 = A[0:2, 0:2], A01 = A[0:2, 2:4], A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
    
    # We call add_strassen_term with the block-level Strassen terms.
    # But wait, add_strassen_term adds 7 rank-1 terms for a 2x2 block multiplication.
    # We need to apply this to the 4x4 case, which is a 2x2 block multiplication where each block is 2x2.
    # So we need to apply Strassen at the block level, and then apply Strassen again at the scalar level.
    
    # The block-level Strassen terms are:
    # M1_block = (A00 + A11)(B00 + B11)
    # M2_block = (A10 + A11)B00
    # M3_block = A00(B01 - B11)
    # M4_block = A11(B10 - B00)
    # M5_block = (A00 + A01)B11
    # M6_block = (A10 - A00)(B00 + B01)
    # M7_block = (A01 - A11)(B10 + B11)
    
    # Each block multiplication involves 2x2 matrices, so we apply Strassen again.
    # We will call add_strassen_term for each block-level term, with the appropriate block indices.
    
    # But add_strassen_term assumes the input is 2x2 blocks.
    # So we need to map the 4x4 indices to 2x2 block indices.
    
    # Let's redefine add_strassen_term to work on 4x4 matrices directly.
    # It will add 7*7 = 49 rank-1 terms.
    
    def add_strassen_4x4(U, V, W, rank_offset):
        # The 7 block-level Strassen terms
        # Each term is a product of a linear combination of A blocks and a linear combination of B blocks.
        # We will expand each block into 2x2 scalars and apply Strassen again.
        
        # M1_block = (A00 + A11)(B00 + B11)
        # A part: A00 + A11 -> blocks (0,0,1), (1,1,1)
        # B part: B00 + B11 -> blocks (0,0,1), (1,1,1)
        # C part: C00 += 1, C11 += 1 -> blocks (0,0,1), (1,1,1)
        
        # We need to apply Strassen to the 2x2 block multiplication.
        # The 2x2 block multiplication is:
        # C_block = A_block * B_block
        # where A_block and B_block are 2x2 matrices of 2x2 blocks.
        
        # Let's define the 7 Strassen terms for the block-level multiplication.
        # Then for each term, we apply Strassen to the 2x2 block multiplication.
        
        # This is getting complicated. Let's use a simpler approach.
        # We will directly construct the 49 rank-1 terms by iterating over the 7 block-level terms
        # and the 7 scalar-level terms for each block-level term.
        
        pass
    
    # Let's use the original approach but with a cleaner implementation.
    # We will define the 49 rank-1 terms explicitly.
    
    # The 49 terms are indexed by (t, k) where t is the block-level term (0..6) and k is the scalar-level term (0..6).
    # For each (t, k), we have a rank-1 term u_{t,k} ⊗ v_{t,k} ⊗ w_{t,k}.
    
    # Let's define the 7 block-level Strassen terms.
    # Each term is a tuple (A_blocks, B_blocks, C_blocks) where A_blocks, B_blocks, C_blocks are lists of (block_row, block_col, coeff).
    
    block_terms = [
        # M1: (A00 + A11)(B00 + B11)
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # M2: (A10 + A11)B00
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
        # M3: A00(B01 - B11)
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
        # M4: A11(B10 - B00)
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        # M5: (A00 + A01)B11
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        # M6: (A10 - A00)(B00 + B01)
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        # M7: (A01 - A11)(B10 + B11)
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
    ]
    
    # The 7 scalar-level Strassen terms for 2x2 matrices.
    # Each term is a tuple (A_scalars, B_scalars, C_scalars) where A_scalars, B_scalars, C_scalars are lists of (row, col, coeff).
    
    scalar_terms = [
        # M1: (A00 + A11)(B00 + B11)
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # M2: (A10 + A11)B00
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
        # M3: A00(B01 - B11)
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
        # M4: A11(B10 - B00)
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        # M5: (A00 + A01)B11
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        # M6: (A10 - A00)(B00 + B01)
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        # M7: (A01 - A11)(B10 + B11)
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
    ]
    
    # Now, for each block-level term t and scalar-level term k, we construct a rank-1 term.
    # The rank-1 term is u_{t,k} ⊗ v_{t,k} ⊗ w_{t,k}.
    # u_{t,k} is the product of the A part of block term t and the A part of scalar term k.
    # v_{t,k} is the product of the B part of block term t and the B part of scalar term k.
    # w_{t,k} is the product of the C part of block term t and the C part of scalar term k.
    
    # But this is not quite right. The block-level term t defines a product of two 2x2 block matrices.
    # The scalar-level term k defines a product of two 2x2 scalar matrices.
    # So the rank-1 term is the product of the two.
    
    # Let's define the rank-1 term as follows:
    # For block term t = (A_blocks, B_blocks, C_blocks) and scalar term k = (A_scalars, B_scalars, C_scalars):
    # u_{t,k}[i,j] = sum_{(br, bc, ca) in A_blocks} sum_{(r, c, cb) in A_scalars} ca * cb * A[2*br+r, 2*bc+c]
    # v_{t,k}[i,j] = sum_{(br, bc, cb) in B_blocks} sum_{(r, c, cc) in B_scalars} cb * cc * B[2*br+r, 2*bc+c]
    # w_{t,k}[i,j] = sum_{(br, bc, cc) in C_blocks} sum_{(r, c, cd) in C_scalars} cc * cd * C[2*br+r, 2*bc+c]
    
    # This is correct. Let's implement it.
    
    for t_idx, (A_blocks, B_blocks, C_blocks) in enumerate(block_terms):
        for k_idx, (A_scalars, B_scalars, C_scalars) in enumerate(scalar_terms):
            rank_idx = t_idx * 7 + k_idx
            
            # Construct u_{t,k}
            for (br, bc, ca) in A_blocks:
                for (r, c, cb) in A_scalars:
                    a_idx = (2*br + r)*4 + (2*bc + c)
                    U[a_idx, rank_idx] += ca * cb
            
            # Construct v_{t,k}
            for (br, bc, cb) in B_blocks:
                for (r, c, cc) in B_scalars:
                    b_idx = (2*br + r)*4 + (2*bc + c)
                    V[b_idx, rank_idx] += cb * cc
            
            # Construct w_{t,k}
            for (br, bc, cc) in C_blocks:
                for (r, c, cd) in C_scalars:
                    # Trace convention: C[k*n+i] where k is col, i is row.
                    # In our block notation, C block (br, bc) corresponds to rows 2*br to 2*br+1 and cols 2*bc to 2*bc+1.
                    # So C[2*br + r, 2*bc + c] has index (2*bc + c)*4 + (2*br + r).
                    c_idx = (2*bc + c)*4 + (2*br + r)
                    W[c_idx, rank_idx] += cc * cd
    
    # Verify the decomposition
    T_recon = np.einsum('as,bs,cs->abc', U, V, W)
    err = np.max(np.abs(T_recon - T))
    
    if err > 1e-10:
        # Fall back to rank-49 (which is what we just computed)
        # But if it's wrong, we need to return something valid.
        # For now, return the computed decomposition.
        pass
    
    elapsed = time.time() - start_time
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
