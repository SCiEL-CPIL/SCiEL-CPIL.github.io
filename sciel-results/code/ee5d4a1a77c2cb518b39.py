# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Find rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    start_time = time.time()
    
    n, m, p = 4, 4, 6
    
    # Use rank-8 decomposition for 2x2*2x3, giving total rank 7*8=56
    # This requires a valid rank-8 decomposition for [2,2,3]
    
    # Let's use a direct approach: build the tensor and find a decomposition
    # A is 2x2, B is 2x3, C is 2x3
    # C[0,0] = a*e + c*h
    # C[0,1] = a*f + c*i
    # C[0,2] = a*g + c*j
    # C[1,0] = d*e + d*h  (wait, C[1,0] = d*e + d*h? No)
    # C[1,0] = d*e + f*h? No.
    # C = A @ B where A = [[a,b],[c,d]], B = [[e,f,g],[h,i,j]]
    # C[0,0] = a*e + b*h
    # C[0,1] = a*f + b*i
    # C[0,2] = a*g + b*j
    # C[1,0] = c*e + d*h
    # C[1,1] = c*f + d*i
    # C[1,2] = c*g + d*j
    
    # We need 6 outputs, each is a bilinear form in 8 variables.
    # Schoolbook uses 12 terms (each of the 6 outputs has 2 terms).
    # Strassen-style can reduce this.
    
    # Known rank for [2,2,3] is 8. Let's construct it.
    # Using the method of "splitting" the columns:
    # Treat B as having two 2x2 blocks and one 2x1 block, but that's not clean.
    
    # Alternative: use the fact that we can do [2,2,3] with rank 8
    # by using 4 terms for the first two columns (Strassen for [2,2,2])
    # and 4 terms for the third column.
    # But that gives rank 7+4=11, not 8.
    
    # Actually, the rank of [2,2,3] is indeed 8. Here's a valid decomposition:
    # Let's use the following 8 bilinear forms:
    # t1 = (a+d)(e+h)
    # t2 = (a+c)(e+h)  
    # t3 = (b+d)(f+i)
    # t4 = (b+c)(f+i)
    # t5 = (a-d)(e-h)
    # t6 = (b-d)(f-i)
    # t7 = (a+b)(g+i)
    # t8 = (c+d)(g+j)
    
    # Let me verify this carefully:
    # C[0,0] should be a*e + b*h
    # From t1,t2,t5: t1 - t2 + t5 = (a+d)(e+h) - (a+c)(e+h) + (a-d)(e-h)
    # = (e+h)[a+d-a-c] + (a-d)(e-h)
    # = (e+h)(d-c) + ae - ah - de + dh
    # = de - ce + eh - ch + ae - ah - de + dh
    # = ae - ce + eh - ch - ah + dh
    # This should equal ae + bh. Not matching.
    
    # Let me try a different approach. I'll use a verified rank-8 scheme.
    # Actually, let me just use rank 11 (7 Strassen + 4 schoolbook for the extra column)
    # for the inner block, giving total rank 7*11 = 77.
    
    # But wait, the current program already does this and gets score 1.2468.
    # Let me try to find a better outer decomposition or a valid rank-8 inner decomposition.
    
    # For now, let's implement the known working rank-77 solution cleanly.
    
    rank = 77
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper functions for global indices
    def get_global_A(block_id, r, s):
        """block_id: 0..3, r,s: 0..1 (local 2x2 block coords)"""
        I = (block_id // 2) * 2 + r
        J = (block_id % 2) * 2 + s
        return I, J
    
    def get_global_B(block_id, s, k):
        """block_id: 0..3, s: 0..1 (local row), k: 0..2 (local col)"""
        J = (block_id // 2) * 2 + s
        K = (block_id % 2) * 3 + k
        return J, K
    
    def get_global_C(block_id, r, k):
        """block_id: 0..3, r: 0..1 (local row), k: 0..2 (local col)"""
        I = (block_id // 2) * 2 + r
        K = (block_id % 2) * 3 + k
        return I, K
    
    # Outer Strassen steps for 4x4 * 4x6
    # A is 4x4, split into 2x2 blocks of 2x2: A00, A01, A10, A11
    # B is 4x6, split into 2x2 blocks of 2x3: B00, B01, B10, B11
    # C is 4x6, split into 2x2 blocks of 2x3: C00, C01, C10, C11
    #
    # Strassen for [2,2,2] -> [2,2,2]:
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)(B00)     -> C10 += M2, C11 -= M2
    # M3 = (A00)(B01-B11)     -> C01 += M3, C11 += M3
    # M4 = (A11)(B10-B00)     -> C00 += M4, C10 += M4
    # M5 = (A00+A01)(B11)     -> C01 += M5, C00 -= M5
    # M6 = (A10-A00)(B00+B01) -> C01 -= M6, C10 += M6
    # M7 = (A01-A11)(B10+B11) -> C00 += M7, C10 -= M7
    #
    # For [2,2,3] blocks, the outer structure is similar but B and C blocks are 2x3.
    
    # Each outer step involves adding/subtracting 2x2 blocks of A, 2x2 blocks of B (which are 2x3),
    # and 2x2 blocks of C (which are 2x3).
    
    # The 7 outer steps:
    # Step 0: M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # Step 1: M2 = (A10+A11)(B00)     -> C10 += M2, C11 -= M2
    # Step 2: M3 = (A00)(B01-B11)     -> C01 += M3, C11 += M3
    # Step 3: M4 = (A11)(B10-B00)     -> C00 += M4, C10 += M4
    # Step 4: M5 = (A00+A01)(B11)     -> C01 += M5, C00 -= M5
    # Step 5: M6 = (A10-A00)(B00+B01) -> C01 -= M6, C10 += M6
    # Step 6: M7 = (A01-A11)(B10+B11) -> C00 += M7, C10 -= M7
    
    # For each outer step, we need to compute the product of sums/differences of 2x2 A-blocks
    # with sums/differences of 2x3 B-blocks, which gives a 2x3 C-block contribution.
    
    # Each such product can be computed with the inner decomposition (rank 11 for [2,2,3]).
    
    # Inner Strassen for [2,2,2] (first two columns of the 2x3 blocks):
    inner_strassen = [
        # M1 = (a+d)(e+h) -> C00 += M1, C10 += M1
        (([(0,0,1), (1,1,1)], [(0,0,1), (1,0,1)], [(0,0,1), (1,0,1)])),
        # M2 = (c+d)(e+h) -> C10 += M2, C11 -= M2  (wait, need to be careful)
        # Actually for [2,2,2]:
        # M1 = (a+d)(e+h) -> C00 += M1, C10 += M1
        # M2 = (c+d)(e)   -> C10 += M2, C11 -= M2  ... this doesn't look right either.
        # Let me use the standard Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)(B00)
        # M3 = (A00)(B01-B11)
        # M4 = (A11)(B10-B00)
        # M5 = (A00+A01)(B11)
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        #
        # Where A = [[A00,A01],[A10,A11]], B = [[B00,B01],[B10,B11]], C = [[C00,C01],[C10,C11]]
        #
        # For scalar 2x2:
        # C00 = M1 - M4 + M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 - M7
    ]
    
    # Let me restart with a cleaner approach. I'll directly construct the rank-77 decomposition
    # by iterating over all 7 outer steps and for each, using 11 inner terms.
    
    # Clear matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 7 outer Strassen steps for [2,2,2]->[2,2,3] blocks
    # Each step: (A_coeffs, B_coeffs, C_coeffs)
    # A_coeffs: list of (block_id, coeff) for 2x2 blocks of A
    # B_coeffs: list of (block_id, coeff) for 2x3 blocks of B
    # C_coeffs: list of (block_id, coeff) for 2x3 blocks of C
    
    outer_steps = [
        # Step 0: M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        (([(0,1),(1,1)], [(0,1),(1,1)], [(0,1),(1,1)])),
        # Step 1: M2 = (A10+A11)(B00) -> C10 += M2, C11 -= M2
        (([(2,1),(3,1)], [(0,1)], [(2,1),(-1,1)])),  # This is wrong format
    ]
    
    # Let me redefine more carefully.
    # Block IDs:
    # A blocks: 0=A00, 1=A01, 2=A10, 3=A11 (each 2x2)
    # B blocks: 0=B00, 1=B01, 2=B10, 3=B11 (each 2x3)
    # C blocks: 0=C00, 1=C01, 2=C10, 3=C11 (each 2x3)
    
    outer_steps = [
        # Step 0: (A00+A11)(B00+B11) -> C00 +=, C11 +=
        (([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)])),
        # Step 1: (A10+A11)(B00) -> C10 +=, C11 -=
        (([(2,1),(3,1)], [(0,1)], [(2,1),(3,-1)])),
        # Step 2: (A00)(B01-B11) -> C01 +=, C11 +=
        (([(0,1)], [(1,1),(3,-1)], [(1,1),(3,1)])),
        # Step 3: (A11)(B10-B00) -> C00 +=, C10 +=
        (([(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)])),
        # Step 4: (A00+A01)(B11) -> C01 +=, C00 -=
        (([(0,1),(1,1)], [(3,1)], [(1,1),(0,-1)])),
        # Step 5: (A10-A00)(B00+B01) -> C01 -=, C10 +=
        (([(2,1),(0,-1)], [(0,1),(1,1)], [(1,-1),(2,1)])),
        # Step 6: (A01-A11)(B10+B11) -> C00 +=, C10 -=
        (([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1),(2,-1)])),
    ]
    
    # For each outer step, we need to compute the product of the combined A block (2x2)
    # and the combined B block (2x3), contributing to the combined C blocks.
    # The combined A block is a 2x2 matrix where each element is a sum of A entries.
    # The combined B block is a 2x3 matrix where each element is a sum of B entries.
    # The product is a 2x3 matrix.
    
    # We'll use the inner decomposition to compute each 2x2 * 2x3 product.
    # Inner decomposition for [2,2,3]: rank 11 (7 Strassen for cols 0,1 + 4 schoolbook for col 2)
    
    # Strassen inner terms for [2,2,2] (columns 0 and 1):
    # M1 = (a+d)(e+h) -> C00 += M1, C10 += M1
    # M2 = (c+d)(e)   -> C10 += M2, C11 -= M2
    # M3 = (a)(f-i)   -> C01 += M3, C11 += M3
    # M4 = (d)(h-e)   -> C00 += M4, C10 += M4
    # M5 = (a+b)(i)   -> C01 += M5, C00 -= M5
    # M6 = (c-a)(e+f) -> C01 -= M6, C10 += M6
    # M7 = (b-d)(h+i) -> C00 += M7, C10 -= M7
    
    # Schoolbook inner terms for column 2:
    # M8 = (a)(j) -> C02 += M8
    # M9 = (b)(j) -> C02 += M9
    # M10 = (c)(j) -> C12 += M10
    # M11 = (d)(j) -> C12 += M11
    
    # Where A = [[a,b],[c,d]], B = [[e,f,g],[h,i,j]], C = A@B
    
    inner_terms = [
        # (A_coeffs, B_coeffs, C_coeffs) for each of the 11 terms
        # A_coeffs: list of ((row,col), coeff)
        # B_coeffs: list of ((row,col), coeff)
        # C_coeffs: list of ((row,col), coeff)
        
        # M1 = (a+d)(e+h)
        (([(0,0,1),(1,1,1)], [(0,0,1),(1,0,1)], [(0,0,1),(1,0,1)])),
        # M2 = (c+d)(e)
        (([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)])),
        # M3 = (a)(f-i)
        (([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)])),
        # M4 = (d)(h-e)
        (([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)])),
        # M5 = (a+b)(i)
        (([(0,0,1),(0,1,1)], [(1,1,1)], [(0,1,1),(0,0,-1)])),
        # M6 = (c-a)(e+f)
        (([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(0,1,-1),(1,0,1)])),
        # M7 = (b-d)(h+i)
        (([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1),(1,0,-1)])),
        # M8 = (a)(j)
        (([(0,0,1)], [(0,2,1)], [(0,2,1)])),
        # M9 = (b)(j)
        (([(0,1,1)], [(0,2,1)], [(0,2,1)])),
        # M10 = (c)(j)
        (([(1,0,1)], [(0,2,1)], [(1,2,1)])),
        # M11 = (d)(j)
        (([(1,1,1)], [(0,2,1)], [(1,2,1)])),
    ]
    
    # Now build the full decomposition
    for step_idx, (A_blocks, B_blocks, C_blocks) in enumerate(outer_steps):
        base_rank = step_idx * 11
        
        for term_idx, (A_coeffs, B_coeffs, C_coeffs) in enumerate(inner_terms):
            rank_idx = base_rank + term_idx
            
            # Add contributions to U, V, W
            for (r, c, a_coeff) in A_coeffs:
                for (block_id, b_coeff) in A_blocks:
                    I, J = get_global_A(block_id, r, c)
                    U[I * m + J][rank_idx] += a_coeff * b_coeff
            
            for (r, c, b_coeff) in B_coeffs:
                for (block_id, b_coeff_outer) in B_blocks:
                    J, K = get_global_B(block_id, r, c)
                    V[J * p + K][rank_idx] += b_coeff * b_coeff_outer
            
            for (r, c, c_coeff) in C_coeffs:
                for (block_id, c_coeff_outer) in C_blocks:
                    I, K = get_global_C(block_id, r, c)
                    W[K * n + I][rank_idx] += c_coeff * c_coeff_outer
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
