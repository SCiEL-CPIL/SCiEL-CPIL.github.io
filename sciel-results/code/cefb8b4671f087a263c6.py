"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    t0 = time.time()
    deadline = t0 + 120.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 4, 4, 7

    # We want to decompose the tensor T[i,j,k] (i in [0,n), j in [0,m), k in [0,p))
    # as sum over r of U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

    # Strategy:
    # Split p into p1=4 and p2=3.
    # Part 1: 4x4x4 matrix multiplication (i,j,k1 with k1 in [0,4))
    # Part 2: 4x4x3 matrix multiplication (i,j,k2 with k2 in [0,3))
    #
    # For Part 1, use Strassen's rank-49 decomposition for 4x4x4.
    # For Part 2, use schoolbook rank-48 decomposition for 4x4x3.
    # Total rank = 49 + 48 = 97.

    def strassen_4x4x4():
        """Return U, V, W for 4x4x4 matrix mult via Strassen (rank 49)."""
        # Strassen for 2x2: 7 multiplications.
        # For 4x4, split into 2x2 blocks: 16 blocks, each 2x2.
        # Apply Strassen to each pair of 2x2 blocks.
        # This gives rank 7 * 16 = 112? No, that's not right.
        # Actually, Strassen 4x4 can be done with rank 49 by applying
        # Strassen to the 2x2 block structure, where each "multiplication"
        # is itself a 2x2 multiplication done via Strassen's 7 terms.
        # Wait, that gives 7*7 = 49 for a 4x4 if we nest Strassen.
        # Actually, the standard Strassen for 4x4 with 2x2 subblocks:
        # We have 4 2x2 blocks in A, 4 in B, 4 in C.
        # A = [[A11, A12],[A21, A22]], B = [[B11, B12],[B21, B22]]
        # C = A*B, where each Ci is 2x2.
        # Strassen: 7 products of 2x2 matrices.
        # Each 2x2 product uses 7 scalar multiplications (Strassen 2x2).
        # Total: 7 * 7 = 49 scalar multiplications.
        # So rank 49 is correct for Strassen 4x4x4.
        
        # Let's build this explicitly.
        # Indices: i in [0,4), j in [0,4), k in [0,4)
        # U has shape (16, 49), V has shape (16, 49), W has shape (16, 49)
        
        # Strassen 2x2:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
        
        # For 4x4, A and B are 4x4, split into 2x2 blocks.
        # Let's index the 2x2 subblocks: A_ab where a,b in {1,2} (using 1-based)
        # A_11, A_12, A_21, A_22 each 2x2.
        # Similarly for B.
        # C_ab = sum_c A_ac * B_cb where the product is 2x2 matrix mult.
        
        # The 7 Strassen "multiplications" for the 4x4 level are:
        # S1 = (A_11 + A_22)(B_11 + B_22)
        # S2 = (A_21 + A_22)B_11
        # S3 = A_11(B_12 - B_22)
        # S4 = A_22(B_21 - B_11)
        # S5 = (A_11 + A_12)B_22
        # S6 = (A_21 - A_11)(B_11 + B_12)
        # S7 = (A_12 - A_22)(B_21 + B_22)
        
        # Each S_m is a 2x2 matrix product, which we decompose via Strassen 2x2.
        # So each S_m contributes 7 scalar multiplications.
        # Total: 7 * 7 = 49.
        
        # Let's define the 49 terms.
        # For each of the 7 Strassen-4x4 terms (m1..m7), we have a 2x2 product.
        # Each 2x2 product has 7 scalar terms.
        
        # Let's build U, V, W.
        # U[i*m+j, r] = 1 if the (i,j) element of A is involved in term r.
        # V[j*p+k, r] = 1 if the (j,k) element of B is involved in term r.
        # W[k*n+i, r] = 1 if the (k,i) element of C is produced by term r.
        
        # For 4x4x4: n=4, m=4, p=4.
        # U shape: (16, 49), V shape: (16, 49), W shape: (16, 49).
        
        U = np.zeros((16, 49), dtype=int)
        V = np.zeros((16, 49), dtype=int)
        W = np.zeros((16, 49), dtype=int)
        
        # Strassen 2x2 helper: given the 2x2 A block indices and B block indices,
        # and the base column offset, fill in the 7 terms.
        # A block is at position (a_block, b_block) in the 4x4, where a_block, b_block in {0,1}
        # The 2x2 subblock A[a_block*2:(a_block+1)*2, b_block*2:(b_block+1)*2]
        # For U: row index is i*m+j = i*4+j.
        # For V: row index is j*p+k = j*4+k.
        # For W: row index is k*n+i = k*4+i.
        
        def fill_strassen_2x2(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, col_offset, signs_a, signs_b, signs_c):
            """
            Fill in 7 Strassen terms for a 2x2 product.
            a_rows, a_cols: list of 4 row/col indices for the A 2x2 block.
            b_rows, b_cols: list of 4 row/col indices for the B 2x2 block.
            c_rows, c_cols: list of 4 row/col indices for the C 2x2 block.
            col_offset: base column offset in U, V, W.
            signs_a, signs_b: not used directly, we'll hardcode the Strassen structure.
            """
            # For a 2x2 product:
            # A = [[a00, a01], [a10, a11]]
            # B = [[b00, b01], [b10, b11]]
            # C = A*B
            # C00 = a00*b00 + a01*b10
            # C01 = a00*b01 + a01*b11
            # C10 = a10*b00 + a11*b10
            # C11 = a10*b01 + a11*b11
            #
            # Strassen:
            # M1 = (a00+a11)(b00+b11)
            # M2 = (a10+a11)b00
            # M3 = a00(b01-b11)
            # M4 = a11(b10-b00)
            # M5 = (a00+a01)b11
            # M6 = (a10-a00)(b00+b01)
            # M7 = (a01-a11)(b10+b11)
            # C00 = M1+M4-M5+M7
            # C01 = M3+M5
            # C10 = M2+M4
            # C11 = M1-M2+M3+M6
            
            # The 7 terms correspond to:
            # Term 0: M1
            # Term 1: M2
            # Term 2: M3
            # Term 3: M4
            # Term 4: M5
            # Term 5: M6
            # Term 6: M7
            
            # For each term, we need to identify which elements of A, B, C are involved.
            # And what the coefficient is (can be 1 or -1).
            
            # Let's define the terms as (a_elements, b_elements, c_elements)
            # where each is a list of (index_in_2x2, coefficient).
            # index_in_2x2: 0=a00, 1=a01, 2=a10, 3=a11 (row-major)
            
            terms = [
                # M1 = (a00+a11)(b00+b11)
                ([ (0,1), (3,1) ], [ (0,1), (3,1) ], [ (0,1), (1,1), (2,-1), (3,1) ]),  # C00+=M1, C01+=M1? No.
                # Wait, C00 = M1+M4-M5+M7, so M1 contributes +1 to C00.
                # C11 = M1-M2+M3+M6, so M1 contributes +1 to C11.
                # M1 does NOT contribute to C01 or C10.
                # So c_elements for M1: [(0,1), (3,1)] where 0=C00, 3=C11.
            ]
            
            # Let me redo this more carefully.
            # For each of the 7 Strassen terms, I'll define:
            # - Which A elements are multiplied (with signs)
            # - Which B elements are multiplied (with signs)
            # - Which C elements are produced (with signs)
            
            # M1 = (a00+a11)(b00+b11)
            #   A: a00(+1), a11(+1)
            #   B: b00(+1), b11(+1)
            #   C: C00(+1), C11(+1)
            
            # M2 = (a10+a11)b00
            #   A: a10(+1), a11(+1)
            #   B: b00(+1)
            #   C: C10(+1), C11(-1)
            
            # M3 = a00(b01-b11)
            #   A: a00(+1)
            #   B: b01(+1), b11(-1)
            #   C: C01(+1), C11(+1)
            
            # M4 = a11(b10-b00)
            #   A: a11(+1)
            #   B: b10(+1), b00(-1)
            #   C: C00(+1), C10(+1)
            
            # M5 = (a00+a01)b11
            #   A: a00(+1), a01(+1)
            #   B: b11(+1)
            #   C: C00(-1), C01(+1)
            
            # M6 = (a10-a00)(b00+b01)
            #   A: a10(+1), a00(-1)
            #   B: b00(+1), b01(+1)
            #   C: C11(+1)
            
            # M7 = (a01-a11)(b10+b11)
            #   A: a01(+1), a11(-1)
            #   B: b10(+1), b11(+1)
            #   C: C00(+1)
            
            # Let me verify:
            # C00 = M1+M4-M5+M7 = (a00+a11)(b00+b11) + a11(b10-b00) - (a00+a01)b11 + (a01-a11)(b10+b11)
            #     = a00b00 + a00b11 + a11b00 + a11b11 + a11b10 - a11b00 - a00b11 - a01b11 + a01b10 + a01b11 - a11b10 - a11b11
            #     = a00b00 + a01b10  ✓
            
            # C01 = M3+M5 = a00(b01-b11) + (a00+a01)b11 = a00b01 - a00b11 + a00b11 + a01b11 = a00b01 + a01b11  ✓
            
            # C10 = M2+M4 = (a10+a11)b00 + a11(b10-b00) = a10b00 + a11b00 + a11b10 - a11b00 = a10b00 + a11b10  ✓
            
            # C11 = M1-M2+M3+M6 = (a00+a11)(b00+b11) - (a10+a11)b00 + a00(b01-b11) + (a10-a00)(b00+b01)
            #     = a00b00 + a00b11 + a11b00 + a11b11 - a10b00 - a11b00 + a00b01 - a00b11 + a10b00 + a10b01 - a00b00 - a00b01
            #     = a10b01 + a11b11  ✓
            
            # Good. Now let's define the 7 terms.
            
            # Index mapping for 2x2:
            # 0: (0,0) -> a00, b00, C00
            # 1: (0,1) -> a01, b01, C01
            # 2: (1,0) -> a10, b10, C10
            # 3: (1,1) -> a11, b11, C11
            
            strassen_terms = [
                # (A_elements, B_elements, C_elements)
                # Each element is (idx, coeff)
                ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),  # M1
                ([(2,1),(3,1)], [(0,1)], [(2,1),(3,-1)]),        # M2
                ([(0,1)], [(1,1),(3,-1)], [(1,1),(3,1)]),        # M3
                ([(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)]),        # M4
                ([(0,1),(1,1)], [(3,1)], [(0,-1),(1,1)]),        # M5
                ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),        # M6
                ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),        # M7
            ]
            
            for t_idx, (a_elems, b_elems, c_elems) in enumerate(strassen_terms):
                col = col_offset + t_idx
                for (a_idx, coeff) in a_elems:
                    ai = a_rows[a_idx]
                    aj = a_cols[a_idx]
                    U[ai*4 + aj, col] += coeff
                for (b_idx, coeff) in b_elems:
                    bi = b_rows[b_idx]
                    bj = b_cols[b_idx]
                    V[bi*4 + bj, col] += coeff
                for (c_idx, coeff) in c_elems:
                    ci = c_rows[c_idx]
                    cj = c_cols[c_idx]
                    W[cj*4 + ci, col] += coeff  # W[k*n+i, r], k=cj, i=ci
        
        # Now apply this to the 4x4 Strassen decomposition.
        # The 4x4 is split into 2x2 blocks:
        # A_00 = A[0:2, 0:2], A_01 = A[0:2, 2:4]
        # A_10 = A[2:4, 0:2], A_11 = A[2:4, 2:4]
        # Similarly for B.
        
        # Strassen 4x4 terms:
        # S1 = (A_00 + A_11)(B_00 + B_11)
        # S2 = (A_10 + A_11)B_00
        # S3 = A_00(B_01 - B_11)
        # S4 = A_11(B_10 - B_00)
        # S5 = (A_00 + A_01)B_11
        # S6 = (A_10 - A_00)(B_00 + B_01)
        # S7 = (A_01 - A_11)(B_10 + B_11)
        
        # Each S_m is a 2x2 product. We decompose each via Strassen 2x2 (7 terms).
        # Total: 7 * 7 = 49 terms.
        
        # For each S_m, we need to define the A block and B block.
        # The A block for S_m may be a sum of 2x2 blocks.
        # But since we're building a linear decomposition, we can distribute.
        # Actually, for the tensor decomposition, each "multiplication" term
        # corresponds to a rank-1 term. The Strassen algorithm combines
        # multiple rank-1 terms.
        
        # Let me think about this differently.
        # The tensor for 4x4x4 matrix mult has 64 non-zero entries (one for each (i,j,k) 
        # where C[i,k] = sum_j A[i,j]*B[j,k]).
        # Actually, C[i,k] = sum_j A[i,j] * B[j,k], so for each (i,j,k), 
        # the tensor entry T[i,j,k] = 1 (in the standard basis).
        # Wait, the tensor is T_{i,j,k} = delta_{i,k} in the output... no.
        # The matrix multiplication tensor T is defined such that:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # So T[i,j,k] = 1 for all i,j,k (in the sense that the (i,k)-th entry of C
        # depends on A[i,j] and B[j,k] for each j).
        # Actually, T is a 4x4x4 tensor where T[i,j,k] = 1 if the (i,k) output
        # depends on A[i,j] and B[j,k]. Since C[i,k] = sum_j A[i,j]*B[j,k],
        # T[i,j,k] = 1 for all i,j,k.
        
        # Hmm, that's not quite right either. Let me reconsider.
        # The tensor T has entries T[i,j,k] such that:
        # C_{ik} = sum_j T[i,j,k] A_{ij} B_{jk}
        # For standard matrix mult, T[i,j,k] = 1 for all i,j,k.
        # So the tensor is the all-ones 4x4x4 tensor? No, that can't be right.
        # Actually, C_{ik} = sum_j A_{ij} B_{jk}, so T[i,j,k] = 1 for all i,j,k.
        # Yes, the 4x4x4 matrix multiplication tensor is the all-ones tensor of shape (4,4,4).
        # Wait, that's not a tensor in the usual sense. Let me re-examine.
        
        # The matrix multiplication tensor T_{n,n,n} is defined as:
        # T_{ijk} = 1 if k = j (in some formulations)
        # Actually, I think I'm confusing things. Let me look at the definition again.
        
        # The task says: "exactly decompose the matrix multiplication tensor"
        # With the convention W[k*n+i, r].
        # So C[i,k] = sum_{j,r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # For matrix mult: C[i,k] = sum_j A[i,j] * B[j,k]
        # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # for the coefficient of A[i,j]*B[j,k] in C[i,k].
        # This means the tensor T[i,j,k] = 1 for all i,j,k (in the 4x4x4 case).
        # Wait, no. The coefficient of A[i,j]*B[j,k] in C[i,k] is 1, and
        # the coefficient of A[i',j']*B[j',k'] in C[i,k] is 0 unless i'=i and k'=k.
        # So T[i,j,k] = 1 if the term A[i,j]*B[j,k] appears in C[i,k], which is always.
        # But T is indexed as T[i,j,k] where i is row of A, j is col of A / row of B,
        # k is col of B. And the output is C[i,k].
        # So the tensor entry T[i,j,k] should be 1 if A[i,j]*B[j,k] contributes to C[i,k].
        # This is true for all i,j,k, so T[i,j,k] = 1 for all i,j,k.
        # But that would mean the tensor is all-ones, which has rank 1, which is wrong.
        
        # I think I'm misunderstanding the tensor structure. Let me re-read the task.
        # "Matrix multiplication 4,4,7"
        # "Find a rank-minimal exact tensor decomposition for multiplying a 4 by 4 matrix
        # by a 4 by 7 matrix."
        # "The output factor follows the official trace convention W[k*n+i,r]."
        
        # So A is 4x4, B is 4x7, C is 4x7.
        # C[i,k] = sum_j A[i,j] * B[j,k] where i in [0,4), j in [0,4), k in [0,7).
        # The tensor T[i,j,k] = 1 if A[i,j]*B[j,k] contributes to C[i,k].
        # This is true for all i,j,k, so T[i,j,k] = 1 for all i,j,k in [0,4)x[0,4)x[0,7).
        
        # But the rank of the all-ones tensor is 1, which contradicts known results.
        # So I must be misunderstanding the tensor definition.
        
        # Actually, I think the tensor is defined such that:
        # C[i,k] = sum_{j,r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # And this must equal sum_j A[i,j] * B[j,k].
        # So the "tensor" is the set of coefficients:
        # For each (i,j,k), the coefficient of A[i,j]*B[j,k] in C[i,k] is 1.
        # And for any other combination, it's 0.
        # But since C[i,k] only depends on A[i,j] and B[j,k] (same i and k),
        # the tensor T[i,j,k] = 1 for all i,j,k.
        
        # I think the confusion is that the tensor is not "all ones" in the sense of
        # a single tensor, but rather the structure is such that each output C[i,k]
        # is a bilinear form in A and B. The tensor decomposition is a way to write
        # the bilinear map as a sum of rank-1 terms.
        
        # OK let me just trust the Strassen decomposition and implement it.
        # The key insight is that Strassen's algorithm for 4x4 matrix multiplication
        # uses 49 scalar multiplications, which corresponds to a rank-49 decomposition.
        
        # Let me implement this by directly encoding the Strassen algorithm.
        
        # Actually, let me take a step back. The current program uses schoolbook
        # with rank 112. The inspiration says Strassen 4x4x4 rank 49 + schoolbook
        # 4x4x3 rank 48 = 97. Let me just implement that.
        
        # For the 4x4x4 part, I'll use a known rank-49 decomposition.
        # For the 4x4x3 part, I'll use schoolbook (rank 48).
        
        # Let me build the schoolbook part first, as it's simpler.
        return U, V, W

    def schoolbook(n, m, p):
        """Return U, V, W for nxmxp matrix mult via schoolbook (rank n*m*p)."""
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=int)
        V = np.zeros((m*p, rank), dtype=int)
        W = np.zeros((p*n, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W

    # Part 1: 4x4x4 via Strassen rank 49
    # Part 2: 4x4x3 via schoolbook rank 48
    
    # For the 4x4x4 Strassen, let me implement it properly.
    # I'll define the 49 terms explicitly.
    
    def strassen_444_decomp():
        """
        Return U (16x49), V (16x49), W (16x49) for 4x4x4 matrix mult.
        Using Strassen's algorithm with 49 scalar multiplications.
        """
        U = np.zeros((16, 49), dtype=int)
        V = np.zeros((16, 49), dtype=int)
        W = np.zeros((16, 49), dtype=int)
        
        # The 4x4 matrix is split into 2x2 blocks.
        # A = [[A00, A01], [A10, A11]] where each Aij is 2x2.
        # B = [[B00, B01], [B10, B11]]
        # C = A*B = [[C00, C01], [C10, C11]]
        
        # Strassen:
        # S1 = (A00+A11)(B00+B11)
        # S2 = (A10+A11)B00
        # S3 = A00(B01-B11)
        # S4 = A11(B10-B00)
        # S5 = (A00+A01)B11
        # S6 = (A10-A00)(B00+B01)
        # S7 = (A01-A11)(B10+B11)
        
        # C00 = S1+S4-S5+S7
        # C01 = S3+S5
        # C10 = S2+S4
        # C11 = S1-S2+S3+S6
        
        # Each Sm is a 2x2 matrix product. We decompose each via Strassen 2x2.
        
        # For a 2x2 product X*Y where X = [[x00,x01],[x10,x11]], Y = [[y00,y01],[y10,y11]]:
        # Z = X*Y
        # Z00 = x00*y00 + x01*y10
        # Z01 = x00*y01 + x01*y11
        # Z10 = x10*y00 + x11*y10
        # Z11 = x10*y01 + x11*y11
        #
        # Strassen 2x2:
        # M1 = (x00+x11)(y00+y11)
        # M2 = (x10+x11)y00
        # M3 = x00(y01-y11)
        # M4 = x11(y10-y00)
        # M5 = (x00+x01)y11
        # M6 = (x10-x00)(y00+y01)
        # M7 = (x01-x11)(y10+y11)
        #
        # Z00 = M1+M4-M5+M7
        # Z01 = M3+M5
        # Z10 = M2+M4
        # Z11 = M1-M2+M3+M6
        
        # For each of the 7 Strassen-4x4 terms, we have a 2x2 product.
        # The X and Y matrices are linear combinations of 2x2 blocks of A and B.
        # Since the decomposition is linear, we can distribute.
        
        # Let me define a helper that, given the coefficients for X and Y
        # (as linear combinations of A-blocks and B-blocks), fills in the
        # 7 Strassen-2x2 terms.
        
        # Actually, this is getting complex. Let me just hardcode all 49 terms.
        
        # For S1 = (A00+A11)(B00+B11):
        # X = A00+A11, Y = B00+B11
        # X00 = A00_00 + A11_00, X01 = A00_01 + A11_01
        # X10 = A00_10 + A11_10, X11 = A00_11 + A11_11
        # Y00 = B00_00 + B11_00, Y01 = B00_01 + B11_01
        # Y10 = B00_10 + B11_10, Y11 = B00_11 + B11_11
        
        # A00 is at A[0:2, 0:2], so A00_00 = A[0,0], A00_01 = A[0,1],
        # A00_10 = A[1,0], A00_11 = A[1,1]
        # A11 is at A[2:4, 2:4], so A11_00 = A[2,2], A11_01 = A[2,3],
        # A11_10 = A[3,2], A11_11 = A[3,3]
        
        # Similarly for B blocks.
        
        # This is very tedious to do by hand for all 49 terms.
        # Let me instead implement it programmatically.
        
        def block_rows(block_idx):
            """Return the row indices for a 2x2 block."""
            r = (block_idx // 2) * 2
            return [r, r+1]
        
        def block_cols(block_idx):
            """Return the col indices for a 2x2 block."""
            c = (block_idx % 2) * 2
            return [c, c+1]
        
        # Block indices: 0=A00, 1=A01, 2=A10, 3=A11 (and similarly for B)
        # A00: rows [0,1], cols [0,1]
        # A01: rows [0,1], cols [2,3]
        # A10: rows [2,3], cols [0,1]
        # A11: rows [2,3], cols [2,3]
        
        # For each Strassen-4x4 term, define X and Y as linear combinations of blocks.
        # X = sum of (coeff, block_idx) for A blocks
        # Y = sum of (coeff, block_idx) for B blocks
        
        strassen4_terms = [
            # S1 = (A00+A11)(B00+B11)
            ([(1, 0), (1, 3)], [(1, 0), (1, 3)]),
            # S2 = (A10+A11)B00
            ([(1, 2), (1, 3)], [(1, 0)]),
            # S3 = A00(B01-B11)
            ([(1, 0)], [(1, 1), (-1, 3)]),
            # S4 = A11(B10-B00)
            ([(1, 3)], [(1, 2), (-1, 0)]),
            # S5 = (A00+A01)B11
            ([(1, 0), (1, 1)], [(1, 3)]),
            # S6 = (A10-A00)(B00+B01)
            ([(1, 2), (-1, 0)], [(1, 0), (1, 1)]),
            # S7 = (A01-A11)(B10+B11)
            ([(1, 1), (-1, 3)], [(1, 2), (1, 3)]),
        ]
        
        for s_idx, (x_blocks, y_blocks) in enumerate(strassen4_terms):
            # x_blocks: list of (coeff, block_idx) for X
            # y_blocks: list of (coeff, block_idx) for Y
            # For each of the 7 Strassen-2x2 terms, we need to determine
            # which A and B elements are involved.
            
            # Build X as a 2x2 matrix of linear combinations.
            # X[0,0] = sum of coeff * A_block[block_idx][0,0]
            # etc.
            
            # For each of the 7 Strassen-2x2 terms, the X elements involved are:
            # M1: X00, X11
            # M2: X10, X11
            # M3: X00
            # M4: X11
            # M5: X00, X01
            # M6: X10, X00
            # M7: X01, X11
            
            # And Y elements:
            # M1: Y00, Y11
            # M2: Y00
            # M3: Y01, Y11
            # M4: Y10, Y00
            # M5: Y11
            # M6: Y00, Y01
            # M7: Y10, Y11
            
            # And C elements (output of the 2x2 product, which is one of C00,C01,C10,C11):
            # The 2x2 product is S_s, which contributes to C blocks as per Strassen-4x4.
            # S1 -> C00(+1), C11(+1)
            # S2 -> C10(+1), C11(-1)
            # S3 -> C01(+1), C11(+1)
            # S4 -> C00(+1), C10(+1)
            # S5 -> C00(-1), C01(+1)
            # S6 -> C11(+1)
            # S7 -> C00(+1)
            
            c_contribs = [
                [(0, 1), (3, 1)],   # S1 -> C00(+1), C11(+1)
                [(2, 1), (3, -1)],  # S2 -> C10(+1), C11(-1)
                [(1, 1), (3, 1)],   # S3 -> C01(+1), C11(+1)
                [(0, 1), (2, 1)],   # S4 -> C00(+1), C10(+1)
                [(0, -1), (1, 1)],  # S5 -> C00(-1), C01(+1)
                [(3, 1)],           # S6 -> C11(+1)
                [(0, 1)],           # S7 -> C00(+1)
            ]
            
            x_elem_indices = [
                [0, 3],  # M1: X00, X11
                [2, 3],  # M2: X10, X11
                [0],     # M3: X00
                [3],     # M4: X11
                [0, 1],  # M5: X00, X01
                [2, 0],  # M6: X10, X00
                [1, 3],  # M7: X01, X11
            ]
            
            y_elem_indices = [
                [0, 3],  # M1: Y00, Y11
                [0],     # M2: Y00
                [1, 3],  # M3: Y01, Y11
                [2, 0],  # M4: Y10, Y00
                [3],     # M5: Y11
                [0, 1],  # M6: Y00, Y01
                [2, 3],  # M7: Y10, Y11
            ]
            
            # Coefficients for X elements in each Strassen-2x2 term:
            x_elem_coeffs = [
                [1, 1],  # M1
                [1, 1],  # M2
                [1],     # M3
                [1],     # M4
                [1, 1],  # M5
                [1, -1], # M6
                [1, -1], # M7
            ]
            
            y_elem_coeffs = [
                [1, 1],  # M1
                [1],     # M2
                [1, -1], # M3
                [1, -1], # M4
                [1],     # M5
                [1, 1],  # M6
                [1, 1],  # M7
            ]
            
            for m_idx in range(7):
                col = s_idx * 7 + m_idx
                # X elements
                for e_idx, elem_idx in enumerate(x_elem_indices[m_idx]):
                    coeff_x = x_elem_coeffs[m_idx][e_idx]
                    # X[elem_idx] is a linear combination of A blocks
                    for (block_coeff, block_idx) in x_blocks:
                        total_coeff = coeff_x * block_coeff
                        if total_coeff == 0:
                            continue
                        # Get the A element at position elem_idx in block block_idx
                        # elem_idx: 0=X00, 1=X01, 2=X10, 3=X11
                        # block_idx: 0=A00, 1=A01, 2=A10, 3=A11
                        br = (block_idx // 2) * 2
                        bc = (block_idx % 2) * 2
                        r_off = elem_idx // 2  # 0 for 0,1; 1 for 2,3
                        c_off = elem_idx % 2
                        ai = br + r_off
                        aj = bc + c_off
                        U[ai*4 + aj, col] += total_coeff
                # Y elements
                for e_idx, elem_idx in enumerate(y_elem_indices[m_idx]):
                    coeff_y = y_elem_coeffs[m_idx][e_idx]
                    for (block_coeff, block_idx) in y_blocks:
                        total_coeff = coeff_y * block_coeff
                        if total_coeff == 0:
                            continue
                        br = (block_idx // 2) * 2
                        bc = (block_idx % 2) * 2
                        r_off = elem_idx // 2
                        c_off = elem_idx % 2
                        bi = br + r_off
                        bj = bc + c_off
                        V[bi*4 + bj, col] += total_coeff
                # C elements
                for (c_block_idx, c_coeff) in c_contribs[s_idx]:
                    # c_block_idx: 0=C00, 1=C01, 2=C10, 3=C11
                    # The 2x2 product S_s produces a 2x2 matrix, and
                    # this contributes to C block c_block_idx with coefficient c_coeff.
                    # For each element of the 2x2 product, we need to map to C.
                    # The 2x2 product Z = X*Y has Z00, Z01, Z10, Z11.
                    # Z00 contributes to C[c_block][0,0]
                    # Z01 contributes to C[c_block][0,1]
                    # Z10 contributes to C[c_block][1,0]
                    # Z11 contributes to C[c_block][1,1]
                    # But in our Strassen-2x2 decomposition, each M term
                    # contributes to specific Z elements.
                    # We need to track which Z elements each M term produces.
                    
                    # For Strassen-2x2:
                    # M1 -> Z00(+1), Z11(+1)
                    # M2 -> Z10(+1), Z11(-1)
                    # M3 -> Z01(+1), Z11(+1)
                    # M4 -> Z00(+1), Z10(+1)
                    # M5 -> Z00(-1), Z01(+1)
                    # M6 -> Z11(+1)
                    # M7 -> Z00(+1)
                    
                    z_contribs = [
                        [(0, 1), (3, 1)],   # M1
                        [(2, 1), (3, -1)],  # M2
                        [(1, 1), (3, 1)],   # M3
                        [(0, 1), (2, 1)],   # M4
                        [(0, -1), (1, 1)],  # M5
                        [(3, 1)],           # M6
                        [(0, 1)],           # M7
                    ]
                    
                    for (z_elem, z_coeff) in z_contribs[m_idx]:
                        total_coeff = c_coeff * z_coeff
                        if total_coeff == 0:
                            continue
                        # Map z_elem to C position
                        # z_elem: 0=Z00, 1=Z01, 2=Z10, 3=Z11
                        # c_block_idx: 0=C00, 1=C01, 2=C10, 3=C11
                        # C block row: c_block_idx // 2
                        # C block col: c_block_idx % 2
                        cr = (c_block_idx // 2) * 2 + (z_elem // 2)
                        cc = (c_block_idx % 2) * 2 + (z_elem % 2)
                        # W[k*n+i, r] where k=cc, i=cr
                        W[cc*4 + cr, col] += total_coeff
        
        return U, V, W

    # Build the 4x4x4 Strassen part
    U1, V1, W1 = strassen_444_decomp()
    
    # Build the 4x4x3 schoolbook part
    U2, V2, W2 = schoolbook(4, 4, 3)
    
    # Combine:
    # U = [U1, U2] -> shape (16, 49+48) = (16, 97)
    # V = [V1, V2] -> shape (16, 49+48) = (16, 97)
    # W = [W1, W2] -> shape (16+12, 49+48) = (28, 97)
    #   W1 has shape (16, 49) for k in [0,4)
    #   W2 has shape (12, 48) for k in [0,3)
    #   But the full W should have shape (28, 97) for k in [0,7)
    #   For k in [0,4): W1 contributes
    #   For k in [4,7): W2 contributes (shifted by 4)
    
    U = np.hstack([U1, U2])
    V = np.hstack([V1, V2])
    
    # W1 is for k in [0,4), W2 is for k in [0,3) but we need k in [4,7)
    # So we shift W2's rows by 4.
    W_full = np.zeros((28, 97), dtype=int)
    W_full[0:16, 0:49] = W1
    W_full[4:7, 49:97] = W2  # k in [4,7) corresponds to k in [0,3) in W2
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W_full.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

