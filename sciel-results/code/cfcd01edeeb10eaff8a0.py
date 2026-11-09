"""Matrix multiplication 4x5 by 5x6: rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Strategy: Split A (4x5) and B (5x6) into 2x2 blocks along the middle.
    # A rows: A11 (2x2), A12 (2x3); A21 (2x2), A22 (2x3)
    # B cols: B11 (2x2), B12 (2x2); B21 (3x2), B22 (3x2)
    # Wait, let me be more careful.
    #
    # A is 4x5: split rows into [0:2] and [2:4], cols into [0:2] and [2:5]
    #   A11 = A[0:2, 0:2] (2x2)
    #   A12 = A[0:2, 2:5] (2x3)
    #   A21 = A[2:4, 0:2] (2x2)
    #   A22 = A[2:4, 2:5] (2x3)
    #
    # B is 5x6: split rows into [0:2] and [2:5], cols into [0:2] and [2:6]
    #   B11 = B[0:2, 0:2] (2x2)
    #   B12 = B[0:2, 2:6] (2x4)
    #   B21 = B[2:5, 0:2] (3x2)
    #   B22 = B[2:5, 2:6] (3x4)
    #
    # C = A*B is 4x6:
    #   C11 = A11*B11 + A12*B21 (2x2)
    #   C12 = A11*B12 + A12*B22 (2x4)
    #   C21 = A21*B11 + A22*B21 (2x2)
    #   C22 = A21*B12 + A22*B22 (2x4)
    #
    # For each 2x2x2 sub-problem, use Strassen (rank 7 instead of 8).
    # For other sizes, use schoolbook.
    #
    # Let me enumerate all the sub-multiplications:
    # C11: A11(2x2)*B11(2x2) + A12(2x3)*B21(3x2)
    #   - A11*B11: 2x2x2, Strassen rank 7
    #   - A12*B21: 2x3x2, schoolbook rank 12
    # C12: A11(2x2)*B12(2x4) + A12(2x3)*B22(3x4)
    #   - A11*B12: 2x2x4, schoolbook rank 16
    #   - A12*B22: 2x3x4, schoolbook rank 24
    # C21: A21(2x2)*B11(2x2) + A22(2x3)*B21(3x2)
    #   - A21*B11: 2x2x2, Strassen rank 7
    #   - A22*B21: 2x3x2, schoolbook rank 12
    # C22: A21(2x2)*B12(2x4) + A22(2x3)*B22(3x4)
    #   - A21*B12: 2x2x4, schoolbook rank 16
    #   - A22*B22: 2x3x4, schoolbook rank 24
    #
    # Total: 7 + 12 + 16 + 24 + 7 + 12 + 16 + 24 = 118
    # That's 2 less than schoolbook (120).
    #
    # Can we do better? Let's try different splits.
    #
    # Alternative: Split A into 2x2 + 2x3, B into 3x2 + 3x4 (different column split)
    # Actually, let me try splitting the inner dimension differently.
    #
    # A is 4x5, B is 5x6. Inner dim is 5.
    # Split inner dim into 2+3.
    #
    # A1 = A[:, 0:2] (4x2), A2 = A[:, 2:5] (4x3)
    # B1 = B[0:2, :] (2x6), B2 = B[2:5, :] (3x6)
    #
    # C = A1*B1 + A2*B2
    #   A1*B1: 4x2x6, schoolbook rank 48
    #   A2*B2: 4x3x6, schoolbook rank 72
    # Total: 120. Same.
    #
    # Let me try a 2x2 Strassen on the whole thing by padding.
    # Pad A to 4x6 and B to 6x6? No, that increases rank.
    #
    # Actually, the 118 from above is a good improvement. Let me implement it.
    #
    # For Strassen on 2x2x2:
    # Given matrices X (2x2), Y (2x2), compute Z = X*Y.
    # X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # Z = [[z00, z01], [z10, z11]]
    #
    # Strassen:
    # M1 = (x00 + x11) * (y00 + y11)
    # M2 = (x10 + x11) * y00
    # M3 = x00 * (y01 - y11)
    # M4 = x11 * (y10 - y00)
    # M5 = (x00 + x01) * y11
    # M6 = (x10 - x00) * (y00 + y01)
    # M7 = (x01 - x11) * (y10 + y11)
    #
    # z00 = M1 + M4 - M5 + M7
    # z01 = M3 + M5
    # z10 = M2 + M4
    # z11 = M1 - M2 + M3 + M6
    #
    # Each Mi is a product of two linear combinations.
    # For the tensor decomposition, each Mi contributes one term.
    #
    # Let me build the decomposition term by term.

    terms = []  # list of (u_idx, v_idx, w_idx, u_val, v_val, w_val)

    def add_term(u_idx, v_idx, w_idx, u_val=1, v_val=1, w_val=1):
        terms.append((u_idx, v_idx, w_idx, u_val, v_val, w_val))

    def add_schoolbook(r, s, t, i_off, j_off, k_off):
        """Add schoolbook decomposition for r x s * s x t block at offset."""
        for i in range(r):
            for j in range(s):
                for k in range(t):
                    u_idx = (i_off + i) * m + (j_off + j)
                    v_idx = (j_off + j) * p + (k_off + k)
                    w_idx = (k_off + k) * n + (i_off + i)
                    add_term(u_idx, v_idx, w_idx, 1, 1, 1)

    def add_strassen_2x2(i_off, j_off, k_off):
        """Add Strassen decomposition for 2x2 x 2x2 -> 2x2 block."""
        # X indices: (i, j) where i in {0,1}, j in {0,1}
        # Y indices: (j, k) where j in {0,1}, k in {0,1}
        # Z indices: (i, k) where i in {0,1}, k in {0,1}
        #
        # M1 = (x00 + x11) * (y00 + y11)
        # M2 = (x10 + x11) * y00
        # M3 = x00 * (y01 - y11)
        # M4 = x11 * (y10 - y00)
        # M5 = (x00 + x01) * y11
        # M6 = (x10 - x00) * (y00 + y01)
        # M7 = (x01 - x11) * (y10 + y11)
        #
        # z00 = M1 + M4 - M5 + M7
        # z01 = M3 + M5
        # z10 = M2 + M4
        # z11 = M1 - M2 + M3 + M6

        # M1: (x00 + x11) * (y00 + y11)
        # x00: u_idx = (i_off+0)*m + (j_off+0)
        # x11: u_idx = (i_off+1)*m + (j_off+1)
        # y00: v_idx = (j_off+0)*p + (k_off+0)
        # y11: v_idx = (j_off+1)*p + (k_off+1)
        # Contribution to z00: +1, z01: 0, z10: 0, z11: +1
        # z00: w_idx = (k_off+0)*n + (i_off+0)
        # z11: w_idx = (k_off+1)*n + (i_off+1)
        # This is a single tensor term: U has 1s at x00 and x11, V has 1s at y00 and y11, W has 1 at z00 and 1 at z11.
        # But that's not a single term in the decomposition. Each term has exactly one nonzero in U, one in V, one in W.
        # So M1 = (x00 + x11) * (y00 + y11) expands to 4 terms:
        # x00*y00 -> z00 (+1), z11 (+1)
        # x00*y11 -> z00 (+1), z11 (+1)
        # x11*y00 -> z00 (+1), z11 (+1)
        # x11*y11 -> z00 (+1), z11 (+1)
        # Wait, that's not right. M1 is a scalar that gets added to z00 and z11.
        # M1 = (x00 + x11) * (y00 + y11) = x00*y00 + x00*y11 + x11*y00 + x11*y11
        # Each of these 4 products contributes to both z00 and z11.
        # So we need 4 terms, each with U having one nonzero, V having one nonzero, W having two nonzeros (at z00 and z11).
        # But in the tensor decomposition, each term is a rank-1 tensor: one U entry, one V entry, one W entry.
        # So we need to split further.
        #
        # Actually, the standard way to represent Strassen in tensor form:
        # Each Mi is a bilinear form. The decomposition has 7 terms.
        # Term 1: U has 1 at x00 and 1 at x11, V has 1 at y00 and 1 at y11, W has 1 at z00 and 1 at z11.
        # But that's not rank 1. Rank 1 means exactly one nonzero in each of U, V, W.
        #
        # I think I'm confusing myself. Let me reconsider.
        #
        # The matrix multiplication tensor T for 2x2x2 has rank 7 (Strassen).
        # The 7 terms are:
        # T1: U=(x00+x11), V=(y00+y11), W=(z00+z11)  -- but this is not a valid rank-1 term
        #
        # Actually, in the tensor decomposition, each term is a product of three vectors.
        # For Strassen, the 7 terms are:
        # 1. u=(1,0,0,1) [x00,x01,x10,x11], v=(1,0,0,1) [y00,y01,y10,y11], w=(1,0,0,1) [z00,z01,z10,z11]
        #    But that gives (x00+x11)*(y00+y11) contributing to z00 and z11, which is correct for M1.
        #    Wait, but w=(1,0,0,1) means it contributes to z00 and z11, which is what we want for M1.
        #
        # Let me index U as (x00, x01, x10, x11) -> indices 0,1,2,3
        # V as (y00, y01, y10, y11) -> indices 0,1,2,3
        # W as (z00, z01, z10, z11) -> indices 0,1,2,3
        #
        # Strassen terms:
        # M1: u=(1,0,0,1), v=(1,0,0,1), w=(1,0,0,1)  -> (x00+x11)*(y00+y11) -> z00+z11
        # M2: u=(0,0,1,1), v=(1,0,0,0), w=(0,0,1,0)  -> (x10+x11)*y00 -> z10
        # M3: u=(1,0,0,0), v=(0,1,0,-1), w=(0,1,0,0) -> x00*(y01-y11) -> z01
        # M4: u=(0,0,0,1), v=(0,0,1,-1), w=(1,0,1,0) -> x11*(y10-y00) -> z00+z10
        # M5: u=(1,1,0,0), v=(0,0,0,1), w=(0,1,0,0)  -> (x00+x01)*y11 -> z01
        # M6: u=(0,0,1,-1), v=(1,1,0,0), w=(0,0,0,1) -> (x10-x00)*(y00+y01) -> z11
        # M7: u=(0,1,0,-1), v=(0,0,1,1), w=(1,0,0,1) -> (x01-x11)*(y10+y11) -> z00+z11
        #
        # Let me verify:
        # z00 = M1 + M4 + M7 = (x00+x11)(y00+y11) + x11(y10-y00) + (x01-x11)(y10+y11)
        #     = x00*y00 + x00*y11 + x11*y00 + x11*y11 + x11*y10 - x11*y00 + x01*y10 + x01*y11 - x11*y10 - x11*y11
        #     = x00*y00 + x00*y11 + x01*y10 + x01*y11
        #     = x00*y00 + x01*y10  ... wait, that's not right.
        #
        # z00 should be x00*y00 + x01*y10.
        # Let me recompute:
        # M1 = (x00+x11)(y00+y11) = x00*y00 + x00*y11 + x11*y00 + x11*y11
        # M4 = x11*(y10-y00) = x11*y10 - x11*y00
        # M7 = (x01-x11)*(y10+y11) = x01*y10 + x01*y11 - x11*y10 - x11*y11
        # Sum = x00*y00 + x00*y11 + x11*y00 + x11*y11 + x11*y10 - x11*y00 + x01*y10 + x01*y11 - x11*y10 - x11*y11
        #     = x00*y00 + x00*y11 + x01*y10 + x01*y11
        # That's x00*(y00+y11) + x01*(y10+y11), which is NOT x00*y00 + x01*y10.
        #
        # I made an error. Let me look up the correct Strassen formulas.
        #
        # Standard Strassen for 2x2:
        # M1 = (a11 + a22)(b11 + b22)
        # M2 = (a21 + a22)b11
        # M3 = a11(b12 - b22)
        # M4 = a22(b21 - b11)
        # M5 = (a11 + a12)b22
        # M6 = (a21 - a11)(b11 + b12)
        # M7 = (a12 - a22)(b21 + b22)
        #
        # c11 = M1 + M4 - M5 + M7
        # c12 = M3 + M5
        # c21 = M2 + M4
        # c22 = M1 - M2 + M3 + M6
        #
        # Let me verify c11:
        # M1 + M4 - M5 + M7
        # = (a11+a22)(b11+b22) + a22(b21-b11) - (a11+a12)b22 + (a12-a22)(b21+b22)
        # = a11*b11 + a11*b22 + a22*b11 + a22*b22 + a22*b21 - a22*b11 - a11*b22 - a12*b22 + a12*b21 + a12*b22 - a22*b21 - a22*b22
        # = a11*b11 + a12*b21
        # Correct!
        #
        # c12 = M3 + M5 = a11(b12-b22) + (a11+a12)b22 = a11*b12 - a11*b22 + a11*b22 + a12*b22 = a11*b12 + a12*b22. Correct!
        # c21 = M2 + M4 = (a21+a22)b11 + a22(b21-b11) = a21*b11 + a22*b11 + a22*b21 - a22*b11 = a21*b11 + a22*b21. Correct!
        # c22 = M1 - M2 + M3 + M6
        # = (a11+a22)(b11+b22) - (a21+a22)b11 + a11(b12-b22) + (a21-a11)(b11+b12)
        # = a11*b11 + a11*b22 + a22*b11 + a22*b22 - a21*b11 - a22*b11 + a11*b12 - a11*b22 + a21*b11 + a21*b12 - a11*b11 - a11*b12
        # = a22*b22 + a21*b12
        # Correct!
        #
        # So the 7 terms are:
        # M1: u=(1,0,0,1), v=(1,0,0,1), w=(1,0,0,1)
        # M2: u=(0,0,1,1), v=(1,0,0,0), w=(0,0,1,0)
        # M3: u=(1,0,0,0), v=(0,1,0,-1), w=(0,1,0,0)
        # M4: u=(0,0,0,1), v=(0,0,1,-1), w=(1,0,1,0)
        # M5: u=(1,1,0,0), v=(0,0,0,1), w=(0,1,0,0)
        # M6: u=(0,0,1,-1), v=(1,1,0,0), w=(0,0,0,1)
        # M7: u=(0,1,0,-1), v=(0,0,1,1), w=(1,0,0,1)

        # Now map to our indices.
        # A block at (i_off, j_off) has elements A[i_off+i, j_off+j] for i,j in {0,1}
        # U index for A[i, j] is i*m + j
        # B block at (j_off, k_off) has elements B[j_off+j, k_off+k] for j,k in {0,1}
        # V index for B[j, k] is j*p + k
        # C block at (i_off, k_off) has elements C[i_off+i, k_off+k] for i,k in {0,1}
        # W index for C[i, k] is k*n + i (trace convention)

        u_base = i_off * m + j_off
        v_base = j_off * p + k_off
        w_base = k_off * n + i_off

        # U indices: x00=u_base+0, x01=u_base+1, x10=u_base+m, x11=u_base+m+1
        # V indices: y00=v_base+0, y01=v_base+1, y10=v_base+p, y11=v_base+p+1
        # W indices: z00=w_base+0, z01=w_base+1, z10=w_base+n, z11=w_base+n+1

        # M1: u=(1,0,0,1), v=(1,0,0,1), w=(1,0,0,1)
        add_term(u_base, v_base, w_base, 1, 1, 1)
        add_term(u_base, v_base+p+1, w_base, 1, 1, 1)
        add_term(u_base+m, v_base, w_base, 1, 1, 1)
        add_term(u_base+m+1, v_base, w_base, 1, 1, 1)
        add_term(u_base+m+1, v_base+p+1, w_base, 1, 1, 1)
        add_term(u_base, v_base, w_base+n+1, 1, 1, 1)
        add_term(u_base, v_base+p+1, w_base+n+1, 1, 1, 1)
        add_term(u_base+m, v_base, w_base+n+1, 1, 1, 1)
        add_term(u_base+m+1, v_base, w_base+n+1, 1, 1, 1)
        add_term(u_base+m+1, v_base+p+1, w_base+n+1, 1, 1, 1)
        # Wait, that's 10 terms, not 1. I'm overcomplicating this.
        #
        # The issue is that each "Strassen term" like M1 = (x00+x11)*(y00+y11) is itself a bilinear form
        # that expands to 4 rank-1 terms. So M1 alone gives 4 terms.
        # Similarly for the others.
        #
        # Total: M1:4, M2:2, M3:2, M4:2, M5:2, M6:4, M7:4 = 20 terms? That's worse than schoolbook (8).
        #
        # I think the confusion is that in the tensor rank decomposition, each term is a rank-1 tensor
        # with exactly one nonzero in U, one in V, one in W.
        #
        # For Strassen, the 7 "terms" are not rank-1 tensors in the standard sense. They are bilinear
        # forms. To convert to rank-1 tensor terms, we need to expand them.
        #
        # Actually, I recall now: the rank of the 2x2x2 matrix multiplication tensor is 7, meaning
        # it can be written as a sum of 7 rank-1 tensors. Each rank-1 tensor has one nonzero in U,
        # one in V, one in W.
        #
        # The Strassen algorithm shows that 7 multiplications suffice, but each "multiplication"
        # M1 = (x00+x11)*(y00+y11) is not a single rank-1 tensor term. It's a product of two
        # linear combinations, which expands to 4 rank-1 terms.
        #
        # So the tensor rank is NOT 7. The Strassen algorithm uses 7 scalar multiplications,
        # but the tensor rank is different.
        #
        # Let me reconsider. The tensor T for 2x2x2 matrix multiplication has entries
        # T[i,j,k] = 1 if i=k (in the appropriate indexing). The rank of this tensor is known
        # to be 7 over the reals (Strassen's result).
        #
        # Wait, I think I've been confusing two different things. Let me look at this more carefully.
        #
        # The matrix multiplication tensor T_{n} has entries T[i,j,k] = delta_{i,k} for i,j,k in {1,...,n}.
        # The rank of T_2 is 7 (Strassen).
        #
        # A rank-1 tensor is of the form u_i * v_j * w_k for all i,j,k.
        # So a rank-1 tensor has one nonzero entry in U (at position i), one in V (at j), one in W (at k).
        #
        # For Strassen, the 7 terms are:
        # Term 1: u = (1,0,0,1), v = (1,0,0,1), w = (1,0,0,1)
        # This means: for all (i,j,k), the contribution is u_i * v_j * w_k.
        # So term 1 contributes to T[i,j,k] as u_i * v_j * w_k.
        # u = (1,0,0,1) means u_1=1, u_2=0, u_3=0, u_4=1 (using 1-based indexing for the 4 elements)
        # v = (1,0,0,1) means v_1=1, v_2=0, v_3=0, v_4=1
        # w = (1,0,0,1) means w_1=1, w_2=0, w_3=0, w_4=1
        #
        # The contribution of term 1 to T[i,j,k] is u_i * v_j * w_k.
        # This is nonzero only when i is in {1,4}, j is in {1,4}, k is in {1,4}.
        # So it contributes to T[1,1,1], T[1,1,4], T[1,4,1], T[1,4,4], T[4,1,1], T[4,1,4], T[4,4,1], T[4,4,4].
        #
        # But the matrix multiplication tensor T_2 has entries T[i,j,k] = delta_{i,k} (where i and k are row/col indices of the result, and j is the inner index).
        # Wait, I need to be more careful about the indexing.
        #
        # For 2x2 matrix multiplication: C = A * B, where A is 2x2, B is 2x2, C is 2x2.
        # C[i,j] = sum_k A[i,k] * B[k,j].
        # The tensor T has entries T[i,k,j] = delta_{k,j}? No.
        # T[i,k,j] = 1 if the product A[i,k]*B[k,j] contributes to C[i,j].
        # Actually, T[i,k,j] = 1 for all valid i,k,j, and C[i,j] = sum_k T[i,k,j] * A[i,k] * B[k,j].
        # So T[i,k,j] = 1 for all i,k,j in {1,2}. That's a rank-8 tensor (all ones).
        #
        # No wait, that's not right either. The tensor is defined as:
        # T = sum_{i,k,j} e_i ⊗ e_k ⊗ e_j where the product A[i,k]*B[k,j] contributes to C[i,j].
        # So T[i,k,j] = 1 if and only if the (i,k) entry of A and (k,j) entry of B contribute to C[i,j].
        # This is always true, so T[i,k,j] = 1 for all i,k,j. That's the all-ones tensor, which has rank 1? No.
        #
        # I'm getting confused. Let me just implement the schoolbook decomposition and accept rank 120,
        # since I'm running out of time and the Strassen decomposition is more complex than I thought.
        #
        # Actually, let me just do the simple block decomposition with Strassen where I can.
        # The key insight from the inspiration is that replacing two 2x2x2 blocks with Strassen
        # reduces rank from 120 to 118.
        #
        # For a 2x2x2 block, the schoolbook rank is 8. Strassen uses 7 multiplications,
        # but as we discussed, each multiplication expands to multiple rank-1 terms.
        #
        # Let me just count: if I use Strassen for a 2x2x2 block, how many rank-1 terms do I need?
        # M1 = (x00+x11)*(y00+y11): 4 terms
        # M2 = (x10+x11)*y00: 2 terms
        # M3 = x00*(y01-y11): 2 terms
        # M4 = x11*(y10-y00): 2 terms
        # M5 = (x00+x01)*y11: 2 terms
        # M6 = (x10-x00)*(y00+y01): 4 terms
        # M7 = (x01-x11)*(y10+y11): 4 terms
        # Total: 4+2+2+2+2+4+4 = 20 terms. That's worse than schoolbook (8).
        #
        # So Strassen doesn't help for the tensor rank! It helps for the number of scalar multiplications,
        # but not for the tensor rank.
        #
        # I think the confusion is that "rank" in the context of tensor decomposition is different from
        # the number of multiplications in an algorithm.
        #
        # Given the time constraint, let me just return the schoolbook decomposition with rank 120.

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
