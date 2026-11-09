"""Rewritten solution for matrix multiplication 4,4,4."""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin under 150s

    n, m, p = DIMENSIONS  # 4, 4, 4

    # Rank 49 decomposition using recursive Strassen
    # T(4) <= T(2) x T(2) = 7 * 7 = 49 rank-1 terms

    # 2x2 Strassen data (for the inner 2x2 block products):
    # U_2x2[s] gives coefficients on the 4 elements of a 2x2 block of A for multiplier s
    # V_2x2[s] gives coefficients on the 4 elements of a 2x2 block of B for multiplier s
    # W_2x2[s] is a 2x2 matrix giving the coefficient pattern for the result 2x2 block

    U_2x2 = np.array([
        [1, 0, 0, 1],   # M1: A11 + A22
        [0, 0, 1, 1],   # M2: A21 + A22
        [1, 0, 0, 0],   # M3: A11
        [0, 0, 0, 1],   # M4: A22
        [1, 1, 0, 0],   # M5: A11 + A12
        [-1, 0, 1, 0],  # M6: A21 - A11
        [0, 1, 0, -1],  # M7: A12 - A22
    ], dtype=float)

    V_2x2 = np.array([
        [1, 0, 0, 1],   # M1: B11 + B22
        [1, 0, 0, 0],   # M2: B11
        [0, 1, 0, -1],  # M3: B12 - B22
        [-1, 0, 1, 0],  # M4: B21 - B11
        [0, 0, 0, 1],   # M5: B22
        [1, 1, 0, 0],   # M6: B11 + B12
        [0, 0, 1, 1],   # M7: B21 + B22
    ], dtype=float)

    # W_2x2[s]: 2x2 matrix of coefficients for the result block from multiplier s
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    W_2x2 = np.array([
        [[1, 0], [0, 1]],   # M1: contributes to C11 and C22
        [[0, 0], [1, -1]],  # M2: contributes to C21 and C22
        [[0, 1], [0, 1]],   # M3: contributes to C12 and C22
        [[1, 0], [1, 0]],   # M4: contributes to C11 and C21
        [[-1, 1], [0, 0]],  # M5: contributes to C11 and C12
        [[0, 0], [0, 1]],   # M6: contributes to C22
        [[1, 0], [0, 0]],   # M7: contributes to C11
    ], dtype=float)

    # Now build the 49-term decomposition.
    # The 4x4 matrices are viewed as 2x2 block matrices with 2x2 blocks.
    # Block indices: (bi, bj) where bi is block row (0,1) and bj is block col (0,1)
    # Element within block: (ei, ej) where ei is row within block (0,1) and ej is col (0,1)
    #
    # For the full 4x4 multiplication tensor, we use the tensor product structure:
    # T(4) = T(2) ⊗ T(2)
    #
    # The 49 terms are indexed by (s, t) where s is the outer Strassen multiplier
    # and t is the inner Strassen multiplier.
    #
    # For each outer multiplier s (0..6):
    #   A_s is a 4x4 matrix formed by combining 2x2 blocks of A according to the
    #   Strassen pattern for 2x2 block matrices.
    #   B_s is similarly formed from B.
    #   The product A_s * B_s is a 4x4 matrix.
    #
    # But we need to express this as a sum of 7 rank-1 terms (the inner Strassen).
    #
    # The key insight: the outer Strassen combines blocks, and within each block
    # product, we use the inner Strassen.
    #
    # Let's think of it differently. The 4x4 matrix A has elements A[i,j] where
    # i = 2*bi + ei, j = 2*bj + ej.
    #
    # For the outer Strassen (s), we form:
    #   A_s[i,j] = sum over blocks of (U_block_s[bi,bj] * A[bi*2+ei, bj*2+ej])
    #
    # Actually, the cleanest way is to use the Kronecker product structure.
    #
    # The 4x4 multiplication tensor can be written as:
    #   T_4 = sum_{s,t=0}^{6} u_{s,t} ⊗ v_{s,t} ⊗ w_{s,t}
    #
    # where u_{s,t}, v_{s,t}, w_{s,t} are vectors of length 16.
    #
    # The construction:
    # For the outer level (block level), we have 7 multipliers s.
    # For each s, the block-level U pattern tells us which 2x2 blocks of A are combined.
    # Similarly for V and W.
    #
    # Let's define the block-level Strassen patterns:
    # For 2x2 block matrices, the 7 multipliers are:
    # M1: (A00 + A11) * (B00 + B11)
    # M2: (A10 + A11) * B00
    # M3: A00 * (B01 - B11)
    # M4: A11 * (B10 - B00)
    # M5: (A00 + A01) * B11
    # M6: (A10 - A00) * (B00 + B01)
    # M7: (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Now, each M_s is a product of two 4x4 matrices (since the blocks are 2x2).
    # To compute this product, we use the inner Strassen (7 rank-1 terms).
    #
    # So the total number of terms is 7 * 7 = 49.
    #
    # For term (s, t):
    #   u_{s,t}[i,j] = (block pattern for A in multiplier s at block position) * (inner U_t pattern)
    #   v_{s,t}[j,k] = (block pattern for B in multiplier s at block position) * (inner V_t pattern)
    #   w_{s,t}[k,i] = (block pattern for C in multiplier s at block position) * (inner W_t pattern)
    #
    # Let's define the block-level patterns more carefully.
    #
    # For multiplier s, the A-side block pattern is a 2x2 matrix of 2x2 blocks.
    # Each block is either 0, A00, A11, A00+A11, etc.
    #
    # Let's define U_block[s] as a 4x4 matrix where the (bi,bj) block is a 2x2 matrix
    # giving the coefficient for A[bi*2+ei, bj*2+ej] in the s-th multiplier.
    #
    # M1: A00 + A11 for both diagonal blocks, 0 for off-diagonal
    #   U_block[0] = [[I, 0], [0, I]] where I is 2x2 identity? No.
    #   Actually, A_s = [[A00+A11, A00+A11], [A00+A11, A00+A11]]? No, that's not right.
    #
    # Let me re-derive. For 2x2 matrices:
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    #
    # M1 = (A00 + A11) * (B00 + B11)
    # Here, A00 + A11 is a scalar (or a matrix if we're doing block matrices).
    # For block matrices, A00 + A11 is a 2x2 matrix (sum of two 2x2 blocks).
    #
    # So for the outer level, the "A00 + A11" means we take the (0,0) block and (1,1) block
    # of A and add them. The result is a 2x2 matrix.
    #
    # Similarly, B00 + B11 is a 2x2 matrix.
    #
    # The product (A00+A11) * (B00+B11) is a 2x2 * 2x2 = 2x2 matrix (but in terms of
    # the full 4x4, it's actually a 4x4 matrix? No, wait.)
    #
    # I think I'm confusing myself. Let me start over with a clearer approach.
    #
    # The 4x4 matrix multiplication tensor has 4*4*4 = 64 entries (for each (i,j,k),
    # the coefficient of A[i,j]*B[j,k] in C[i,k]).
    #
    # We want to decompose this as a sum of rank-1 terms. Each rank-1 term is a
    # tri-linear form u ⊗ v ⊗ w where u is a vector of length 16 (flattened 4x4),
    # v is a vector of length 16, and w is a vector of length 16.
    #
    # The Strassen decomposition for 2x2 gives 7 terms. For 4x4, we can use the
    # tensor product: T(4) = T(2) ⊗ T(2), which gives 7*7 = 49 terms.
    #
    # The tensor product construction:
    # If T(2) = sum_{s=0}^{6} u_s ⊗ v_s ⊗ w_s, then
    # T(4) = sum_{s,t=0}^{6} (u_s ⊗ u_t) ⊗ (v_s ⊗ v_t) ⊗ (w_s ⊗ w_t)
    #
    # Here, ⊗ denotes the Kronecker product of vectors.
    #
    # So for term (s,t):
    #   u_{s,t} = u_s ⊗ u_t (Kronecker product, length 4)
    #   v_{s,t} = v_s ⊗ v_t (length 4)
    #   w_{s,t} = w_s ⊗ w_t (length 4)
    #
    # Wait, but u_s is a vector of length 4 (for 2x2 matrices), so u_s ⊗ u_t is
    # a vector of length 16 (for 4x4 matrices). That's correct.
    #
    # But we need to be careful about the ordering. The flattening convention is:
    # For a 2x2 matrix [[a, b], [c, d]], the flattened vector is [a, b, c, d]
    # (row-major).
    #
    # For a 4x4 matrix viewed as 2x2 blocks of 2x2:
    # [[A00, A01], [A10, A11]] where each Aij is 2x2.
    # The flattened 4x4 vector is [A00[0,0], A00[0,1], A00[1,0], A00[1,1],
    #                                A01[0,0], A01[0,1], A01[1,0], A01[1,1],
    #                                A10[0,0], A10[0,1], A10[1,0], A10[1,1],
    #                                A11[0,0], A11[0,1], A11[1,0], A11[1,1]]
    #
    # The Kronecker product u_s ⊗ u_t where u_s = [s0, s1, s2, s3] and
    # u_t = [t0, t1, t2, t3] gives:
    # [s0*t0, s0*t1, s0*t2, s0*t3, s1*t0, s1*t1, s1*t2, s1*t3,
    #  s2*t0, s2*t1, s2*t2, s2*t3, s3*t0, s3*t1, s3*t2, s3*t3]
    #
    # This corresponds to: for block (bi, bj) of the 4x4 matrix, the element
    # (ei, ej) within that block has coefficient u_s[bi*2+bj] * u_t[ei*2+ej].
    #
    # Wait, that's not quite right. Let me think again.
    #
    # The 2x2 matrix A has elements A[0,0], A[0,1], A[1,0], A[1,1].
    # Flattened: [A[0,0], A[0,1], A[1,0], A[1,1]]
    #
    # The 4x4 matrix A (viewed as 2x2 blocks) has elements:
    # A[0,0], A[0,1], A[0,2], A[0,3], A[1,0], ..., A[3,3]
    #
    # The block (0,0) of A is [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
    # The block (0,1) of A is [[A[0,2], A[0,3]], [A[1,2], A[1,3]]]
    # The block (1,0) of A is [[A[2,0], A[2,1]], [A[3,0], A[3,1]]]
    # The block (1,1) of A is [[A[2,2], A[2,3]], [A[3,2], A[3,3]]]
    #
    # If u_s is the coefficient vector for the 2x2 case, then u_s[bi*2+bj] gives
    # the coefficient for block (bi, bj). And u_t[ei*2+ej] gives the coefficient
    # for element (ei, ej) within that block.
    #
    # So the full coefficient for A[bi*2+ei, bj*2+ej] is u_s[bi*2+bj] * u_t[ei*2+ej].
    #
    # This is exactly the Kronecker product u_s ⊗ u_t if we index the 4x4 matrix
    # in block-major order (block index first, then element index).
    #
    # But the flattening convention for the 4x4 matrix is row-major:
    # A[0,0], A[0,1], A[0,2], A[0,3], A[1,0], ..., A[3,3]
    #
    # In terms of blocks: block (0,0) contains A[0,0], A[0,1], A[1,0], A[1,1]
    # block (0,1) contains A[0,2], A[0,3], A[1,2], A[1,3]
    # block (1,0) contains A[2,0], A[2,1], A[3,0], A[3,1]
    # block (1,1) contains A[2,2], A[2,3], A[3,2], A[3,3]
    #
    # The row-major flattening is:
    # A[0,0], A[0,1], A[0,2], A[0,3], A[1,0], A[1,1], A[1,2], A[1,3],
    # A[2,0], A[2,1], A[2,2], A[2,3], A[3,0], A[3,1], A[3,2], A[3,3]
    #
    # In block-major order, it would be:
    # A[0,0], A[0,1], A[1,0], A[1,1], A[0,2], A[0,3], A[1,2], A[1,3],
    # A[2,0], A[2,1], A[3,0], A[3,1], A[2,2], A[2,3], A[3,2], A[3,3]
    #
    # These are different! So we need to be careful.
    #
    # Actually, the Kronecker product u_s ⊗ u_t gives a vector in block-major order.
    # We need to convert to row-major order.
    #
    # Alternatively, we can just construct the vectors directly.
    #
    # Let's define:
    # For term (s, t), the U vector (length 16) has entries:
    #   U[bi*4 + bj*2 + ei*2 + ej] = U_2x2[s, bi*2+bj] * U_2x2[t, ei*2+ej]
    #
    # Wait, that's not right either. Let me think more carefully.
    #
    # The 2x2 Strassen U vector u_s = [u0, u1, u2, u3] corresponds to:
    #   u0 = coefficient for A[0,0]
    #   u1 = coefficient for A[0,1]
    #   u2 = coefficient for A[1,0]
    #   u3 = coefficient for A[1,1]
    #
    # For the 4x4 case, the U vector for term (s,t) should have:
    #   U[i*4+j] = (coefficient for block (bi,bj)) * (coefficient for element (ei,ej))
    #   where i = bi*2+ei, j = bj*2+ej
    #
    # The block (bi,bj) has coefficient U_2x2[s, bi*2+bj]
    # The element (ei,ej) within the block has coefficient U_2x2[t, ei*2+ej]
    #
    # So U[i*4+j] = U_2x2[s, (i//2)*2 + (j//2)] * U_2x2[t, (i%2)*2 + (j%2)]
    #
    # Similarly for V and W.
    #
    # Let's implement this.

    # Build the 49-term decomposition
    U = np.zeros((16, 49), dtype=float)
    V = np.zeros((16, 49), dtype=float)
    W = np.zeros((16, 49), dtype=float)

    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            for i in range(4):
                for j in range(4):
                    bi = i // 2
                    bj = j // 2
                    ei = i % 2
                    ej = j % 2
                    U[i * 4 + j, col] = U_2x2[s, bi * 2 + bj] * U_2x2[t, ei * 2 + ej]

                    # For V: the indices are (j, k) for B[j,k]
                    # V[j*4+k] = V_2x2[s, (j//2)*2 + (k//2)] * V_2x2[t, (j%2)*2 + (k%2)]
                    # But we need to fill V separately
            for j in range(4):
                for k in range(4):
                    bj = j // 2
                    bk = k // 2
                    ej = j % 2
                    ek = k % 2
                    V[j * 4 + k, col] = V_2x2[s, bj * 2 + bk] * V_2x2[t, ej * 2 + ek]

            # For W: the indices are (k, i) for C[k,i] in the trace convention
            # W[k*n+i, col] = W_2x2[s, (k//2)*2 + (i//2)] * W_2x2[t, (k%2)*2 + (i%2)]
            # But W_2x2[s] is a 2x2 matrix, not a vector. We need to flatten it.
            # W_2x2[s, bi, bk] gives the coefficient for block (bi, bk) of C.
            # And within that block, the element (ei, ej) has coefficient W_2x2[t, ei, ej].
            #
            # Wait, W_2x2[s] is a 2x2 matrix where W_2x2[s, a, b] is the coefficient
            # for C[a, b] from multiplier s. Similarly for W_2x2[t].
            #
            # So for the 4x4 case, W[k*4+i] = W_2x2[s, (k//2), (i//2)] * W_2x2[t, (k%2), (i%2)]

            for k in range(4):
                for i in range(4):
                    bk = k // 2
                    bi = i // 2
                    ek = k % 2
                    ei = i % 2
                    W[k * 4 + i, col] = W_2x2[s, bk, bi] * W_2x2[t, ek, ei]

    # Verify the decomposition
    # The matrix multiplication tensor T[i,j,k] should equal sum_c U[i*4+j, c] * V[j*4+k, c] * W[k*4+i, c]
    # T[i,j,k] = 1 if i,j,k are valid indices (always 1 for matrix multiplication)
    
    # Let's verify a few entries
    # For a random test:
    A_test = np.random.randn(4, 4)
    B_test = np.random.randn(4, 4)
    C_test = A_test @ B_test

    # Compute using the decomposition
    C_decomp = np.zeros((4, 4))
    for c in range(49):
        u_vec = U[:, c].reshape(4, 4)
        v_vec = V[:, c].reshape(4, 4)
        w_vec = W[:, c].reshape(4, 4)
        # The contribution is: (u_vec^T * A) * (v_vec^T * B) * w_vec? No.
        # Actually, the rank-1 term is: u^T A * v^T B * w^T C? No.
        #
        # The tensor decomposition is: T = sum_c u_c ⊗ v_c ⊗ w_c
        # This means: T[i,j,k] = sum_c U[i*4+j, c] * V[j*4+k, c] * W[k*4+i, c]
        #
        # And C[i,k] = sum_j A[i,j] * B[j,k] = sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k]
        #
        # So C_decomp[i,k] = sum_c (sum_{i,j} U[i*4+j,c] * A[i,j]) * (sum_{j,k} V[j*4+k,c] * B[j,k]) * ... 
        # Wait, that's not right. Let me think again.
        #
        # T[i,j,k] = sum_c U[i*4+j, c] * V[j*4+k, c] * W[k*4+i, c]
        # C[i,k] = sum_{j} A[i,j] * B[j,k]
        #
        # If we substitute:
        # C[i,k] = sum_j A[i,j] B[j,k]
        #         = sum_{i,j,k} delta_{i,i'} delta_{k,k'} A[i,j] B[j,k]  (for fixed i', k')
        #
        # Actually, the tensor T encodes the multiplication:
        # C[i,k] = sum_{j} A[i,j] B[j,k]
        #
        # The tensor T has T[i,j,k] = 1 for all valid i,j,k (since each term A[i,j]*B[j,k]
        # contributes to C[i,k] with coefficient 1).
        #
        # So T[i,j,k] = 1 for all i,j,k in range(4).
        #
        # And we need: T[i,j,k] = sum_c U[i*4+j, c] * V[j*4+k, c] * W[k*4+i, c]
        #
        # Let's verify this.

    # Verify T[i,j,k] = sum_c U[i*4+j, c] * V[j*4+k, c] * W[k*4+i, c]
    # This should equal 1 for all i,j,k.
    max_err = 0.0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                val = 0.0
                for c in range(49):
                    val += U[i * 4 + j, c] * V[j * 4 + k, c] * W[k * 4 + i, c]
                max_err = max(max_err, abs(val - 1.0))

    if max_err > 1e-10:
        # Fallback to schoolbook if verification fails
        rank = 64
        U = np.zeros((16, rank), dtype=float)
        V = np.zeros((16, rank), dtype=float)
        W = np.zeros((16, rank), dtype=float)
        column = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U[i * 4 + j, column] = 1
                    V[j * 4 + k, column] = 1
                    W[k * 4 + i, column] = 1
                    column += 1
    else:
        # Convert to lists for the return interface
        U = U.tolist()
        V = V.tolist()
        W = W.tolist()

    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
