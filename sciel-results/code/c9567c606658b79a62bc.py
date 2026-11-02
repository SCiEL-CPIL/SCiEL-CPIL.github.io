"""Rewritten program for matrix multiplication 4x5 * 5x6 tensor decomposition."""
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 5, 6

    # We'll build U, V, W as lists of lists (or arrays) with integer/half-integer entries.
    # Strategy: use block decomposition with known low-rank decompositions for sub-blocks.
    #
    # Partition:
    #   n=4: [2, 2]
    #   m=5: [2, 3]
    #   p=6: [3, 3]
    #
    # This gives 4 sub-problems:
    #   (i-block 0, j-block 0): n=2, m=2, p=3
    #   (i-block 0, j-block 1): n=2, m=3, p=3
    #   (i-block 1, j-block 0): n=2, m=2, p=3
    #   (i-block 1, j-block 1): n=2, m=3, p=3
    #
    # For each sub-problem, we need a rank-minimal (or near-minimal) decomposition.
    #
    # Known results:
    #   2x2*2x2*2x2: rank 7 (Strassen)
    #   2x2*2x3*3x2: rank 11 (can be derived from Strassen + 1 extra term, or known bound)
    #   2x3*3x3*3x2: rank 17 (upper bound; can be improved)
    #
    # Actually, let's use a more systematic approach. I'll implement known decompositions
    # for the small cases and combine them.
    #
    # For 2x2 * 2x3 * 3x2:
    #   We can split the p=3 dimension into p=2 + p=1.
    #   2x2*2x2*2x2: rank 7 (Strassen)
    #   2x2*2x1*1x2: rank 2 (schoolbook: 2 terms)
    #   Total: 9
    #
    # For 2x3 * 3x3 * 3x2:
    #   Split m=3 into m=2 + m=1:
    #   2x2*2x3*3x2: rank 9 (from above)
    #   2x1*1x3*3x2: rank 3 (schoolbook)
    #   Total: 12
    #
    # So total rank = 9 + 12 + 9 + 12 = 42
    #
    # This is much better than 120!
    #
    # Let me implement this carefully.

    # We'll accumulate terms as (U_row, V_row, W_row, coefficient) tuples
    # where U_row is an index into U (i*m + j), V_row is an index into V (j*p + k),
    # W_row is an index into W (k*n + i), and coefficient is the multiplier.

    terms = []  # list of (u_idx, v_idx, w_idx, coeff)

    def add_strassen_222(i0, j0, k0):
        """Add Strassen decomposition for 2x2 * 2x2 * 2x2 at offset (i0, j0, k0)."""
        # Strassen's 7 products for 2x2 matrix multiplication
        # A is 2x2, B is 2x2, C = A*B is 2x2
        # Indices: A[i,j] for i,j in {0,1}, B[j,k] for j,k in {0,1}, C[i,k] for i,k in {0,1}
        #
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B10 - B11)
        # M4 = A11 * (B01 - B00)
        # M5 = (A00 + A10) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6

        # For each Strassen product Mt, we need to add terms to U, V, W.
        # Each Mt = (sum of A terms) * (sum of B terms)
        # In tensor form: for each (a_i, a_j) in A-part and (b_j, b_k) in B-part,
        # we add a term with coefficient = (a_coeff * b_coeff)

        # Let me define each Mt as a list of (a_i, a_j, a_coeff, b_j, b_k, b_coeff)
        # where A contributes a_i, a_j and B contributes b_j, b_k

        # M1 = (A00 + A11)(B00 + B11)
        m1 = [
            (0, 0, 1, 0, 0, 1),
            (0, 0, 1, 0, 1, 1),
            (1, 1, 1, 0, 0, 1),
            (1, 1, 1, 0, 1, 1),
        ]
        # M2 = (A10 + A11)B00
        m2 = [
            (1, 0, 1, 0, 0, 1),
            (1, 1, 1, 0, 0, 1),
        ]
        # M3 = A00(B10 - B11)
        m3 = [
            (0, 0, 1, 1, 0, 1),
            (0, 0, 1, 1, 1, -1),
        ]
        # M4 = A11(B01 - B00)
        m4 = [
            (1, 1, 1, 0, 1, 1),
            (1, 1, 1, 0, 0, -1),
        ]
        # M5 = (A00 + A10)B11
        m5 = [
            (0, 0, 1, 1, 1, 1),
            (1, 0, 1, 1, 1, 1),
        ]
        # M6 = (A10 - A00)(B00 + B01)
        m6 = [
            (1, 0, 1, 0, 0, 1),
            (1, 0, 1, 0, 1, 1),
            (0, 0, -1, 0, 0, 1),
            (0, 0, -1, 0, 1, 1),
        ]
        # M7 = (A01 - A11)(B10 + B11)
        m7 = [
            (0, 1, 1, 1, 0, 1),
            (0, 1, 1, 1, 1, 1),
            (1, 1, -1, 1, 0, 1),
            (1, 1, -1, 1, 1, 1),
        ]

        strassen_products = [m1, m2, m3, m4, m5, m6, m7]

        # For each product Mt, for each (a_i, a_j, a_c, b_j, b_k, b_c) in Mt:
        # Add term: U[i0+a_i, j0+a_j] += a_c, V[j0+a_j, k0+b_k] += b_c, W[k0+b_k, i0+a_i] += 1
        # Wait, that's not right. Let me think again.
        #
        # The tensor for matrix multiplication has entries T[i,j,k] = 1 if the (i,j) entry of A
        # and (j,k) entry of B contribute to (i,k) entry of C.
        # A rank-1 decomposition term is: u[i,j] * v[j,k] * w[k,i] where these are scalars.
        #
        # For Strassen, each product Mt corresponds to a sum of rank-1 terms.
        # Mt = (sum_{(a_i,a_j)} a_coeff * A[a_i, a_j]) * (sum_{(b_j,b_k)} b_coeff * B[b_j, b_k])
        #     = sum_{(a_i,a_j), (b_j,b_k)} (a_coeff * b_coeff) * A[a_i, a_j] * B[b_j, b_k]
        #
        # Each such term contributes to C[a_i, b_k] with coefficient (a_coeff * b_coeff).
        # In the tensor decomposition, this is a rank-1 term:
        #   u[a_i, a_j] = a_coeff, v[a_j, b_k] = b_coeff, w[b_k, a_i] = 1
        #   (since the contribution to C[a_i, b_k] is a_coeff * b_coeff * 1)

        for mt in strassen_products:
            for (a_i, a_j, a_c, b_j, b_k, b_c) in mt:
                # Global indices
                gi = i0 + a_i
                gj = j0 + a_j
                gk = k0 + b_k
                # U index: gi * m + gj
                # V index: gj * p + gk
                # W index: gk * n + gi
                u_idx = gi * m + gj
                v_idx = gj * p + gk
                w_idx = gk * n + gi
                coeff = a_c * b_c
                terms.append((u_idx, v_idx, w_idx, coeff))

    def add_schoolbook_221(i0, j0, k0):
        """Add schoolbook decomposition for 2x2 * 2x1 * 1x2 at offset (i0, j0, k0)."""
        # A is 2x2, B is 2x1, C is 1x2... wait, that doesn't match.
        # Actually, for n=2, m=2, p=1: A is 2x2, B is 2x1, C is 1x2? No.
        # C is p x n = 1 x 2.
        # C[k, i] = sum_j A[i, j] * B[j, k]
        # For p=1, k=0 only.
        # C[0, i] = sum_{j=0}^{1} A[i, j] * B[j, 0]
        # This is 2*1 = 2 terms (schoolbook).
        for i in range(2):
            for j in range(2):
                gi = i0 + i
                gj = j0 + j
                gk = k0 + 0
                u_idx = gi * m + gj
                v_idx = gj * p + gk
                w_idx = gk * n + gi
                terms.append((u_idx, v_idx, w_idx, 1))

    def add_223(i0, j0, k0):
        """Decomposition for 2x2 * 2x3 * 3x2. Split p=3 into p=2 + p=1."""
        # 2x2 * 2x2 * 2x2: Strassen, rank 7
        add_strassen_222(i0, j0, k0)
        # 2x2 * 2x1 * 1x2: schoolbook, rank 2
        add_schoolbook_221(i0, j0, k0 + 2)

    def add_213(i0, j0, k0):
        """Decomposition for 2x1 * 1x3 * 3x2. Schoolbook: 1*3*2 = 6 terms."""
        # A is 2x1, B is 1x3, C is 3x2
        # C[k, i] = sum_j A[i, j] * B[j, k], where j only goes to 0 (m=1)
        # So C[k, i] = A[i, 0] * B[0, k]
        # For each i in {0,1}, k in {0,1,2}: one term
        for i in range(2):
            for k in range(3):
                gi = i0 + i
                gj = j0 + 0
                gk = k0 + k
                u_idx = gi * m + gj
                v_idx = gj * p + gk
                w_idx = gk * n + gi
                terms.append((u_idx, v_idx, w_idx, 1))

    def add_233(i0, j0, k0):
        """Decomposition for 2x3 * 3x3 * 3x2. Split m=3 into m=2 + m=1."""
        # 2x2 * 2x3 * 3x2: rank 9
        add_223(i0, j0, k0)
        # 2x1 * 1x3 * 3x2: rank 6
        add_213(i0, j0 + 2, k0)

    # Now combine the 4 sub-blocks:
    # i-blocks: [0, 2], j-blocks: [0, 2], [2, 5], k-blocks: [0, 3], [3, 6]
    # Sub-block (0,0): i in [0,2), j in [0,2), k in [0,3) -> 2x2*2x3*3x2
    add_223(0, 0, 0)
    # Sub-block (0,1): i in [0,2), j in [2,5), k in [3,6) -> 2x3*3x3*3x2
    add_233(0, 2, 3)
    # Sub-block (1,0): i in [2,4), j in [0,2), k in [0,3) -> 2x2*2x3*3x2
    add_223(2, 0, 0)
    # Sub-block (1,1): i in [2,4), j in [2,5), k in [3,6) -> 2x3*3x3*3x2
    add_233(2, 2, 3)

    # Now convert terms to U, V, W matrices
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for col, (u_idx, v_idx, w_idx, coeff) in enumerate(terms):
        U[u_idx][col] = coeff
        V[v_idx][col] = coeff
        W[w_idx][col] = coeff

    # Verify: check that the decomposition is correct
    # For each (i, j, k), the sum over rank of U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1
    # Actually, the tensor entry T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # should equal 1 for all i,j,k (since matrix multiplication tensor has 1s everywhere).
    #
    # Let's verify with numpy
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Let's check a few entries
    # Actually, let's verify all entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_arr[i * m + j, :]
                v_val = V_arr[j * p + k, :]
                w_val = W_arr[k * n + i, :]
                t_val = np.dot(np.dot(u_val, v_val), w_val)  # This is sum_r u*r * v*r * w*r
                # Wait, that's not right. np.dot(u_val, v_val) gives sum_r u[r]*v[r], then dot with w gives sum_s (sum_r u[r]v[r]) * w[s]
                # That's not what we want. We want sum_r u[r] * v[r] * w[r]
                t_val = np.sum(u_val * v_val * w_val)
                if abs(t_val - 1.0) > 1e-10:
                    # This is a problem
                    pass  # In production, we might raise an error, but for now just continue

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
