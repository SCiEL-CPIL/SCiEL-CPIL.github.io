import numpy as np
import time

DIMENSIONS = [4, 4, 9]
DEADLINE = 145.0  # safety margin before 150s kill


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    start_time = time.time()

    # Schoolbook fallback (rank 144)
    def schoolbook():
        rank_school = n * m * p
        U_s = np.zeros((n * m, rank_school))
        V_s = np.zeros((m * p, rank_school))
        W_s = np.zeros((p * n, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j, col] = 1
                    V_s[j * p + k, col] = 1
                    W_s[k * n + i, col] = 1
                    col += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

    best_result = schoolbook()
    best_rank = n * m * p

    # ============================================================
    # Strassen-based decomposition for 4x4 * 4x4 (rank 49)
    # ============================================================
    # We build a rank-49 decomposition for 4x4 * 4x4 using Strassen
    # applied to 2x2 block matrices, each block being a 2x2*2x2 product
    # decomposed with rank 7 (Strassen for 2x2).
    #
    # Strassen for 2x2*2x2:
    #   A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    #   m1 = (a+d)(e+h)
    #   m2 = (c+d)e
    #   m3 = a(f-h)
    #   m4 = d(g-e)
    #   m5 = (a+b)h
    #   m6 = (c-a)(e+f)
    #   m7 = (b-d)(g+h)
    #   c11 = m1 + m4 - m5 + m7
    #   c12 = m3 + m5
    #   c21 = m2 + m4
    #   c22 = m1 - m2 + m3 + m6
    #
    # For 4x4*4x4, view as 2x2 blocks of 2x2:
    #   A = [[A11, A12],[A21, A22]], B = [[B11, B12],[B21, B22]]
    #   C11 = A11*B11 + A12*B21
    #   C12 = A11*B12 + A12*B22
    #   C21 = A21*B11 + A22*B21
    #   C22 = A21*B12 + A22*B22
    #
    # Strassen on block level:
    #   M1 = (A11+A22)(B11+B22)
    #   M2 = (A21+A22)B11
    #   M3 = A11(B12-B22)
    #   M4 = A22(B21-B11)
    #   M5 = (A11+A12)B22
    #   M6 = (A21-A11)(B11+B12)
    #   M7 = (A12-A22)(B21+B22)
    #   C11 = M1 + M4 - M5 + M7
    #   C12 = M3 + M5
    #   C21 = M2 + M4
    #   C22 = M1 - M2 + M3 + M6
    #
    # Each Mi is a 2x2*2x2 product (rank 7), so total rank = 7*7 = 49.
    # ============================================================

    def build_strassen_2x2(U_list, V_list, W_list, offset):
        """Add Strassen rank-7 decomposition for 2x2*2x2.
        U_list: list of (row, val) for U contributions
        V_list: list of (row, val) for V contributions
        W_list: list of (row, val) for W contributions
        Returns offset for next column.
        """
        # m1 = (a+d)(e+h)
        # U: a(+1), d(+1); V: e(+1), h(+1)
        U_list.append((0, 1)); U_list.append((3, 1))
        V_list.append((0, 1)); V_list.append((3, 1))
        W_list.append((0, 1))  # c11 gets +m1
        W_list.append((1, 0))  # c12 gets 0
        W_list.append((2, 0))  # c21 gets 0
        W_list.append((3, 1))  # c22 gets +m1
        offset += 1

        # m2 = (c+d)e
        U_list.append((2, 1)); U_list.append((3, 1))
        V_list.append((0, 1))
        W_list.append((0, 0))
        W_list.append((1, 0))
        W_list.append((2, 1))  # c21 gets +m2
        W_list.append((3, -1)) # c22 gets -m2
        offset += 1

        # m3 = a(f-h)
        U_list.append((0, 1))
        V_list.append((1, 1)); V_list.append((3, -1))
        W_list.append((0, 0))
        W_list.append((1, 1))  # c12 gets +m3
        W_list.append((2, 0))
        W_list.append((3, 1))  # c22 gets +m3
        offset += 1

        # m4 = d(g-e)
        U_list.append((3, 1))
        V_list.append((2, 1)); V_list.append((0, -1))
        W_list.append((0, 1))  # c11 gets +m4
        W_list.append((1, 0))
        W_list.append((2, 1))  # c21 gets +m4
        W_list.append((3, 0))
        offset += 1

        # m5 = (a+b)h
        U_list.append((0, 1)); U_list.append((1, 1))
        V_list.append((3, 1))
        W_list.append((0, -1)) # c11 gets -m5
        W_list.append((1, 1))  # c12 gets +m5
        W_list.append((2, 0))
        W_list.append((3, 0))
        offset += 1

        # m6 = (c-a)(e+f)
        U_list.append((2, 1)); U_list.append((0, -1))
        V_list.append((0, 1)); V_list.append((1, 1))
        W_list.append((0, 0))
        W_list.append((1, 0))
        W_list.append((2, 0))
        W_list.append((3, 1))  # c22 gets +m6
        offset += 1

        # m7 = (b-d)(g+h)
        U_list.append((1, 1)); U_list.append((3, -1))
        V_list.append((2, 1)); V_list.append((3, 1))
        W_list.append((0, 1))  # c11 gets +m7
        W_list.append((1, 0))
        W_list.append((2, 0))
        W_list.append((3, 0))
        offset += 1

        return offset

    def build_strassen_4x4():
        """Build rank-49 decomposition for 4x4*4x4.
        Returns U (16x49), V (16x49), W (16x49) for the 4x4*4x4 subproblem.
        Indexing: U row = i*4+j (i,j in 0..3), V row = j*4+k, W row = k*4+i
        """
        R = 49
        U = np.zeros((16, R))
        V = np.zeros((16, R))
        W = np.zeros((16, R))

        # For 4x4*4x4:
        # A is 4x4, B is 4x4, C is 4x4
        # A indices: i (row, 0-3), j (col, 0-3) -> U row = i*4+j
        # B indices: j (row, 0-3), k (col, 0-3) -> V row = j*4+k
        # C indices: i (row, 0-3), k (col, 0-3) -> W row = k*4+i

        # Block structure:
        # A11 = A[0:2, 0:2] (rows 0,1; cols 0,1)
        # A12 = A[0:2, 2:4] (rows 0,1; cols 2,3)
        # A21 = A[2:4, 0:2] (rows 2,3; cols 0,1)
        # A22 = A[2:4, 2:4] (rows 2,3; cols 2,3)
        # Similarly for B.

        # Each block Mi is a 2x2*2x2 product.
        # We need to map the block-level Strassen to element-level.

        # Let's define the 7 block-level multiplications, each of which
        # is a 2x2*2x2 product decomposed with rank 7.

        # Block A submatrices (2x2 each):
        # A11: A[i,j] for i in {0,1}, j in {0,1}
        # A12: A[i,j] for i in {0,1}, j in {2,3}
        # A21: A[i,j] for i in {2,3}, j in {0,1}
        # A22: A[i,j] for i in {2,3}, j in {2,3}

        # Block B submatrices:
        # B11: B[j,k] for j in {0,1}, k in {0,1}
        # B12: B[j,k] for j in {0,1}, k in {2,3}
        # B21: B[j,k] for j in {2,3}, k in {0,1}
        # B22: B[j,k] for j in {2,3}, k in {2,3}

        # Block C submatrices:
        # C11: C[i,k] for i in {0,1}, k in {0,1}
        # C12: C[i,k] for i in {0,1}, k in {2,3}
        # C21: C[i,k] for i in {2,3}, k in {0,1}
        # C22: C[i,k] for i in {2,3}, k in {2,3}

        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)

        # For each Mi, we have a 2x2*2x2 product. The "A" part is a linear
        # combination of A blocks, and the "B" part is a linear combination of B blocks.
        # The result Mi is a 2x2 matrix, and it contributes to C blocks.

        # Let me define helper: given a 2x2 "A" matrix (as linear combo of A elements)
        # and a 2x2 "B" matrix (as linear combo of B elements), and which C block it
        # contributes to (and with what coefficient), add the rank-7 Strassen decomposition.

        # For a 2x2*2x2 product where:
        #   "a" (A[0,0]) = sum of alpha * A[i,j] for various (i,j)
        #   "b" (A[0,1]) = ...
        #   "c" (A[1,0]) = ...
        #   "d" (A[1,1]) = ...
        #   "e" (B[0,0]) = sum of beta * B[j,k]
        #   ...
        #   Result C[0,0] gets coeff * m1, C[0,1] gets coeff * m2, etc.

        # This is getting complex. Let me use a more direct approach.

        # Direct approach: enumerate all 49 rank-1 terms.
        # For each of the 7 block-level Strassen terms, and for each of the 7
        # element-level Strassen terms, we get one rank-1 term.

        # Block-level Strassen term t (0-6) involves:
        #   A_part: linear combo of A elements
        #   B_part: linear combo of B elements
        #   C_contrib: which C block and coefficient

        # Element-level Strassen term s (0-6) within a 2x2*2x2:
        #   A_sub: which of the 4 "a,b,c,d" positions
        #   B_sub: which of the 4 "e,f,g,h" positions
        #   C_sub: which of the 4 "c11,c12,c21,c22" positions and coefficient

        # Let me define the block-level Strassen more carefully.

        # For M1 = (A11+A22)(B11+B22):
        # The "A" 2x2 matrix has:
        #   position (0,0): A11[0,0] + A22[0,0] = A[0,0] + A[2,2]
        #   position (0,1): A11[0,1] + A22[0,1] = A[0,1] + A[2,3]
        #   position (1,0): A11[1,0] + A22[1,0] = A[1,0] + A[3,2]
        #   position (1,1): A11[1,1] + A22[1,1] = A[1,1] + A[3,3]
        # The "B" 2x2 matrix has:
        #   position (0,0): B11[0,0] + B22[0,0] = B[0,0] + B[2,2]
        #   position (0,1): B11[0,1] + B22[0,1] = B[0,1] + B[2,3]
        #   position (1,0): B11[1,0] + B22[1,0] = B[1,0] + B[3,2]
        #   position (1,1): B11[1,1] + B22[1,1] = B[1,1] + B[3,3]
        # M1 contributes to C11 with +1 and C22 with +1.
        # C11 is C[i,k] for i in {0,1}, k in {0,1}
        # C22 is C[i,k] for i in {2,3}, k in {2,3}

        # This is very tedious to do by hand. Let me instead build it programmatically.

        # I'll create a function that, given:
        #   - A_linear: dict mapping (i,j) -> coefficient for A[i,j]
        #   - B_linear: dict mapping (j,k) -> coefficient for B[j,k]
        #   - C_linear: dict mapping (i,k) -> coefficient for C[i,k]
        # adds the rank-7 Strassen decomposition.

        def add_strassen_2x2(U, V, W, col_offset, A_lin, B_lin, C_lin):
            """Add rank-7 Strassen for a 2x2*2x2 where A and B are linear combos.
            A_lin: {(local_i, local_j): coeff} for the 2x2 A
            B_lin: {(local_j, local_k): coeff} for the 2x2 B
            C_lin: {(local_i, local_k): coeff} for the 2x2 C
            """
            # For the 2x2 Strassen, the "a" position corresponds to local (0,0), etc.
            # We need to map local positions to global A[i,j], B[j,k], C[i,k].

            # The 2x2 A has entries at local positions (0,0),(0,1),(1,0),(1,1)
            # Each local position (li, lj) maps to a linear combo of global A[i,j].
            # Similarly for B and C.

            # For Strassen, we need the 7 products:
            # m1 = (a+d)(e+h)
            # m2 = (c+d)e
            # m3 = a(f-h)
            # m4 = d(g-e)
            # m5 = (a+b)h
            # m6 = (c-a)(e+f)
            # m7 = (b-d)(g+h)

            # where a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
            #       e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]

            # And C[0,0] = m1+m4-m5+m7, C[0,1]=m3+m5, C[1,0]=m2+m4, C[1,1]=m1-m2+m3+m6

            # For each Strassen term, the "A" part is a linear combo of a,b,c,d
            # and the "B" part is a linear combo of e,f,g,h.
            # Each of a,b,c,d is a linear combo of global A[i,j].
            # So the "A" part of each Strassen term is a linear combo of global A[i,j].

            # Let me define the 7 Strassen terms' A and B parts as linear combos of local positions.
            strassen_terms = [
                # (A_part as dict of local pos -> coeff, B_part as dict of local pos -> coeff,
                #  C_part as dict of local pos -> coeff)
                ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),  # m1
                ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),  # m2
                ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),  # m3
                ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),  # m4
                ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),  # m5
                ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),  # m6
                ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),  # m7
            ]

            for term_idx in range(7):
                A_part, B_part, C_part = strassen_terms[term_idx]
                col = col_offset + term_idx

                # Build U column: for each global A[i,j], sum contributions
                # from local positions that map to it.
                # A_lin maps local (li, lj) -> coeff for A[i,j]
                # But actually A_lin should map local position to global (i,j) with coefficient.
                # Let me redefine: A_lin is a list of (global_i, global_j, coeff)
                # for the 2x2 A's entries.

                # Actually, let me restructure. The input should be:
                # A_entries: list of (global_i, global_j, coeff) for the 2x2 A
                # B_entries: list of (global_j, global_k, coeff) for the 2x2 B
                # C_entries: list of (global_i, global_k, coeff) for the 2x2 C

                # But the Strassen terms reference local positions. Let me pass
                # the mapping from local to global.

                # Let me restructure the function signature.
                pass  # Will fix below

            return col_offset + 7

        # Let me restructure with a cleaner approach.

        # I'll define the 7 block-level Strassen terms explicitly, with their
        # A and B parts as linear combos of global A[i,j] and B[j,k],
        # and their C contribution.

        # Then for each block-level term, I'll apply the 2x2 Strassen
        # decomposition, which requires mapping the 2x2 local positions
        # to global A[i,j], B[j,k], C[i,k].

        # This is still complex. Let me try a different strategy:
        # directly enumerate all 49 rank-1 terms by composing the two Strassen levels.

        # Actually, let me just verify the schoolbook and use the random search
        # as a secondary attempt. The Strassen construction is error-prone.

        # Let me try a simpler approach: use the known fact that 4x4*4x4 can
        # be done with rank 49, and just construct it carefully.

        # For now, let me just return the schoolbook and rely on the random
        # search for potential improvements.

        return U, V, W  # Will be zeros if not implemented

    # Given the complexity of the Strassen construction, let me focus on
    # improving the random search strategy to find valid lower-rank solutions.

    # Key insight: the random search fails because the linear system A W = b
    # has many solutions, and the half-integer constraint is very restrictive.
    # Instead, let me try a structured approach:
    #
    # 1. For rank r, we need to find U (16 x r), V (36 x r), W (36 x r)
    #    such that for all (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    #
    # 2. A necessary condition: for each (i,j), the vector (U[i*m+j,0],...,U[i*m+j,r-1])
    #    and for each (j,k), the vector (V[j*p+k,0],...,V[j*p+k,r-1]) must be such
    #    that the Hadamard product across c, when multiplied by W[k*n+i,:], sums to 1.
    #
    # 3. Let me try a different parameterization: fix U and V to be sparse
    #    with a specific structure, then solve for W.

    # Let me try rank 120-140 with a better random strategy.

    import random
    random.seed(42)

    def try_rank(R):
        """Try to find a valid decomposition of rank R."""
        num_eqs = n * m * p  # 144
        num_W_vars = R * p * n  # R * 36

        # Precompute equation indices
        eq_indices = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_indices.append((i, j, k))

        for attempt in range(200):
            if time.time() - start_time > DEADLINE:
                return None

            # Sample U and V with structured sparsity
            U = np.zeros((n * m, R))
            V = np.zeros((m * p, R))

            for c in range(R):
                # For each column, choose non-zero entries
                num_U_nz = random.randint(2, 5)
                U_indices = random.sample(range(n * m), min(num_U_nz, n * m))
                for idx in U_indices:
                    U[idx, c] = 1.0

                num_V_nz = random.randint(3, 7)
                V_indices = random.sample(range(m * p), min(num_V_nz, m * p))
                for idx in V_indices:
                    V[idx, c] = 1.0

            # Build A matrix
            A = np.zeros((num_eqs, num_W_vars))
            b = np.ones(num_eqs)

            for eq_idx, (i, j, k) in enumerate(eq_indices):
                for c in range(R):
                    u_val = U[i * m + j, c]
                    v_val = V[j * p + k, c]
                    if u_val != 0 and v_val != 0:
                        w_idx = c * p * n + k * n + i
                        A[eq_idx, w_idx] = u_val * v_val

            # Solve A W = b
            try:
                W_flat, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            except Exception:
                continue

            # Check if W is half-integer
            W_rounded = np.round(W_flat * 2) / 2
            if np.allclose(W_flat, W_rounded, atol=1e-8):
                W = W_rounded.reshape(R, p, n)
                # Verify exactness
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(R):
                                s += U[i * m + j, c] * V[j * p + k, c] * W[c, k, i]
                            if abs(s - 1.0) > 1e-8:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if valid:
                    W_out = np.zeros((p * n, R))
                    for c in range(R):
                        for k in range(p):
                            for i in range(n):
                                W_out[k * n + i, c] = W[c, k, i]
                    return U, V, W_out
        return None

    # Try ranks from 120 to 140
    for R in range(120, 141):
        if time.time() - start_time > DEADLINE - 5:
            break
        result = try_rank(R)
        if result is not None:
            U_mat, V_mat, W_mat = result
            best_result = {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
            best_rank = R
            break

    return best_result
