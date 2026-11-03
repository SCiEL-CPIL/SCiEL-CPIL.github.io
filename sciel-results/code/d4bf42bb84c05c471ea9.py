"""Matrix multiplication 5x5x5 with Strassen-based rank minimization."""
import time
import random

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    if (n, m, p) != (5, 5, 5):
        # Fallback to schoolbook for other sizes
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

    # For 5x5x5: use Strassen for 4x4x4 block (rank 49) + schoolbook for border
    # Block 4x4: indices 0..3 in each dimension
    # Border: index 4 in each dimension
    # C = [[A00][A01][A02][A03][A04],
    #      [A10][A11][A12][A13][A14],
    #      [A20][A21][A22][A23][A24],
    #      [A30][A31][A32][A33][A34],
    #      [A40][A41][A42][A43][A44]]
    # B = [[B00][B01][B02][B03][B04],
    #      [B10][B11][B12][B13][B14],
    #      [B20][B21][B22][B23][B24],
    #      [B30][B31][B32][B33][B34],
    #      [B40][B41][B42][B43][B44]]
    # C[i][j] = sum_k A[i][k]*B[k][j]
    #
    # Split: indices 0-3 are "core", index 4 is "border"
    # C_core (4x4) = A_core*B_core + A_core*B_border + A_border*B_core + A_border*B_border
    # C_border_top (4x1) = A_core*B_border_right(4x1 col 4 of B) ... wait
    #
    # Let me be precise:
    # A is 5x5, B is 5x5, C is 5x5.
    # Core: rows/cols 0..3. Border: row/col 4.
    #
    # C[i][j] for i,j in 0..3 (core-core):
    #   = sum_{k=0}^3 A[i][k]*B[k][j] + A[i][4]*B[4][j]
    #
    # C[i][4] for i in 0..3 (core-border row):
    #   = sum_{k=0}^3 A[i][k]*B[k][4] + A[i][4]*B[4][4]
    #
    # C[4][j] for j in 0..3 (border-core col):
    #   = sum_{k=0}^3 A[4][k]*B[k][j] + A[4][4]*B[4][j]
    #
    # C[4][4] (border-border):
    #   = sum_{k=0}^3 A[4][k]*B[k][4] + A[4][4]*B[4][4]
    #
    # Total terms:
    # Core-core: 4x4x4 = 64 schoolbook, but Strassen gives 49
    # Core-border row (A[i][k]*B[k][4] for i,k in 0..3, j=4): 4*4=16
    # Border-core col (A[4][k]*B[k][j] for k,j in 0..3, i=4): 4*4=16
    # A[i][4]*B[4][j] for i in 0..3, j in 0..3: 16
    # A[4][4]*B[4][j] for j in 0..3: 4
    # A[i][4]*B[4][4] for i in 0..3: 4
    # A[4][k]*B[k][4] for k in 0..3: 4
    # A[4][4]*B[4][4]: 1
    #
    # Total non-core-core: 16+16+16+4+4+4+1 = 61
    # Core-core with Strassen: 49
    # Total: 49 + 61 = 110

    # Strassen 4x4x4 terms
    # Strassen for 2x2: 7 multiplications
    # For 4x4: apply Strassen to get rank 49
    #
    # Strassen recursion:
    # 1x1: 1 term (just multiply)
    # 2x2: 7 terms
    # 4x4: 7 terms each of which is a 2x2 multiply -> 7*7 = 49 terms
    #
    # Each term in the 4x4 decomposition is a rank-1 tensor:
    # U_col[a], V_col[b], W_col[c] where a,b,c are in [0,16) (flattened 4x4)
    #
    # Let's build the Strassen decomposition for 4x4x4.
    #
    # Strassen for 2x2:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B10-B11)
    # P4 = A11(B01-B10)
    # P5 = (A00+A10)B11
    # P6 = (A01-A00)(B00+B10)
    # P7 = (A10-A11)(B01+B11)
    #
    # C00 = P1 - P4 + P6
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P3 + P7
    #
    # For the rank-1 decomposition, each Pi is itself a rank-1 tensor if the
    # sub-entries are scalars. But when we apply Strassen to 2x2 blocks, each
    # Pi involves a 2x2 matrix multiplication which itself has rank 4 in schoolbook
    # or rank 7 in Strassen.
    #
    # For 4x4: split into 2x2 blocks. Apply Strassen formula. Each Pi is a
    # 2x2 matrix multiply, which we further decompose using Strassen (rank 7).
    # Total: 7 * 7 = 49 rank-1 terms.

    # Let me define the 49 terms explicitly.
    # Indexing: for a 4x4 matrix, element (i,j) where i,j in {0,1,2,3}
    # Block (bi, bj) contains elements (2*bi+di, 2*bj+dj) for di,dj in {0,1}
    #
    # Strassen level 1 (2x2 blocks):
    # Let blocks be A00, A01, A10, A11 (each 2x2)
    # P1 = (A00+A11)(B00+B11)  -> rank 7
    # P2 = (A10+A11)B00        -> rank 7
    # P3 = A00(B10-B11)        -> rank 7
    # P4 = A11(B01-B10)        -> rank 7
    # P5 = (A00+A10)B11        -> rank 7
    # P6 = (A01-A00)(B00+B10)  -> rank 7
    # P7 = (A10-A11)(B01+B11)  -> rank 7
    #
    # Each P has 7 rank-1 terms from inner Strassen.
    #
    # Inner Strassen for a 2x2 multiply X*Y:
    # Q1 = (X00+X11)(Y00+Y11)
    # Q2 = (X10+X11)Y00
    # Q3 = X00(Y10-Y11)
    # Q4 = X11(Y01-Y10)
    # Q5 = (X00+X10)Y11
    # Q6 = (X01-X00)(Y00+Y10)
    # Q7 = (X10-X11)(Y01+Y11)
    #
    # Result:
    # R00 = Q1 - Q4 + Q6
    # R01 = Q3 + Q5
    # R10 = Q2 + Q4
    # R11 = Q1 - Q3 + Q7
    #
    # Each Q is a scalar multiply: X_ik * Y_kj, which is rank 1.
    #
    # So for 4x4, each of the 7 outer Strassen multiplications contributes
    # 7 inner rank-1 terms, totaling 49.

    # Let me build this systematically.
    # For a rank-1 term in the 4x4x4 decomposition:
    # U[row] = 1, V[col] = 1, W[row] = 1 with coefficients.
    # Actually, each rank-1 term is: c * e_i^U * e_j^V * e_k^W
    # where the tensor entry T[a,b,c] = U[a,term]*V[b,term]*W[c,term]
    # and sum over terms = T for all (a,b,c).

    # T[a,b,c] for matrix mult = delta_{a_i, b_j} * delta_{b_i, c_j} ...
    # Actually T[a,b,c] = 1 if a and b form a valid (i,j) and b and c form valid (k,l)
    # with matching j=k, else 0.
    # For 4x4: a in [0,16), b in [0,16), c in [0,16)
    # a = 4*i + j (row i, col j of A)
    # b = 4*k + l (row k, col l of B)  -- wait, b indexes B which is m x p = 4x4
    # c = 4*p_row + n_col? No.
    #
    # Let me use the standard indexing:
    # U has shape (n*m, rank) = (16, rank): index = i*m + j where i in [0,n), j in [0,m)
    # V has shape (m*p, rank) = (16, rank): index = j*p + k where j in [0,m), k in [0,p)
    # W has shape (p*n, rank) = (16, rank): index = k*n + i where k in [0,p), i in [0,n)
    #
    # For n=m=p=4:
    # U[i*4+j], V[j*4+k], W[k*4+i]
    #
    # The tensor T[U_idx, V_idx, W_idx] = delta(j_A, j_B) * delta(k_B, k_C)
    # where U_idx = i_A*4 + j_A, V_idx = j_B*4 + k_B, W_idx = k_C*4 + i_C
    # T = 1 if j_A == j_B and k_B == k_C, else 0.
    #
    # A rank-1 term: U_col[a]=x, V_col[b]=y, W_col[c]=z contributes x*y*z to T[a,b,c].
    # For it to be valid: we need x*y*z = T[a,b,c] for the single (a,b,c) it covers.
    # Since T is 0 or 1, and entries are 0 or 1 (or -1 for Strassen),
    # each rank-1 term in Strassen has U=±1 at one position, V=±1 at one, W=±1 at one,
    # but the coefficient pattern gives the linear combination.

    # Actually in Strassen, the decomposition is:
    # C = sum_t (sum_a u_a A[a]) (sum_b v_b B[b]) (sum_c w_c)^T E_c
    # where the sum is over terms t, and each term is a rank-1 tensor.
    #
    # More precisely: C = sum_t U_t A V_t W_t^T
    # where U_t is a diagonal (or linear combo) matrix, V_t similar, W_t is a row vector.
    # In the flattened tensor form:
    # T[u_idx, v_idx, w_idx] = sum_t U[u_idx, t] * V[v_idx, t] * W[w_idx, t]
    #
    # For Strassen 2x2: the 7 terms have specific U, V, W patterns.
    #
    # Let me just hardcode the 49 terms for 4x4x4 Strassen.
    #
    # I'll use a recursive approach to generate them.

    def strassen_2x2_terms():
        """Return 7 (u, v, w) for 2x2x2 Strassen.
        u: dict from (i,j) in A to coefficient
        v: dict from (k,l) in B to coefficient
        w: dict from (i,l) in C to coefficient
        """
        # P1 = (A00+A11)(B00+B11)
        # C00 += P1, C11 += P1
        t1_u = {(0,0): 1, (1,1): 1}
        t1_v = {(0,0): 1, (1,1): 1}
        t1_w = {(0,0): 1, (1,1): 1}
        # Wait, this isn't quite right. Let me think again.
        #
        # The matrix product C = A*B where A, B, C are 2x2.
        # C[i,j] = sum_k A[i,k]*B[k,j]
        #
        # Strassen computes 7 products:
        # P1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
        # P2 = (A[1,0]+A[1,1])*B[0,0]
        # P3 = A[0,0]*(B[1,0]-B[1,1])
        # P4 = A[1,1]*(B[0,1]-B[1,0])
        # P5 = (A[0,0]+A[1,0])*B[1,1]
        # P6 = (A[0,1]-A[0,0])*(B[0,0]+B[1,0])
        # P7 = (A[1,0]-A[1,1])*(B[0,1]+B[1,1])
        #
        # C[0,0] = P1 - P4 + P6
        # C[0,1] = P3 + P5
        # C[1,0] = P2 + P4
        # C[1,1] = P1 - P3 + P7
        #
        # Each P is a scalar product of two linear combos.
        # In tensor form: P = (sum_a u_a A[a]) * (sum_b v_b B[b])
        # This is a rank-1 "bilinear form" on A and B, giving a scalar.
        # Then C is a linear combo of these scalars placed in C.
        #
        # So for the full 2x2x2 tensor T[a_idx, b_idx, c_idx]:
        # T = sum_t (sum_a u_t[a] * A[a]) * (sum_b v_t[b] * B[b]) * (sum_c w_t[c] * C_basis[c])
        # where C_basis[c] is the indicator for C element c.
        #
        # Actually: C = sum_t P_t * M_t where M_t is a fixed 2x2 matrix
        # (the "placement" matrix). And P_t = (U_t . A) * (V_t . B) where U_t, V_t
        # are linear functionals on A and B.
        #
        # In flattened tensor:
        # T[a, b, c] = sum_t u_t[a] * v_t[b] * w_t[c]
        # where u_t[a] is the coefficient of A element a in the t-th functional,
        # v_t[b] similar for B, and w_t[c] is the coefficient placing the product
        # into C element c.
        #
        # For P1: u1 = {A00:1, A11:1}, v1 = {B00:1, B11:1}
        # P1 contributes to C00 with +1 and C11 with +1.
        # So w1 = {C00:1, C11:1}
        #
        # For P2: u2 = {A10:1, A11:1}, v2 = {B00:1}
        # P2 contributes to C10 with +1.
        # w2 = {C10:1}
        #
        # etc.

        terms = []
        # P1
        terms.append(({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}))
        # P2
        terms.append(({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}))
        # P3
        terms.append(({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1}))
        # P4
        terms.append(({(1,1):1}, {(0,1):1, (1,0):-1}, {(0,0):-1, (1,0):1}))
        # P5
        terms.append(({(0,0):1, (1,0):1}, {(1,1):1}, {(0,1):1}))
        # P6
        terms.append(({(0,1):1, (0,0):-1}, {(0,0):1, (1,0):1}, {(0,0):1}))
        # P7
        terms.append(({(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1}, {(1,1):1}))
        return terms

    def build_4x4_strassen():
        """Build 49 rank-1 terms for 4x4x4 matrix multiplication via Strassen.
        Returns list of (u_dict, v_dict, w_dict) where dicts map (i,j) to coeff.
        u_dict: A element (i,j) -> coeff, i,j in 0..3
        v_dict: B element (k,l) -> coeff, k,l in 0..3
        w_dict: C element (i,l) -> coeff, i,l in 0..3
        """
        terms_2x2 = strassen_2x2_terms()
        terms_4x4 = []
        for outer in range(7):
            ou, ov, ow = terms_2x2[outer]
            # ou, ov, ow are dicts from block indices (0,0),(0,1),(1,0),(1,1) to coeff
            # For each inner term, combine
            for inner in range(7):
                iu, iv, iw = terms_2x2[inner]
                # iu: element within a 2x2 block -> coeff
                # We need to map: outer block (bi, bj) has sub-block at (di, dj)
                # A element (2*bi+di, 2*bj+dj) gets coeff ou[(bi,bj)] * iu[(di,dj)]
                # B element (2*bk+dk, 2*bl+dl) gets coeff ov[(bk,bl)] * iv[(dk,dl)]
                # C element (2*bi+di, 2*bl+dl) gets coeff ow[(bi,bl)] * iw[(di,dl)]

                new_u = {}
                for (bi, bj), cu in ou.items():
                    for (di, dj), ci in iu.items():
                        key = (2*bi + di, 2*bj + dj)
                        new_u[key] = new_u.get(key, 0) + cu * ci

                new_v = {}
                for (bk, bl), cv in ov.items():
                    for (dk, dl), ci in iv.items():
                        key = (2*bk + dk, 2*bl + dl)
                        new_v[key] = new_v.get(key, 0) + cv * ci

                new_w = {}
                for (bi, bl), cw in ow.items():
                    for (di, dl), ci in iw.items():
                        key = (2*bi + di, 2*bl + dl)
                        new_w[key] = new_w.get(key, 0) + cw * ci

                # Remove zero coefficients
                new_u = {k: v for k, v in new_u.items() if v != 0}
                new_v = {k: v for k, v in new_v.items() if v != 0}
                new_w = {k: v for k, v in new_w.items() if v != 0}

                terms_4x4.append((new_u, new_v, new_w))
        return terms_4x4

    terms_4x4 = build_4x4_strassen()
    assert len(terms_4x4) == 49

    # Now build the full 5x5x5 decomposition.
    # Core (0-3) x Core (0-3) x Core (0-3): 49 terms from Strassen
    # Border terms: 61 terms (schoolbook)
    #
    # U indices: i*5+j for i,j in 0..4
    # V indices: j*5+k for j,k in 0..4
    # W indices: k*5+i for k,i in 0..4

    total_rank = 49 + 61  # 110
    U = [[0] * total_rank for _ in range(25)]
    V = [[0] * total_rank for _ in range(25)]
    W = [[0] * total_rank for _ in range(25)]

    col = 0
    # Core-core terms from Strassen
    for (u_dict, v_dict, w_dict) in terms_4x4:
        for (i, j), cu in u_dict.items():
            if 0 <= i < 4 and 0 <= j < 4:
                U[i*5+j][col] = cu
        for (k, l), cv in v_dict.items():
            if 0 <= k < 4 and 0 <= l < 4:
                V[k*5+l][col] = cv
        for (i2, l2), cw in w_dict.items():
            if 0 <= i2 < 4 and 0 <= l2 < 4:
                W[l2*5+i2][col] = cw
        col += 1

    # Border terms (schoolbook)
    # C[i][4] for i in 0..3: sum_k A[i][k]*B[k][4] + A[i][4]*B[4][4]
    #   Terms: A[i][k]*B[k][4] for i,k in 0..3 (16 terms)
    for i in range(4):
        for k in range(4):
            U[i*5+k][col] = 1
            V[k*5+4][col] = 1
            W[4*5+i][col] = 1
            col += 1

    # C[4][j] for j in 0..3: sum_k A[4][k]*B[k][j] (16 terms)
    for k in range(4):
        for j in range(4):
            U[4*5+k][col] = 1
            V[k*5+j][col] = 1
            W[j*5+4][col] = 1
            col += 1

    # A[i][4]*B[4][j] for i in 0..3, j in 0..3 (16 terms)
    for i in range(4):
        for j in range(4):
            U[i*5+4][col] = 1
            V[4*5+j][col] = 1
            W[j*5+i][col] = 1
            col += 1

    # A[4][4]*B[4][j] for j in 0..3 (4 terms)
    for j in range(4):
        U[4*5+4][col] = 1
        V[4*5+j][col] = 1
        W[j*5+4][col] = 1
        col += 1

    # A[i][4]*B[4][4] for i in 0..3 (4 terms)
    for i in range(4):
        U[i*5+4][col] = 1
        V[4*5+4][col] = 1
        W[4*5+i][col] = 1
        col += 1

    # A[4][k]*B[k][4] for k in 0..3 (4 terms)
    for k in range(4):
        U[4*5+k][col] = 1
        V[k*5+4][col] = 1
        W[4*5+4][col] = 1
        col += 1

    # A[4][4]*B[4][4] (1 term)
    U[4*5+4][col] = 1
    V[4*5+4][col] = 1
    W[4*5+4][col] = 1
    col += 1

    assert col == total_rank, f"Expected {total_rank} columns, got {col}"

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
