import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 4, 7]


def construct():
    start = time.time()
    deadline = start + 110.0  # safety margin under 120s kill

    n, m, p = DIMENSIONS  # A: n x m, B: m x p, C: n x p

    def _elapsed_ok():
        return (time.time() - start) < deadline

    # We will attempt to find a low-rank decomposition using a structured
    # approach: block decomposition + Strassen-like recursion where applicable.
    #
    # For 2x4 * 4x7:
    #   We can split B into two 4x3 and 4x4 blocks? Or use the fact that
    #   2x4 * 4x7 = 2x7.
    #
    # Strategy: Use a combination of:
    #   1. Schoolbook for small blocks
    #   2. Strassen for 2x2 blocks (rank 7 instead of 8)
    #
    # Let's think about block structure:
    #   A is 2x4, B is 4x7.
    #   We can view A as [A1 | A2] where A1 is 2x2, A2 is 2x2.
    #   B as [[B1], [B2]] where B1 is 2x7, B2 is 2x7.
    #   C = A1*B1 + A2*B2 (each product is 2x7)
    #
    # For each 2x2 * 2x7 product, we can further decompose:
    #   2x2 * 2x7: split the 7 into 4+3.
    #   2x2 * 2x4: split 4 into 2+2, giving two 2x2*2x2 products.
    #   2x2 * 2x3: schoolbook gives 2*2*3=12, but maybe we can do better?
    #
    # Actually, let's use a more systematic approach. We'll build the decomposition
    # by combining smaller known decompositions.
    #
    # Key insight: The tensor for (n,m,p) multiplication can be built from
    # decompositions of smaller problems via Kronecker product or block composition.
    #
    # Let's define helper functions to create decompositions for base cases
    # and combine them.

    # Base case: 1x1 * 1x1 -> rank 1
    def base_111():
        return [[1]], [[1]], [[1]]

    # Schoolbook for (n,m,p): rank = n*m*p
    def schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        t = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][t] = 1
                    V[j * p + k][t] = 1
                    W[k * n + i][t] = 1
                    t += 1
        return U, V, W

    # Strassen for 2x2 * 2x2: rank 7
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    def strassen_222():
        # Each M is a product of two linear combinations of A entries and B entries.
        # We need to express each M as a sum of rank-1 terms.
        # Since each M is (linear combo of A) * (linear combo of B), and both are
        # 2x2 matrices, each M contributes 2*2=4 rank-1 terms to the 2x2*2x2 tensor.
        # But wait - we want to share the computation. The key is that M1..M7 are
        # computed once, and then combined.
        #
        # Let's define the 7 intermediate products. Each is a 2x2 matrix product
        # of two 2x2 matrices (which are linear combinations of A and B entries).
        #
        # For M1 = (A11+A22)(B11+B22):
        #   The matrix (A11+A22) has entries:
        #     [A11[0,0]+A22[0,0], A11[0,1]+A22[0,1]]
        #     [A21[0,0]+A22[1,0], A21[0,1]+A22[1,1]]
        #   Wait, I'm confusing myself. Let me use proper indexing.
        #
        # A is 2x2: A[i][j] for i,j in {0,1}
        # B is 2x2: B[j][k] for j,k in {0,1}
        #
        # A11 = A[0][0], A12 = A[0][1], A21 = A[1][0], A22 = A[1][1]
        # B11 = B[0][0], B12 = B[0][1], B21 = B[1][0], B22 = B[1][1]
        #
        # M1 = (A11+A22) * (B11+B22)  -- but these are scalars! So M1 is scalar.
        # Wait no, in Strassen for 2x2, the "blocks" are scalars. So each M is scalar.
        #
        # So we have 7 scalar products, each of which is a product of two linear
        # combinations. Each scalar product is rank 1 in the tensor sense.
        #
        # Let me define the 7 linear combinations for A and B:
        # A-side:
        #   a1 = A11 + A22
        #   a2 = A21 + A22
        #   a3 = A11
        #   a4 = A22
        #   a5 = A11 + A12
        #   a6 = A21 - A11
        #   a7 = A12 - A22
        #
        # B-side:
        #   b1 = B11 + B22
        #   b2 = B11
        #   b3 = B21 - B22
        #   b4 = B11 - B21
        #   b5 = B22
        #   b6 = B11 + B12
        #   b7 = B21 + B22
        #
        # Products:
        #   M1 = a1 * b1
        #   M2 = a2 * b2
        #   M3 = a3 * b3
        #   M4 = a4 * b4
        #   M5 = a5 * b5
        #   M6 = a6 * b6
        #   M7 = a7 * b7
        #
        # C entries:
        #   C11 = M1 + M4 - M5 + M7
        #   C12 = M3 + M5
        #   C21 = M2 + M4
        #   C22 = M1 - M2 + M3 + M6
        #
        # Now, to build the tensor decomposition:
        # U has 4 rows (A entries: A11, A12, A21, A22)
        # V has 4 rows (B entries: B11, B12, B21, B22)
        # W has 4 rows (C entries: C11, C12, C21, C22)
        #
        # For each of the 7 products Mt = at * bt:
        #   at is a linear combination of A entries, bt is a linear combination of B entries.
        #   We need to expand at * bt into rank-1 terms.
        #
        # Since at = sum_i alpha_i * A_i and bt = sum_j beta_j * B_j,
        #   at * bt = sum_i sum_j alpha_i * beta_j * A_i * B_j
        #   Each A_i * B_j is a rank-1 term in the tensor.
        #
        # But we want to share the computation of Mt. The standard approach is:
        #   Each Mt is a single "node" in the computation. We create one rank-1 term
        #   for each (A_i, B_j) pair that appears in any Mt.
        #
        # Actually, the cleaner way: for each Mt, we add rank-1 terms for each
        # (i,j) pair where alpha_i != 0 and beta_j != 0. The contribution to C
        # comes from the W matrix.
        #
        # Let me just enumerate all (i,j) pairs across all 7 products.

        # A indices: 0=A11, 1=A12, 2=A21, 3=A22
        # B indices: 0=B11, 1=B12, 2=B21, 3=B22
        # C indices: 0=C11, 1=C12, 2=C21, 3=C22

        # Define the 7 products as lists of (a_coeff, b_coeff) for each A_i, B_j
        products = []

        # M1: a1*b1, a1 = A11+A22, b1 = B11+B22
        products.append([(1, 0, 1, 0),  # A11 * B11
                         (1, 0, 1, 1),  # A11 * B22
                         (0, 1, 1, 0),  # A22 * B11
                         (0, 1, 1, 1)]) # A22 * B22

        # M2: a2*b2, a2 = A21+A22, b2 = B11
        products.append([(2, 0, 0, 0),  # A21 * B11
                         (3, 0, 0, 0)]) # A22 * B11

        # M3: a3*b3, a3 = A11, b3 = B21-B22
        products.append([(0, 0, 2, 1),  # A11 * B21
                         (0, 0, 2, -1)]) # A11 * (-B22)

        # M4: a4*b4, a4 = A22, b4 = B11-B21
        products.append([(3, 0, 0, 1),  # A22 * B11
                         (3, 0, 2, -1)]) # A22 * (-B21)

        # M5: a5*b5, a5 = A11+A12, b5 = B22
        products.append([(0, 0, 0, 1),  # A11 * B22
                         (1, 0, 0, 1)]) # A12 * B22

        # M6: a6*b6, a6 = A21-A11, b6 = B11+B12
        products.append([(2, 0, 0, 1),  # A21 * B11
                         (2, 0, 1, 1),  # A21 * B12
                         (0, 0, 0, -1), # (-A11) * B11
                         (0, 0, 1, -1)]) # (-A11) * B12

        # M7: a7*b7, a7 = A12-A22, b7 = B21+B22
        products.append([(1, 0, 2, 1),  # A12 * B21
                         (1, 0, 3, 1),  # A12 * B22
                         (3, 0, 2, -1), # (-A22) * B21
                         (3, 0, 3, -1)]) # (-A22) * B22

        # Now build U, V, W
        # Each term in each product contributes one rank-1 term.
        # Total terms: sum of lengths of all products
        total_terms = sum(len(prod) for prod in products)
        U = [[0] * total_terms for _ in range(4)]
        V = [[0] * total_terms for _ in range(4)]
        W = [[0] * total_terms for _ in range(4)]

        t = 0
        for prod_idx, prod in enumerate(products):
            for (ai, bi, coeff_a, coeff_b) in prod:
                U[ai][t] = coeff_a
                V[bi][t] = coeff_b
                # W contributions: we'll fill these in after
                t += 1

        # Now fill W based on C formulas
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        #
        # For each term t in product prod_idx, the contribution to C[c] is:
        #   W[c][t] = (coeff for c in the formula for M_{prod_idx})
        #
        # Let's map product index to C contributions:
        # prod 0 = M1: C11 += 1, C22 += 1
        # prod 1 = M2: C21 += 1, C22 += -1
        # prod 2 = M3: C12 += 1, C22 += 1
        # prod 3 = M4: C11 += 1, C21 += 1
        # prod 4 = M5: C11 += -1, C12 += 1
        # prod 5 = M6: C22 += 1
        # prod 6 = M7: C11 += 1

        c_contribs = {
            0: {0: 1, 3: 1},   # M1 -> C11, C22
            1: {2: 1, 3: -1},  # M2 -> C21, C22
            2: {1: 1, 3: 1},   # M3 -> C12, C22
            3: {0: 1, 2: 1},   # M4 -> C11, C21
            4: {0: -1, 1: 1},  # M5 -> C11, C12
            5: {3: 1},         # M6 -> C22
            6: {0: 1},         # M7 -> C11
        }

        t = 0
        for prod_idx, prod in enumerate(products):
            for (ai, bi, coeff_a, coeff_b) in prod:
                for c, wval in c_contribs[prod_idx].items():
                    W[c][t] = wval
                t += 1

        return U, V, W

    # Combine decompositions: if we have a decomposition for (n1,m1,p1) and
    # (n2,m2,p2), we can combine them for block matrix multiplication.
    #
    # For 2x4 * 4x7:
    #   Split as [2x2 | 2x2] * [[2x7],[2x7]]
    #   C = A1*B1 + A2*B2
    #   where A1,A2 are 2x2, B1,B2 are 2x7.
    #
    # For 2x2 * 2x7:
    #   Split B into [2x4 | 2x3]
    #   C = A*B_left + A*B_right
    #   A*B_left: 2x2 * 2x4
    #   A*B_right: 2x2 * 2x3
    #
    # For 2x2 * 2x4:
    #   Split B into [2x2 | 2x2]
    #   C = A*B1 + A*B2
    #   Each is 2x2 * 2x2 -> Strassen rank 7 each, total 14
    #
    # For 2x2 * 2x3:
    #   Schoolbook: 2*2*3 = 12
    #   Can we do better? Maybe split 3 into 1+2:
    #   C = A*B1 + A*B2 where B1 is 2x1, B2 is 2x2
    #   2x2*2x1: schoolbook 4, or maybe 3?
    #   2x2*2x2: Strassen 7
    #   Total: 4+7=11 or 3+7=10
    #
    # Let's try 2x2*2x1 with rank 3 (if possible) or just use 4.
    # Actually, 2x2*2x1 is just 4 scalar products, rank 4.
    #
    # So for 2x2*2x3: 4+7=11
    # For 2x2*2x4: 7+7=14
    # For 2x2*2x7: 14+11=25
    # For 2x4*4x7: 25+25=50
    #
    # That's better than 56! Let's implement this.

    def combine_add(U1, V1, W1, U2, V2, W2):
        """Combine two decompositions by addition (for C = C1 + C2)."""
        r1 = len(W1[0]) if W1 else 0
        r2 = len(W2[0]) if W2 else 0
        r = r1 + r2
        U = [row1 + [0]*r2 for row1 in U1] + [row2 + [0]*r1 for row2 in U2]
        V = [row1 + [0]*r2 for row1 in V1] + [row2 + [0]*r1 for row2 in V2]
        W = [row1 + [0]*r2 for row1 in W1] + [row2 + [0]*r1 for row2 in W2]
        return U, V, W

    def combine_block_A(U, V, W, n1, n2, m, p):
        """
        Combine for C = A1*B + A2*B where A1 is n1xm, A2 is n2xm.
        U has (n1+m) rows... wait, this is getting complicated.
        Let me just build the full decomposition directly.
        """
        pass

    # Let me just build the 2x4*4x7 decomposition directly using the block structure.

    # For 2x2 * 2x2: use Strassen (rank 7)
    # For 2x2 * 2x1: schoolbook (rank 4)
    # For 2x2 * 2x3: 2x2*2x1 + 2x2*2x2 = 4+7=11
    # For 2x2 * 2x4: 2x2*2x2 + 2x2*2x2 = 7+7=14
    # For 2x2 * 2x7: 2x2*2x4 + 2x2*2x3 = 14+11=25
    # For 2x4 * 4x7: 2x2*2x7 + 2x2*2x7 = 25+25=50

    # But we need to handle the indexing correctly. Let me build it step by step.

    def make_222():
        """Return U,V,W for 2x2*2x2 with Strassen, rank 7."""
        return strassen_222()

    def make_221():
        """Return U,V,W for 2x2*2x1, schoolbook rank 4."""
        U, V, W = schoolbook(2, 2, 1)
        return U, V, W

    def make_223():
        """Return U,V,W for 2x2*2x3, rank 11."""
        # C = A*B1 + A*B2 where B1 is 2x1, B2 is 2x2
        U1, V1, W1 = make_221()  # 2x2*2x1
        U2, V2, W2 = make_222()  # 2x2*2x2
        # Combine: U has 4 rows (A entries), V has 3 rows (B entries: 1 from B1, 2 from B2)
        # W has 2 rows (C entries: 1 from C1, 1 from C2)
        #
        # U1: 4 rows, 4 cols (A11,A12,A21,A22)
        # V1: 2 rows, 4 cols (B11,B21) -- wait, B is 2x1, so B has 2 entries
        # Actually for 2x2*2x1: A is 2x2 (4 entries), B is 2x1 (2 entries), C is 2x1 (2 entries)
        # U: 4 rows, V: 2 rows, W: 2 rows
        #
        # U2: 4 rows, 7 cols
        # V2: 4 rows, 7 cols (B is 2x2, 4 entries)
        # W2: 4 rows, 7 cols (C is 2x2, 4 entries)
        #
        # Combined:
        # U: 4 rows, 11 cols
        # V: 6 rows, 11 cols (2 from V1, 4 from V2)
        # W: 6 rows, 11 cols (2 from W1, 4 from W2)

        U = []
        for i in range(4):
            U.append(U1[i] + U2[i])
        V = []
        for i in range(2):
            V.append(V1[i] + [0]*7)
        for i in range(4):
            V.append([0]*4 + V2[i])
        W = []
        for i in range(2):
            W.append(W1[i] + [0]*7)
        for i in range(4):
            W.append([0]*4 + W2[i])
        return U, V, W

    def make_224():
        """Return U,V,W for 2x2*2x4, rank 14."""
        # C = A*B1 + A*B2 where B1 is 2x2, B2 is 2x2
        U1, V1, W1 = make_222()
        U2, V2, W2 = make_222()
        U = []
        for i in range(4):
            U.append(U1[i] + U2[i])
        V = []
        for i in range(4):
            V.append(V1[i] + [0]*7)
        for i in range(4):
            V.append([0]*7 + V2[i])
        W = []
        for i in range(4):
            W.append(W1[i] + [0]*7)
        for i in range(4):
            W.append([0]*7 + W2[i])
        return U, V, W

    def make_227():
        """Return U,V,W for 2x2*2x7, rank 25."""
        # C = A*B1 + A*B2 where B1 is 2x4, B2 is 2x3
        U1, V1, W1 = make_224()  # 2x2*2x4
        U2, V2, W2 = make_223()  # 2x2*2x3
        U = []
        for i in range(4):
            U.append(U1[i] + U2[i])
        V = []
        for i in range(8):
            V.append(V1[i] + [0]*11)
        for i in range(6):
            V.append([0]*14 + V2[i])
        W = []
        for i in range(8):
            W.append(W1[i] + [0]*11)
        for i in range(6):
            W.append([0]*14 + W2[i])
        return U, V, W

    def make_247():
        """Return U,V,W for 2x4*4x7, rank 50."""
        # C = A1*B1 + A2*B2 where A1,A2 are 2x2, B1,B2 are 2x7
        U1, V1, W1 = make_227()  # 2x2*2x7
        U2, V2, W2 = make_227()  # 2x2*2x7
        # U: 8 rows (A1 has 4, A2 has 4)
        # V: 14 rows (B1 has 14, B2 has 14)
        # W: 14 rows (C1 has 14, C2 has 14)
        U = []
        for i in range(4):
            U.append(U1[i] + [0]*25)
        for i in range(4):
            U.append([0]*25 + U2[i])
        V = []
        for i in range(14):
            V.append(V1[i] + [0]*25)
        for i in range(14):
            V.append([0]*25 + V2[i])
        W = []
        for i in range(14):
            W.append(W1[i] + [0]*25)
        for i in range(14):
            W.append([0]*25 + W2[i])
        return U, V, W

    try:
        if not _elapsed_ok():
            raise TimeoutError()

        U, V, W = make_247()
        rank = len(U[0])

        # Verify with a random test
        rng = np.random.default_rng(42)
        A = rng.integers(-3, 4, size=(n, m))
        B = rng.integers(-3, 4, size=(m, p))
        ref = A @ B

        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)

        # C[i,k] = sum_t U[i*m+j, t] * A[i,j] * V[j*p+k, t] * B[j,k] * W[k*n+i, t]
        # More precisely: C_flat[k*n+i] = sum_t (sum_j U[i*m+j,t]*A[i,j]) * (sum_j V[j*p+k,t]*B[j,k]) * W[k*n+i,t]
        # Wait, that's not right. The tensor is:
        # T[i,j,k] = sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t]
        # And C[i,k] = sum_j A[i,j] * B[j,k] = sum_j sum_t ... 
        # Actually: C[i,k] = sum_j A[i,j] * B[j,k]
        # And the decomposition says: C[i,k] = sum_t (sum_j U[i*m+j,t] * A[i,j]) * ... 
        # No, the standard tensor decomposition is:
        # C[i,k] = sum_j A[i,j] * B[j,k] = sum_t (sum_j U[i*m+j,t] * A[i,j]) * (sum_j V[j*p+k,t] * B[j,k]) * W[k*n+i,t]
        # That's not right either. The correct form is:
        # C[i,k] = sum_t U[i*m+*, t] contracted with A, V[*, j*p+k, t] contracted with B, times W[k*n+i, t]
        # More precisely:
        # Let a_t = sum_j U[i*m+j, t] * A[i,j]  -- this depends on i, so it's per i
        # Let b_t = sum_j V[j*p+k, t] * B[j,k]  -- this depends on k, so it's per k
        # Then C[i,k] = sum_t a_t(i) * b_t(k) * W[k*n+i, t]
        # That's not a simple tensor contraction. Let me re-examine.
        #
        # The matrix multiplication tensor T has T[i,j,k] = delta such that
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # The rank-r decomposition is:
        # T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
        # where U, V, W are the factor matrices.
        #
        # So to verify: for each (i,j,k), check that
        # 1 = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
        # (since T[i,j,k] = 1 if we think of A[i,j]*B[j,k] contributing to C[i,k])
        #
        # Wait, T[i,j,k] is the coefficient of A[i,j]*B[j,k] in C[i,k], which is 1.
        # So we need: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1 for all i,j,k.

        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for t in range(rank):
                        s += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                    if abs(s - 1.0) > 1e-9:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break

        if ok:
            return {"U": U, "V": V, "W": W}
        else:
            # Fall back to schoolbook
            U, V, W = schoolbook(n, m, p)
            return {"U": U, "V": V, "W": W}

    except Exception as e:
        # Fall back to schoolbook
        U, V, W = schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
