"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p  # 60

    deadline = time.time() + 115

    # Hybrid construction: use Strassen 2x2x2 blocks where possible,
    # schoolbook for the rest. This reduces rank from 60 to 58.

    # Strassen 2x2x2 formulas (for C = A * B where A is 2x2, B is 2x2):
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # We'll track which (i,j,k) triples are covered by Strassen blocks
    # and which terms (with coefficients) they contribute.
    # Each term is indexed by a column index.

    term_count = 0
    # U[(i,j)] = list of (coeff, term_index)
    # V[(j,k)] = list of (coeff, term_index)
    # W[(k,i)] = list of (coeff, term_index)
    U = {}
    V = {}
    W = {}

    def add_strassen_block(i0, j0, k0):
        """Add Strassen 2x2x2 block starting at (i0, j0, k0).
        Covers (i0,i0+1) x (j0,j0+1) x (k0,k0+1).
        Returns number of terms added (7)."""
        nonlocal term_count
        base = term_count

        # M1 = (A[i0,j0] + A[i0+1,j0+1]) * (B[j0,k0] + B[j0+1,k0+1])
        t = base; term_count += 1
        U.setdefault((i0, j0), []).append((1, t))
        U.setdefault((i0+1, j0+1), []).append((1, t))
        V.setdefault((j0, k0), []).append((1, t))
        V.setdefault((j0+1, k0+1), []).append((1, t))
        W.setdefault((k0, i0), []).append((1, t))
        W.setdefault((k0+1, i0+1), []).append((1, t))

        # M2 = (A[i0+1,j0] + A[i0+1,j0+1]) * B[j0,k0]
        t = base + 1; term_count += 1
        U.setdefault((i0+1, j0), []).append((1, t))
        U.setdefault((i0+1, j0+1), []).append((1, t))
        V.setdefault((j0, k0), []).append((1, t))
        W.setdefault((k0, i0+1), []).append((1, t))

        # M3 = A[i0,j0] * (B[j0,k0+1] - B[j0+1,k0+1])
        t = base + 2; term_count += 1
        U.setdefault((i0, j0), []).append((1, t))
        V.setdefault((j0, k0+1), []).append((1, t))
        V.setdefault((j0+1, k0+1), []).append((-1, t))
        W.setdefault((k0+1, i0), []).append((1, t))

        # M4 = A[i0+1,j0+1] * (B[j0+1,k0] - B[j0,k0])
        t = base + 3; term_count += 1
        U.setdefault((i0+1, j0+1), []).append((1, t))
        V.setdefault((j0+1, k0), []).append((1, t))
        V.setdefault((j0, k0), []).append((-1, t))
        W.setdefault((k0, i0+1), []).append((1, t))

        # M5 = (A[i0,j0] + A[i0,j0+1]) * B[j0+1,k0+1]
        t = base + 4; term_count += 1
        U.setdefault((i0, j0), []).append((1, t))
        U.setdefault((i0, j0+1), []).append((1, t))
        V.setdefault((j0+1, k0+1), []).append((1, t))
        W.setdefault((k0+1, i0), []).append((1, t))

        # M6 = (A[i0+1,j0] - A[i0,j0]) * (B[j0,k0] + B[j0,k0+1])
        t = base + 5; term_count += 1
        U.setdefault((i0+1, j0), []).append((1, t))
        U.setdefault((i0, j0), []).append((-1, t))
        V.setdefault((j0, k0), []).append((1, t))
        V.setdefault((j0, k0+1), []).append((1, t))
        W.setdefault((k0, i0+1), []).append((1, t))

        # M7 = (A[i0,j0+1] - A[i0+1,j0+1]) * (B[j0+1,k0] + B[j0+1,k0+1])
        t = base + 6; term_count += 1
        U.setdefault((i0, j0+1), []).append((1, t))
        U.setdefault((i0+1, j0+1), []).append((-1, t))
        V.setdefault((j0+1, k0), []).append((1, t))
        V.setdefault((j0+1, k0+1), []).append((1, t))
        W.setdefault((k0+1, i0), []).append((1, t))

        return 7

    # Determine which (i,j,k) triples are covered by Strassen blocks
    covered = set()

    # Add Strassen block 1: i0=0, j0=0, k0=0
    # Covers (i,j,k) in {0,1} x {0,1} x {0,1}
    add_strassen_block(0, 0, 0)
    for i in range(0, 2):
        for j in range(0, 2):
            for k in range(0, 2):
                covered.add((i, j, k))

    # Add Strassen block 2: i0=1, j0=2, k0=3
    # Covers (i,j,k) in {1,2} x {2,3} x {3,4}
    add_strassen_block(1, 2, 3)
    for i in range(1, 3):
        for j in range(2, 4):
            for k in range(3, 5):
                covered.add((i, j, k))

    # Add schoolbook terms for all other (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in covered:
                    continue
                t = term_count
                term_count += 1
                U.setdefault((i, j), []).append((1, t))
                V.setdefault((j, k), []).append((1, t))
                W.setdefault((k, i), []).append((1, t))

    rank = term_count

    # Convert to matrix format
    U_mat = [[0] * rank for _ in range(n * m)]
    V_mat = [[0] * rank for _ in range(m * p)]
    W_mat = [[0] * rank for _ in range(p * n)]

    for (i, j), coeffs in U.items():
        for coeff, t in coeffs:
            U_mat[i * m + j][t] = coeff

    for (j, k), coeffs in V.items():
        for coeff, t in coeffs:
            V_mat[j * p + k][t] = coeff

    for (k, i), coeffs in W.items():
        for coeff, t in coeffs:
            W_mat[k * n + i][t] = coeff

    return {"U": U_mat, "V": V_mat, "W": W_mat}
# EVOLVE-BLOCK-END
