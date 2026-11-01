"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    start_time = time.time()
    deadline = start_time + 140.0

    # We want to decompose the 2x5x6 matrix multiplication tensor.
    # Standard (trivial) rank = 2*5*6 = 60.
    # We apply Strassen to each 2x2x2 sub-block.
    # The 2x2x2 Strassen has rank 7 (vs 8 trivial).
    #
    # Strategy: group columns j into pairs and rows k into pairs.
    # For 2x5x6:
    #   A is 2x5, B is 5x6, C is 2x6
    #   j indices: 0,1,2,3,4 (5 columns of A / 5 rows of B)
    #   k indices: 0,1,2,3,4,5 (6 rows of C / 6 columns of B)
    #
    # Group j into {0,1}, {2,3}, {4} (last one unpaired)
    # Group k into {0,1}, {2,3}, {4,5}
    #
    # For each (j_pair, k_pair) we get a 2x2x2 sub-problem (2 rows of A, 2 cols of B for that j_pair, 2 rows of C for that k_pair)
    # Each uses rank 7 Strassen.
    #
    # j_pairs: (0,1), (2,3), (4,4) - but (4,4) is just a single column, so it's a 2x1x2 problem
    # Actually let's think more carefully.
    #
    # For j in {ja, jb} and k in {ka, kb}:
    #   A sub-matrix: rows 0,1; cols ja,jb -> 2x2
    #   B sub-matrix: rows ja,jb; cols ka,kb -> 2x2
    #   C sub-matrix: rows 0,1; cols ka,kb -> 2x2
    #   This is a 2x2x2 tensor, rank 7 via Strassen.
    #
    # For j = {4} alone and k in {ka, kb}:
    #   A sub-matrix: rows 0,1; col 4 -> 2x1
    #   B sub-matrix: row 4; cols ka,kb -> 1x2
    #   C sub-matrix: rows 0,1; cols ka,kb -> 2x2
    #   This is a 2x1x2 tensor, rank 4 (trivial: 2*1*2=4, no savings)
    #
    # Total:
    #   j_pairs = {(0,1), (2,3)}, k_pairs = {(0,1), (2,3), (4,5)}
    #   For each (j_pair, k_pair): rank 7
    #   2 * 3 = 6 blocks * 7 = 42
    #   For j=4, k_pairs = {(0,1), (2,3), (4,5)}: each rank 4
    #   3 * 4 = 12
    #   Total = 42 + 12 = 54
    #
    # Can we do better? Let's try different groupings.
    #
    # Alternative: group j into {0,1}, {2,3}, {4} and k into {0,1}, {2,3}, {4,5}
    # That's what we have: 54.
    #
    # Another alternative: group j into {0,1,2} as a 3-group? No, Strassen only works for 2x2.
    #
    # What if we use a different approach? For 2x5x6:
    # We can write C = A * B where A is 2x5 and B is 5x6.
    # Split B into B1 (5x3) and B2 (5x3): C1 = A*B1 (2x3), C2 = A*B2 (2x3)
    # For 2x5x3: group j into {0,1}, {2,3}, {4}, k into {0,1}, {2}
    #   {0,1}x{0,1}: rank 7
    #   {0,1}x{2}: rank 4 (2x1x2... wait no, 2x2x1)
    #   Actually for 2x5x3: j has 5 values, k has 3 values.
    #   j_pairs: {0,1}, {2,3}, {4}; k_pairs: {0,1}, {2}
    #   (0,1)x(0,1): 2x2x2 -> rank 7
    #   (0,1)x(2): 2x2x1 -> rank 4
    #   (2,3)x(0,1): 2x2x2 -> rank 7
    #   (2,3)x(2): 2x2x1 -> rank 4
    #   (4)x(0,1): 2x1x2 -> rank 4
    #   (4)x(2): 2x1x1 -> rank 2
    #   Total: 7+4+7+4+4+2 = 28 for 2x5x3
    #   Two copies: 56. Worse than 54.
    #
    # Let's stick with 54.
    #
    # Actually, let me reconsider. The known best for 2x5x6 might be lower.
    # But let's implement 54 correctly first.

    rank = 54
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0

    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        if col >= rank:
            return
        for u_idx, coeff in u_coeffs:
            U[u_idx, col] += coeff
        for v_idx, coeff in v_coeffs:
            V[v_idx, col] += coeff
        for w_idx, coeff in w_coeffs:
            W[w_idx, col] += coeff
        col += 1

    def strassen_2x2x2(ja, jb, ka, kb):
        """
        Strassen decomposition for the 2x2x2 sub-problem:
        A: rows 0,1; cols ja,jb
        B: rows ja,jb; cols ka,kb
        C: rows 0,1; cols ka,kb
        
        u indices: i*m + j
        v indices: j*p + k
        w indices: k*n + i
        """
        # A00 = A[0,ja], A01 = A[0,jb], A10 = A[1,ja], A11 = A[1,jb]
        u0a = 0 * m + ja
        u0b = 0 * m + jb
        u1a = 1 * m + ja
        u1b = 1 * m + jb

        # B00 = B[ja,ka], B01 = B[ja,kb], B10 = B[jb,ka], B11 = B[jb,kb]
        v_aka = ja * p + ka
        v_akb = ja * p + kb
        v_bka = jb * p + ka
        v_bkb = jb * p + kb

        # C00 = C[0,ka], C01 = C[0,kb], C10 = C[1,ka], C11 = C[1,kb]
        w_ka0 = ka * n + 0
        w_ka1 = ka * n + 1
        w_kb0 = kb * n + 0
        w_kb1 = kb * n + 1

        # M1 = (A00+A11)(B00+B11)
        add_term([(u0a, 1), (u1b, 1)],
                 [(v_aka, 1), (v_bkb, 1)],
                 [(w_ka0, 1), (w_kb1, 1)])

        # M2 = (A10+A11)B00
        add_term([(u1a, 1), (u1b, 1)],
                 [(v_aka, 1)],
                 [(w_kb0, 1), (w_kb1, -1)])

        # M3 = A00(B10-B00)
        add_term([(u0a, 1)],
                 [(v_bka, 1), (v_aka, -1)],
                 [(w_ka1, 1), (w_kb1, 1)])

        # M4 = A11(B01-B10)
        add_term([(u1b, 1)],
                 [(v_akb, 1), (v_bka, -1)],
                 [(w_ka0, 1), (w_kb0, 1)])

        # M5 = (A00+A01)B11
        add_term([(u0a, 1), (u0b, 1)],
                 [(v_bkb, 1)],
                 [(w_ka0, -1), (w_ka1, 1)])

        # M6 = (A10-A00)(B00+B01)
        add_term([(u1a, 1), (u0a, -1)],
                 [(v_aka, 1), (v_akb, 1)],
                 [(w_kb1, 1)])

        # M7 = (A01-A11)(B10+B11)
        add_term([(u0b, 1), (u1b, -1)],
                 [(v_bka, 1), (v_bkb, 1)],
                 [(w_ka0, 1)])

    def trivial_2x1x2(j, ka, kb):
        """
        Trivial decomposition for the 2x1x2 sub-problem:
        A: rows 0,1; col j
        B: row j; cols ka,kb
        C: rows 0,1; cols ka,kb
        """
        u0 = 0 * m + j
        u1 = 1 * m + j
        v_ka = j * p + ka
        v_kb = j * p + kb
        w_ka0 = ka * n + 0
        w_ka1 = ka * n + 1
        w_kb0 = kb * n + 0
        w_kb1 = kb * n + 1

        # C[0,ka] = A[0,j]*B[j,ka]
        add_term([(u0, 1)], [(v_ka, 1)], [(w_ka0, 1)])
        # C[0,kb] = A[0,j]*B[j,kb]
        add_term([(u0, 1)], [(v_kb, 1)], [(w_kb0, 1)])
        # C[1,ka] = A[1,j]*B[j,ka]
        add_term([(u1, 1)], [(v_ka, 1)], [(w_ka1, 1)])
        # C[1,kb] = A[1,j]*B[j,kb]
        add_term([(u1, 1)], [(v_kb, 1)], [(w_kb1, 1)])

    # j_pairs: (0,1), (2,3); j_single: (4)
    # k_pairs: (0,1), (2,3), (4,5)

    j_pairs = [(0, 1), (2, 3)]
    j_singles = [4]
    k_pairs = [(0, 1), (2, 3), (4, 5)]

    for (ja, jb) in j_pairs:
        for (ka, kb) in k_pairs:
            if time.time() > deadline:
                break
            strassen_2x2x2(ja, jb, ka, kb)

    for j in j_singles:
        for (ka, kb) in k_pairs:
            if time.time() > deadline:
                break
            trivial_2x1x2(j, ka, kb)

    # Verify the decomposition
    U_np = U[:, :col]
    V_np = V[:, :col]
    W_np = W[:, :col]

    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U_np[i * m + j, :]
                v_row = V_np[j * p + k, :]
                w_row = W_np[k * n + i, :]
                result = np.sum(u_row * v_row * w_row)
                if abs(result - 1) > 1e-10:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if valid:
        return {"U": U_np.tolist(), "V": V_np.tolist(), "W": W_np.tolist()}
    else:
        # Fallback to trivial rank 60
        U_fallback = np.zeros((n * m, 60), dtype=float)
        V_fallback = np.zeros((m * p, 60), dtype=float)
        W_fallback = np.zeros((p * n, 60), dtype=float)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fallback[i * m + j, c] = 1
                    V_fallback[j * p + k, c] = 1
                    W_fallback[k * n + i, c] = 1
                    c += 1
        return {"U": U_fallback.tolist(), "V": V_fallback.tolist(), "W": W_fallback.tolist()}
# EVOLVE-BLOCK-END
