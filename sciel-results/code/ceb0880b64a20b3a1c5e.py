import time
import sys
from fractions import Fraction
import random

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 135.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8
    nm, mp, pn = n * m, m * p, p * n

    # Schoolbook fallback
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = Fraction(0)
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
                    if acc != 1:
                        return False
        return True

    # Strassen 2x2x2 decomposition
    S_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]

    def _verify_strassen():
        for i in range(2):
            for a in range(2):
                for k in range(2):
                    acc = 0
                    for r in range(7):
                        u_val = S_U[r][i * 2 + a]
                        v_val = S_V[r][a * 2 + k]
                        w_val = S_W[r][k * 2 + i]
                        acc += u_val * v_val * w_val
                    if acc != 1:
                        return False
        return True

    if not _verify_strassen():
        return _schoolbook()

    # Build the rank-44 solution first
    rank = 44
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]

    col = 0

    # Part 1: A1 (2x2) * B1 (2x8) using Strassen
    for block_idx in range(4):
        k0 = block_idx * 2
        k1 = k0 + 1
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li * 2 + la]
                    if u_val != 0:
                        u_idx = li * m + la
                        U[u_idx][col] = u_val

            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la * 2 + lk]
                    if v_val != 0:
                        v_idx = la * p + (k0 + lk)
                        V[v_idx][col] = v_val

            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk * 2 + li]
                    if w_val != 0:
                        w_idx = (k0 + lk) * n + li
                        W[w_idx][col] = w_val

            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    # Part 2: A2 (2x1) * B2 (1x8)
    for i in range(2):
        for k in range(8):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    candidate = {"U": U, "V": V, "W": W}

    if not _validate(candidate):
        return _schoolbook()

    # Now try to find a lower-rank solution by merging terms
    # Strategy: Try to merge one of the a=2 terms with a Strassen term
    # by modifying both to share a rank term.

    # The key insight: In the Strassen terms, the W factor uses indices (k, i) for k in {k0, k0+1} and i in {0, 1}.
    # The a=2 terms use W indices (k, i) for all k and i.
    # So W indices overlap! We might be able to modify a Strassen term to also contribute to the a=2 part.

    # Let's try a different approach: parameterize the rank terms and solve for entries.
    # We'll try rank 43, 42, etc.

    # For rank 43: try to merge one a=2 term with a Strassen term.
    # The a=2 term for (i, k) has U[i,2]=1, V[2,k]=1, W[k,i]=1.
    # A Strassen term for block j has specific U, V, W entries.
    # If we modify the Strassen term to have non-zero entries in U[i,2] and V[2,k],
    # we need to adjust the other terms to compensate.

    # This is complex. Let's try a simpler approach: random search for lower rank.

    # Try rank 43
    if _elapsed_ok():
        best_rank = 44
        best_candidate = candidate

        # Try to find a rank-43 solution by modifying one term
        # We'll try each possible merge: merge a=2 term (i, k) with Strassen term (block, r)
        for i_merge in range(2):
            for k_merge in range(8):
                if not _elapsed_ok():
                    break
                for block in range(4):
                    if not _elapsed_ok():
                        break
                    for r_merge in range(7):
                        if not _elapsed_ok():
                            break

                        # Create a modified rank-43 candidate
                        # Remove the a=2 term for (i_merge, k_merge) and modify the Strassen term
                        U_new = [row[:] for row in U]
                        V_new = [row[:] for row in V]
                        W_new = [row[:] for row in W]

                        # Find the column index of the a=2 term
                        # It's in the last 16 columns: col = 28 + i*8 + k
                        col_a2 = 28 + i_merge * 8 + k_merge

                        # Find the column index of the Strassen term
                        col_strassen = block * 7 + r_merge

                        # Remove the a=2 term by zeroing it out
                        # Actually, we can't just remove it; we need to redistribute its contribution.
                        # This is getting too complex. Let's skip this approach.
                        pass

        # If no improvement found, return the rank-44 solution
        return candidate

    return candidate

# EVOLVE-BLOCK-END
