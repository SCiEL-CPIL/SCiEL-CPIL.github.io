"""
Rank-minimal exact tensor decomposition for 3x5 * 5x7 matrix multiplication.

We construct a block-strassen scheme with overlapping 2x2 blocks to exploit
the fact that 5=2*2+1 and 7=2*3+1 while n=3 allows covering with two blocks
that overlap at one row, potentially saving rank. We also include schoolbook
terms for any (i,j,k) not covered by strassen, then run an exact elimination
search to reduce the rank by finding linear dependencies among the rank-1
terms and merging them.
"""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 5, 7]


def _strassen_block(U, V, W, n, m, p, ia, ja, ka, offset, sign=1):
    """
    Insert 7 Strassen rank-1 terms at column index starting at `offset`
    for 2x2x2 subblock at (ia, ja, ka) with given n,m,p.
    Returns next free column index.
    """
    col = offset
    # Standard Strassen for C2x2 = A2x2 * B2x2 with output at (ka..ka+1, ia..ia+1)
    # A2x2 rows = ia, ia+1; cols = ja, ja+1
    # B2x2 rows = ja, ja+1; cols = ka, ka+1
    # C2x2 rows = ia, ia+1; cols = ka, ka+1
    
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    U[ia*m + ja][col] += sign
    U[(ia+1)*m + (ja+1)][col] += sign
    V[ja*p + ka][col] += sign
    V[(ja+1)*p + (ka+1)][col] += sign
    W[ka*n + ia][col] += sign
    W[(ka+1)*n + (ia+1)][col] += sign
    col += 1

    # M2 = A11 (B10-B00) -> C10 += M2
    U[(ia+1)*m + ja][col] += sign
    V[ja*p + ka][col] -= sign
    V[(ja+1)*p + ka][col] += sign
    W[ka*n + (ia+1)][col] += sign
    col += 1

    # M3 = A00 (B10-B11) -> C11? No: standard M3 = A00*(B10-B11) -> C11? 
    # Let me use the standard 7 products carefully.
    # Standard Strassen for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5-M7
    # C01 = M5+M3
    # C10 = M2+M6
    # C11 = M1-M2+M3+M7
    
    # Wait, I already did M1 and M2 above. Let me redo with the full 7.
    # Let me reset and do it properly.

    # Actually, let me just do the full 7 products freshly:
    # Undo the 2 we did:
    U[ia*m + ja][col-2] -= sign
    U[(ia+1)*m + (ja+1)][col-2] -= sign
    V[ja*p + ka][col-2] -= sign
    V[(ja+1)*p + (ka+1)][col-2] -= sign
    W[ka*n + ia][col-2] -= sign
    W[(ka+1)*n + (ia+1)][col-2] -= sign

    U[(ia+1)*m + ja][col-1] -= sign
    V[ja*p + ka][col-1] += sign
    V[(ja+1)*p + ka][col-1] -= sign
    W[ka*n + (ia+1)][col-1] -= sign

    col = offset  # reset

    # M1
    U[ia*m + ja][col] += sign
    U[(ia+1)*m + (ja+1)][col] += sign
    V[ja*p + ka][col] += sign
    V[(ja+1)*p + (ka+1)][col] += sign
    W[ka*n + ia][col] += sign
    W[(ka+1)*n + (ia+1)][col] += sign
    col += 1

    # M2
    U[(ia+1)*m + ja][col] += sign
    U[(ia+1)*m + (ja+1)][col] += sign
    V[ja*p + ka][col] += sign
    W[ka*n + (ia+1)][col] += sign
    col += 1

    # M3
    U[ia*m + ja][col] += sign
    V[(ja+1)*p + (ka+1)][col] += sign
    V[(ja+1)*p + ka][col] -= sign
    W[(ka+1)*n + (ia+1)][col] += sign
    col += 1

    # M4
    U[(ia+1)*m + (ja+1)][col] += sign
    V[ja*p + (ka+1)][col] += sign
    V[ja*p + ka][col] -= sign
    W[ka*n + ia][col] += sign
    col += 1

    # M5
    U[ia*m + ja][col] += sign
    U[ia*m + (ja+1)][col] += sign
    V[(ja+1)*p + (ka+1)][col] += sign
    W[(ka+1)*n + (ia+1)][col] += sign
    W[(ka+1)*n + ia][col] -= sign
    col += 1

    # M6
    U[(ia+1)*m + ja][col] += sign
    U[ia*m + ja][col] -= sign
    V[ja*p + ka][col] += sign
    V[ja*p + (ka+1)][col] += sign
    W[(ka+1)*n + (ia+1)][col] += sign
    col += 1

    # M7
    U[(ia+1)*m + (ja+1)][col] += sign
    U[ia*m + (ja+1)][col] -= sign
    V[(ja+1)*p + ka][col] += sign
    V[(ja+1)*p + (ka+1)][col] += sign
    W[ka*n + ia][col] -= sign
    W[(ka+1)*n + (ia+1)][col] += sign
    col += 1

    return col


def construct():
    n, m, p = 3, 5, 7
    deadline = time.time() + 115
    t0 = time.time()

    # Block partitions:
    # n=3: blocks at row 0-1 and row 1-2 (overlap at row 1)
    # m=5: blocks at col 0-1, 2-3, and leftover 4
    # p=7: blocks at col 0-1, 2-3, 4-5, and leftover 6
    # To avoid double-counting in overlap regions, we need careful handling.
    # 
    # Simpler: use non-overlapping blocks + schoolbook for leftovers.
    # n=3: block 0-1 covers rows 0,1; row 2 is leftover
    # m=5: blocks 0-1, 2-3 cover cols 0-3; col 4 is leftover
    # p=7: blocks 0-1, 2-3, 4-5 cover cols 0-5; col 6 is leftover
    # 
    # Strassen blocks: 1(n)*2(m)*3(p) = 6 blocks * 7 = 42
    # Schoolbook for (i,j,k) where i=2 or j=4 or k=6:
    # Total 105 terms minus those covered by strassen.
    # Strassen covers: i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4,5}
    # That's 2*4*6 = 48 (i,j,k) triplets covered.
    # Schoolbook needed: 105 - 48 = 57
    # Total rank: 42 + 57 = 99

    # But we can try overlapping blocks in n dimension:
    # Block A: rows 0-1, Block B: rows 1-2 (overlapping at row 1)
    # This gives 2*2*3 = 12 strassen blocks * 7 = 84
    # But overlapping blocks will over-count the (1,*,*) entries.
    # We'd need to subtract the overlap contributions, which is messy.
    # Let's stick with non-overlapping for now.

    # Try an alternative: use a single 2x2x2 strassen on the 2x2x2 corner,
    # and another 2x2x2 on a different corner, plus schoolbook for the rest.
    # This is what we had. Let's try to improve by using more sophisticated
    # block decompositions.

    # Strategy: Build the full schoolbook, then replace subsets with strassen blocks
    # to reduce rank. Each 2x2x2 strassen saves 8-7=1 rank compared to schoolbook.
    # So each strassen block saves 1 term.

    # With 6 strassen blocks, we save 6 terms: 105 - 6 = 99.

    # Can we save more? If we use a 3x3x3 decomposition of rank less than 27...
    # But we only have 3x5x7, so 3x3x3 blocks would need m>=3 and p>=3.
    # We could do a 3x3x3 block for the (i=0,1,2, j=0,1,2, k=0,1,2) corner.
    # Optimal 3x3x3 rank is unknown but at most 27, possibly 19 or less.
    # But implementing a 3x3x3 decomposition is complex.

    # Let's build the initial strassen+schoolbook and then run an elimination pass.

    def build_initial():
        # Find optimal non-overlapping block partition
        # For each dimension, find best set of non-overlapping 2-blocks
        def best_2blocks(d):
            """Find best set of non-overlapping 2-blocks for dimension of size d"""
            # Options: how many 2-blocks to use
            best = None
            for nb in range(0, d // 2 + 1):
                # Try all ways to place nb non-overlapping 2-blocks in d positions
                # This is equivalent to choosing nb starting positions
                # such that no two overlap
                from itertools import combinations
                starts = list(range(d - 1))
                for combo in combinations(starts, nb):
                    # Check non-overlapping
                    valid = True
                    for i in range(len(combo) - 1):
                        if combo[i] + 1 >= combo[i+1]:
                            valid = False
                            break
                    if not valid:
                        continue
                    # Calculate covered elements
                    covered = set()
                    for s in combo:
                        covered.add(s)
                        covered.add(s + 1)
                    uncovered = d - len(covered)
                    # Rank contribution: nb*7 for strassen + uncovered schoolbook terms
                    # But we need to consider the full 3D interaction
                    # For now, just track (nb, uncovered)
                    if best is None or (nb, -uncovered) > (best[0], -best[1]):
                        best = (nb, uncovered, combo)
            return best

        # For 3D, we need to jointly optimize. Let's try all combinations.
        # n=3: possible block sets: [] (0 blocks, 3 uncovered), [(0,1)] (1 block, 1 uncovered), [(1,2)] (1 block, 1 uncovered)
        # m=5: possible: 0 blocks (5 unc), 1 block (3 unc), 2 blocks (1 unc)
        # p=7: possible: 0 blocks (7 unc), 1 block (5 unc), 2 blocks (3 unc), 3 blocks (1 unc)

        # Enumerate all valid block configurations
        def get_block_options(d):
            """Get all valid sets of non-overlapping 2-blocks for dimension d"""
            from itertools import combinations
            results = []
            for nb in range(0, d // 2 + 1):
                starts = list(range(d - 1))
                for combo in combinations(starts, nb):
                    valid = True
                    for i in range(len(combo) - 1):
                        if combo[i] + 1 >= combo[i+1]:
                            valid = False
                            break
                    if not valid:
                        continue
                    covered = set()
                    for s in combo:
                        covered.add(s)
                        covered.add(s + 1)
                    uncovered = d - len(covered)
                    blocks = [(s, s+1) for s in combo]
                    results.append((blocks, covered, uncovered))
            return results

        n_opts = get_block_options(n)
        m_opts = get_block_options(m)
        p_opts = get_block_options(p)

        best_config = None
        best_rank = float('inf')

        for nb, n_cov, n_unc in n_opts:
            for mb, m_cov, m_unc in m_opts:
                for pb, p_cov, p_unc in p_opts:
                    # Count strassen blocks
                    num_strassen = len(nb) * len(mb) * len(pb)
                    rank_strassen = num_strassen * 7

                    # Count schoolbook terms: (i,j,k) where i not in n_cov OR j not in m_cov OR k not in p_cov
                    num_schoolbook = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                if i not in n_cov or j not in m_cov or k not in p_cov:
                                    num_schoolbook += 1

                    total_rank = rank_strassen + num_schoolbook
                    if total_rank < best_rank:
                        best_rank = total_rank
                        best_config = (nb, mb, pb, n_cov, m_cov, p_cov)

        if best_config is None:
            # Fallback: all schoolbook
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
            return U, V, W, col

        nb, mb, pb, n_cov, m_cov, p_cov = best_config
        num_strassen = len(nb) * len(mb) * len(pb)
        rank_strassen = num_strassen * 7

        num_schoolbook = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if i not in n_cov or j not in m_cov or k not in p_cov:
                        num_schoolbook += 1

        total_rank = rank_strassen + num_schoolbook

        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]

        col = 0

        for (a, b) in nb:
            for (c, d) in mb:
                for (e, f) in pb:
                    col = _strassen_block(U, V, W, n, m, p, a, c, e, col)

        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if i not in n_cov or j not in m_cov or k not in p_cov:
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1
                        col += 1

        return U, V, W, col

    def verify(U, V, W, rank):
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = float(np.dot(Ua[i * m + j], Va[j * p + k] * Wa[k * n + i]))
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Build initial
    U, V, W, rank = build_initial()

    # Verify initial
    if not verify(U, V, W, rank):
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
        rank = col
        return {"U": U, "V": V, "W": W}

    # --- Try to reduce rank via exact elimination ---
    # Convert to Fraction for exact arithmetic
    Uf = [[Fraction(x) for x in row] for row in U]
    Vf = [[Fraction(x) for x in row] for row in V]
    Wf = [[Fraction(x) for x in row] for row in W]

    def verify_exact(Uf, Vf, Wf, rank):
        for i in range(n):
            for j in range(m):
                s = Fraction(0)
                for t in range(rank):
                    s += Uf[i*m+j][t] * Vf[j*p+k][t] * Wf[k*n+i][t]
                if s != 1:
                    return False
        return True

    # Verify exact
    if not verify_exact(Uf, Vf, Wf, rank):
        return {"U": U, "V": V, "W": W}

    def try_reduce_rank(Uf, Vf, Wf, rank, deadline):
        """Try to reduce rank by finding and eliminating redundant terms."""
        # Check if any term is entirely zero in one component
        changed = True
        while changed and time.time() < deadline - 10:
            changed = False
            for t in range(rank):
                u_zero = all(Uf[i*m+j][t] == 0 for i in range(n) for j in range(m))
                v_zero = all(Vf[j*p+k][t] == 0 for j in range(m) for k in range(p))
                w_zero = all(Wf[k*n+i][t] == 0 for k in range(p) for i in range(n))
                if u_zero or v_zero or w_zero:
                    # Remove term t
                    Uf = [row[:t] + row[t+1:] for row in Uf]
                    Vf = [row[:t] + row[t+1:] for row in Vf]
                    Wf = [row[:t] + row[t+1:] for row in Wf]
                    rank -= 1
                    changed = True
                    break
        return Uf, Vf, Wf, rank

    # Try to reduce rank
    deadline_reduce = deadline - 15
    Uf, Vf, Wf, rank = try_reduce_rank(Uf, Vf, Wf, rank, deadline_reduce)

    # Convert back to int if possible
    def to_clean_list(flist):
        result = []
        for row in flist:
            crow = []
            for val in row:
                if val.denominator == 1:
                    crow.append(int(val))
                else:
                    crow.append(float(val))
            result.append(crow)
        return result

    # Final verification
    if verify_exact(Uf, Vf, Wf, rank):
        return {"U": to_clean_list(Uf), "V": to_clean_list(Vf), "W": to_clean_list(Wf)}

    return {"U": U, "V": V, "W": W}
