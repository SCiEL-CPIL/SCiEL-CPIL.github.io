import time
import random
from itertools import combinations

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Rank = 105 - (number of disjoint 2x2x2 Strassen blocks).
    Find a maximum packing of disjoint blocks via random restarts + local search
    (2-for-1 swap moves) on a bitset overlap structure.
    """
    start_time = time.time()
    deadline = start_time + 112.0  # safety margin before 120s / 150s kill

    n, m, p = DIMENSIONS  # 3, 5, 7
    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21
    total = n * m * p  # 105

    # Precompute all blocks: each = (params, frozenset of (i,j,k), bitmask)
    blocks = []
    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for j1 in range(m - 1):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p - 1):
                        for k2 in range(k1 + 1, p):
                            mask = 0
                            triples = []
                            for i in (i1, i2):
                                for j in (j1, j2):
                                    for k in (k1, k2):
                                        idx = (i * m + j) * p + k
                                        mask |= (1 << idx)
                                        triples.append((i, j, k))
                            blocks.append(((i1, i2, j1, j2, k1, k2), frozenset(triples), mask))

    B = len(blocks)
    # Overlap matrix as bitmasks: ov[a] has bit b set if blocks a,b share a triple
    ov = [0] * B
    for a in range(B):
        ma = blocks[a][2]
        bits = 0
        for b in range(B):
            if a != b and (ma & blocks[b][2]):
                bits |= (1 << b)
        ov[a] = bits

    def greedy(packing):
        used = 0
        res = []
        for b in packing:
            if (used & blocks[b][2]) == 0:
                used |= blocks[b][2]
                res.append(b)
        return res

    def local_search(packing):
        used = 0
        for b in packing:
            used |= blocks[b][2]
        improved = True
        while improved:
            improved = False
            cur_len = len(packing)
            for a in range(B):
                if (used & blocks[a][2]) == 0:
                    # free block; try to swap out one current block to fit a plus more
                    for c in packing:
                        # remove c
                        used2 = used & ~blocks[c][2]
                        if (used2 & blocks[a][2]) == 0:
                            used2 |= blocks[a][2]
                            # try to add another free block d
                            for d in range(B):
                                if (used2 & blocks[d][2]) == 0 and d not in packing and d != a:
                                    packing2 = [x for x in packing if x != c]
                                    packing2.append(a)
                                    packing2.append(d)
                                    return packing2
                        break  # only try one removal per a to save time
        return packing

    best = []
    best_len = 0

    rng = random.Random(1234567)
    order = list(range(B))
    trial = 0
    while True:
        if time.time() > deadline:
            break
        trial += 1
        # random ordering
        rng.shuffle(order)
        cand = greedy(order)
        if len(cand) > best_len:
            best = cand
            best_len = len(cand)
            if best_len == 13:
                break
        # local search on a copy
        cand2 = local_search(list(cand))
        if len(cand2) > best_len:
            best = cand2
            best_len = len(cand2)
            if best_len == 13:
                break
        # occasional re-greedy from the improved set ordering
        if trial % 3 == 0 and time.time() > deadline - 8.0:
            break

    # Build the matrices from the best packing
    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_entries, v_entries, w_entries):
        U_cols.append([0] * U_rows)
        V_cols.append([0] * V_rows)
        W_cols.append([0] * W_rows)
        for idx, val in u_entries.items():
            U_cols[-1][idx] = val
        for idx, val in v_entries.items():
            V_cols[-1][idx] = val
        for idx, val in w_entries.items():
            W_cols[-1][idx] = val

    def add_strassen_block(i1, i2, j1, j2, k1, k2):
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2

        b11 = j1 * p + k1
        b12 = j1 * p + k2
        b21 = j2 * p + k1
        b22 = j2 * p + k2

        c11 = k1 * n + i1
        c12 = k2 * n + i1
        c21 = k1 * n + i2
        c22 = k2 * n + i2

        # M1 = (a11+a22)*(b11+b22)
        add_term({a11: 1, a22: 1}, {b11: 1, b22: 1}, {c11: 1, c22: 1})
        # M2 = (a21+a22)*b11
        add_term({a21: 1, a22: 1}, {b11: 1}, {c21: 1, c22: -1})
        # M3 = a11*(b12-b22)
        add_term({a11: 1}, {b12: 1, b22: -1}, {c12: 1, c22: 1})
        # M4 = a22*(b21-b11)
        add_term({a22: 1}, {b21: 1, b11: -1}, {c11: 1, c21: 1})
        # M5 = (a11+a12)*b22
        add_term({a11: 1, a12: 1}, {b22: 1}, {c11: -1, c12: 1})
        # M6 = (a21-a11)*(b11+b12)
        add_term({a21: 1, a11: -1}, {b11: 1, b12: 1}, {c22: 1})
        # M7 = (a12-a22)*(b21+b22)
        add_term({a12: 1, a22: -1}, {b21: 1, b22: 1}, {c11: 1})

    covered = set()
    for bidx in best:
        i1, i2, j1, j2, k1, k2 = blocks[bidx][0]
        add_strassen_block(i1, i2, j1, j2, k1, k2)
        for t in blocks[bidx][1]:
            covered.add(t)

    # Schoolbook terms for uncovered triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    add_term({i * m + j: 1}, {j * p + k: 1}, {k * n + i: 1})

    rank = len(U_cols)

    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    for c in range(rank):
        for r in range(U_rows):
            U[r][c] = U_cols[c][r]
        for r in range(V_rows):
            V[r][c] = V_cols[c][r]
        for r in range(W_rows):
            W[r][c] = W_cols[c][r]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
