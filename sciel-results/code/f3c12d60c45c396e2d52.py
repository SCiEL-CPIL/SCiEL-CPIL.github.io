import time
import random
import numpy as np

_START = time.time()
_LIMIT = 130.0  # safety margin below the 150s kill

def _lw_colors():
    """Lazebnik-Woldar C4-free 5-colouring of K_26."""
    n = 26
    col = np.zeros((n, n), dtype=np.int8)
    # field F_25 = GF(5^2)
    p = 5
    m = 2
    # irreducible polynomial x^2 - 2  (2 is non-square mod 5)
    def add(a, b):
        return (a + b) % 25
    def mul(a, b):
        # represent a = a0 + a1*alpha, b = b0 + b1*alpha, alpha^2 = 2
        a0, a1 = a % p, a // p
        b0, b1 = b % p, b // p
        c0 = (a0 * b0 + 2 * a1 * b1) % p
        c1 = (a0 * b1 + a1 * b0) % p
        return c0 + c1 * p
    def inv(a):
        for x in range(25):
            if mul(a, x) == 1:
                return x
        return 0
    for i in range(n):
        for j in range(i + 1, n):
            diff = add(i, 25 - j)  # i - j in F_25
            d = inv(diff)          # 1/(i-j)
            c = d % 5
            col[i, j] = c
            col[j, i] = c
    return n, col

def _build_nbrs(col):
    n = col.shape[0]
    nbrs = [set() for _ in range(n)]
    for i in range(n):
        ci = col[i]
        for j in range(n):
            if j != i and ci[j] != 0:
                nbrs[i].add(j)
    return nbrs

def _conflict_count(col, nbrs):
    n = col.shape[0]
    total = 0
    for i in range(n):
        for c in range(5):
            # count pairs of common-neighbors
            pass
    # faster: for each colour, for each vertex, count C(|Nc(v)|,2)
    # A C4 exists iff there are two vertices with >=2 common neighbours in a colour.
    # Equivalently sum over v of C(|Nc(v)|,2) counts each C4 twice.
    for i in range(n):
        for c in range(5):
            cnt = 0
            for j in range(n):
                if j != i and col[i, j] == c:
                    cnt += 1
            total += cnt * (cnt - 1) // 2
    return total // 2  # each C4 counted twice

def _try_extend(col26, seed):
    """Try to extend the 26-vertex colouring to n=27 (and 28)."""
    rng = random.Random(seed)
    n0 = 26
    for target in (27, 28):
        # start fresh from 26
        col = col26.copy()
        # add vertex 26
        col = np.zeros((target, target), dtype=np.int8)
        col[:26, :26] = col26
        # initialise edges from 26 to 0..25
        for j in range(26):
            c = rng.randrange(5)
            col[26, j] = c
            col[j, 26] = c
        if target == 28:
            for j in range(27):
                c = rng.randrange(5)
                col[27, j] = c
                col[j, 27] = c

        nbrs = _build_nbrs(col)
        # local search to eliminate C4s
        n = target
        for step in range(40000):
            if time.time() - _START > _LIMIT:
                break
            # find a random edge
            i = rng.randrange(n)
            j = rng.randrange(n)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            old = col[i, j]
            # try each new colour
            best_c = -1
            best_gain = 0
            # compute common-neighbour counts for i and j in each colour
            # For changing edge (i,j) from old to c:
            #   remove i from N_old(j), j from N_old(i)
            #   add i to N_c(j), j to N_c(i)
            # A conflict is created if i and j have >=2 common neighbours in colour c,
            #   OR if some other vertex k has >=2 common neighbours with i in colour c after adding j, etc.
            # Simpler: just recompute conflict delta by checking affected vertices.
            # Affected vertices: i, j, and any common neighbour in old or c.
            # We'll do a targeted check.
            for c in range(5):
                if c == old:
                    continue
                # tentatively: does adding edge (i,j) in colour c create a C4?
                # C4 involving i,j in colour c: need two common neighbours of i and j in colour c.
                common_c = 0
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if col[i, k] == c and col[j, k] == c:
                        common_c += 1
                        if common_c >= 2:
                            break
                if common_c >= 2:
                    continue
                # Also, does removing edge (i,j) from old and adding to c create C4s
                # that do NOT involve both i and j? A C4 is a 4-cycle; changing one edge
                # can only affect C4s that use that edge. So only C4s using (i,j) matter.
                # A C4 using edge (i,j) in colour c is exactly a pair of common neighbours.
                # So the above check is sufficient.
                # Gain = number of C4s removed in old colour that used edge (i,j).
                # A C4 in old colour using (i,j): pair of common neighbours of i,j in old colour.
                common_old = 0
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if col[i, k] == old and col[j, k] == old:
                        common_old += 1
                gain = common_old
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            if best_c >= 0:
                col[i, j] = best_c
                col[j, i] = best_c
                nbrs[i].discard(j)
                nbrs[j].discard(i)
                nbrs[i].add(j) if best_c != 0 else None
                nbrs[j].add(i) if best_c != 0 else None
                # rebuild is expensive; instead track via col directly in conflict check
            else:
                # no improving move at this edge; try a few more random edges
                continue
        # final conflict check
        if time.time() - _START > _LIMIT:
            break
        cc = _conflict_count(col, nbrs)
        if cc == 0:
            return target, col
    return 26, col26

def construct():
    t0 = time.time()
    n0, col26 = _lw_colors()
    # verify seed is valid
    if time.time() - t0 > _LIMIT:
        n = n0
        colors = [int(col26[i, j]) for i in range(n) for j in range(i + 1, n)]
        return {'n': n, 'colors': colors}
    best_n = n0
    best_col = col26
    # try several random seeds for extension
    for seed in range(40):
        if time.time() - t0 > _LIMIT:
            break
        tn, tcol = _try_extend(col26, seed)
        if tn > best_n:
            best_n = tn
            best_col = tcol
            if best_n >= 28:
                break
    n = best_n
    colors = [int(best_col[i, j]) for i in range(n) for j in range(i + 1, n)]
    return {'n': n, 'colors': colors}
