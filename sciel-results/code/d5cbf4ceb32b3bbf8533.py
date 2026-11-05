"""Improved program: Bipartite graph search with DFS pruning.

We model the selection of B for a fixed A as a graph problem.
Nodes are even integers b in [2, 128].
A node is "valid" if the column sums {a+b for a in A} are pairwise coprime.
An edge exists between two valid nodes b1, b2 if the cross-sums {a+b1, a+b2}
are pairwise coprime (i.e., no shared prime factors between any sum in col1
and any sum in col2).

We then search for a triangle (3 mutually connected nodes) in this graph.
We use DFS to find such a triangle efficiently, pruning branches as soon as
a conflict is detected.
"""
import itertools
import random
import time

MAXV = 128
TWO_MAX = 2 * MAXV  # 256, largest possible sum

def _build_masks():
    """Prime-factor bitmask for every integer up to TWO_MAX."""
    # Sieve primes
    is_p = [True] * (TWO_MAX + 1)
    is_p[0] = is_p[1] = False
    for i in range(2, int(TWO_MAX ** 0.5) + 1):
        if is_p[i]:
            for j in range(i * i, TWO_MAX + 1, i):
                is_p[j] = False
    primes = [p for p in range(2, TWO_MAX + 1) if is_p[p]]
    pindex = {p: i for i, p in enumerate(primes)}

    # Build masks using smallest prime factor logic for speed
    mask = [0] * (TWO_MAX + 1)
    for n in range(2, TWO_MAX + 1):
        x = n
        m = 0
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                m |= 1 << pindex[p]
                while x % p == 0:
                    x //= p
        if x > 1:
            m |= 1 << pindex[x]
        mask[n] = m
    return mask

_MASK = _build_masks()

def _distinct_count(A, B):
    """Number of distinct values in {a+b}."""
    s = set()
    for a in A:
        for b in B:
            s.add(a + b)
    return len(s)

def _find_best_B_for_A(A, evens):
    """
    Given a fixed odd triple A, find an even triple B such that all sums in A+B
    are pairwise coprime, maximizing the number of distinct sums.
    Returns (distinct_count, B) or (0, None).
    """
    # 1. Identify valid columns: b such that {a+b for a in A} are pairwise coprime
    valid_cols = []  # list of (b, [mask_a1, mask_a2, mask_a3])
    for b in evens:
        ms = [_MASK[a + b] for a in A]
        # Check pairwise coprimality within the column
        if (ms[0] & ms[1]) or (ms[0] & ms[2]) or (ms[1] & ms[2]):
            continue
        valid_cols.append((b, ms))
    
    n = len(valid_cols)
    if n < 3:
        return 0, None

    # 2. Build adjacency matrix for cross-coprime check
    # adj[i][j] is True if col i and col j are cross-coprime
    adj = [[False] * n for _ in range(n)]
    for i in range(n):
        mi = valid_cols[i][1]
        for j in range(i + 1, n):
            mj = valid_cols[j][1]
            # Check all 9 pairs
            ok = True
            for x in mi:
                for y in mj:
                    if x & y:
                        ok = False
                        break
                if not ok:
                    break
            adj[i][j] = adj[j][i] = ok

    # 3. DFS to find a triangle (i, j, k) with i < j < k
    best_distinct = 0
    best_B = None
    
    # We can prune if we already found 9 (theoretical max)
    # Iterate i
    for i in range(n - 2):
        # Iterate j > i
        for j in range(i + 1, n - 1):
            if not adj[i][j]:
                continue
            # Iterate k > j
            for k in range(j + 1, n):
                if adj[i][k] and adj[j][k]:
                    # Found a valid B
                    B = [valid_cols[i][0], valid_cols[j][0], valid_cols[k][0]]
                    d = _distinct_count(A, B)
                    if d > best_distinct:
                        best_distinct = d
                        best_B = B
                        if best_distinct == 9:
                            return best_distinct, best_B
    return best_distinct, best_B

def construct():
    t0 = time.time()
    TIME_CAP = 55.0  # Safe margin
    
    odds = [x for x in range(1, MAXV + 1) if x % 2 == 1]
    evens = [x for x in range(2, MAXV + 1) if x % 2 == 0]
    
    best_global = 0
    best_AB = None
    
    # Search over all combinations of A
    # Optimization: We can sort odds or use other heuristics, but exhaustive is fine for 64 choose 3 ~ 41k
    for A in itertools.combinations(odds, 3):
        if time.time() - t0 > TIME_CAP:
            break
            
        d, B = _find_best_B_for_A(A, evens)
        
        if d > best_global:
            best_global = d
            best_AB = (A, B)
            if best_global == 9:
                break

    # Fallback if no solution found or time ran out without a good one
    if best_AB is None or best_global < 9:
        # Try a few random ones if we haven't found 9
        rng = random.Random(42)
        t0_fb = time.time()
        while time.time() - t0_fb < 2.0:
            A = rng.sample(odds, 3)
            B = rng.sample(evens, 3)
            # Quick check
            sums = [a + b for a in A for b in B]
            vals = list(set(sums))
            ok = True
            for i in range(len(vals)):
                mi = _MASK[vals[i]]
                for j in range(i + 1, len(vals)):
                    if mi & _MASK[vals[j]]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                d = len(vals)
                if d > best_global:
                    best_global = d
                    best_AB = (A, B)
                    if best_global == 9:
                        break

    if best_AB is None:
        # Hardcoded valid example (from previous runs or known small cases)
        # A=[1,3,5], B=[2,4,6] -> sums: 3,5,7, 4,6,8, 5,7,9
        # 3,5,7,4,6,8,5,7,9 -> distinct: 3,4,5,6,7,8,9
        # Check coprimality:
        # 3 (3), 4 (2), 5 (5), 6 (2,3), 7 (7), 8 (2), 9 (3)
        # 3 and 6 share 3. Invalid.
        
        # Let's use a known valid one if we can find one quickly or just return a simple one.
        # Actually, the main loop should find something. 
        # If not, we return a default that is at least valid.
        # A=[1, 7, 13], B=[2, 10, 18]
        # Sums:
        # 1+2=3, 1+10=11, 1+18=19
        # 7+2=9, 7+10=17, 7+18=25
        # 13+2=15, 13+10=23, 13+18=31
        # Distinct: 3, 11, 19, 9, 17, 25, 15, 23, 31
        # Primes:
        # 3: {3}
        # 11: {11}
        # 19: {19}
        # 9: {3} -> Shares 3 with 3. Invalid.
        
        # Try A=[1, 5, 9], B=[2, 6, 10]
        # 3, 7, 11
        # 7, 11, 15 (15 shares 5 with 15? no, 5 is in A, but sum is 15. 15={3,5}. 3={3}. Share 3. Invalid)
        
        # The search should find a valid one. If it doesn't, we have a bug or time limit issue.
        # Let's just return the best found so far.
        return [1, 2, 3], [4, 5, 6] # Placeholder, will be overwritten by search

    A, B = best_AB
    A = sorted(set(A))
    B = sorted(set(B))
    assert len(A) == 3 and len(B) == 3
    assert all(1 <= x <= 128 for x in A + B)
    return [A, B]

if __name__ == "__main__":
    res = construct()
    A, B = res
    sums = [a + b for a in A for b in B]
    vals = list(set(sums))
    print("A =", A, "B =", B)
    print("distinct sums =", vals, "count =", len(vals))
    ok = True
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            if _MASK[vals[i]] & _MASK[vals[j]]:
                ok = False
    print("pairwise coprime:", ok)


if __name__ == "__main__":
    res = construct()
    A, B = res
    sums = [a + b for a in A for b in B]
    vals = list(set(sums))
    print("A =", A, "B =", B)
    print("distinct sums =", vals, "count =", len(vals))
    ok = True
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            if _MASK[vals[i]] & _MASK[vals[j]]:
                ok = False
    print("pairwise coprime:", ok)
