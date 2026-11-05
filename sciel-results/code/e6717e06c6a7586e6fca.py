"""Fresh design: hypergraph independent-set heuristic for no-equal-gcd triples."""

from math import gcd
import time
import random
from itertools import combinations

# Precompute the gcd table and the conflict matrix once.
# conflict[i][j] = 1 if i and j are in some bad triple (i.e. they cannot coexist
# in any valid set without checking the third element). Actually we need the
# triple condition, so we precompute which pairs "conflict" in the sense that
# there exists a third number making a bad triple. But the cleanest is to just
# test triples on the fly using a precomputed gcd table.

N = 128

# Build gcd table
_gcd = [[0]*(N+1) for _ in range(N+1)]
for i in range(1, N+1):
    for j in range(i, N+1):
        g = gcd(i, j)
        _gcd[i][j] = g
        _gcd[j][i] = g

# Precompute, for each pair (a,b) with a<b, the set of c such that (a,b,c) is a
# bad triple. This lets us do O(1)-ish conflict checks.
# bad_pair_conflict[a][b] = bitmask/set of c values that would form a bad triple.
# Since N=128, a pair (a,b) can conflict with at most N-2 values. We store a
# frozenset for each ordered pair index.

# For speed, represent "selected" as a boolean list and check a new element x by
# verifying that for every pair (a,b) in selected, adding x is not a bad triple.
# That is O(k^2) per insertion where k is current size. With ~90 elements that's
# ~8000 checks per insertion, ~720k total per construction. Fine for many restarts.

# Precompute bad-triple membership: bad[a][b][c] is True if (a,b,c) is a bad triple.
# Use a dict keyed by (min, mid, max) to save memory, or a 3D structure.
# N=128 -> 128^3/6 ~ 350k triples. Store as a set of frozensets? Too slow to hash.
# Instead store as a boolean 3D list indexed by (a,b,c) with a<b<c.

# We'll store bad as a list of dicts: bad[a][b] = set of c > b such that (a,b,c) bad, for a<b.
bad = [dict() for _ in range(N+1)]
for a in range(1, N+1):
    for b in range(a+1, N+1):
        g_ab = _gcd[a][b]
        bad_c = set()
        for c in range(b+1, N+1):
            g_ac = _gcd[a][c]
            g_bc = _gcd[b][c]
            if g_ab == g_ac == g_bc:
                bad_c.add(c)
        if bad_c:
            bad[a][b] = bad_c

# Now a triple (a,b,c) with a<b<c is bad iff c in bad[a][b].

def _is_bad_triple(a, b, c):
    """Return True if a,b,c (distinct) form a bad triple."""
    # normalize
    if a == b or b == c or a == c:
        return False
    if a > b:
        a, b = b, a
    if b > c:
        b, c = c, b
    if a > b:
        a, b = b, a
    # now a <= b <= c, and distinct so a < b < c
    return c in bad.get(a, {}).get(b, set())


def _conflict_count(x, selected_set):
    """Count how many pairs in selected_set would form a bad triple with x."""
    cnt = 0
    sel = list(selected_set)
    n = len(sel)
    for i in range(n):
        a = sel[i]
        for j in range(i+1, n):
            b = sel[j]
            if _is_bad_triple(a, b, x):
                cnt += 1
    return cnt


def _greedy_construct(rng, degree_order):
    """Build a valid set by greedily adding elements in a given order,
    skipping any that would create a bad triple."""
    selected = set()
    for x in degree_order:
        # check all pairs
        ok = True
        sel = list(selected)
        n = len(sel)
        for i in range(n):
            a = sel[i]
            for j in range(i+1, n):
                b = sel[j]
                if _is_bad_triple(a, b, x):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            selected.add(x)
    return selected


def _local_search_swap(selected_set, rng, max_iter=200):
    """Try to improve by swapping one element out for one in."""
    selected = set(selected_set)
    current_size = len(selected)
    improved = True
    iters = 0
    while improved and iters < max_iter:
        improved = False
        iters += 1
        # try adding each candidate not in selected
        for x in range(1, N+1):
            if x in selected:
                continue
            # check if x can be added
            ok = True
            sel = list(selected)
            n = len(sel)
            for i in range(n):
                a = sel[i]
                for j in range(i+1, n):
                    b = sel[j]
                    if _is_bad_triple(a, b, x):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                # but we want a swap that increases size, so if x can be added
                # directly we just add it (no swap needed)
                selected.add(x)
                current_size += 1
                improved = True
                break
        if improved:
            continue
        # try swaps: remove one, add one
        for x in range(1, N+1):
            if x in selected:
                continue
            # find an element y in selected such that removing y allows adding x
            # and the resulting set is valid and larger (size stays same, so we
            # need net gain: actually a swap keeps size same; to grow we need
            # removal that enables a previously-blocked addition).
            # Simpler: try removing each y and see if x becomes addable.
            for y in list(selected):
                new_set = selected - {y}
                # check if x can be added to new_set
                ok = True
                sel = list(new_set)
                n = len(sel)
                for i in range(n):
                    a = sel[i]
                    for j in range(i+1, n):
                        b = sel[j]
                        if _is_bad_triple(a, b, x):
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    new_set.add(x)
                    # size is same as before (removed y, added x)
                    # only accept if it leads to further improvement potential;
                    # for now accept any swap that keeps validity (neutral) to
                    # escape local optima, but prefer ones that reduce future
                    # conflicts. We'll accept to explore.
                    selected = new_set
                    improved = True
                    break
            if improved:
                break
    return selected


def construct():
    start = time.time()
    deadline = start + 80.0  # safety margin under 90s

    # Precompute a base degree: number of bad triples each element participates in.
    # degree[x] = number of pairs (a,b) such that (a,b,x) is bad.
    degree = [0]*(N+1)
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            g_ab = _gcd[a][b]
            for c in range(b+1, N+1):
                if _gcd[a][c] == g_ab and _gcd[b][c] == g_ab:
                    degree[a] += 1
                    degree[b] += 1
                    degree[c] += 1

    best_set = set()
    best_size = 0

    rng = random.Random(12345)

    # Multiple restarts with different orderings.
    restart = 0
    while True:
        if time.time() > deadline - 5.0:
            break
        restart += 1

        # Build an ordering: sort by degree ascending, then randomize ties.
        elems = list(range(1, N+1))
        # Add a random perturbation to degree to diversify.
        perturbed = [(degree[x] + rng.random()*0.5, rng.random(), x) for x in elems]
        perturbed.sort()
        order = [x for _, _, x in perturbed]

        s = _greedy_construct(rng, order)
        if len(s) > best_size:
            best_set = set(s)
            best_size = len(s)

        # Local search on the best candidate so far (or on this one if it's best).
        if time.time() > deadline - 10.0:
            break
        s2 = _local_search_swap(s, rng, max_iter=100)
        if len(s2) > best_size:
            best_set = set(s2)
            best_size = len(s2)

        # Also periodically run local search on the current best to keep improving.
        if restart % 3 == 0 and time.time() > deadline - 15.0:
            break
        if restart % 3 == 0:
            s3 = _local_search_swap(best_set, rng, max_iter=100)
            if len(s3) > best_size:
                best_set = set(s3)
                best_size = len(s3)

    # Final local search pass on the best set if time permits.
    if time.time() < deadline - 3.0:
        s4 = _local_search_swap(best_set, rng, max_iter=300)
        if len(s4) > best_size:
            best_set = set(s4)
            best_size = len(s4)

    # Ensure nonempty
    if not best_set:
        best_set = {1}

    return sorted(best_set)


if __name__ == "__main__":
    print(construct())
