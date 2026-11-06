"""Branch-and-bound maximum independent set solver for equal-gcd triple avoidance."""
import time
from math import gcd

def construct():
    t0 = time.time()
    DEADLINE = 85.0
    N = 128
    N1 = N + 1

    # Precompute gcd matrix
    gcd_mat = [[0] * N1 for _ in range(N1)]
    for i in range(1, N1):
        for j in range(i + 1, N1):
            g = gcd(i, j)
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g

    # Precompute all bad triples
    bad_triples = []
    for a in range(1, N1):
        for b in range(a + 1, N1):
            g_ab = gcd_mat[a][b]
            for c in range(b + 1, N1):
                if gcd_mat[a][c] == g_ab and gcd_mat[b][c] == g_ab:
                    bad_triples.append((a, b, c))

    # Build conflict graph: two numbers are adjacent if they form a bad triple with some third number
    # Actually, we need: for the independent set formulation, we want to find the largest set S
    # such that no three elements in S form a bad triple.
    # This is a maximum independent set in a 3-uniform hypergraph.
    # For branch-and-bound, we can use the following approach:
    # - Order elements by some heuristic
    # - At each step, decide to include or exclude the next element
    # - Upper bound: use the fact that if we have a conflict graph where edges connect pairs
    #   that cannot both be in S (because they'd form a bad triple with some third element in S),
    #   then the remaining candidates form a graph, and we can use greedy coloring or clique bounds.

    # Simpler approach: use a recursive branch-and-bound with memoization on the set of
    # remaining candidates, using a tight upper bound.

    # For each element x, precompute the set of elements that conflict with x
    # (i.e., pairs (a,b) such that {a,b,x} is a bad triple)
    # Actually, let's precompute: for each pair (a,b), the set of x such that {a,b,x} is bad.
    # And for each x, the set of pairs (a,b) that would be completed by x.

    # Let's use a different approach: 
    # The key insight is that we can model this as finding a maximum set with no bad triple.
    # We'll use a recursive algorithm that processes elements in a fixed order.

    # Precompute: for each element x, and for each pair (a,b) with a<b, 
    # whether {a,b,x} is a bad triple.
    # bad_with_pair[x][a][b] = True if {a,b,x} is bad
    # This is O(N^3) space which is 128^3 = 2M, too much.

    # Alternative: for each x, precompute the set of (a,b) pairs that x completes.
    # completed_pairs[x] = set of (a,b) with a<b such that {a,b,x} is bad
    completed_pairs = [set() for _ in range(N1)]
    for a, b, c in bad_triples:
        completed_pairs[a].add((b, c))
        completed_pairs[b].add((a, c))
        completed_pairs[c].add((a, b))

    # For branch-and-bound, we'll use the following:
    # - Process elements in a fixed order (e.g., by number of bad triples they're in, ascending)
    # - Maintain the current set S as a frozenset
    # - Upper bound: |S| + (number of remaining elements that can potentially be added)
    #   But this is loose. Better: use the fact that the remaining elements form a hypergraph,
    #   and we can compute a greedy upper bound.

    # Let's use a simpler but effective approach:
    # 1. Get a good initial solution via greedy
    # 2. Use branch-and-bound with the bound: |S| + |R| where R is the set of remaining
    #    elements that don't immediately conflict with S
    # 3. Order elements by "safety" (fewer conflicts first)

    # Precompute conflict degree for each element
    triple_count = [0] * N1
    for a, b, c in bad_triples:
        triple_count[a] += 1
        triple_count[b] += 1
        triple_count[c] += 1

    # Order elements by triple count (ascending), then by value
    order = sorted(range(1, N1), key=lambda x: (triple_count[x], x))

    # Precompute for each x, the set of pairs in the "prefix" that x would complete
    # This is complex. Let's use a simpler bound.

    # Greedy initial solution
    def greedy():
        S = set()
        for x in order:
            # Check if x can be added to S
            can_add = True
            for a in S:
                for b in S:
                    if a < b and (a, b) in completed_pairs[x]:
                        can_add = False
                        break
                if not can_add:
                    break
            if can_add:
                S.add(x)
        return S

    best_set = greedy()
    best_size = len(best_set)

    # Branch and bound
    # We'll use a recursive function with the following state:
    # - idx: index in the order array
    # - S: current set (as a frozenset for hashing, but we'll use a list for efficiency)
    # - S_set: set version of S for fast lookup

    # For the upper bound, we'll use:
    # |S| + (number of remaining elements from idx to end that don't conflict with S)
    # This is an overestimate, but it's fast to compute.

    # Actually, let's use a better bound: for the remaining elements, compute the maximum
    # independent set in the induced subhypergraph. This is NP-hard, but for small sets
    # (say, < 20 elements), we can compute it exactly.

    # For now, let's use the simple bound and hope the ordering helps.

    import sys
    sys.setrecursionlimit(10000)

    def can_add_to(S_set, x):
        """Check if x can be added to S_set without creating a bad triple."""
        for a in S_set:
            for b in S_set:
                if a < b and (a, b) in completed_pairs[x]:
                    return False
        return True

    def branch_and_bound(idx, S_list, S_set):
        nonlocal best_size, best_set
        if time.time() > DEADLINE - 1:
            return

        # Upper bound: current size + remaining elements
        remaining = len(order) - idx
        if len(S_list) + remaining <= best_size:
            return

        if idx == len(order):
            if len(S_list) > best_size:
                best_size = len(S_list)
                best_set = set(S_list)
            return

        x = order[idx]

        # Option 1: exclude x
        branch_and_bound(idx + 1, S_list, S_set)

        # Option 2: include x, if possible
        if can_add_to(S_set, x):
            S_list.append(x)
            S_set.add(x)
            branch_and_bound(idx + 1, S_list, S_set)
            S_list.pop()
            S_set.remove(x)

    # This is too slow for n=128. Let's use a different approach.
    # Use a local search with 2-swap moves, starting from the greedy solution.

    def local_search(S, max_iters=1000):
        S = sorted(S)
        S_set = set(S)
        remaining = [x for x in range(1, N1) if x not in S_set]
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < DEADLINE - 3:
            improved = False
            iters += 1

            # Try to add an element
            for x in remaining[:]:
                if can_add_to(S_set, x):
                    S.append(x)
                    S.sort()
                    S_set.add(x)
                    remaining.remove(x)
                    improved = True
                    break

            # Try 2-swap: remove one from S, add one from remaining
            if not improved and remaining:
                for i in range(len(S)):
                    a = S[i]
                    for j in range(len(remaining)):
                        b = remaining[j]
                        # Try swap: remove a, add b
                        new_S_set = S_set.copy()
                        new_S_set.remove(a)
                        new_S_set.add(b)
                        # Check validity
                        valid = True
                        for x in new_S_set:
                            for y in new_S_set:
                                if x < y and (x, y) in completed_pairs[b] or (x, y) in completed_pairs[a]:
                                    # This check is wrong. Let's redo.
                                    pass
                        # Simpler: just check if the new set is valid
                        valid = True
                        for x in new_S_set:
                            for y in new_S_set:
                                if x < y:
                                    for z in new_S_set:
                                        if z > y and (x, y) in completed_pairs[z]:
                                            valid = False
                                            break
                                if not valid:
                                    break
                            if not valid:
                                break
                        if valid:
                            S[i] = b
                            S.sort()
                            S_set = new_S_set
                            remaining.remove(b)
                            remaining.append(a)
                            improved = True
                            break
                    if improved:
                        break

        return S

    # Run local search
    best_set = local_search(list(best_set))
    best_size = len(best_set)

    # Try multiple random restarts
    import random
    for seed in range(20):
        if time.time() > DEADLINE - 5:
            break
        random.seed(seed)
        S_rand = set()
        remaining = list(range(1, N1))
        random.shuffle(remaining)
        for x in remaining:
            if can_add_to(S_rand, x):
                S_rand.add(x)
        S_rand = local_search(list(S_rand), max_iters=200)
        if len(S_rand) > best_size:
            best_size = len(S_rand)
            best_set = S_rand

    if not best_set:
        best_set = {1}

    return sorted(best_set)


# Wrap to ensure nonempty
_orig_construct = construct
def construct():
    result = _orig_construct()
    if not result:
        return [1]
    return result
