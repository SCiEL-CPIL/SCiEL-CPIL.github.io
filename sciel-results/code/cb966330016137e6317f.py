"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random
from itertools import combinations
from collections import defaultdict, Counter

def construct():
    deadline = time.time() + 80.0
    n = 128

    def lcm(a, b):
        return a * b // math.gcd(a, b)

    # Precompute lcm for all pairs
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_table[i][j] = lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]

    # Precompute bad_triple: for each pair (a,b), the set of x such that (a,b,x) is a bad triple
    # A triple (a,b,x) is bad if lcm(a,b) == lcm(a,x) == lcm(b,x)
    bad_triple = {}
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            l_ab = lcm_table[a][b]
            bads = set()
            for x in range(1, n + 1):
                if x == a or x == b:
                    continue
                if lcm_table[a][x] == l_ab and lcm_table[b][x] == l_ab:
                    bads.add(x)
            bad_triple[(a, b)] = bads
            bad_triple[(b, a)] = bads

    # Also precompute all bad triples as a list of frozensets for the removal strategy
    bad_triple_list = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            for x in bad_triple[(a, b)]:
                if x > b:
                    bad_triple_list.append(frozenset({a, b, x}))
                elif x < a:
                    bad_triple_list.append(frozenset({x, a, b}))
                # x between a and b already covered by other pairs

    # Deduplicate
    bad_triple_set = set(bad_triple_list)

    def can_add(S, x):
        """Check if x can be added to set S without creating a bad triple."""
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                a = S_list[i]
                b = S_list[j]
                if x in bad_triple[(a, b)]:
                    return False
        return True

    def is_valid(S):
        """Check if set S has no bad triples."""
        S_list = sorted(S)
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                for k in range(j + 1, len(S_list)):
                    a, b, c = S_list[i], S_list[j], S_list[k]
                    if lcm_table[a][b] == lcm_table[a][c] == lcm_table[b][c]:
                        return False
        return True

    def greedy_add(order):
        """Greedy: add elements in given order if they don't conflict."""
        S = set()
        for x in order:
            if x not in S and can_add(S, x):
                S.add(x)
        return S

    def greedy_remove():
        """Greedy vertex cover: remove elements to hit all bad triples, then add back."""
        # Count how many bad triples each element is in
        elem_count = Counter()
        for t in bad_triple_set:
            for x in t:
                elem_count[x] += 1

        # Greedy: repeatedly pick the element in the most uncovered triples, remove it
        removed = set()
        uncovered = set(bad_triple_set)
        elem_triples = defaultdict(set)
        for t in bad_triple_set:
            for x in t:
                elem_triples[x].add(t)

        iteration = 0
        while uncovered and iteration < 200:
            if time.time() > deadline - 5:
                break
            iteration += 1
            best_x = None
            best_size = 0
            for x in elem_triples:
                if x in removed:
                    continue
                size = len(elem_triples[x] & uncovered)
                if size > best_size:
                    best_size = size
                    best_x = x
            if best_x is None or best_size == 0:
                break
            removed.add(best_x)
            uncovered -= elem_triples[best_x]

        result = set(range(1, n + 1)) - removed

        # Try to add back removed elements if possible
        for x in sorted(removed, key=lambda v: -elem_count[v]):
            if time.time() > deadline - 3:
                break
            if can_add(result, x):
                result.add(x)
                removed.discard(x)

        return result

    def local_search(S, max_iters=2000):
        """Local search: try adding, 1-swap, and 2-swap moves."""
        S = set(S)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            if time.time() > deadline - 2:
                break
            improved = False
            iters += 1

            # Try adding each missing element
            for x in range(1, n + 1):
                if x not in S and can_add(S, x):
                    S.add(x)
                    improved = True
                    break

            if improved:
                continue

            # Try 1-swap: remove one element, add a different one
            S_list = list(S)
            for i in range(len(S_list)):
                if time.time() > deadline - 2:
                    break
                a = S_list[i]
                S2 = S - {a}
                for x in range(1, n + 1):
                    if x not in S2 and can_add(S2, x):
                        S = S2 | {x}
                        improved = True
                        break
                if improved:
                    break

            if improved:
                continue

            # Try 2-swap: remove two elements, add two different ones
            S_list = list(S)
            for i in range(len(S_list)):
                if time.time() > deadline - 2:
                    break
                for j in range(i + 1, len(S_list)):
                    if time.time() > deadline - 2:
                        break
                    a, b = S_list[i], S_list[j]
                    S2 = S - {a, b}
                    added = 0
                    new_S = set(S2)
                    for x in range(1, n + 1):
                        if x not in new_S and can_add(new_S, x):
                            new_S.add(x)
                            added += 1
                            if added == 2:
                                break
                    if len(new_S) > len(S):
                        S = new_S
                        improved = True
                        break
                if improved:
                    break

        return S

    # Try multiple strategies
    best_S = set()

    # Strategy 1: Greedy add ascending
    S1 = greedy_add(range(1, n + 1))
    S1 = local_search(S1)
    if len(S1) > len(best_S):
        best_S = S1

    # Strategy 2: Greedy add descending
    S2 = greedy_add(range(n, 0, -1))
    S2 = local_search(S2)
    if len(S2) > len(best_S):
        best_S = S2

    # Strategy 3: Greedy remove then add back
    S3 = greedy_remove()
    S3 = local_search(S3)
    if len(S3) > len(best_S):
        best_S = S3

    # Strategy 4: Random restarts with greedy add
    random.seed(42)
    for restart in range(30):
        if time.time() > deadline - 5:
            break
        order = list(range(1, n + 1))
        random.shuffle(order)
        S = greedy_add(order)
        S = local_search(S)
        if len(S) > len(best_S):
            best_S = S

    # Strategy 5: Random restarts with greedy remove
    for restart in range(10):
        if time.time() > deadline - 3:
            break
        # Perturb the greedy remove by randomizing tie-breaking
        S = greedy_remove()
        # Randomly swap some elements
        S_list = list(S)
        for _ in range(5):
            if len(S_list) < 2:
                break
            i, j = random.sample(range(len(S_list)), 2)
            a, b = S_list[i], S_list[j]
            S2 = S - {a}
            if can_add(S2, b):
                S = S2 | {b}
        S = local_search(S)
        if len(S) > len(best_S):
            best_S = S

    # Final verification
    if not is_valid(best_S):
        best_S = {1}

    if not best_S:
        best_S = {1}

    return sorted(best_S)
# EVOLVE-BLOCK-END
