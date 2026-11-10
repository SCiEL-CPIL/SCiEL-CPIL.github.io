"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
import random

def construct():
    deadline = time.time() + 85  # 5s safety margin under 90s kill
    N = 128

    # Precompute GCD matrix
    gcd_mat = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i, N+1):
            g = gcd(i, j)
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g

    # Precompute bad_pairs: for each vertex v (0-based), list of pairs (a,b) 
    # such that (v+1, a+1, b+1) form a bad triple (all pairwise gcds equal)
    bad_pairs = [[] for _ in range(N)]
    for a in range(N):
        for b in range(a+1, N):
            ga = gcd(a+1, b+1)
            for c in range(N):
                if c == a or c == b:
                    continue
                if gcd(a+1, c+1) == ga and gcd(b+1, c+1) == ga:
                    bad_pairs[c].append((a, b))
                    bad_pairs[a].append((b, c))
                    bad_pairs[b].append((a, c))

    # Precompute lookup: bad_pair_lookup[v][u] = list of w such that (u,w) in bad_pairs[v]
    bad_pair_lookup = [{} for _ in range(N)]
    for v in range(N):
        for (a, b) in bad_pairs[v]:
            if a not in bad_pair_lookup[v]:
                bad_pair_lookup[v][a] = []
            bad_pair_lookup[v][a].append(b)
            if b not in bad_pair_lookup[v]:
                bad_pair_lookup[v][b] = []
            bad_pair_lookup[v][b].append(a)

    def compute_blocked(selected_set):
        """Compute blocked counts for all vertices given a selected set."""
        blocked = [0] * N
        for v in selected_set:
            for u in range(N):
                if u == v or u in selected_set:
                    continue
                if v in bad_pair_lookup[u]:
                    for w in bad_pair_lookup[u][v]:
                        if w in selected_set:
                            blocked[u] += 1
        return blocked

    def greedy_construction(order):
        """Greedy: add vertices in given order if valid."""
        selected = []
        selected_set = set()
        blocked = [0] * N
        for v in order:
            if v in selected_set:
                continue
            if blocked[v] == 0:
                selected.append(v)
                selected_set.add(v)
                for u in range(N):
                    if u == v or u in selected_set:
                        continue
                    if v in bad_pair_lookup[u]:
                        for w in bad_pair_lookup[u][v]:
                            if w in selected_set:
                                blocked[u] += 1
        return selected

    def local_improve_add(subset):
        """Try to add vertices that can fit."""
        s = set(subset)
        blocked = compute_blocked(s)
        improved = True
        while improved:
            improved = False
            for v in range(N):
                if v in s:
                    continue
                if blocked[v] == 0:
                    s.add(v)
                    improved = True
                    for u in range(N):
                        if u == v or u in s:
                            continue
                        if v in bad_pair_lookup[u]:
                            for w in bad_pair_lookup[u][v]:
                                if w in s:
                                    blocked[u] += 1
        return list(s)

    def swap_improve(subset, rng, max_swaps=30):
        """Try 1-for-1 swaps to improve."""
        s = set(subset)
        if not s:
            return list(s)
        blocked = compute_blocked(s)
        best_size = len(s)
        best_set = None
        elements = list(s)
        rng.shuffle(elements)
        
        for rem in elements[:max_swaps]:
            if time.time() > deadline:
                break
            s2 = s - {rem}
            blocked2 = compute_blocked(s2)
            # Greedily add
            for v in range(N):
                if v in s2:
                    continue
                if blocked2[v] == 0:
                    s2.add(v)
                    for u in range(N):
                        if u == v or u in s2:
                            continue
                        if v in bad_pair_lookup[u]:
                            for w in bad_pair_lookup[u][v]:
                                if w in s2:
                                    blocked2[u] += 1
            if len(s2) > best_size:
                best_size = len(s2)
                best_set = s2.copy()
        
        if best_set is not None:
            return list(best_set)
        return list(s)

    def multi_swap_improve(subset, rng, num_rounds=5):
        """Multiple rounds of swap improvement."""
        current = list(subset)
        for _ in range(num_rounds):
            if time.time() > deadline:
                break
            improved = swap_improve(current, rng, max_swaps=20)
            if len(improved) <= len(current):
                break
            current = improved
        return current

    def perturbation(subset, rng, num_remove=2):
        """Remove some random elements to escape local minima."""
        s = list(subset)
        if len(s) <= num_remove:
            return s
        to_remove = rng.sample(s, num_remove)
        return [x for x in s if x not in set(to_remove)]

    best = []
    random.seed(42)
    rng = random.Random(12345)

    # Strategy 1: Multiple random greedy passes
    for trial in range(200):
        if time.time() > deadline:
            break
        order = list(range(N))
        rng.shuffle(order)
        candidate = greedy_construction(order)
        candidate = local_improve_add(candidate)
        if len(candidate) > len(best):
            best = candidate

    # Strategy 2: Heuristic orderings
    for order in [list(range(N)), list(range(N-1, -1, -1)),
                  sorted(range(N), key=lambda x: len(bad_pairs[x])),
                  sorted(range(N), key=lambda x: -len(bad_pairs[x])),
                  sorted(range(N), key=lambda x: x % 7),
                  sorted(range(N), key=lambda x: x % 5)]:
        if time.time() > deadline:
            break
        candidate = greedy_construction(order)
        candidate = local_improve_add(candidate)
        if len(candidate) > len(best):
            best = candidate

    # Strategy 3: Intensive local search on best
    for trial in range(50):
        if time.time() > deadline:
            break
        if not best:
            break
        improved = multi_swap_improve(best, rng, num_rounds=5)
        if len(improved) > len(best):
            best = improved
        else:
            break

    # Strategy 4: Perturbation + re-optimization
    for trial in range(80):
        if time.time() > deadline:
            break
        if not best:
            break
        perturbed = perturbation(best, rng, num_remove=rng.choice([1, 2, 3]))
        candidate = local_improve_add(perturbed)
        candidate = multi_swap_improve(candidate, rng, num_rounds=3)
        if len(candidate) > len(best):
            best = candidate

    # Strategy 5: Fresh random starts with full local search
    for trial in range(100):
        if time.time() > deadline:
            break
        order = list(range(N))
        rng.shuffle(order)
        candidate = greedy_construction(order)
        candidate = local_improve_add(candidate)
        candidate = multi_swap_improve(candidate, rng, num_rounds=4)
        if len(candidate) > len(best):
            best = candidate

    # Final polish
    if best:
        final = local_improve_add(best)
        if len(final) > len(best):
            best = final

    if not best:
        best = local_improve_add(greedy_construction(list(range(N))))

    return [v + 1 for v in sorted(best)]
# EVOLVE-BLOCK-END
