"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations
import random

def construct():
    deadline = time.time() + 75  # 15s safety margin under 90s kill
    N = 128
    # Precompute GCD matrix
    gcd_mat = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i, N+1):
            g = gcd(i, j)
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g

    # Precompute bad_pairs: for each v, list of (a,b) such that (v,a,b) is a bad triple
    # v, a, b are 0-based indices
    bad_pairs = [[] for _ in range(N)]
    for a in range(N):
        for b in range(a+1, N):
            ga = gcd(a+1, b+1)
            for c in range(N):
                if c == a or c == b:
                    continue
                if gcd(a+1, c+1) == ga and gcd(b+1, c+1) == ga:
                    # (a,b,c) is a bad triple
                    bad_pairs[c].append((a,b))
                    bad_pairs[a].append((b,c))
                    bad_pairs[b].append((a,c))

    # Precompute bitmask for each vertex for quick "is in set" check
    # vertex v corresponds to bit v
    
    def is_valid_addition(v, S_mask):
        """Check if vertex v can be added to set S_mask without creating a bad triple."""
        for (a, b) in bad_pairs[v]:
            if (S_mask & (1 << a)) and (S_mask & (1 << b)):
                return False
        return True

    def greedy_construction(order):
        """Greedy: add vertices in given order if valid."""
        S_mask = 0
        count = 0
        for v in order:
            if is_valid_addition(v, S_mask):
                S_mask |= (1 << v)
                count += 1
        return S_mask, count

    def local_improve(S_mask, count, rng):
        """Local search: try to add vertices, then try 1-for-2 swaps."""
        improved = True
        while improved:
            improved = False
            # Phase 1: Add any valid vertex
            for v in range(N):
                if not (S_mask & (1 << v)):
                    if is_valid_addition(v, S_mask):
                        S_mask |= (1 << v)
                        count += 1
                        improved = True
            # Phase 2: 1-for-2 Swap
            # Try removing each vertex and seeing if we can add more than 1
            for v in range(N):
                if S_mask & (1 << v):
                    S_old = S_mask
                    S_mask ^= (1 << v)
                    count_temp = count - 1
                    # Try to add as many as possible
                    for u in range(N):
                        if not (S_mask & (1 << u)):
                            if is_valid_addition(u, S_mask):
                                S_mask |= (1 << u)
                                count_temp += 1
                    if count_temp > count:
                        count = count_temp
                        improved = True
                    else:
                        S_mask = S_old
        return S_mask, count

    best_mask = 0
    best_count = 0
    random.seed(42)
    rng = random.Random(12345)

    # Multi-start greedy
    for trial in range(200):
        if time.time() > deadline:
            break
        order = list(range(N))
        random.shuffle(order)
        mask, count = greedy_construction(order)
        mask, count = local_improve(mask, count, rng)
        if count > best_count:
            best_count = count
            best_mask = mask

    # Also try sorted orderings
    for order in [list(range(N)), list(range(N-1, -1, -1)),
                  sorted(range(N), key=lambda x: len(bad_pairs[x])),
                  sorted(range(N), key=lambda x: -len(bad_pairs[x]))]:
        if time.time() > deadline:
            break
        mask, count = greedy_construction(order)
        mask, count = local_improve(mask, count, rng)
        if count > best_count:
            best_count = count
            best_mask = mask

    # If we have time, do more aggressive perturbation on the best solution
    if best_count > 0 and time.time() < deadline - 10:
        for trial in range(50):
            if time.time() > deadline:
                break
            # Perturb: remove 5 random vertices from best_mask
            S_mask = best_mask
            count = best_count
            # Remove 5 random
            vertices_in_set = [i for i in range(N) if S_mask & (1 << i)]
            if len(vertices_in_set) < 5:
                continue
            to_remove = rng.sample(vertices_in_set, 5)
            for v in to_remove:
                S_mask ^= (1 << v)
                count -= 1
            # Local improve
            S_mask, count = local_improve(S_mask, count, rng)
            if count > best_count:
                best_count = count
                best_mask = S_mask

    # Convert back to 1-based list
    result = [i + 1 for i in range(N) if best_mask & (1 << i)]
    if not result:
        # Fallback
        result = [1]
    return result
# EVOLVE-BLOCK-END
