import math
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # safety margin before 60s

    # Diagonal constraint: 2a must not be a perfect square
    # 2a = k^2 => a = k^2/2, k even => a = 2*m^2
    def is_square(n):
        if n < 0:
            return False
        r = math.isqrt(n)
        return r * r == n

    # Exclude diagonal-violating numbers
    excluded_diag = set()
    for m in range(1, 20):
        a = 2 * m * m
        if 1 <= a <= 256:
            excluded_diag.add(a)
    # excluded_diag = {2,8,18,32,50,72,98,128,162,200,242}

    candidates = [x for x in range(1, 257) if x not in excluded_diag]
    # 245 candidates

    # Build conflict graph: edge between a,b (a<b) if a+b is a perfect square
    # Also self-conflict already handled by diagonal exclusion
    n = len(candidates)
    cand_set = set(candidates)

    # Precompute for each candidate, which other candidates conflict
    # conflicts[i] = set of indices j such that candidates[i]+candidates[j] is a square
    conflicts = [set() for _ in range(n)]
    for i in range(n):
        ai = candidates[i]
        for j in range(i + 1, n):
            aj = candidates[j]
            s = ai + aj
            if is_square(s):
                conflicts[i].add(j)
                conflicts[j].add(i)

    def is_valid_subset(subset_indices):
        """Check if subset of indices is valid (no conflicts among them)."""
        sset = set(subset_indices)
        for i in sset:
            for j in conflicts[i]:
                if j in sset:
                    return False
        return True

    def greedy(order):
        """Greedy: process in given order, add if no conflict with already added."""
        added = set()
        for i in order:
            # check if i conflicts with any already added
            conflict = False
            for j in conflicts[i]:
                if j in added:
                    conflict = True
                    break
            if not conflict:
                added.add(i)
        return added

    def local_search(subset, max_iters=100000):
        """Try to improve by: adding excluded vertices, swapping."""
        sset = set(subset)
        excluded = [i for i in range(n) if i not in sset]
        improved = True
        iters = 0
        while improved and iters < max_iters and (time.time() - start_time) < DEADLINE:
            improved = False
            iters += 1
            # Try to add each excluded vertex
            for i in excluded:
                if i in sset:
                    continue
                # Check if i conflicts with any in sset
                can_add = True
                for j in conflicts[i]:
                    if j in sset:
                        can_add = False
                        break
                if can_add:
                    sset.add(i)
                    excluded.remove(i)
                    improved = True
                    break  # restart scan
            # Try swaps: remove one, add one
            if not improved:
                for i in sset:
                    for j in excluded:
                        if j in sset:
                            continue
                        # Remove i, add j: check j has no conflict with sset-{i}
                        can_add = True
                        for k in conflicts[j]:
                            if k in sset and k != i:
                                can_add = False
                                break
                        if can_add:
                            # This swap is valid; but only useful if it enables more adds
                            # Simple swap: remove i, add j (same size, might enable future adds)
                            # Only do if it doesn't reduce size (it doesn't)
                            # Actually skip pure swaps unless they enable growth
                            pass
        return sset

    # Multiple greedy strategies
    best = set()
    best_size = 0

    # Strategy 1: natural order
    order1 = list(range(n))
    s1 = greedy(order1)
    s1 = local_search(s1, max_iters=50000)
    if len(s1) > best_size:
        best = s1
        best_size = len(s1)

    # Strategy 2: reverse order
    order2 = list(range(n - 1, -1, -1))
    s2 = greedy(order2)
    s2 = local_search(s2, max_iters=50000)
    if len(s2) > best_size:
        best = s2
        best_size = len(s2)

    # Strategy 3: by degree (fewest conflicts first)
    degrees = [len(conflicts[i]) for i in range(n)]
    order3 = sorted(range(n), key=lambda i: degrees[i])
    s3 = greedy(order3)
    s3 = local_search(s3, max_iters=50000)
    if len(s3) > best_size:
        best = s3
        best_size = len(s3)

    # Strategy 4: by degree (most conflicts first)
    order4 = sorted(range(n), key=lambda i: -degrees[i])
    s4 = greedy(order4)
    s4 = local_search(s4, max_iters=50000)
    if len(s4) > best_size:
        best = s4
        best_size = len(s4)

    # Randomized restarts
    random.seed(42)
    for restart in range(50):
        if (time.time() - start_time) > DEADLINE:
            break
        order = list(range(n))
        random.shuffle(order)
        s = greedy(order)
        s = local_search(s, max_iters=20000)
        if len(s) > best_size:
            best = s
            best_size = len(s)

    # More aggressive local search on best
    if (time.time() - start_time) < DEADLINE:
        best = local_search(best, max_iters=200000)
        best_size = len(best)

    # Convert back to values
    result = sorted([candidates[i] for i in best])
    if not result:
        result = [1]
    return result
