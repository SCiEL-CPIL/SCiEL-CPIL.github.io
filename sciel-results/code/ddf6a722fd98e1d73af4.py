import random
import time
import itertools
from collections import Counter

def _is_sidon_fast(arr):
    """Fast Sidon check for a list of ints (unordered pair sums incl. diagonals unique)."""
    seen = set()
    n = len(arr)
    for i in range(n):
        ai = arr[i]
        for j in range(i, n):
            s = ai + arr[j]
            if s in seen:
                return False
            seen.add(s)
    return True

def _max_sidon_size(ambient):
    """Exact largest Sidon subset size for a 12-element ambient set.
    Uses bitmask DP with subset-monotonicity pruning:
    if a subset is NOT Sidon, its supersets are NOT Sidon.
    We enumerate subsets in descending size and return the first Sidon one."""
    n = len(ambient)
    # Precompute pair-sums for the full ambient as a reference is not enough;
    # we need per-subset. Use a bitmask loop.
    # For n=12, 4096 subsets. For each, check Sidon.
    # Optimization: precompute for each pair (i,j) the sum, and for each subset
    # quickly detect duplicate sums.
    # Simpler: iterate subsets descending, check.
    # Order subsets by size descending.
    # Build list of (size, mask) sorted by size desc.
    masks_by_size = [[] for _ in range(n + 1)]
    for mask in range(1, 1 << n):
        masks_by_size[mask.bit_count()].append(mask)
    # Precompute element index list for each mask? We can iterate bits.
    # For each mask, check Sidon.
    # To speed up, precompute the multiset of pair sums per mask via DP is complex.
    # Just do direct check; 4096 * O(k^2) is fine.
    for k in range(n, 0, -1):
        for mask in masks_by_size[k]:
            # Extract elements
            elems = []
            m = mask
            idx = 0
            while m:
                if m & 1:
                    elems.append(ambient[idx])
                m >>= 1
                idx += 1
            if _is_sidon_fast(elems):
                return k
    return 0

def construct():
    start = time.time()
    deadline = start + 80.0  # safety margin (90s hard kill)
    rng = random.Random(12345)

    best_set = list(range(12))
    best_val = _max_sidon_size(best_set)

    # The theoretical minimum for 12 distinct ints: any 4-element set has 10 pair sums
    # (incl diagonals) and the sums must be in [2*min, 2*max]. For small ambient ranges,
    # 4 is often impossible. We aim for 3.

    def try_candidate(cand):
        nonlocal best_set, best_val
        val = _max_sidon_size(cand)
        if val < best_val:
            best_set = list(cand)
            best_val = val
            if best_val == 3:
                return True
        return False

    # === Strategy: conflict-graph greedy ===
    # For a candidate ambient set A, define conflict graph: edge (i,j) if there exist
    # (i',j') != (i,j) with i'+j' = i+j (unordered, allowing i=i').
    # A set with many collisions has small largest Sidon subset.
    # We want to CHOOSE A (12 elements) to maximize collision density.

    # Heuristic: pick elements that create many duplicate pair sums.
    # Start with a "dense" seed and grow by adding the element that maximizes
    # new sum-collisions with existing elements.

    def greedy_dense_set(seed_elements, target=12, pool=None):
        """Grow a set by adding elements that create the most sum collisions."""
        if pool is None:
            pool = list(range(0, 129))
        current = list(seed_elements)
        # Precompute current pair sums (with multiplicity)
        cur_sums = Counter()
        for i in range(len(current)):
            for j in range(i, len(current)):
                cur_sums[current[i] + current[j]] += 1
        while len(current) < target:
            best_elem = None
            best_score = -1
            # Score: number of new sum-collisions this element would create
            for x in pool:
                if x in current:
                    continue
                # Count how many pair sums involving x already exist in cur_sums
                score = 0
                for y in current:
                    s = x + y
                    if cur_sums.get(s, 0) >= 1:
                        score += 1
                    # Also x+x
                sxx = x + x
                if cur_sums.get(sxx, 0) >= 1:
                    score += 1
                if score > best_score:
                    best_score = score
                    best_elem = x
            if best_elem is None:
                break
            current.append(best_elem)
            # Update cur_sums
            for y in current[:-1]:
                cur_sums[best_elem + y] += 1
            cur_sums[best_elem + best_elem] += 1
        return current

    # Try many random seeds and greedy growth
    for trial in range(300):
        if time.time() > deadline:
            break
        # Random seed of 3-4 elements from a small range
        seed_size = rng.randint(3, 5)
        seed = rng.sample(range(0, 30), seed_size)
        # Pool: small range for density
        pool = list(range(0, 40))
        cand = greedy_dense_set(seed, 12, pool)
        if len(cand) == 12:
            if try_candidate(cand):
                return best_set

    # === Local search on best_set ===
    # Try replacing one element with another to reduce max_sidon_size.
    if best_val > 3 and time.time() < deadline - 10.0:
        current = list(best_set)
        current_val = best_val
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            # Try all single-element swaps with elements in [0, 60]
            for idx in range(12):
                if time.time() > deadline - 10.0:
                    break
                old = current[idx]
                for new in range(0, 61):
                    if new == old or new in current:
                        continue
                    current[idx] = new
                    val = _max_sidon_size(current)
                    if val < current_val:
                        current_val = val
                        best_set = list(current)
                        best_val = val
                        improved = True
                        if best_val == 3:
                            return best_set
                    else:
                        current[idx] = old
                current[idx] = old

    # === Random restarts with small-range bias ===
    for _ in range(20000):
        if time.time() > deadline:
            break
        # Sample 12 from [0, R] with small R
        R = rng.randint(15, 50)
        cand = rng.sample(range(0, R + 1), 12)
        if try_candidate(cand):
            return best_set

    return best_set
