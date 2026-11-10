"""Rewritten program: 3-uniform hypergraph (3-AP) greedy MIS + bitset-DP full-constraint verifier with repair."""
import random
import time

def _is_3ap_free_add(x, S):
    """Return True if adding x to set S keeps it free of 3-term arithmetic progressions.
    A 3-AP (a,b,c) with 2a=b+c is a conflict (a is the mean of the pair {b,c}).
    When adding x, new conflicts are 3-APs containing x:
      - x is the mean: exists b,c in S with b+c == 2x
      - x is an endpoint: exists a in S with 2a == x + c for some c in S
    """
    two_x = 2 * x
    for b in S:
        c = two_x - b
        if c in S and c != b:
            return False
    for a in S:
        c = 2 * a - x
        # c != a and c != x are guaranteed (a in S, x not in S, a != x)
        if c in S:
            return False
    return True


def _element_is_mean(a, S):
    """Return True if a is the exact arithmetic mean of some subset of S\{a} of size >= 2.
    Uses a per-count bitset DP (Python big ints) over subset sums."""
    others = [x for x in S if x != a]
    n = len(others)
    if n < 2:
        return False
    max_sum = sum(others)
    mask = (1 << (max_sum + 1)) - 1
    dp = [0] * (n + 1)
    dp[0] = 1  # bit 0 set: empty subset sum 0
    for e in others:
        for c in range(n - 1, -1, -1):
            if dp[c]:
                dp[c + 1] |= (dp[c] << e) & mask
    for c in range(2, n + 1):
        target = c * a
        if target <= max_sum and ((dp[c] >> target) & 1):
            return True
    return False


def _is_fully_valid(S):
    for a in S:
        if _element_is_mean(a, S):
            return False
    return True


def _repair(S):
    """Remove any element that is the mean of a subset of the others, until valid."""
    S = set(S)
    guard = len(S) + 2
    while guard > 0:
        guard -= 1
        removed = False
        for a in list(S):
            if _element_is_mean(a, S):
                S.remove(a)
                removed = True
                break
        if not removed:
            break
    return S


def _greedy_3ap_free(order):
    S = set()
    for x in order:
        if _is_3ap_free_add(x, S):
            S.add(x)
    return S


def _calculate_3ap_degrees(n=512):
    """Calculate the number of 3-APs each element is part of."""
    degrees = [0] * (n + 1)
    # For each mean a, count pairs (b,c)
    for a in range(1, n + 1):
        for b in range(1, n + 1):
            c = 2 * a - b
            if c > 0 and c <= n and c != b:
                # (a,b,c) is a 3-AP. a is mean.
                # This contributes to degree of a, b, c.
                degrees[a] += 1
                degrees[b] += 1
                degrees[c] += 1
    return degrees


def _local_search(S, max_iterations=100):
    """Try to improve set S by adding or swapping elements."""
    S = set(S)
    if not _is_fully_valid(S):
        S = _repair(S)
    
    improved = True
    iters = 0
    while improved and iters < max_iterations:
        iters += 1
        improved = False
        
        # Try to add an element
        for x in range(1, 513):
            if x in S:
                continue
            new_S = S | {x}
            if _is_fully_valid(new_S):
                S = new_S
                improved = True
                break
        
        if improved:
            continue
        
        # Try to swap an element
        for y in list(S):
            for x in range(1, 513):
                if x in S:
                    continue
                new_S = (S - {y}) | {x}
                if _is_fully_valid(new_S):
                    S = new_S
                    improved = True
                    break
            if improved:
                break
                
    return S


def construct():
    rng = random.Random(20240601)
    universe = list(range(1, 513))
    
    # Precompute degrees for heuristic ordering
    degrees = _calculate_3ap_degrees(512)
    
    candidates = []
    
    # 1. Random restarts
    n_random_restarts = 300
    for _ in range(n_random_restarts):
        order = universe[:]
        rng.shuffle(order)
        candidates.append(_greedy_3ap_free(order))
        
    # 2. Degree-sorted restarts (Low degree first)
    n_degree_restarts = 50
    for _ in range(n_degree_restarts):
        # Add some randomness to the sorted order to avoid getting stuck
        order = sorted(universe, key=lambda x: degrees[x] + rng.random() * 10)
        candidates.append(_greedy_3ap_free(order))
        
    # 3. High degree first (maybe?)
    for _ in range(20):
        order = sorted(universe, key=lambda x: -degrees[x] + rng.random() * 10)
        candidates.append(_greedy_3ap_free(order))

    # Sort by size (desc), dedupe, keep top unique
    candidates.sort(key=len, reverse=True)
    seen = set()
    unique = []
    for S in candidates:
        key = frozenset(S)
        if key not in seen:
            seen.add(key)
            unique.append(S)
        if len(unique) >= 20:
            break

    best_valid = set()
    # Apply local search to top candidates
    for S in unique:
        # First, ensure it's valid (repair)
        R = _repair(S)
        # Then, try to improve with local search
        improved_S = _local_search(R)
        if len(improved_S) > len(best_valid):
            best_valid = improved_S
            
    # Final safety verification
    if not _is_fully_valid(best_valid):
        best_valid = _repair(best_valid)

    # Fallback (should not trigger): powers of two are valid
    if not best_valid:
        best_valid = {1, 2, 4, 8, 16, 32, 64, 128, 256}

    # Cap at 48 (defensive) and return sorted distinct ints in [1, 512]
    result = sorted(x for x in best_valid if 1 <= x <= 512)[:48]
    return result
