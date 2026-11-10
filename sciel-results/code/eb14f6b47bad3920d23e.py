"""Rewritten program: 3-uniform hypergraph (3-AP) greedy MIS + bitset-DP full-constraint verifier with repair."""
import random

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
    """Return True if a is the exact arithmetic mean of some subset of S\\{a} of size >= 2.
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

def _local_search_3ap(S):
    """Local search to maximize size of 3-AP free set S."""
    S = set(S)
    improved = True
    while improved:
        improved = False
        # Try to add any element
        for x in range(1, 513):
            if x not in S and _is_3ap_free_add(x, S):
                S.add(x)
                improved = True
                break
        if not improved:
            # Try to swap: remove y, add x
            # We only do this if it might lead to a larger set later
            # Simple heuristic: try to swap a 'central' element with a 'peripheral' one?
            # Just try random swaps until no improvement in size is possible in next step
            # To keep it fast, we limit swaps
            for y in list(S):
                S.remove(y)
                added = False
                for x in range(1, 513):
                    if x != y and x not in S and _is_3ap_free_add(x, S):
                        S.add(x)
                        added = True
                        break
                if not added:
                    S.add(y) # revert
                else:
                    improved = True
                    break
    return S

def _greedy_3ap_free(order):
    S = set()
    for x in order:
        if _is_3ap_free_add(x, S):
            S.add(x)
    return S


def construct():
    rng = random.Random(20240601)
    universe = list(range(1, 513))

    candidates = []
    n_restarts = 200 # Reduced to allow local search
    for _ in range(n_restarts):
        order = universe[:]
        rng.shuffle(order)
        S = _greedy_3ap_free(order)
        S = _local_search_3ap(S)
        candidates.append(S)

    # Sort by size (desc), dedupe, keep top unique
    candidates.sort(key=len, reverse=True)
    seen = set()
    unique = []
    for S in candidates:
        key = frozenset(S)
        if key not in seen:
            seen.add(key)
            unique.append(S)
        if len(unique) >= 10:
            break

    best_valid = set()
    for S in unique:
        R = _repair(S)
        if len(R) > len(best_valid):
            best_valid = R

    # Final safety verification
    if not _is_fully_valid(best_valid):
        best_valid = _repair(best_valid)

    # Fallback (should not trigger): powers of two are valid
    if not best_valid:
        best_valid = {1, 2, 4, 8, 16, 32, 64, 128, 256}

    # Cap at 48 (defensive) and return sorted distinct ints in [1, 512]
    result = sorted(x for x in best_valid if 1 <= x <= 512)[:48]
    return result
