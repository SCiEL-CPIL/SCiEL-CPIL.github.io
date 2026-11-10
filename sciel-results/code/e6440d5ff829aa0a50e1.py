"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

N = 256
_DEADLINE = 55.0  # safety margin before 60s kill

# EVOLVE-BLOCK-START

def _sidon_sums(S):
    """Return the set of all pairwise sums a+b for a<=b in S, or None if not Sidon."""
    sums = set()
    n = len(S)
    for i in range(n):
        a = S[i]
        for j in range(i, n):
            s = a + S[j]
            if s in sums:
                return None
            sums.add(s)
    return sums

def _is_sidon(S):
    return _sidon_sums(S) is not None

def _is_maximal(S, N, sums=None):
    """Check if S is inclusion-maximal Sidon in [1,N]. Reuse precomputed sums if given."""
    if sums is None:
        sums = _sidon_sums(S)
        if sums is None:
            return False
    S_set = set(S)
    n = len(S)
    for x in range(1, N + 1):
        if x in S_set:
            continue
        # New sums when adding x: {2x} U {x + a : a in S}
        # Must be distinct from each other and from existing sums.
        # Distinctness among new sums: x+a == x+b iff a==b (fine); 2x == x+a iff x==a (x not in S).
        # So new sums are always mutually distinct. Just check against existing sums.
        if (2 * x) in sums:
            continue
        blocked = False
        for a in S:
            if (x + a) in sums:
                blocked = True
                break
        if not blocked:
            return False  # x can be added -> not maximal
    return True

def _sidon_and_maximal(S, N):
    if not S:
        return False
    sums = _sidon_sums(S)
    if sums is None:
        return False
    return _is_maximal(S, N, sums)

def _greedy_sidon(randomize=False, seed=None):
    """Build a Sidon set greedily. If randomize, shuffle candidate order."""
    if seed is not None:
        random.seed(seed)
    selected = []
    sums = set()
    candidates = list(range(1, N + 1))
    if randomize:
        random.shuffle(candidates)
    for x in candidates:
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    selected.sort()
    return selected

def _try_minimize_by_removal(S, N, deadline=None):
    """Try to remove elements while keeping Sidon + maximal. Returns minimal size set found."""
    S = list(S)
    improved = True
    while improved:
        improved = False
        sums = _sidon_sums(S)
        if sums is None:
            break
        for i in range(len(S)):
            if deadline is not None and time.time() > deadline:
                return S
            candidate = S[:i] + S[i+1:]
            if _sidon_and_maximal(candidate, N):
                S = candidate
                improved = True
                break
    return S

def _local_search_swap(S, N, max_swaps=30, deadline=None):
    """Try to improve Sidon set by swapping elements to reduce size or find better configuration."""
    S = list(S)
    S_set = set(S)
    for _ in range(max_swaps):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(len(S)):
            for x in range(1, N + 1):
                if x in S_set:
                    continue
                new_S = S[:i] + S[i+1:] + [x]
                if len(set(new_S)) != len(new_S):
                    continue
                if _sidon_and_maximal(new_S, N):
                    S = new_S
                    S_set = set(new_S)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return S

def _random_greedy_variants(seed):
    """Build a Sidon set with a randomized ordering strategy derived from seed."""
    rng = random.Random(seed)
    selected = []
    sums = set()
    remaining = list(range(1, N + 1))
    while remaining:
        # pick a random candidate from remaining, but bias toward lower values sometimes
        if rng.random() < 0.5:
            idx = rng.randrange(len(remaining))
            x = remaining.pop(idx)
        else:
            x = remaining.pop(0)
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    selected.sort()
    return selected

def construct():
    start_time = time.time()
    best_set = None
    best_size = N + 1

    # Phase 1: Try several random greedy constructions, then minimize
    for seed in range(8000):
        elapsed = time.time() - start_time
        if elapsed > _DEADLINE - 5:  # leave 5s safety
            break
        if seed < 2000:
            S = _greedy_sidon(randomize=True, seed=seed)
        else:
            S = _random_greedy_variants(seed)
        if len(S) < best_size:
            # Try to minimize further
            S_min = _try_minimize_by_removal(S, N, deadline=_DEADLINE - 3)
            if _sidon_and_maximal(S_min, N) and len(S_min) < best_size:
                best_size = len(S_min)
                best_set = S_min
                # Try local search swap to further improve
                S_swapped = _local_search_swap(S_min, N, max_swaps=15, deadline=_DEADLINE - 2)
                if _sidon_and_maximal(S_swapped, N) and len(S_swapped) < best_size:
                    best_size = len(S_swapped)
                    best_set = S_swapped
            if elapsed > _DEADLINE - 10:
                break

    # If nothing found, use deterministic greedy
    if best_set is None:
        S = _greedy_sidon(randomize=False)
        S_min = _try_minimize_by_removal(S, N)
        if _sidon_and_maximal(S_min, N):
            best_set = S_min
        else:
            best_set = S

    return best_set
# EVOLVE-BLOCK-END
