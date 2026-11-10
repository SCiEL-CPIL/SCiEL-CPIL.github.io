import time
import random
import sys

N = 256
_DEADLINE = 55.0

def _sidon_sums_arr(S):
    """Return array of sums or None if not Sidon."""
    sums = [False] * (2 * N + 1)
    n = len(S)
    for i in range(n):
        a = S[i]
        for j in range(i, n):
            s = a + S[j]
            if sums[s]:
                return None
            sums[s] = True
    return sums

def _is_maximal_arr(S, sums_arr):
    """Check if S is inclusion-maximal Sidon in [1,N] using sums array."""
    S_set = set(S)
    blocked = [False] * (N + 1)
    
    # From 2x in sums
    for s in range(2, 2 * N + 1):
        if sums_arr[s]:
            if s % 2 == 0:
                x = s // 2
                if 1 <= x <= N:
                    blocked[x] = True
                    
    # From x + a in sums => x = s - a
    for s in range(2, 2 * N + 1):
        if sums_arr[s]:
            for a in S:
                x = s - a
                if 1 <= x <= N:
                    blocked[x] = True
                    
    for x in range(1, N + 1):
        if x not in S_set and not blocked[x]:
            return False
    return True

def _sidon_and_maximal(S, N):
    if not S:
        return False
    sums = _sidon_sums_arr(S)
    if sums is None:
        return False
    return _is_maximal_arr(S, sums)

def _greedy_sidon(order=None):
    """Build a Sidon set greedily given an explicit order of candidates."""
    if order is None:
        order = list(range(1, N + 1))
    selected = []
    sums = [False] * (2 * N + 1)
    for x in order:
        ok = True
        # Check 2x
        if sums[2 * x]:
            ok = False
        else:
            # Check x + a
            for a in selected:
                if sums[x + a]:
                    ok = False
                    break
        if ok:
            sums[2 * x] = True
            for a in selected:
                sums[x + a] = True
            selected.append(x)
    selected.sort()
    return selected

def _try_minimize_by_removal(S, N, deadline=None):
    """Try to remove elements while keeping Sidon + maximal."""
    S = list(S)
    improved = True
    while improved:
        improved = False
        sums = _sidon_sums_arr(S)
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

def _local_search_swap(S, N, max_swaps=50, deadline=None):
    """Try to improve Sidon set by swapping elements."""
    S = list(S)
    S_set = set(S)
    for _ in range(max_swaps):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(len(S)):
            if deadline is not None and time.time() > deadline:
                break
            for x in range(1, N + 1):
                if x in S_set:
                    continue
                new_S = S[:i] + S[i+1:] + [x]
                sums = _sidon_sums_arr(new_S)
                if sums is None:
                    continue
                if _is_maximal_arr(new_S, sums):
                    S = new_S
                    S_set = set(S)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return S

def construct():
    start_time = time.time()
    best_set = None
    best_size = N + 1
    rng = random.Random(42)
    
    # Phase 1: Try diverse random orderings
    for trial in range(5000):
        elapsed = time.time() - start_time
        if elapsed > _DEADLINE - 5:
            break
        
        # Create diverse orderings
        if trial % 4 == 0:
            # Pure random shuffle
            order = list(range(1, N + 1))
            rng.shuffle(order)
        elif trial % 4 == 1:
            # Random with slight bias toward larger numbers
            order = list(range(1, N + 1))
            for i in range(len(order) - 1, 0, -1):
                j = rng.randint(0, i)
                if rng.random() < 0.3:
                    # Prefer swapping with later (larger) elements
                    j = min(i, rng.randint(i // 2, i))
                order[i], order[j] = order[j], order[i]
        elif trial % 4 == 2:
            # Interleave small and large
            small = list(range(1, N // 2 + 1))
            large = list(range(N // 2 + 1, N + 1))
            rng.shuffle(small)
            rng.shuffle(large)
            order = []
            for i in range(len(small)):
                order.append(small[i])
                if i < len(large):
                    order.append(large[i])
            rng.shuffle(order)
        else:
            # Random but with some structure - pick random pivot and sort by distance
            pivot = rng.randint(1, N)
            order = sorted(range(1, N + 1), key=lambda x: abs(x - pivot) + rng.random() * 10)
        
        S = _greedy_sidon(order)
        
        if len(S) < best_size:
            S_min = _try_minimize_by_removal(S, N, deadline=_DEADLINE - 3)
            if _sidon_and_maximal(S_min, N) and len(S_min) < best_size:
                best_size = len(S_min)
                best_set = S_min
                # Try local search
                S_swapped = _local_search_swap(S_min, N, max_swaps=10, deadline=_DEADLINE - 2)
                if _sidon_and_maximal(S_swapped, N) and len(S_swapped) < best_size:
                    best_size = len(S_swapped)
                    best_set = S_swapped
    
    if best_set is None:
        S = _greedy_sidon()
        S_min = _try_minimize_by_removal(S, N)
        if _sidon_and_maximal(S_min, N):
            best_set = S_min
        else:
            best_set = S

    return best_set
