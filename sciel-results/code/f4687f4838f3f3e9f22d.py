import time
import random
import sys

N = 256
_DEADLINE = 55.0

def _sidon_and_maximal_fast(S, N):
    """Fast Sidon + maximality check using bitmasks."""
    if not S:
        return False
    
    n = len(S)
    S_set = set(S)
    
    # Check Sidon property using sum bitmask
    sum_mask = 0
    for i in range(n):
        a = S[i]
        for j in range(i, n):
            s = a + S[j]
            bit = 1 << s
            if sum_mask & bit:
                return False
            sum_mask |= bit
    
    # Check maximality: every missing x must be blocked
    # x is blocked if: 2x is a sum, or x+a is a sum for some a in S
    for x in range(1, N + 1):
        if x in S_set:
            continue
        # Check if 2x is in sums
        if sum_mask & (1 << (2 * x)):
            continue
        # Check if x + a is in sums for any a in S
        blocked = False
        for a in S:
            if sum_mask & (1 << (x + a)):
                blocked = True
                break
        if not blocked:
            return False
    return True

def _greedy_sidon(order=None):
    """Build a Sidon set greedily given an explicit order of candidates."""
    if order is None:
        order = list(range(1, N + 1))
    selected = []
    sum_mask = 0
    for x in order:
        ok = True
        # Check 2x
        if sum_mask & (1 << (2 * x)):
            ok = False
        else:
            # Check x + a
            for a in selected:
                if sum_mask & (1 << (x + a)):
                    ok = False
                    break
        if ok:
            sum_mask |= (1 << (2 * x))
            for a in selected:
                sum_mask |= (1 << (x + a))
            selected.append(x)
    selected.sort()
    return selected

def _try_minimize_by_removal(S, N, deadline=None):
    """Try to remove elements while keeping Sidon + maximal."""
    S = list(S)
    improved = True
    while improved:
        improved = False
        for i in range(len(S)):
            if deadline is not None and time.time() > deadline:
                return S
            candidate = S[:i] + S[i+1:]
            if _sidon_and_maximal_fast(candidate, N):
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
                if _sidon_and_maximal_fast(new_S, N):
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
    for trial in range(20000):
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
            if _sidon_and_maximal_fast(S_min, N) and len(S_min) < best_size:
                best_size = len(S_min)
                best_set = S_min
                # Try local search
                S_swapped = _local_search_swap(S_min, N, max_swaps=10, deadline=_DEADLINE - 2)
                if _sidon_and_maximal_fast(S_swapped, N) and len(S_swapped) < best_size:
                    best_size = len(S_swapped)
                    best_set = S_swapped
    
    if best_set is None:
        S = _greedy_sidon()
        S_min = _try_minimize_by_removal(S, N)
        if _sidon_and_maximal_fast(S_min, N):
            best_set = S_min
        else:
            best_set = S

    return best_set
