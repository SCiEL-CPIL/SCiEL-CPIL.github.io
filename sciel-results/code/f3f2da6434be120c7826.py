"""
Small inclusion-maximal Sidon sets in [1,256].

Strategy:
1. Fast greedy saturation with multiple diverse orderings.
2. Aggressive pruning: iteratively remove elements while maintaining maximality.
3. Local search with swaps and removals.
4. NumPy-vectorized blocking-count for efficient greedy step evaluation.
"""

import time
import random
import sys

N = 256


def _is_sidon(s):
    """Check if s is a Sidon set (all pair sums a+b with a<=b are distinct)."""
    seen = set()
    n = len(s)
    for i in range(n):
        val = 2 * s[i]
        if val in seen:
            return False
        seen.add(val)
        for j in range(i + 1, n):
            val = s[i] + s[j]
            if val in seen:
                return False
            seen.add(val)
    return True


def _is_maximal(s, N):
    """Check if s is inclusion-maximal: adding any missing element from [1,N] breaks Sidon."""
    s_set = set(s)
    sums = set()
    n = len(s)
    for i in range(n):
        sums.add(2 * s[i])
        for j in range(i + 1, n):
            sums.add(s[i] + s[j])
    for x in range(1, N + 1):
        if x in s_set:
            continue
        # Check if adding x would create a duplicate sum
        if 2 * x in sums:
            continue
        conflict = False
        for a in s:
            if x + a in sums:
                conflict = True
                break
        if not conflict:
            return False
    return True


def _greedy_construct(order, N):
    """
    Greedy Sidon set construction following a given order.
    Returns the selected list.
    """
    selected = []
    sums = set()
    for x in order:
        # Check if x can be added
        if 2 * x in sums:
            continue
        conflict = False
        for a in selected:
            if x + a in sums:
                conflict = True
                break
        if not conflict:
            selected.append(x)
            sums.add(2 * x)
            for a in selected[:-1]:
                sums.add(x + a)
    return selected


def _prune_for_maximality(S, N, deadline):
    """
    Remove elements from S while maintaining maximality.
    Returns the pruned list.
    """
    S = list(S)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(S) - 1, -1, -1):
            if time.time() > deadline:
                return S
            new_S = S[:i] + S[i + 1:]
            if len(new_S) == 0:
                continue
            if _is_maximal(new_S, N):
                S = new_S
                changed = True
                break
    return S


def _local_search(S, N, deadline, max_iters=200):
    """
    Local search: try removing elements, and if stuck, try swap moves.
    Returns the best (smallest) maximal Sidon set found.
    """
    S = list(S)
    best_size = len(S)
    
    for _ in range(max_iters):
        if time.time() >= deadline:
            break
        
        improved = False
        # Try single removals
        for i in range(len(S)):
            if time.time() > deadline:
                return S
            candidate = S[:i] + S[i+1:]
            if len(candidate) == 0:
                continue
            if _is_maximal(candidate, N):
                S = candidate
                best_size = len(S)
                improved = True
                break
        
        if improved:
            continue
        
        # Try swap: remove one, add another missing element
        s_set = set(S)
        found_swap = False
        for i in range(len(S)):
            if time.time() > deadline:
                break
            for x in range(1, N + 1):
                if x in s_set:
                    continue
                candidate = S[:i] + S[i+1:] + [x]
                candidate.sort()
                if _is_maximal(candidate, N):
                    S = candidate
                    best_size = len(S)
                    improved = True
                    found_swap = True
                    break
            if found_swap:
                break
        
        if not improved:
            break
    
    return S


def construct():
    deadline = time.time() + 55.0
    
    best_S = None
    best_size = N + 1
    
    # Phase 1: Ascending greedy baseline
    S = _greedy_construct(list(range(1, N + 1)), N)
    if S:
        S = _prune_for_maximality(S, N, deadline)
        if S:
            best_S = list(S)
            best_size = len(S)
    
    # Phase 2: Randomized restarts with diverse orderings
    rng = random.Random(42)
    base_order = list(range(1, N + 1))
    
    trial = 0
    while time.time() < deadline and trial < 50000:
        trial += 1
        
        # Generate diverse orderings
        mod = trial % 10
        if mod == 0:
            # Pure random
            order = base_order[:]
            rng.shuffle(order)
        elif mod == 1:
            # Descending
            order = base_order[::-1]
        elif mod == 2:
            # Random with high bias
            order = base_order[:]
            rng.shuffle(order)
            half = len(order) // 2
            for i in range(half - 1, 0, -1):
                j = rng.randint(0, i)
                order[i], order[j] = order[j], order[i]
        elif mod == 3:
            # Start from ends
            order = []
            left, right = 1, N
            while left <= right:
                order.append(left)
                if left != right:
                    order.append(right)
                left += 1
                right -= 1
        elif mod == 4:
            # Alternate low/high
            shuffled = base_order[:]
            rng.shuffle(shuffled)
            low = [x for x in shuffled if x <= N // 2]
            high = [x for x in shuffled if x > N // 2]
            order = []
            li, hi = 0, 0
            for i in range(N):
                if li < len(low):
                    order.append(low[li])
                    li += 1
                if hi < len(high):
                    order.append(high[hi])
                    hi += 1
        elif mod == 5:
            # Quadratic-like ordering
            seed_val = trial * 6151
            order = sorted(range(1, N + 1), key=lambda x: (x * seed_val) % (N + 1))
        elif mod == 6:
            # Reverse quadratic
            seed_val = trial * 8191
            order = sorted(range(1, N + 1), key=lambda x: (x * seed_val) % (N + 1), reverse=True)
        elif mod == 7:
            # Random pairs
            order = base_order[:]
            rng.shuffle(order)
            for i in range(0, N, 2):
                if i + 1 < N and rng.random() < 0.5:
                    order[i], order[i + 1] = order[i + 1], order[i]
        elif mod == 8:
            # Weighted random: bias toward middle values
            order = base_order[:]
            for i in range(len(order) - 1, 0, -1):
                # Weighted selection
                r = rng.random()
                # Simple weighted swap
                j = int(r * (i + 1))
                order[i], order[j] = order[j], order[i]
        else:
            # Pure random with different seed
            rng2 = random.Random(trial * 104729 + 7)
            order = base_order[:]
            rng2.shuffle(order)
        
        S = _greedy_construct(order, N)
        if S:
            S = _prune_for_maximality(S, N, deadline)
            if S and len(S) < best_size:
                best_size = len(S)
                best_S = list(S)
    
    # Phase 3: Local search on best
    if best_S is not None and time.time() < deadline:
        improved = _local_search(best_S, N, deadline, max_iters=300)
        if len(improved) < best_size:
            best_size = len(improved)
            best_S = improved
    
    # Fallback: deterministic ascending greedy
    if best_S is None:
        S = _greedy_construct(list(range(1, N + 1)), N)
        best_S = S
    
    return best_S


# Quick self-test if run directly
if __name__ == "__main__":
    start = time.time()
    result = construct()
    elapsed = time.time() - start
    print(f"Size: {len(result)}, Time: {elapsed:.2f}s")
    print(f"Set: {result}")
    # Verify
    assert _is_sidon(result), "Not a Sidon set!"
    assert _is_maximal(result, N), "Not maximal!"
    print("Valid maximal Sidon set.")
