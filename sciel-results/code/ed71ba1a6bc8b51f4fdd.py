"""Additive bases with few representations - optimized construction."""
import time
import random
import numpy as np

N = 128
TARGET_LO = 128
TARGET_HI = 256
TARGETS = list(range(TARGET_LO, TARGET_HI + 1))  # 129 targets
NUM_TARGETS = len(TARGETS)

def _elapsed():
    return time.time() - _START_TIME

def _pair_counts(A):
    """Compute representation counts for each target in [128,256].
    A is a sorted list of ints in [0,128].
    Returns a list of 129 counts (index 0 -> target 128, index 128 -> target 256).
    """
    counts = [0] * NUM_TARGETS
    n = len(A)
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            s = ai + A[j]
            if TARGET_LO <= s <= TARGET_HI:
                counts[s - TARGET_LO] += 1
    return counts

def _peak(counts):
    return max(counts)

def _is_valid(counts):
    return all(c > 0 for c in counts)

def _score(counts):
    """Lower is better. Return peak multiplicity, or infinity if invalid."""
    if not _is_valid(counts):
        return float('inf')
    return _peak(counts)

def _try_swap(A, counts, i, x_new):
    """Try replacing A[i] with x_new. Returns (new_counts, new_peak) or None if invalid.
    A must be sorted. x_new must not already be in A (except at position i).
    """
    # Remove A[i]'s contributions, add x_new's contributions
    ai = A[i]
    if x_new == ai:
        return None
    # Check x_new not in A (other than position i)
    # Since A is sorted, check neighbors
    if i > 0 and A[i-1] == x_new:
        return None
    if i < len(A) - 1 and A[i+1] == x_new:
        return None
    # Also need to check that x_new isn't elsewhere - but since sorted and we checked neighbors,
    # and x_new != ai, if x_new is in A it must be adjacent in sorted order... actually no.
    # Let's just check: since A is sorted, x_new can only appear at most once, and if it's not
    # at position i and not adjacent, it's not in A. Wait, that's not right either.
    # Actually for a sorted list, if x_new is in A, it's at a unique position. We checked
    # positions i-1 and i+1. But x_new could be at another position. Let me think...
    # If A is sorted and x_new != A[i], then x_new is either < A[i] or > A[i].
    # If x_new < A[i], it can only be at positions 0..i-1. We checked i-1.
    # If x_new > A[i], it can only be at positions i+1..n-1. We checked i+1.
    # But what if x_new < A[i] and is at position i-2? That's possible!
    # So we need a proper check. For efficiency, let's use a set.
    pass  # We'll handle this differently

def _try_swap2(A_set, A_sorted, counts, i, x_new):
    """Try replacing A_sorted[i] with x_new. A_set is a set for O(1) lookup.
    Returns (new_counts, new_peak) or None.
    """
    ai = A_sorted[i]
    if x_new == ai:
        return None
    if x_new in A_set:
        return None
    
    n = len(A_sorted)
    new_counts = list(counts)  # copy
    
    # Remove contributions of ai
    for j in range(n):
        if j == i:
            continue
        aj = A_sorted[j]
        # Pair (ai, aj) with i < j or j < i, but since we're removing ai from position i,
        # we need to consider all pairs involving ai
        s = ai + aj
        if TARGET_LO <= s <= TARGET_HI:
            new_counts[s - TARGET_LO] -= 1
    
    # Also remove pair (ai, ai) if i was paired with itself
    # In our counting, pair (i,i) is counted once. So:
    s = ai + ai
    if TARGET_LO <= s <= TARGET_HI:
        new_counts[s - TARGET_LO] -= 1
    
    # Add contributions of x_new
    for j in range(n):
        if j == i:
            continue
        aj = A_sorted[j]
        s = x_new + aj
        if TARGET_LO <= s <= TARGET_HI:
            new_counts[s - TARGET_LO] += 1
    
    # Add pair (x_new, x_new)
    s = x_new + x_new
    if TARGET_LO <= s <= TARGET_HI:
        new_counts[s - TARGET_LO] += 1
    
    # Check validity
    if any(c < 0 for c in new_counts):
        return None
    
    peak = max(new_counts)
    return (new_counts, peak)

def _local_search(A_sorted, max_time=None):
    """Perform swap-based local search to minimize peak multiplicity.
    Returns (best_A, best_peak, best_counts).
    """
    A_set = set(A_sorted)
    counts = _pair_counts(A_sorted)
    current_peak = _peak(counts)
    
    if not _is_valid(counts):
        return (A_sorted, current_peak, counts)
    
    best_A = list(A_sorted)
    best_peak = current_peak
    best_counts = list(counts)
    
    n = len(A_sorted)
    improved = True
    while improved and (max_time is None or _elapsed() < max_time):
        improved = False
        # Try all possible swaps
        for i in range(n):
            ai = A_sorted[i]
            # Try replacing with each value in [0, 128] not in A
            # For efficiency, focus on values that might help
            for x_new in range(0, N + 1):
                if x_new == ai or x_new in A_set:
                    continue
                result = _try_swap2(A_set, A_sorted, counts, i, x_new)
                if result is None:
                    continue
                new_counts, new_peak = result
                if new_peak < current_peak:
                    # Accept
                    A_sorted[i] = x_new
                    A_set.discard(ai)
                    A_set.add(x_new)
                    # Re-sort
                    A_sorted.sort()
                    # Rebuild set (already done)
                    counts = new_counts
                    current_peak = new_peak
                    improved = True
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_A = list(A_sorted)
                        best_counts = list(counts)
                    break  # restart from beginning after a move
        # If we made a move, loop again
    
    return (best_A, best_peak, best_counts)

def _generate_initial_A(size, rng):
    """Generate an initial random set of given size from [0, 128]."""
    return sorted(rng.sample(range(0, N + 1), size))

def _greedy_build(target_size, rng):
    """Build a set greedily to cover all targets, then try to minimize peak."""
    A = []
    A_set = set()
    counts = [0] * NUM_TARGETS
    
    # First, ensure coverage: add elements until all targets are covered
    # Start with some random elements
    remaining = list(range(0, N + 1))
    rng.shuffle(remaining)
    
    # Add elements one by one, prioritizing those that help uncovered targets
    max_iter = 200
    for _ in range(max_iter):
        if all(c > 0 for c in counts):
            break
        # Find the target with minimum count
        min_target_idx = min(range(NUM_TARGETS), key=lambda i: counts[i])
        min_target = TARGET_LO + min_target_idx
        
        # Find a candidate x in [0,128] not in A that helps cover min_target
        best_x = None
        best_gain = -1
        for x in remaining:
            if x in A_set:
                continue
            # How many new pairs does x create that hit uncovered targets?
            gain = 0
            for y in A:
                s = x + y
                if TARGET_LO <= s <= TARGET_HI and counts[s - TARGET_LO] == 0:
                    gain += 1
            # Also x+x
            s = x + x
            if TARGET_LO <= s <= TARGET_HI and counts[s - TARGET_LO] == 0:
                gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_x = x
        
        if best_x is None or best_gain <= 0:
            # Pick a random remaining element
            if remaining:
                best_x = remaining.pop()
            else:
                break
        
        if best_x in remaining:
            remaining.remove(best_x)
        
        A.append(best_x)
        A_set.add(best_x)
        A.sort()
        
        # Update counts
        n = len(A)
        for i in range(n):
            for j in range(i, n):
                s = A[i] + A[j]
                if TARGET_LO <= s <= TARGET_HI:
                    counts[s - TARGET_LO] += 1
    
    # Now try to reduce size or improve peak
    # Remove redundant elements
    changed = True
    while changed and len(A) > 10:
        changed = False
        for i in range(len(A)):
            # Try removing A[i]
            ai = A[i]
            new_counts = list(counts)
            n = len(A)
            for j in range(n):
                if j == i:
                    continue
                aj = A[j]
                s = ai + aj
                if TARGET_LO <= s <= TARGET_HI:
                    new_counts[s - TARGET_LO] -= 1
            s = ai + ai
            if TARGET_LO <= s <= TARGET_HI:
                new_counts[s - TARGET_LO] -= 1
            
            if all(c > 0 for c in new_counts):
                # Can remove
                A.pop(i)
                A_set.discard(ai)
                counts = new_counts
                changed = True
                break
    
    return A, counts

def construct():
    global _START_TIME
    _START_TIME = time.time()
    DEADLINE = 80.0  # safety margin before 90s kill
    
    rng = random.Random(42)
    
    best_A = None
    best_peak = float('inf')
    best_counts = None
    
    # Strategy: try various sizes, use local search
    # Smaller sets have fewer pairs, potentially more uniform distribution
    # Size 35-50 range is promising based on knowledge
    
    sizes_to_try = [35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
    
    # For each size, do multiple restarts
    restarts_per_size = 5
    
    for size in sizes_to_try:
        if _elapsed() > DEADLINE - 5:
            break
        
        for restart in range(restarts_per_size):
            if _elapsed() > DEADLINE - 3:
                break
            
            # Generate initial set
            A = _generate_initial_A(size, rng)
            counts = _pair_counts(A)
            
            # If not valid, try to fix by adding elements
            if not _is_valid(counts):
                # Add elements to cover gaps
                for _ in range(20):
                    if _is_valid(counts):
                        break
                    # Find uncovered target
                    uncovered = [i for i in range(NUM_TARGETS) if counts[i] == 0]
                    if not uncovered:
                        break
                    # Pick a random x to add
                    candidates = [x for x in range(0, N + 1) if x not in set(A)]
                    if not candidates:
                        break
                    x = rng.choice(candidates)
                    A.append(x)
                    A.sort()
                    counts = _pair_counts(A)
            
            if not _is_valid(counts):
                continue
            
            peak = _peak(counts)
            
            # Local search
            A_sorted = sorted(A)
            A_sorted, peak, counts = _local_search(A_sorted, max_time=DEADLINE - 3)
            
            if peak < best_peak:
                best_peak = peak
                best_A = list(A_sorted)
                best_counts = list(counts)
            
            # Early termination if we hit a good peak
            if best_peak <= 2:
                break
        
        if best_peak <= 2:
            break
    
    # If we still don't have a valid set, fall back to the seed
    if best_A is None or not _is_valid(best_counts):
        # Use the seed: [64, 128]
        best_A = list(range(64, 129))
        best_counts = _pair_counts(best_A)
        best_peak = _peak(best_counts)
    
    return best_A

# Module-level time tracking
_START_TIME = time.time()
