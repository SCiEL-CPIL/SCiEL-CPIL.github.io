import time
import random
import numpy as np

N = 128
TARGET_LO = 128
TARGET_HI = 256
NUM_TARGETS = 129  # targets 128..256 inclusive

_START_TIME = time.time()
DEADLINE = 55.0  # Safety margin before 60s kill (evaluator kills at 90s, but prompt says 60s limit for evaluation? "Each evaluation must finish within 60 s". I will use 50s to be safe.)

def _elapsed():
    return time.time() - _START_TIME

def _pair_counts(A):
    """Compute representation counts for each target in [128,256]."""
    counts = np.zeros(NUM_TARGETS, dtype=np.int32)
    n = len(A)
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            s = ai + A[j]
            if TARGET_LO <= s <= TARGET_HI:
                counts[s - TARGET_LO] += 1
    return counts

def _is_valid(counts):
    return np.min(counts) > 0

def _get_peak(counts):
    return np.max(counts)

def _try_swap(A, A_set, counts, i, x_new):
    """
    Try replacing A[i] with x_new.
    Returns (new_counts, new_peak) or None if invalid.
    """
    ai = A[i]
    if x_new == ai or x_new in A_set:
        return None

    n = len(A)
    new_counts = counts.copy()

    # Remove contributions of ai
    # Pair (ai, ai)
    s = ai + ai
    if TARGET_LO <= s <= TARGET_HI:
        new_counts[s - TARGET_LO] -= 1

    # Pairs (ai, aj) for j != i
    for j in range(n):
        if j == i:
            continue
        aj = A[j]
        s = ai + aj
        if TARGET_LO <= s <= TARGET_HI:
            new_counts[s - TARGET_LO] -= 1

    # Add contributions of x_new
    s = x_new + x_new
    if TARGET_LO <= s <= TARGET_HI:
        new_counts[s - TARGET_LO] += 1

    for j in range(n):
        if j == i:
            continue
        aj = A[j]
        s = x_new + aj
        if TARGET_LO <= s <= TARGET_HI:
            new_counts[s - TARGET_LO] += 1

    # Check validity
    if np.min(new_counts) <= 0:
        return None

    peak = np.max(new_counts)
    return (new_counts, peak)

def _local_search(A, max_time=50.0):
    """
    Perform swap-based local search to minimize peak multiplicity.
    Returns (best_A, best_peak).
    """
    A = sorted(A)
    A_set = set(A)
    counts = _pair_counts(A)

    if not _is_valid(counts):
        return (A, float('inf'))

    current_peak = _get_peak(counts)
    best_A = list(A)
    best_peak = current_peak

    n = len(A)
    improved = True
    stall_count = 0
    max_stall = 100  # Allow some neutral moves or stalls before giving up on this configuration

    while improved and _elapsed() < max_time:
        improved = False
        stall_count += 1

        # Shuffle indices to break symmetry in tie-breaking
        indices = list(range(n))
        random.shuffle(indices)

        for i in indices:
            if _elapsed() > max_time:
                break
            ai = A[i]
            
            # Try replacing with each value in [0, 128] not in A
            # We can optimize by only trying values that might help, but let's try all first for robustness
            # To speed up, we can iterate through a shuffled list of candidates
            candidates = [x for x in range(0, N + 1) if x not in A_set and x != ai]
            random.shuffle(candidates)
            
            for x_new in candidates:
                result = _try_swap(A, A_set, counts, i, x_new)
                if result is None:
                    continue
                new_counts, new_peak = result
                
                if new_peak < current_peak:
                    # Accept
                    A[i] = x_new
                    A_set.discard(ai)
                    A_set.add(x_new)
                    A.sort()
                    counts = new_counts
                    current_peak = new_peak
                    improved = True
                    stall_count = 0
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_A = list(A)
                    break  # Restart from beginning after a move

        if not improved and stall_count >= max_stall:
            break

    return (best_A, best_peak)

def _greedy_build(rng, max_time=50.0):
    """
    Build a set greedily. Start with empty, add elements that cover targets.
    """
    A = []
    A_set = set()
    counts = np.zeros(NUM_TARGETS, dtype=np.int32)
    
    # Initial anchors to help cover extremes
    # 0 helps cover low targets if paired with high ones
    # 128 helps cover high targets
    # Let's start with a random small set or specific anchors
    # Actually, let's start empty and let greedy fill it.
    
    # To ensure we don't get stuck in a local minimum of "no coverage",
    # we might need to force some initial elements.
    # Let's start with 0 and 128, and maybe 64.
    initial = [0, 128, 64]
    for x in initial:
        if x not in A_set:
            A.append(x)
            A_set.add(x)
    A.sort()
    counts = _pair_counts(A)
    
    while _elapsed() < max_time:
        if _is_valid(counts):
            break
            
        # Find uncovered targets
        uncovered_indices = np.where(counts == 0)[0]
        if len(uncovered_indices) == 0:
            break
            
        # Pick the target with the "hardest" coverage? Or just the first one?
        # Let's pick the target that has the fewest potential pairs from current A + candidates?
        # Simpler: Pick a random uncovered target and find a best x to cover it.
        target_idx = rng.choice(uncovered_indices)
        target_val = TARGET_LO + target_idx
        
        best_x = None
        best_gain = -1
        
        for x in range(0, N + 1):
            if x in A_set:
                continue
            
            # Calculate how many new targets this x would cover
            gain = 0
            # Pairs with existing A
            for y in A:
                s = x + y
                if TARGET_LO <= s <= TARGET_HI:
                    if counts[s - TARGET_LO] == 0:
                        gain += 1
            # Pair with itself
            s = x + x
            if TARGET_LO <= s <= TARGET_HI:
                if counts[s - TARGET_LO] == 0:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_x = x
                
        if best_x is None or best_gain <= 0:
            # If no single element covers a new target, we might be stuck or need to swap.
            # For greedy build, if we can't add, we stop.
            break
            
        A.append(best_x)
        A_set.add(best_x)
        A.sort()
        counts = _pair_counts(A)
        
    return A, counts

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    rng = random.Random(42)
    
    best_A = None
    best_peak = float('inf')
    
    # Strategy:
    # 1. Generate initial sets of various sizes using random sampling and greedy methods.
    # 2. Apply local search to each.
    # 3. Keep the best.
    
    # Sizes to try. Smaller sizes are preferred if they can cover, as they have fewer pairs.
    # However, if a size is too small, it might not be able to cover all targets.
    # Let's try sizes from 15 to 40.
    
    sizes_to_try = [15, 16, 18, 20, 22, 24, 26, 28, 30, 32, 35, 40]
    
    for size in sizes_to_try:
        if _elapsed() > DEADLINE - 5:
            break
            
        # Multiple restarts for each size
        restarts = 3
        
        for r in range(restarts):
            if _elapsed() > DEADLINE - 2:
                break
            
            # Generate initial A
            if r == 0:
                # Greedy start
                A_init, _ = _greedy_build(rng, max_time=DEADLINE - 10)
                # Adjust size if greedy produced a different size
                # If A_init is too big, we might trim it? Or just use it.
                # If A_init is too small, we can add random elements.
                if len(A_init) < size:
                    # Add random elements
                    A_set = set(A_init)
                    while len(A_init) < size:
                        x = rng.randint(0, N)
                        if x not in A_set:
                            A_init.append(x)
                            A_set.add(x)
                elif len(A_init) > size:
                    # Keep the first 'size' elements? Or random sample?
                    # Random sample is better to maintain diversity
                    A_init = rng.sample(A_init, size)
            else:
                # Random start
                A_init = rng.sample(range(0, N + 1), size)
            
            A_init = sorted(A_init)
            
            # Check validity
            counts = _pair_counts(A_init)
            if not _is_valid(counts):
                # Try to fix coverage by adding elements?
                # Or skip this restart?
                # Let's try a quick fix: add elements that cover gaps.
                A_fixed, _ = _greedy_build(rng, max_time=DEADLINE - 10)
                # Merge A_init and A_fixed?
                # Actually, if A_init is invalid, it's likely a bad start.
                # We can try to repair it by adding elements.
                A_temp = list(A_init)
                A_set = set(A_temp)
                counts_temp = counts
                for _ in range(10): # Max 10 additions
                    if _is_valid(counts_temp):
                        break
                    uncovered = np.where(counts_temp == 0)[0]
                    if len(uncovered) == 0:
                        break
                    target_idx = rng.choice(uncovered)
                    target_val = TARGET_LO + target_idx
                    best_x = None
                    best_gain = -1
                    for x in range(0, N + 1):
                        if x in A_set:
                            continue
                        gain = 0
                        for y in A_temp:
                            s = x + y
                            if TARGET_LO <= s <= TARGET_HI and counts_temp[s - TARGET_LO] == 0:
                                gain += 1
                        s = x + x
                        if TARGET_LO <= s <= TARGET_HI and counts_temp[s - TARGET_LO] == 0:
                            gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_x = x
                    if best_x is None or best_gain <= 0:
                        break
                    A_temp.append(best_x)
                    A_set.add(best_x)
                    A_temp.sort()
                    counts_temp = _pair_counts(A_temp)
                
                if _is_valid(counts_temp):
                    A_init = A_temp
                else:
                    continue # Skip this restart if we couldn't fix it
            
            # Local Search
            A_opt, peak = _local_search(A_init, max_time=DEADLINE - 2)
            
            if peak < best_peak:
                best_peak = peak
                best_A = A_opt
                
            # Early termination if peak is very low
            if best_peak <= 3:
                break
                
        if best_peak <= 3:
            break
            
    # Fallback
    if best_A is None:
        # Use a structured set that is guaranteed to work, though peak might be high
        # Even numbers cover even targets, odd numbers cover odd targets?
        # [0, 2, ..., 128] covers all even targets in [0, 256].
        # But we need [128, 256].
        # If A = {0, 2, ..., 128}, sums are even.
        # Targets 129, 131... are odd. So this doesn't work.
        # We need a set that covers both parities.
        # A = {0, 1, 2, ..., 128} works but peak is high.
        # Let's just return a valid set if we failed.
        best_A = list(range(0, 129, 2)) + [1] # Add 1 to cover odd sums?
        # 1+1=2, 1+2=3...
        # Actually, just use the full range or a dense set.
        best_A = list(range(0, 129))
        
    return best_A
