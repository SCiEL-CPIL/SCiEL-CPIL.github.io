"""Rewritten program with structured sampling + local search for large mean-free subsets."""
# EVOLVE-BLOCK-START
import time
import random
import sys

# --- Constants ---
N_MAX = 512
TARGET_SIZE = 48
MAX_TIME = 85.0  # safety margin within 90s wall-clock

def _is_valid_set(S):
    """
    Check if S (list of distinct positive ints) is valid:
    No element is the exact arithmetic mean of any subset of at least two OTHER elements.
    Uses subset-sum DP with bitsets (Python ints as bitsets).
    """
    n = len(S)
    if n < 3:
        return True
    
    S_set = set(S)
    
    for idx, a in enumerate(S):
        # T = S \ {a}
        T = [x for x in S if x != a]
        n_t = len(T)
        
        # We need: exists k>=2 such that sum(subset of T of size k) == k*a
        # Equivalently: let U = [x - a for x in T]. Exists k>=2 such that sum(subset of U of size k) == 0.
        
        # Bitset DP: dp[k] is a bitset where bit s is set iff sum s achievable with k elements from U.
        # But sums can be negative. Shift by offset.
        # Min sum of k elements: sum of k smallest in U.
        # Max sum of k elements: sum of k largest in U.
        
        # Simpler: just do DP on original sums and check k*a.
        # Max sum in T: sum(T) <= 47*512 = 24064. Bitset size ~24064 bits ~ 3KB.
        # For each a (up to 48 times), DP over 47 elements.
        
        max_possible = sum(T)
        # dp[k]: bitset of achievable sums using k elements
        dp = [0] * (n_t + 1)
        dp[0] = 1  # sum 0 with 0 elements
        
        for x in T:
            for k in range(n_t, 0, -1):
                if dp[k-1]:
                    dp[k] |= (dp[k-1] << x)
        
        # Check for k >= 2
        for k in range(2, n_t + 1):
            target = k * a
            if target > max_possible:
                break
            if (dp[k] >> target) & 1:
                return False
    
    return True

def _is_mean_free_candidate(candidate, existing_set):
    """
    Quick check: is candidate the mean of any subset of existing_set?
    AND: does adding candidate make any existing element a mean?
    For efficiency during greedy construction, we do a partial check.
    
    Actually, for correctness we need full check. But during greedy, we can be smarter:
    - Check 1: candidate is NOT the mean of any subset of existing_set of size >= 2.
    - Check 2: For each y in existing_set, y is NOT the mean of any subset of existing_set U {candidate} of size >= 2.
    
    For check 2, the only new subsets that include candidate. So:
    y is mean of (some subset of existing_set) U {candidate} of total size >= 2.
    This means: (sum(subset of existing_set) + candidate) / k = y for some k >= 2.
    => sum(subset) = k*y - candidate, where the subset has size k-1 >= 1 (since total size k includes candidate).
    Wait, if total size is k, and one element is candidate, then subset of existing_set has size k-1.
    k >= 2, so k-1 >= 1.
    
    So for each y in existing_set, we need to check if exists subset of existing_set of size j >= 1
    such that sum(subset) + candidate = (j+1)*y
    => sum(subset) = (j+1)*y - candidate
    
    This is still expensive. Let's just do the full _is_valid_set for simplicity and correctness.
    With N<=48, it should be fast enough.
    """
    return True  # Placeholder, we'll use full check

def _structured_initial_set():
    """
    Create an initial set using structured sampling to avoid arithmetic progressions.
    Use numbers that are spaced out and not in simple APs.
    Strategy: Start with numbers near 512, 511, 510... but skip some to break patterns.
    Better: Use a set of numbers with large gaps.
    
    Let's try: pick every other number from a high range, then add lower numbers.
    Or use a golden-ratio-like spacing.
    
    Actually, let's just do a greedy from high to low, but with a bias towards numbers
    that are not multiples of small integers relative to others.
    """
    S = []
    S_set = set()
    
    # First, add some large numbers
    for x in range(512, 480, -1):
        if len(S) >= 15:
            break
        new_S = S + [x]
        if _is_valid_set(new_S):
            S.append(x)
            S_set.add(x)
    
    # Now add medium numbers
    for x in range(300, 150, -1):
        if len(S) >= 30:
            break
        if x in S_set:
            continue
        new_S = S + [x]
        if _is_valid_set(new_S):
            S.append(x)
            S_set.add(x)
    
    # Now add small numbers
    for x in range(100, 1, -1):
        if len(S) >= TARGET_SIZE:
            break
        if x in S_set:
            continue
        new_S = S + [x]
        if _is_valid_set(new_S):
            S.append(x)
            S_set.add(x)
    
    return S

def _local_search(S, deadline):
    """
    Local search: try add-back, swap, and removal moves to improve the set.
    """
    S_set = set(S)
    best_set = list(S)
    best_size = len(S)
    stuck_counter = 0
    max_stuck = 100
    
    while time.time() < deadline:
        move_made = False
        
        # Try add-back: add a random number not in S
        if len(S) < TARGET_SIZE:
            candidates = [x for x in range(1, N_MAX + 1) if x not in S_set]
            if candidates:
                # Prefer larger numbers
                candidates.sort(reverse=True)
                # Try a few random ones
                for attempt in range(5):
                    if not candidates:
                        break
                    idx = random.randint(0, min(len(candidates)-1, 20))
                    candidate = candidates[idx]
                    candidates.pop(idx)
                    S_set.add(candidate)
                    new_S = S + [candidate]
                    if _is_valid_set(new_S):
                        S = new_S
                        S_set = set(S)
                        move_made = True
                        stuck_counter = 0
                        if len(S) > best_size:
                            best_set = list(S)
                            best_size = len(S)
                        break
                    else:
                        S_set.discard(candidate)
        
        # Try swap: replace one element with another
        if len(S) > 0:
            for _ in range(3):
                if time.time() > deadline:
                    break
                idx = random.randint(0, len(S)-1)
                x = S[idx]
                y = random.randint(1, N_MAX)
                if y == x or y in S_set:
                    continue
                new_S = [z for z in S if z != x] + [y]
                if _is_valid_set(new_S):
                    S = new_S
                    S_set = set(S)
                    move_made = True
                    stuck_counter = 0
        
        if not move_made:
            stuck_counter += 1
            # Shock move: remove a random element
            if stuck_counter >= max_stuck and len(S) > 0:
                idx = random.randint(0, len(S)-1)
                x = S.pop(idx)
                S_set.discard(x)
                stuck_counter = 0
                move_made = True
        
        if best_size >= TARGET_SIZE:
            break
    
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + MAX_TIME
    
    # Phase 1: Structured initial set
    S_init = _structured_initial_set()
    current_best = list(S_init)
    current_best_size = len(S_init)
    
    # Phase 2: Local search from initial set
    if time.time() < deadline:
        S_ls = _local_search(S_init, deadline)
        if len(S_ls) > current_best_size:
            current_best = S_ls
            current_best_size = len(S_ls)
    
    # Phase 3: Multiple restarts with different random seeds
    for seed in range(20):
        if time.time() >= deadline:
            break
        
        random.seed(seed)
        S_restart = []
        S_restart_set = set()
        
        # Randomized greedy
        candidates = list(range(1, N_MAX + 1))
        random.shuffle(candidates)
        # Bias towards larger numbers
        candidates.sort(key=lambda x: -x + random.randint(0, 100))
        
        for x in candidates:
            if len(S_restart) >= TARGET_SIZE:
                break
            if time.time() >= deadline:
                break
            if x in S_restart_set:
                continue
            new_S = S_restart + [x]
            if _is_valid_set(new_S):
                S_restart.append(x)
                S_restart_set.add(x)
        
        if len(S_restart) > current_best_size:
            current_best = list(S_restart)
            current_best_size = len(S_restart)
        
        # Local search from this restart
        if time.time() < deadline and len(S_restart) < TARGET_SIZE:
            S_ls = _local_search(S_restart, deadline)
            if len(S_ls) > current_best_size:
                current_best = S_ls
                current_best_size = len(S_ls)
        
        if current_best_size >= TARGET_SIZE:
            break
    
    # Final check and return
    return current_best

# EVOLVE-BLOCK-END
