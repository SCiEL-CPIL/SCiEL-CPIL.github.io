"""Rewritten program: bitset subset-sum validation + greedy + local search."""
# EVOLVE-BLOCK-START
import time

N_MAX = 512
TARGET = 48

def _subset_sums_bitset(vals):
    """Return a list/bitset where bit s is 1 iff some subset of vals sums to s.
    Uses a Python int as a bitset. vals are positive ints.
    Returns (bitset_int, max_sum)."""
    bits = 1  # bit 0 set (empty subset)
    max_sum = 0
    for v in vals:
        bits |= (bits << v)
        max_sum += v
    return bits, max_sum

def _is_mean_violation(a, others):
    """Check if a is the exact mean of any subset of `others` of size >=2.
    Uses subset-sum bitsets over `others`. For each k in 2..len(others), check
    if k*a is achievable as a subset sum. But we also must ensure the subset has
    exactly k elements, not just sum k*a. Subset-sum by value alone is not enough
    to fix cardinality. So we need DP over (count, sum) or track counts.
    """
    # We need: exists subset T of others, |T|>=2, sum(T)=|T|*a.
    # DP: dp[c] = bitset of achievable sums using exactly c elements.
    m = len(others)
    # dp[c] is an int bitset of sums achievable with exactly c elements.
    dp = [0] * (m + 1)
    dp[0] = 1  # sum 0 with 0 elements
    for v in others:
        # update backwards in c to avoid reusing v
        for c in range(m, 0, -1):
            dp[c] |= (dp[c-1] << v)
    # Now for each k in 2..m, check if bit (k*a) is set in dp[k]
    for k in range(2, m + 1):
        target = k * a
        if target < 0:
            continue
        if (dp[k] >> target) & 1:
            return True
    return False

def _valid_set(S):
    """Check all elements of S: no element is mean of any subset of >=2 others."""
    S = sorted(S)
    for i, a in enumerate(S):
        others = S[:i] + S[i+1:]
        if _is_mean_violation(a, others):
            return False
    return True

def _can_add(S, x):
    """Check if adding x to S keeps the set valid. S must already be valid.
    We need to check: (1) x is not mean of any subset of S, (2) for each a in S,
    a is not mean of any subset of S\{a} U {x} of size>=2.
    """
    # Check x against S
    if _is_mean_violation(x, S):
        return False
    # Check each a in S with x added
    for a in S:
        others = [y for y in S if y != a] + [x]
        if _is_mean_violation(a, others):
            return False
    return True

def construct():
    deadline = time.time() + 55  # safety margin before 60s kill (evaluator uses 90s but keep margin)
    best = []
    
    # Greedy construction: try to add elements in a smart order
    # Start with a seed, then greedily add
    # Try multiple greedy orders and keep the best
    
    candidates_pool = list(range(1, N_MAX + 1))
    
    best_size = 0
    best_set = []
    
    # Greedy pass 1: ascending
    S = []
    for x in candidates_pool:
        if time.time() > deadline:
            break
        if _can_add(S, x):
            S.append(x)
    if len(S) > best_size:
        best_size = len(S)
        best_set = list(S)
    
    # Greedy pass 2: descending
    S = []
    for x in reversed(candidates_pool):
        if time.time() > deadline:
            break
        if _can_add(S, x):
            S.append(x)
    if len(S) > best_size:
        best_size = len(S)
        best_set = list(S)
    
    # Greedy pass 3: even numbers only
    S = []
    for x in range(2, N_MAX + 1, 2):
        if time.time() > deadline:
            break
        if _can_add(S, x):
            S.append(x)
    if len(S) > best_size:
        best_size = len(S)
        best_set = list(S)
    
    # Greedy pass 4: multiples of 3
    S = []
    for x in range(3, N_MAX + 1, 3):
        if time.time() > deadline:
            break
        if _can_add(S, x):
            S.append(x)
    if len(S) > best_size:
        best_size = len(S)
        best_set = list(S)
    
    # Local search: try to improve best_set by swaps and additions
    current = list(best_set)
    current_size = len(current)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current = sorted(current)
        
        # Try to add any missing element
        for x in range(1, N_MAX + 1):
            if x in current:
                continue
            if time.time() > deadline:
                break
            if _can_add(current, x):
                current.append(x)
                current_size += 1
                improved = True
                current = sorted(current)
                break
        
        if time.time() > deadline:
            break
        
        # Try swaps: remove one, add one (to potentially enable more additions)
        # Only try if we want to maintain size but improve structure
        if len(current) < TARGET:
            # Try to find a swap that allows a net gain
            for i, a in enumerate(current):
                if time.time() > deadline:
                    break
                rest = current[:i] + current[i+1:]
                for x in range(1, N_MAX + 1):
                    if x in current:
                        continue
                    # Check if rest + x is valid
                    if _can_add(rest, x):
                        # Now try to add more
                        new_set = rest + [x]
                        new_set = sorted(new_set)
                        # Try to greedily add more
                        added = 0
                        for y in range(1, N_MAX + 1):
                            if y in new_set:
                                continue
                            if time.time() > deadline:
                                break
                            if _can_add(new_set, y):
                                new_set.append(y)
                                new_set = sorted(new_set)
                                added += 1
                        if len(new_set) > len(current):
                            current = new_set
                            current_size = len(current)
                            improved = True
                            break
                if improved:
                    break
    
    # Final check: ensure validity
    if _valid_set(current):
        best_set = current
    
    return best_set
# EVOLVE-BLOCK-END
