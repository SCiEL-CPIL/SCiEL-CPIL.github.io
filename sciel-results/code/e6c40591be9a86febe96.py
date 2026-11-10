"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _is_valid_bitset(S):
    """Check validity using bitset representation of subset sums per cardinality.
    Returns True if no two different cardinalities share a sum."""
    n = len(S)
    if n == 0:
        return True
    max_sum = sum(S)
    # sums_by_card[k] is a bitset (int) where bit s is set if sum s is achievable with k elements
    sums_by_card = [0] * (n + 1)
    sums_by_card[0] = 1  # bit 0 set: empty subset has sum 0
    
    for val in S:
        # Process in reverse to avoid using the same element twice
        for k in range(n - 1, -1, -1):
            if sums_by_card[k] != 0:
                shifted = sums_by_card[k] << val
                sums_by_card[k + 1] |= shifted
    
    # Check for cross-cardinality collisions
    for k1 in range(n + 1):
        if sums_by_card[k1] == 0:
            continue
        for k2 in range(k1 + 1, n + 1):
            if sums_by_card[k2] == 0:
                continue
            if (sums_by_card[k1] & sums_by_card[k2]) != 0:
                return False
    return True

def _count_conflicts_bitset(S):
    """Count total number of cross-cardinality sum collisions using bitsets."""
    n = len(S)
    if n == 0:
        return 0
    sums_by_card = [0] * (n + 1)
    sums_by_card[0] = 1
    
    for val in S:
        for k in range(n - 1, -1, -1):
            if sums_by_card[k] != 0:
                sums_by_card[k + 1] |= (sums_by_card[k] << val)
    
    total_conflicts = 0
    for k1 in range(n + 1):
        if sums_by_card[k1] == 0:
            continue
        for k2 in range(k1 + 1, n + 1):
            if sums_by_card[k2] == 0:
                continue
            total_conflicts += bin(sums_by_card[k1] & sums_by_card[k2]).count('1')
    return total_conflicts

def _greedy_conflict_removal(initial_set, deadline):
    """Start with initial_set, iteratively remove the element whose removal
    minimizes total cross-cardinality conflicts. Then try to add back elements."""
    current = list(initial_set)
    used = set(current)
    all_candidates = list(range(1, 257))
    
    # Phase 1: Remove elements that cause the most conflicts
    changed = True
    while changed and time.time() < deadline:
        changed = False
        n = len(current)
        if n <= 1:
            break
        
        best_idx = -1
        best_conflicts_after = _count_conflicts_bitset(current)
        
        for i in range(n):
            trial = current[:i] + current[i+1:]
            c = _count_conflicts_bitset(trial)
            if c < best_conflicts_after:
                best_conflicts_after = c
                best_idx = i
        
        if best_idx >= 0:
            removed = current.pop(best_idx)
            used.discard(removed)
            changed = True
    
    # Phase 2: Try to add back elements greedily
    improved = True
    while improved and time.time() < deadline:
        improved = False
        candidates = [x for x in all_candidates if x not in used]
        candidates.sort()
        
        for c in candidates:
            if time.time() > deadline:
                break
            trial = current + [c]
            trial.sort()
            if _is_valid_bitset(trial):
                current = trial
                used.add(c)
                improved = True
                break
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # Strategy 1: Start with full range and use conflict-driven removal
    initial_full = list(range(1, 257))
    result1 = _greedy_conflict_removal(initial_full, deadline - 10)
    
    # Strategy 2: Start with powers-of-2-like structure and extend
    # Use geometric sequence as seed
    geo_seed = []
    v = 1
    while v <= 256:
        geo_seed.append(v)
        v *= 2
    # geo_seed = [1, 2, 4, 8, 16, 32, 64, 128]
    result2 = _greedy_conflict_removal(geo_seed, deadline - 5)
    
    # Strategy 3: Try different starting points with greedy addition (descending)
    result3 = []
    if time.time() < deadline - 3:
        for start_val in [256, 200, 150, 100, 50, 20, 10, 5, 1]:
            if time.time() > deadline - 3:
                break
            current = [start_val]
            used = {start_val}
            
            for val in range(256, 0, -1):
                if val in used or time.time() > deadline - 3:
                    continue
                trial = current + [val]
                trial.sort()
                if _is_valid_bitset(trial):
                    current = trial
                    used.add(val)
            
            if len(current) > len(result3):
                result3 = current
    
    # Pick the best result
    best = max([result1, result2, result3], key=len)
    
    # Final verification and cleanup
    if len(best) > 0:
        # Ensure distinct and in range
        seen = set()
        cleaned = []
        for x in best:
            if isinstance(x, int) and not isinstance(x, bool) and 1 <= x <= 256 and x not in seen:
                cleaned.append(x)
                seen.add(x)
        best = cleaned
        
        # Exact verification
        if best and not _is_valid_bitset(best):
            # Fallback to a known small valid set
            best = [1, 2, 4, 8, 16, 32, 64, 128]
    
    return list(best)
# EVOLVE-BLOCK-END
