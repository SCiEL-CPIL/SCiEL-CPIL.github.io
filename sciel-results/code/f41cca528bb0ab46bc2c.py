import time

# EVOLVE-BLOCK-START

def _gen_smooth(limit):
    """Generate all 2,3,5-smooth numbers up to limit."""
    vals = set()
    p2 = 1
    while p2 <= limit:
        p3 = p2
        while p3 <= limit:
            p5 = p3
            while p5 <= limit:
                vals.add(p5)
                p5 *= 5
            p3 *= 3
        p2 *= 2
    return sorted(v for v in vals if v <= limit)

# Precompute smooth numbers up to 128
_SMOOTH = _gen_smooth(128)
_N = len(_SMOOTH)
_SMOOTH_SET = set(_SMOOTH)

# Precompute divisibility compatibility masks
# _INCOMPAT_MASK[i] has bits set for all j such that _SMOOTH[i] and _SMOOTH[j] are INCOMPATIBLE
# (one divides the other, or they are the same index)
_INCOMPAT_MASK = [0] * _N
for i in range(_N):
    vi = _SMOOTH[i]
    mask = 1 << i  # Incompatible with itself (distinctness)
    for j in range(_N):
        if i == j:
            continue
        vj = _SMOOTH[j]
        if vi > vj:
            if vi % vj == 0:
                mask |= (1 << j)
        else:
            if vj % vi == 0:
                mask |= (1 << j)
    _INCOMPAT_MASK[i] = mask

# Precompute prefix sums for pruning
# _PREFIX[i] = sum of the first i smallest smooth numbers (indices 0..i-1)
_PREFIX = [0] * (_N + 1)
for i in range(_N):
    _PREFIX[i + 1] = _PREFIX[i] + _SMOOTH[i]

# Precompute suffix sums for pruning (sum of largest k available)
# Since we pick from a subset, we need dynamic suffix sums.
# But for initial bounds, we can use the global largest/smallest.
# Let's precompute sorted indices by value descending for search order.
# However, the mask logic requires original indices.

# Optimization: Precompute for each k, the minimum and maximum possible sums
# of k distinct smooth numbers. This is a global bound, not per-subset.
# Min sum for k: sum of first k smallest smooth numbers.
# Max sum for k: sum of last k largest smooth numbers.
_MIN_SUM_K = [0] * (_N + 1)
_MAX_SUM_K = [0] * (_N + 1)
for k in range(1, _N + 1):
    _MIN_SUM_K[k] = _PREFIX[k]
    _MAX_SUM_K[k] = sum(_SMOOTH[-k:])

def construct():
    start_time = time.time()
    DEADLINE = 80.0  # Safety margin before 90s kill

    results = [None] * 129
    total_terms = 0

    for n in range(1, 129):
        if time.time() - start_time > DEADLINE:
            # Fallback for remaining targets if time is up
            results[n] = _quick_fallback(n)
            total_terms += len(results[n])
            continue

        found = None
        # Try k from 1 to 30
        for k in range(1, 31):
            if k > n:
                break
            
            # Quick global bound check
            if n < _MIN_SUM_K[k] or n > _MAX_SUM_K[k]:
                continue

            found = _find_minimal_rep(n, k)
            if found is not None:
                break
        
        if found is None:
            # Should not happen for n<=128, but handle gracefully
            found = _quick_fallback(n)
        
        results[n] = found
        total_terms += len(found)
        
    return results[1:129]

def _find_minimal_rep(target, k):
    """
    Find k distinct smooth numbers summing to target, pairwise non-dividing.
    Returns a list of values or None.
    Uses bitmask-based DFS with strong pruning.
    """
    # We use backtracking.
    # State: remaining_k, remaining_sum, available_mask, current_path
    
    # Pre-filter: only consider smooth numbers <= target.
    # But the mask is global. We can just check value > remaining_sum in loop.
    
    # To speed up, we can precompute the list of available indices sorted by value descending.
    # But availability changes.
    
    # Let's use the mask approach.
    # Initial mask: all bits set.
    full_mask = (1 << _N) - 1
    path = []
    
    # We want to find ANY valid set. The order of trying candidates matters for speed.
    # Trying larger values first usually reduces the remaining sum faster, helping prune.
    # Let's create a list of indices sorted by value descending.
    # indices_desc = sorted(range(_N), key=lambda i: _SMOOTH[i], reverse=True)
    
    # However, we must only pick from available_mask.
    # Iterating over all N indices and checking mask is O(N).
    # N is ~35. This is fast.
    
    def dfs(remaining_k, remaining_sum, available_mask):
        if remaining_k == 0:
            return remaining_sum == 0
        
        if remaining_k > 0 and remaining_sum <= 0:
            return False
            
        # Pruning 1: Not enough items available
        # Count bits in available_mask
        # Python 3.10+ has int.bit_count()
        if available_mask.bit_count() < remaining_k:
            return False
            
        # Pruning 2: Sum bounds
        # Extract available values?
        # To get tight bounds, we need the smallest and largest available values.
        # Let's extract the list of available indices.
        avail_indices = []
        m = available_mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            idx = lsb.bit_length() - 1
            avail_indices.append(idx)
            m ^= lsb
            
        if len(avail_indices) < remaining_k:
            return False
            
        # Sort by value descending to pick big ones first
        avail_indices.sort(key=lambda i: _SMOOTH[i], reverse=True)
        
        # Calculate min and max possible sums from avail_indices
        # Min sum: sum of smallest remaining_k
        # Max sum: sum of largest remaining_k
        # Since avail_indices is sorted desc:
        # Largest are at the start, smallest at the end.
        
        # Optimization: If remaining_k is small, just sum the slices.
        # This is O(k) per node.
        
        min_sum = sum(_SMOOTH[i] for i in avail_indices[-remaining_k:])
        max_sum = sum(_SMOOTH[i] for i in avail_indices[:remaining_k])
        
        if remaining_sum < min_sum or remaining_sum > max_sum:
            return False
            
        # Iterate candidates
        for i in avail_indices:
            v = _SMOOTH[i]
            if v > remaining_sum:
                # Since sorted desc, if v > remaining_sum, all subsequent (smaller) might be valid?
                # No, sorted desc means next are smaller. So we can't break.
                # But if v is too big, we skip.
                continue
                
            # Check if i is in available_mask (it is, by construction)
            # Pick i
            path.append(i)
            
            # Update mask
            new_mask = available_mask & ~_INCOMPAT_MASK[i]
            
            if dfs(remaining_k - 1, remaining_sum - v, new_mask):
                return True
                
            path.pop()
            
        return False
    
    if dfs(k, target, full_mask):
        return [_SMOOTH[i] for i in path]
    else:
        return None

def _quick_fallback(n):
    # If the main search fails (shouldn't happen), return a valid representation.
    if n in _SMOOTH_SET:
        return [n]
    
    # Try 2 terms
    for a in _SMOOTH:
        if a >= n: break
        b = n - a
        if b in _SMOOTH_SET and a != b:
            # Check divisibility
            if a % b != 0 and b % a != 0:
                return [a, b]
                
    # Try 3 terms
    # This is a last resort.
    # We can just call the main search with a higher k limit?
    # Or a simple greedy.
    # Given the constraints, 3 terms should always work for n<=128.
    # Let's try a few combinations.
    for a in _SMOOTH:
        if a >= n: break
        for b in _SMOOTH:
            if b >= a: break # distinct and ordered? Not strictly needed but reduces work
            if a + b >= n: break
            c = n - a - b
            if c in _SMOOTH_SET and c != a and c != b:
                # Check pairwise divisibility
                if (a % b != 0 and b % a != 0 and
                    a % c != 0 and c % a != 0 and
                    b % c != 0 and c % b != 0):
                    return [a, b, c]
                    
    # Should not reach here
    return [1] if n == 1 else [n] # Defensive, might be invalid if n not smooth

# EVOLVE-BLOCK-END
