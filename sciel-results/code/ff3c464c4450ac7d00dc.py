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
            
            # Feasibility check: Can we sum k distinct smooth numbers to n?
            # Minimum sum is sum of k smallest smooth numbers (1, 2, 3, ...)
            # Note: The k smallest smooth numbers are just the first k in _SMOOTH (since 1,2,3,4,5,6... are smooth)
            # Actually, _SMOOTH is sorted, so _PREFIX[k] is the sum of the k smallest.
            if k < _N:
                min_possible_sum = _PREFIX[k]
                if n < min_possible_sum:
                    # If n is smaller than the sum of the k smallest distinct smooth numbers,
                    # it's impossible. But wait, we just need ANY k distinct.
                    # The smallest k distinct positive integers are 1..k.
                    # Are 1..k always smooth? No. But the smallest k smooth numbers are a subset.
                    # Actually, if n < sum(1..k) = k(k+1)/2, it's impossible for ANY distinct positive integers.
                    # So if n < k(k+1)/2, break?
                    # No, we are looking for k terms. If n is too small, we can't pick k distinct positive integers.
                    # The minimum sum of k distinct positive integers is k(k+1)/2.
                    if n < k * (k + 1) // 2:
                        break 
                    
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
    """
    if k == 1:
        if target in _SMOOTH_SET:
            return [target]
        return None

    # DFS with pruning
    # State: remaining_k, remaining_sum, available_mask, current_path
    # available_mask: bits for indices that can still be picked.
    
    # We want to find ANY solution, so we can stop at the first one.
    # To find the "smallest" or "most compact" solution isn't required, just any valid one for this k.
    # However, the order of search might affect performance.
    # Sorting candidates by value descending often helps prune "sum too large" branches early?
    # Or ascending to prune "sum too small"?
    # Let's stick to index order but use strong sum pruning.
    
    # Pre-compute for each possible available_mask? No, mask is 2^35.
    # We will extract available indices in the DFS.
    
    def dfs(remaining_k, remaining_sum, available_mask, current_path):
        if remaining_k == 0:
            if remaining_sum == 0:
                return True
            return False
        
        if remaining_k > 0 and remaining_sum <= 0:
            return False
            
        # Get available indices
        # Optimization: Instead of building a list every time, we can iterate bits.
        # But N is small (~35), so list comprehension is fine.
        avail_indices = []
        m = available_mask
        while m:
            # Get lowest set bit
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            avail_indices.append(idx)
            m ^= lsb
            
        if len(avail_indices) < remaining_k:
            return False
            
        # Pruning bounds
        # Sort avail_indices by value to get min/max sums?
        # _SMOOTH is sorted ascending, so avail_indices sorted by index = sorted by value.
        # So avail_indices is already sorted by value ascending if we iterate bits 0..N-1.
        # But `while m` extracts in increasing index order.
        # So avail_indices[0] is smallest, avail_indices[-1] is largest.
        
        # Min sum: sum of smallest remaining_k available
        min_sum = sum(_SMOOTH[i] for i in avail_indices[:remaining_k])
        if remaining_sum < min_sum:
            return False
            
        # Max sum: sum of largest remaining_k available
        max_sum = sum(_SMOOTH[i] for i in avail_indices[-remaining_k:])
        if remaining_sum > max_sum:
            return False
        
        # Iterate candidates
        # Heuristic: Try larger values first? Or smaller?
        # If we try larger values first, we might hit the target sum faster?
        # Let's try iterating in reverse order (largest to smallest) to see if it helps.
        # Actually, standard DFS order is usually fine.
        # Let's stick to increasing order (smallest to largest) to keep the "min_sum" logic consistent with "prefix" thinking?
        # No, the pruning is done before the loop.
        
        for i in avail_indices:
            v = _SMOOTH[i]
            if v > remaining_sum:
                continue
                
            # Check if picking i is valid with current path?
            # The mask `available_mask` already ensures i is compatible with all previously picked items.
            # Because when we picked a previous item, we masked out its incompatibles.
            
            current_path.append(i)
            
            new_mask = available_mask & ~_INCOMPAT_MASK[i]
            
            if dfs(remaining_k - 1, remaining_sum - v, new_mask, current_path):
                return True
                
            current_path.pop()
            
        return False

    # Initial mask: all bits set
    full_mask = (1 << _N) - 1
    path = []
    
    if dfs(k, target, full_mask, path):
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
        if b in _SMOOTH_SET and a != b and a % b != 0 and b % a != 0:
            return [a, b]
    # Try 3 terms
    # Brute force for fallback
    for a in _SMOOTH:
        if a >= n: break
        for b in _SMOOTH:
            if b <= a: continue
            if a + b >= n: break
            c = n - a - b
            if c in _SMOOTH_SET and c > b:
                # Check divisibility
                if a % b != 0 and b % a != 0 and a % c != 0 and c % a != 0 and b % c != 0 and c % b != 0:
                    return [a, b, c]
    # Last resort: return a single term if possible (invalid if not smooth, but prevents crash)
    # The problem guarantees a solution exists.
    return [1] if n == 1 else [n]

# EVOLVE-BLOCK-END
