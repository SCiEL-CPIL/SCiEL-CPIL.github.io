import time

# Pre-compute smooth numbers
def _generate_smooth(limit):
    smooth = set()
    a = 1
    while a <= limit:
        b = a
        while b <= limit:
            c = b
            while c <= limit:
                smooth.add(c)
                c *= 5
            b *= 3
        a *= 2
    return sorted(x for x in smooth if x <= limit)

SMOOTH = _generate_smooth(128)
SMOOTH_SET = set(SMOOTH)
N_SMOOTH = len(SMOOTH)

# Pre-compute for speed: for each index i, list of indices j > i such that SMOOTH[j] % SMOOTH[i] != 0
# Actually, we just need to check pairwise in DFS.

def _find_min_terms(target):
    """Find minimum-k antichain sum representation of target using smooth numbers."""
    if target in SMOOTH_SET:
        return [target]
    
    # Search for k from 2 upwards
    for k in range(2, 11):
        result = _dfs(target, 0, [], k)
        if result:
            return result
    return None

def _dfs(rem, start_idx, chosen, k):
    """
    Find k distinct smooth numbers > chosen[-1] (if any) that sum to rem,
    forming an antichain with chosen.
    """
    num_chosen = len(chosen)
    remaining = k - num_chosen
    
    if remaining == 0:
        return []
    
    if remaining == 1:
        # The last number is rem
        if rem not in SMOOTH_SET:
            return None
        if chosen and rem <= chosen[-1]:
            return None
        # Check antichain
        for c in chosen:
            if rem % c == 0 or c % rem == 0:
                return None
        return [rem]
    
    # Pruning: min possible sum
    # The smallest 'remaining' smooth numbers greater than chosen[-1]
    min_possible = 0
    count = 0
    idx = start_idx
    if chosen:
        while idx < N_SMOOTH and SMOOTH[idx] <= chosen[-1]:
            idx += 1
    # We need 'remaining' numbers. If we don't have enough, return None
    # But we can just let the loop handle it or check bounds.
    
    # Average bound
    # Each number must be <= rem // remaining
    # Also, each number must be >= SMOOTH[start_idx] (effectively)
    
    for i in range(start_idx, N_SMOOTH):
        val = SMOOTH[i]
        
        # If val is too large, break
        # The remaining 'remaining - 1' numbers must be >= val (since distinct and sorted)
        # Actually, they must be > val.
        # Min sum of remaining-1 numbers is (val + 1) + ... but smooth numbers are not consecutive.
        # Simple bound: val <= rem // remaining
        if val > rem // remaining:
            break
        
        # Check antichain with chosen
        valid = True
        for c in chosen:
            if val % c == 0 or c % val == 0:
                valid = False
                break
        if not valid:
            continue
        
        # Recurse
        res = _dfs(rem - val, i + 1, chosen + [val], k)
        if res is not None:
            return [val] + res
            
    return None

# Pre-compute all minimal representations at module load time
_PRECOMPUTED = {}

def _precompute_all():
    global _PRECOMPUTED
    for n in range(1, 129):
        ans = _find_min_terms(n)
        if ans is None:
            # Fallback: should not happen for 1..128 with k up to 10
            # Try larger k or a different search
            # For safety, just try k up to 20 with a looser search?
            # Actually, 1..128 are small. k=1,2,3,4 should cover all.
            # Let's verify with a simple exhaustive search for fallback
            ans = _fallback_search(n)
        _PRECOMPUTED[n] = ans

def _fallback_search(target):
    # Very slow, but only if needed
    # Try k=1 to 20
    for k in range(1, 21):
        # Use itertools.combinations for small k?
        # Or just the DFS with k
        res = _dfs(target, 0, [], k)
        if res:
            return res
    return [target] if target in SMOOTH_SET else [] # Should not reach here

def construct():
    # Ensure pre-computation is done
    if not _PRECOMPUTED:
        _precompute_all()
    return [_PRECOMPUTED[n] for n in range(1, 129)]

# Run pre-computation at module load
_start_time = time.time()
_precompute_all()
# print(f"Precomputation took {time.time() - _start_time:.4f}s") # Disabled for production
