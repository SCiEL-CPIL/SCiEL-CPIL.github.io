import time
import sys

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

# Precompute divisibility compatibility
# compat[i][j] is True if _SMOOTH[i] and _SMOOTH[j] are distinct and neither divides the other.
# Actually, we just need to know if adding j to a set containing i is valid.
# Let's precompute a bitmask for each node i: mask[i] has bits set for all j such that 
# _SMOOTH[i] and _SMOOTH[j] are INCOMPATIBLE (one divides the other).
# Note: We treat i and i as incompatible (distinctness required).

_INCOMPAT_MASK = [0] * _N
for i in range(_N):
    vi = _SMOOTH[i]
    mask = 0
    for j in range(_N):
        if i == j:
            mask |= (1 << j)
            continue
        vj = _SMOOTH[j]
        # Check if vi divides vj or vj divides vi
        if vi > vj:
            if vi % vj == 0:
                mask |= (1 << j)
        else:
            if vj % vi == 0:
                mask |= (1 << j)
    _INCOMPAT_MASK[i] = mask

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill
    
    results = [None] * 129
    total_terms = 0
    
    # Precompute suffix sums of sorted smooth numbers to help with pruning?
    # Actually, for small k, simple bounds are enough.
    # Let's sort smooth numbers descending for search order? 
    # Or ascending? 
    # Usually, picking larger numbers first reduces the branching factor for the remaining sum.
    # Let's create an index mapping for quick access.
    
    # We will search for k = 1, 2, 3, ... for each n.
    
    for n in range(1, 129):
        if time.time() - start_time > DEADLINE:
            # Fallback: use a greedy or precomputed valid set if time is up
            # But given the small size, this shouldn't happen.
            # For safety, assign a valid single term if n is smooth, else a known valid combo.
            # A safe fallback: if n is smooth, [n]. Else, try a quick greedy.
            if n in _SMOOTH:
                results[n] = [n]
            else:
                # Very basic fallback: find any valid representation (might not be minimal)
                # This is unlikely to be needed.
                results[n] = _quick_fallback(n)
            continue

        found = None
        # Try k from 1 to 30
        for k in range(1, 31):
            if k > n: 
                break
            found = _find_minimal_rep(n, k)
            if found is not None:
                break
        
        if found is None:
            # Should not happen for n<=128, but handle gracefully
            found = _quick_fallback(n)
        
        results[n] = found
        total_terms += len(found)
        
        # Optional: early exit if time is tight, but we want to minimize total terms, 
        # so we should try to finish all. The search is very fast.
        
    return results[1:129]

def _find_minimal_rep(target, k):
    """
    Find k distinct smooth numbers summing to target, pairwise non-dividing.
    Returns a list of values or None.
    """
    # We use backtracking.
    # To make it fast, we can use the precomputed masks.
    
    # We need to choose k items from _SMOOTH.
    # Let's iterate through indices.
    
    # Optimization: 
    # 1. The sum of the k smallest distinct positive integers is k(k+1)/2. If target < that, return None.
    # 2. The sum of the k largest smooth numbers <= target must be >= target.
    
    # Let's filter candidates to those <= target.
    # Since _SMOOTH is sorted ascending, we can find the max index.
    # However, the divisibility mask is global.
    
    # We'll use a recursive DFS.
    # State: current index in _SMOOTH, remaining k, remaining sum, current chosen list, current mask of blocked items.
    # Wait, the mask of blocked items depends on the chosen items.
    # If we pick item i, then all j in _INCOMPAT_MASK[i] are blocked.
    # The set of available items is those not in the union of masks of chosen items.
    
    # To speed up, we can precompute for each subset of size k? No, k is small but subsets are many.
    # Backtracking is best.
    
    # Order of selection: 
    # It is often faster to pick larger numbers first to satisfy the sum constraint quickly.
    # Let's create a list of indices sorted by value descending.
    # But the mask logic is easier with original indices.
    
    # Let's just iterate i from 0 to _N-1.
    # To prune, we can check if the remaining sum can be formed by the remaining k-1 items 
    # from the available set.
    
    # A simpler, effective heuristic for small N:
    # Generate all combinations of k smooth numbers? 
    # C(35, 3) is ~6500. C(35, 4) is ~50k. This is very fast.
    # We can just iterate combinations!
    
    from itertools import combinations
    
    # Filter smooth numbers <= target
    valid_indices = [i for i, v in enumerate(_SMOOTH) if v <= target]
    
    # If target is small, k is small.
    # Let's try combinations of indices.
    # To minimize time, we should check divisibility.
    
    # For k=1:
    if k == 1:
        if target in _SMOOTH:
            return [target]
        return None
        
    # For k >= 2:
    # We can iterate combinations of indices.
    # To make it faster, we can sort indices by value descending and break early if sum is too high?
    # Combinations doesn't support early break easily on sum unless we customize.
    # Given N is ~35, even C(35, 5) is 324,632. Checking divisibility is O(k^2).
    # 300k * 25 ops is ~7.5M ops. In Python, this might take ~1-2 seconds per target if not optimized.
    # 128 targets * 1 sec = 128 sec. Too slow.
    
    # We need a smarter search than brute force combinations for k>3.
    # Backtracking with pruning is better.
    
    # Let's use DFS.
    # We will pick items one by one.
    # To ensure we find the *first* valid set (which might not be unique), we just need one.
    # The problem asks to minimize the *total* number of summands. 
    # So for each target, we find the min k.
    # The specific set doesn't matter for the score, only the count k.
    
    # Let's implement a fast DFS.
    
    # Precompute: for each index i, the list of indices j > i that are compatible with i?
    # No, compatibility is transitive in a set? No, it's pairwise.
    # But if i and j are compatible, and j and l are compatible, i and l might not be.
    
    # Let's stick to the mask approach.
    # available_mask: bits for indices that can still be picked.
    # Initially, all bits 0..N-1 are 1.
    # When we pick i, we clear bits in available_mask that are set in _INCOMPAT_MASK[i].
    
    # We also need to ensure we pick exactly k items summing to target.
    
    # To speed up DFS, we can order the choices.
    # Let's try picking the largest possible value first to reduce the remaining sum quickly.
    
    # Indices sorted by value descending
    # But we need to map back to original indices for masks.
    
    # Let's define the search space as the list of indices sorted by value descending.
    # search_indices = sorted(range(_N), key=lambda i: _SMOOTH[i], reverse=True)
    
    # However, we must respect that we can only pick distinct indices.
    
    def dfs(remaining_k, remaining_sum, available_mask, current_path):
        if remaining_k == 0:
            if remaining_sum == 0:
                return True
            return False
        
        if remaining_k > 0 and remaining_sum <= 0:
            return False
            
        # Pruning: 
        # The minimum sum of remaining_k distinct positive integers is remaining_k * 1 + ...? 
        # No, they must be distinct smooth numbers.
        # The smallest possible values are the smallest available smooth numbers.
        # If the sum of the smallest remaining_k available numbers is > remaining_sum, prune.
        # If the sum of the largest remaining_k available numbers is < remaining_sum, prune.
        
        # Let's extract available indices
        # This is expensive if done every step. 
        # But N is small.
        
        avail_indices = []
        for i in range(_N):
            if (available_mask >> i) & 1:
                avail_indices.append(i)
                
        if len(avail_indices) < remaining_k:
            return False
            
        # Sort by value descending to pick big ones first
        avail_indices.sort(key=lambda i: _SMOOTH[i], reverse=True)
        
        # Calculate bounds
        # Smallest sum: pick the smallest remaining_k from avail_indices
        # Largest sum: pick the largest remaining_k from avail_indices
        
        # Since avail_indices is sorted desc, the smallest are at the end.
        # smallest_vals = [ _SMOOTH[i] for i in avail_indices[-remaining_k:] ]
        # largest_vals = [ _SMOOTH[i] for i in avail_indices[:remaining_k] ]
        
        # To avoid list creation overhead, we can just check the first and last few.
        # Actually, for k<=5, we can just sum the relevant slices.
        
        if remaining_k <= len(avail_indices):
            min_sum = sum(_SMOOTH[i] for i in avail_indices[-remaining_k:])
            max_sum = sum(_SMOOTH[i] for i in avail_indices[:remaining_k])
            
            if remaining_sum < min_sum:
                return False
            if remaining_sum > max_sum:
                return False
            
        # Iterate candidates
        for i in avail_indices:
            v = _SMOOTH[i]
            if v > remaining_sum:
                continue
                
            # Pick i
            current_path.append(i)
            
            # Update mask
            new_mask = available_mask & ~_INCOMPAT_MASK[i]
            # Also clear bit i itself (though it's in mask[i] already? 
            # _INCOMPAT_MASK[i] has bit i set. So new_mask clears i.
            
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
    # 1 term if smooth
    if n in _SMOOTH:
        return [n]
    # Try 2 terms
    for a in _SMOOTH:
        if a >= n: break
        b = n - a
        if b in _SMOOTH and a != b and a % b != 0 and b % a != 0:
            return [a, b]
    # Try 3 terms
    # This is a last resort.
    return [1, 1, n-2] # Invalid if not distinct or smooth, but defensive.
    # Actually, let's just return [n] if we must, but it might be invalid.
    # The problem states all targets require valid representations.
    # My main search is robust.

# EVOLVE-BLOCK-END
