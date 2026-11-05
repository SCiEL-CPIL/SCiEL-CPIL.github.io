import time
from functools import lru_cache

def _generate_smooth(max_val):
    """Generate all 2^a * 3^b * 5^c numbers up to max_val, sorted."""
    smooth = set()
    a = 1
    while a <= max_val:
        b = a
        while b <= max_val:
            c = b
            while c <= max_val:
                smooth.add(c)
                c *= 5
            b *= 3
        a *= 2
    return sorted(smooth)

def construct():
    start_time = time.time()
    deadline = 85.0
    
    smooth_all = _generate_smooth(128)
    n_smooth = len(smooth_all)
    smooth_set = set(smooth_all)
    
    # Map value to index
    val_to_idx = {v: i for i, v in enumerate(smooth_all)}
    
    # Precompute conflict masks
    # conflict_mask[i] is a bitmask of indices j such that smooth_all[i] and smooth_all[j] are not in an antichain
    # i.e., one divides the other.
    conflict_mask = [0] * n_smooth
    for i in range(n_smooth):
        vi = smooth_all[i]
        mask = 0
        for j in range(n_smooth):
            if i == j:
                continue
            vj = smooth_all[j]
            if vi % vj == 0 or vj % vi == 0:
                mask |= (1 << j)
        conflict_mask[i] = mask
        
    answers = []
    
    for n in range(1, 129):
        if time.time() - start_time > deadline:
            # Fallback
            answers.append([n] if n in smooth_set else [1, n-1])
            continue
            
        # Candidates for this target n
        # Only smooth numbers <= n can be part of the sum
        cands = [v for v in smooth_all if v <= n]
        cand_indices = [val_to_idx[v] for v in cands]
        m = len(cands)
        
        found = None
        
        # Try k = 1, 2, ...
        for k in range(1, 31):
            if time.time() - start_time > deadline:
                break
                
            # If k == 1, check if n is smooth
            if k == 1:
                if n in smooth_set:
                    found = [n]
                    break
                continue
            
            # For k >= 2, use DFS
            # We need to select k distinct candidates from cands
            # such that they sum to n and form an antichain
            
            # Optimization: if k == 2, check pairs directly?
            # DFS is general enough.
            
            # Sort candidates in descending order? 
            # Usually helps to pick larger numbers first to reduce the search space for the remainder.
            # However, the conflict structure might be better explored with a specific order.
            # Let's stick to the natural order or descending. Descending is often better for sum constraints.
            
            # Let's use the original indices in smooth_all to maintain conflict masks correctly.
            # We will iterate over subsets of cand_indices.
            
            # To speed up DFS, we can pre-filter candidates that are "too large" or "too small"?
            # Not strictly necessary for n <= 128.
            
            # We will use a recursive DFS that picks indices from cand_indices.
            # To avoid permutations, we pick indices in increasing order of their position in cand_indices.
            # But we need to track the forbidden mask.
            
            # Let's define a helper function for DFS
            # State: current index in cand_indices, remaining sum, remaining terms, forbidden mask, current path
            # Path stores the actual values.
            
            # To make it faster, we can use a set of available candidate indices.
            # But bitmask is faster.
            
            # Let's map cand_indices to a local 0..m-1 range?
            # No, conflict_mask is global. We can just use the global indices.
            
            # Pruning:
            # 1. If remaining sum < 0, return.
            # 2. If remaining terms == 0, check if remaining sum == 0.
            # 3. If we have picked k terms, return.
            # 4. If the sum of the largest (k - depth) candidates available is < remaining sum, prune.
            #    This requires knowing which candidates are available.
            #    Available candidates are those not in forbidden_mask and not already used.
            #    Since we pick in increasing index order, "used" is implicit by the index pointer.
            #    But forbidden_mask makes it complex.
            
            # Alternative: Iterate over combinations?
            # For k=2, combinations is fast.
            # For k=3, combinations might be okay if m is small.
            # m is at most ~70. C(70, 3) is ~54k. Very fast.
            # C(70, 4) is ~916k. Still fast.
            # C(70, 5) is ~12M. Might be slow if done for many n.
            # But we stop at the first k found.
            # Most targets will have small k.
            
            # Let's try combinations for small k, and DFS for larger k if needed.
            # Or just DFS with good pruning.
            
            # Let's use DFS with bitmask pruning.
            
            # Precompute suffix sums of cands sorted descending?
            # If we sort cands descending, we can easily bound the max possible sum.
            
            # Let's create a list of (value, global_idx) for candidates, sorted by value descending.
            cand_info = [(v, val_to_idx[v]) for v in cands]
            cand_info.sort(key=lambda x: -x[0])
            
            # Suffix sums for pruning
            suffix_sums = [0] * (m + 1)
            for i in range(m - 1, -1, -1):
                suffix_sums[i] = suffix_sums[i + 1] + cand_info[i][0]
                
            def dfs(start_idx, remaining_sum, terms_left, forbidden_mask, path):
                if terms_left == 0:
                    if remaining_sum == 0:
                        return True
                    return False
                
                if remaining_sum < 0:
                    return False
                    
                # Pruning: even if we take the largest 'terms_left' candidates from start_idx, can we reach remaining_sum?
                # The largest candidates are at the beginning of cand_info (since sorted desc).
                # But we are constrained to indices >= start_idx to avoid permutations.
                # So the max sum we can add is the sum of the first 'terms_left' elements in cand_info[start_idx:].
                # This is not simply suffix_sums because we can't take elements before start_idx.
                # We can precompute prefix sums of the sorted list?
                # Actually, suffix_sums[i] is sum of cand_info[i..m-1].
                # The max sum we can get from indices >= start_idx is sum of the largest 'terms_left' elements in that range.
                # Since the list is sorted descending, these are cand_info[start_idx], ..., cand_info[start_idx + terms_left - 1].
                if start_idx + terms_left - 1 < m:
                    max_possible = sum(cand_info[i][0] for i in range(start_idx, start_idx + terms_left))
                    if max_possible < remaining_sum:
                        return False
                else:
                    # Not enough candidates left
                    return False

                for i in range(start_idx, m):
                    v, g_idx = cand_info[i]
                    if v > remaining_sum:
                        # Since sorted descending, if this one is too big, the rest are smaller?
                        # No, sorted descending means cand_info[i] <= cand_info[i-1].
                        # So if v > remaining_sum, we can't use v.
                        # But smaller v might be usable. So we continue.
                        continue
                        
                    # Check if v conflicts with any chosen term
                    # The forbidden_mask contains bits for all global indices that conflict with chosen terms.
                    # If bit g_idx is set in forbidden_mask, then v conflicts with some chosen term.
                    if (forbidden_mask >> g_idx) & 1:
                        continue
                        
                    # Also, we must ensure v is not already in path?
                    # Since we iterate i from start_idx, and we only pick increasing i, we won't pick the same index twice.
                    # And distinct values are guaranteed by distinct indices.
                    
                    path.append(v)
                    new_forbidden = forbidden_mask | conflict_mask[g_idx]
                    if dfs(i + 1, remaining_sum - v, terms_left - 1, new_forbidden, path):
                        return True
                    path.pop()
                    
                return False

            path = []
            if dfs(0, n, k, 0, path):
                found = path
                break
                
        if found is None:
            # Should not happen for 1..128
            # Fallback to a valid representation
            # Greedy
            rem = n
            res = []
            for v in sorted(cands, reverse=True):
                if v > rem:
                    continue
                valid = True
                for ct in res:
                    if v % ct == 0 or ct % v == 0:
                        valid = False
                        break
                if valid:
                    res.append(v)
                    rem -= v
                    if rem == 0:
                        break
            if rem == 0:
                found = res
            else:
                # Last resort
                found = [n] if n in smooth_set else [1, n-1]
                
        answers.append(found)
        
    return answers
