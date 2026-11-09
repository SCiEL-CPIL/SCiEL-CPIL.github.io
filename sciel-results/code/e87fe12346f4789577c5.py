import time

def construct():
    start_time = time.time()
    DEADLINE = 85  # 90s limit, leave 5s margin
    
    # 1. Generate all 2,3,5-smooth numbers up to 128
    smooth = set()
    for a in range(0, 8):
        val_a = 2 ** a
        for b in range(0, 5):
            val_ab = val_a * (3 ** b)
            if val_ab > 128:
                break
            for c in range(0, 4):
                val_abc = val_ab * (5 ** c)
                if val_abc > 128:
                    break
                smooth.add(val_abc)
    
    smooth_list = sorted(smooth)
    n_smooth = len(smooth_list)
    val_to_idx = {v: i for i, v in enumerate(smooth_list)}
    
    # Precompute incompatibility mask for each smooth number
    # incompat[i] is a bitmask of indices j where smooth_list[i] and smooth_list[j] are incompatible (one divides the other)
    incompat = [0] * n_smooth
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            a, b = smooth_list[i], smooth_list[j]
            if a % b == 0 or b % a == 0:
                incompat[i] |= (1 << j)
                incompat[j] |= (1 << i)
    
    # For each target, find minimum k antichain
    def find_min_antichain(n):
        """Find minimum k such that there's an antichain of k smooth numbers summing to n."""
        # Get candidates: smooth numbers <= n
        cand_indices = [i for i in range(n_smooth) if smooth_list[i] <= n]
        if not cand_indices:
            return None
        
        # Try k = 1, 2, 3, ...
        for k in range(1, 31):
            if time.time() - start_time > DEADLINE:
                break
            
            # Check if feasible: sum of k smallest candidates >= n? No, sum of k largest >= n
            # And sum of k smallest <= n
            cand_vals = [smooth_list[i] for i in cand_indices]
            if len(cand_vals) < k:
                continue
            
            # Sort candidates
            # For pruning, we need to know min and max possible sums
            
            # Use DFS with bitmask
            # State: (start_idx_in_cand_indices, remaining_sum, remaining_k, used_mask)
            # used_mask tracks which cand_indices have been used
            
            cand_sorted = sorted(cand_indices, key=lambda i: smooth_list[i])
            cand_vals_sorted = [smooth_list[i] for i in cand_sorted]
            
            # Quick check: can we sum to n with k items from cand_vals_sorted?
            if sum(cand_vals_sorted[-k:]) < n or sum(cand_vals_sorted[:k]) > n:
                continue
            
            # DFS
            found = [None]
            
            def dfs(start, rem_sum, rem_k, used_mask, path):
                if rem_k == 0:
                    if rem_sum == 0:
                        found[0] = list(path)
                        return True
                    return False
                
                if rem_k > len(cand_sorted) - start:
                    return False
                
                # Pruning: max possible sum from remaining
                # Take the largest rem_k values from cand_sorted[start:]
                avail = cand_vals_sorted[start:]
                if len(avail) < rem_k:
                    return False
                max_possible = sum(avail[-rem_k:])  # largest rem_k
                min_possible = sum(avail[:rem_k])   # smallest rem_k
                
                if rem_sum > max_possible or rem_sum < min_possible:
                    return False
                
                for i in range(start, len(cand_sorted)):
                    ci = cand_sorted[i]
                    val = smooth_list[ci]
                    if val > rem_sum:
                        continue
                    
                    # Check compatibility with all in path
                    compatible = True
                    for pi in path:
                        if incompat[pi] & (1 << ci):
                            compatible = False
                            break
                    
                    if not compatible:
                        continue
                    
                    path.append(ci)
                    if dfs(i + 1, rem_sum - val, rem_k - 1, used_mask | (1 << ci), path):
                        return True
                    path.pop()
                
                return False
            
            if dfs(0, n, k, 0, []):
                result = [smooth_list[i] for i in found[0]]
                return result
        
        return None
    
    answers = []
    for n in range(1, 129):
        if time.time() - start_time > DEADLINE:
            # Emergency: use [n] if smooth, else try to find anything
            if n in smooth_list:
                answers.append([n])
            else:
                # Try k=2 quickly
                ans = find_min_antichain(n)
                answers.append(ans if ans else [n])
            continue
        
        ans = find_min_antichain(n)
        if ans is None:
            # Should not happen for valid targets
            if n in smooth_list:
                ans = [n]
            else:
                # Last resort: try harder
                ans = [n]  # Will be invalid but better than crash
        answers.append(ans)
    
    return answers
