import time

def construct():
    start_time = time.time()
    DEADLINE = 85
    
    # Generate all 2,3,5-smooth numbers up to 128
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
    
    # Precompute incompatibility (divisibility)
    incompat = [[False] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i != j:
                a, b = smooth_list[i], smooth_list[j]
                if a > b:
                    a, b = b, a
                if b % a == 0:
                    incompat[i][j] = True
                    incompat[j][i] = True
    
    # Precompute compatible masks for fast compatibility check
    # compat_mask[i] is a bitmask of indices compatible with i
    compat_mask = [0] * n_smooth
    for i in range(n_smooth):
        mask = 0
        for j in range(n_smooth):
            if i != j and not incompat[i][j]:
                mask |= (1 << j)
        compat_mask[i] = mask
    
    # Precompute prefix sums for quick min/max sum calculations
    prefix_sum = [0] * (n_smooth + 1)
    for i in range(n_smooth):
        prefix_sum[i + 1] = prefix_sum[i] + smooth_list[i]
    
    def min_sum_from(start, count):
        """Minimum sum of 'count' elements starting from index 'start'"""
        if start + count > n_smooth:
            return float('inf')
        return prefix_sum[start + count] - prefix_sum[start]
    
    def max_sum_from(start, count):
        """Maximum sum of 'count' elements starting from index 'start'"""
        if start + count > n_smooth:
            return float('-inf')
        # Take the largest 'count' elements from smooth_list[start:]
        end = n_smooth
        start_idx = max(start, n_smooth - count)
        return prefix_sum[end] - prefix_sum[start_idx]
    
    min_k_for_sum = {}
    
    # k=1
    for i in range(n_smooth):
        s = smooth_list[i]
        if s <= 128 and s not in min_k_for_sum:
            min_k_for_sum[s] = (1, [s])
            
    # k=2
    for i in range(n_smooth):
        for j in range(i+1, n_smooth):
            if incompat[i][j]:
                continue
            s = smooth_list[i] + smooth_list[j]
            if s > 128:
                break
            if s not in min_k_for_sum:
                min_k_for_sum[s] = (2, [smooth_list[i], smooth_list[j]])
                
    # k=3
    for i in range(n_smooth):
        for j in range(i+1, n_smooth):
            if incompat[i][j]:
                continue
            s2 = smooth_list[i] + smooth_list[j]
            if s2 > 128:
                break
            for k in range(j+1, n_smooth):
                if incompat[i][k] or incompat[j][k]:
                    continue
                s3 = s2 + smooth_list[k]
                if s3 > 128:
                    break
                if s3 not in min_k_for_sum:
                    min_k_for_sum[s3] = (3, [smooth_list[i], smooth_list[j], smooth_list[k]])
                    
    # k=4
    for i in range(n_smooth):
        for j in range(i+1, n_smooth):
            if incompat[i][j]:
                continue
            s2 = smooth_list[i] + smooth_list[j]
            if s2 > 128:
                break
            for k in range(j+1, n_smooth):
                if incompat[i][k] or incompat[j][k]:
                    continue
                s3 = s2 + smooth_list[k]
                if s3 > 128:
                    break
                for l in range(k+1, n_smooth):
                    if incompat[i][l] or incompat[j][l] or incompat[k][l]:
                        continue
                    s4 = s3 + smooth_list[l]
                    if s4 > 128:
                        break
                    if s4 not in min_k_for_sum:
                        min_k_for_sum[s4] = (4, [smooth_list[i], smooth_list[j], smooth_list[k], smooth_list[l]])
                    
    # Fallback for uncovered numbers using backtracking with better pruning
    def find_antichain_sum(n, k):
        """Find k distinct smooth numbers summing to n with no divisibility relation"""
        
        def backtrack(start, rem_sum, rem_k, current_path, used_mask):
            if rem_k == 0:
                if rem_sum == 0:
                    return [smooth_list[idx] for idx in current_path]
                return None
            
            # Pruning: not enough candidates left
            if n_smooth - start < rem_k:
                return None
            
            # Pruning: check if rem_sum is achievable
            min_possible = min_sum_from(start, rem_k)
            if min_possible > rem_sum:
                return None
            max_possible = max_sum_from(start, rem_k)
            if max_possible < rem_sum:
                return None
            
            # Iterate candidates for the next element
            for i in range(start, n_smooth):
                val = smooth_list[i]
                if val > rem_sum:
                    break
                
                # Check compatibility with used elements using bitmask
                if used_mask & ~(compat_mask[i]):
                    continue
                
                # Pruning: can we form rem_sum - val with rem_k-1 numbers from i+1?
                if rem_k - 1 > 0:
                    min_next = min_sum_from(i+1, rem_k-1)
                    if min_next > rem_sum - val:
                        break
                    max_next = max_sum_from(i+1, rem_k-1)
                    if max_next < rem_sum - val:
                        continue
                
                current_path.append(i)
                new_used_mask = used_mask | (1 << i)
                res = backtrack(i+1, rem_sum - val, rem_k - 1, current_path, new_used_mask)
                if res:
                    return res
                current_path.pop()
                
            return None
        
        return backtrack(0, n, k, [], 0)

    answers = []
    for n in range(1, 129):
        if n in min_k_for_sum:
            answers.append(min_k_for_sum[n][1])
        else:
            found = None
            for k in range(5, 31):
                if time.time() - start_time > DEADLINE:
                    break
                r = find_antichain_sum(n, k)
                if r:
                    found = r
                    break
            if found:
                answers.append(found)
            else:
                # Emergency fallback: try to find any valid representation
                # This should rarely be needed
                for k in range(1, 31):
                    r = find_antichain_sum(n, k)
                    if r:
                        found = r
                        break
                if found:
                    answers.append(found)
                else:
                    answers.append([])
                
    return answers
