    import sys
    sys.setrecursionlimit(100000)

    def construct():
        n = 256
        weights = [1.0 / (i + 1) for i in range(n)]
        
        # Precompute forbidden pairs
        # forbidden_by_pair[i][j] is a list of k indices
        forbidden_by_pair = [[[] for _ in range(n)] for _ in range(n)]
        for i in range(n):
            vi = i + 1
            for j in range(i + 1, n):
                vj = j + 1
                ks = []
                k1 = 2 * vj - vi
                if 1 <= k1 <= n and k1 != vi and k1 != vj:
                    ks.append(k1 - 1)
                k2 = 2 * vi - vj
                if 1 <= k2 <= n and k2 != vi and k2 != vj:
                    ks.append(k2 - 1)
                if (vi + vj) % 2 == 0:
                    k3 = (vi + vj) // 2
                    if 1 <= k3 <= n and k3 != vi and k3 != vj:
                        ks.append(k3 - 1)
                
                forbidden_by_pair[i][j] = ks
                forbidden_by_pair[j][i] = ks
        
        best_sum = 0.0
        best_set = []
        
        # Suffix sums of weights for quick upper bound (ignoring forbidden)
        # This is a very fast, loose bound.
        # suffix_sum[i] = sum(weights[i:])
        suffix_sum = [0.0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix_sum[i] = suffix_sum[i + 1] + weights[i]
        
        def dfs(idx, selected_mask, forbidden_mask, curr_sum, selected_list):
            nonlocal best_sum, best_set
            
            # Quick prune: even if we take ALL remaining, can we beat best?
            if curr_sum + suffix_sum[idx] <= best_sum:
                return
            
            # Tighter prune: sum of available
            # Only compute if quick prune didn't trigger?
            # Or always?
            # Computing available sum is O(n).
            # Let's do it.
            ub = curr_sum
            # We can optimize this by maintaining a list of available?
            # No, just loop.
            for i in range(idx, n):
                bit = 1 << i
                if not (selected_mask & bit) and not (forbidden_mask & bit):
                    ub += weights[i]
            
            if ub <= best_sum:
                return
            
            if idx == n:
                if curr_sum > best_sum:
                    best_sum = curr_sum
                    best_set = list(selected_list)
                return
            
            i = idx
            bit = 1 << i
            
            # Option 1: Include i
            if not (selected_mask & bit) and not (forbidden_mask & bit):
                new_selected_mask = selected_mask | bit
                new_forbidden_mask = forbidden_mask
                for j in selected_list:
                    for k in forbidden_by_pair[i][j]:
                        new_forbidden_mask |= (1 << k)
                
                selected_list.append(i)
                dfs(i + 1, new_selected_mask, new_forbidden_mask, curr_sum + weights[i], selected_list)
                selected_list.pop()
            
            # Option 2: Exclude i
            dfs(i + 1, selected_mask, forbidden_mask, curr_sum, selected_list)
        
        dfs(0, 0, 0, 0.0, [])
        
        result = [i + 1 for i in best_set]
        return result
    