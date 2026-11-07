    def has_perfect_matching(m, L):
        if L < 12:
            return False
        
        adj_masks = [0] * 12
        counts = [0] * 12
        
        for i in range(1, 13):
            rem = (-m) % i
            mask = full_mask[i][rem] & ((1 << (L + 1)) - 1)
            if mask == 0:
                return False
            adj_masks[i - 1] = mask
            counts[i - 1] = mask.bit_count()
            
        # Quick check: sum of smallest counts must be >= 12? No.
        # Hall's condition for single nodes is covered by count > 0.
        
        # Sort indices by count
        order = sorted(range(12), key=lambda x: counts[x])
        
        matchR = [-1] * (L + 1)
        
        def dfs(idx, seen):
            mask = adj_masks[idx] & ~seen
            while mask:
                lsb = mask & -mask
                d = lsb.bit_length() - 1
                mask ^= lsb
                if matchR[d] == -1 or dfs(matchR[d], seen | lsb):
                    matchR[d] = idx
                    return True
            return False

        for idx in order:
            if not dfs(idx, 0):
                return False
        return True
