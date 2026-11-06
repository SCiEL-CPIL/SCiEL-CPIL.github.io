    def backtrack(s, sums_bitset, candidates, start_idx, depth, target_size):
        nonlocal best_set, best_size, best_sums_bitset
        
        if time.time() > deadline - 5:
            return False
            
        if depth == target_size:
            best_set = list(s)
            best_size = target_size
            best_sums_bitset = sums_bitset
            return True
            
        # Pruning: not enough candidates left
        if len(candidates) - start_idx < target_size - depth:
            return False
            
        for i in range(start_idx, len(candidates)):
            if time.time() > deadline - 5:
                return False
                
            x = candidates[i]
            valid, delta = can_add_bitset(x, s, sums_bitset)
            if valid:
                new_s = s + [x]
                new_sums = sums_bitset | delta
                if backtrack(new_s, new_sums, candidates, i + 1, depth + 1, target_size):
                    return True
                    
        return False
