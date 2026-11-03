def _solve_dfs(n_rows, k, max_val, rng, time_limit):
    L = k + 1
    
    def backtrack(rows_done, curr_row, used_bits):
        if time.time() > time_limit:
            return None
        
        if rows_done == n_rows:
            return [list(r) for r in rows_done]
        
        if len(curr_row) == L:
            return backtrack(rows_done + [curr_row], [], used_bits)
        
        start_val = curr_row[-1] + 1 if curr_row else 1
        
        # Find valid candidates
        candidates = []
        # We limit the search space to the first few valid values to keep branching factor low
        # and allow backtracking.
        
        for x in range(start_val, min(start_val + 50, max_val + 1)):
            new_diffs = {x - a for a in curr_row}
            mask = 0
            valid = True
            for d in new_diffs:
                if d > max_val:
                    valid = False
                    break
                bit = 1 << (d - 1)
                if used_bits & bit:
                    valid = False
                    break
                mask |= bit
            if valid:
                candidates.append((x, mask))
                if len(candidates) >= 5:
                    break
        
        if not candidates:
            return None
            
        # Randomize order
        rng.shuffle(candidates)
        
        for x, mask in candidates:
            new_curr_row = curr_row + [x]
            new_used_bits = used_bits | mask
            res = backtrack(rows_done, new_curr_row, new_used_bits)
            if res:
                return res
                
        return None

    return backtrack([], [], 0)
