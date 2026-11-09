"""Rewritten program for divisibility-antichain representations by smooth numbers."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 55  # 55s safety margin before 60s limit
    
    # Generate all 2,3,5-smooth numbers up to 128
    smooth = []
    for x in range(1, 129):
        y = x
        for p in (2, 3, 5):
            while y % p == 0:
                y //= p
        if y == 1:
            smooth.append(x)
    
    n_smooth = len(smooth)
    
    # Conflict matrix: conflict[i][j] = True if smooth[i] | smooth[j] or smooth[j] | smooth[i], i != j
    conflict = [[False] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i != j:
                if smooth[i] % smooth[j] == 0 or smooth[j] % smooth[i] == 0:
                    conflict[i][j] = True
    
    # For each target n, find minimum number of smooth terms forming an antichain summing to n
    answers = [None] * 128
    
    for n in range(1, 129):
        if time.time() > deadline:
            break
        
        # Candidate smooth numbers <= n, with their global indices
        cand_vals = []
        cand_idx = []
        for idx, s in enumerate(smooth):
            if s <= n:
                cand_vals.append(s)
                cand_idx.append(idx)
        
        found = False
        for count in range(1, 31):
            if time.time() > deadline:
                break
            
            result = _dfs(cand_vals, cand_idx, conflict, n, count, 0, 0, [], time.time, deadline)
            if result is not None:
                answers[n-1] = result
                found = True
                break
        
        if not found:
            # Should not happen
            answers[n-1] = None
    
    # Fallback for any missing
    for i in range(128):
        if answers[i] is None and time.time() < deadline:
            n = i + 1
            cand_vals = []
            cand_idx = []
            for idx, s in enumerate(smooth):
                if s <= n:
                    cand_vals.append(s)
                    cand_idx.append(idx)
            for count in range(1, 31):
                if time.time() > deadline:
                    break
                result = _dfs(cand_vals, cand_idx, conflict, n, count, 0, 0, [], time.time, deadline)
                if result is not None:
                    answers[i] = result
                    break
            if answers[i] is None:
                if n in smooth:
                    answers[i] = [n]
    
    return answers


def _dfs(cand_vals, cand_idx, conflict, target, count, start, current_sum, current_indices, time_fn, deadline):
    """Find 'count' elements from cand_vals[start:] summing to target, forming antichain."""
    if time_fn() > deadline:
        return None
    
    if len(current_indices) == count:
        if current_sum == target:
            return [cand_vals[i] for i in current_indices]
        return None
    
    remaining = count - len(current_indices)
    if start + remaining > len(cand_vals):
        return None
    
    # Pruning: min and max possible additional sum
    if start + remaining <= len(cand_vals):
        min_add = sum(cand_vals[start:start+remaining])
        if current_sum + min_add > target:
            return None
        max_add = sum(cand_vals[len(cand_vals)-remaining:len(cand_vals)])
        if current_sum + max_add < target:
            return None
    
    for i in range(start, len(cand_vals)):
        if time_fn() > deadline:
            return None
        
        val = cand_vals[i]
        new_sum = current_sum + val
        if new_sum > target:
            continue
        
        # Check antichain with current selections
        ok = True
        for j in current_indices:
            gi = cand_idx[i]
            gj = cand_idx[j]
            if conflict[gi][gj]:
                ok = False
                break
        if not ok:
            continue
        
        current_indices.append(i)
        result = _dfs(cand_vals, cand_idx, conflict, target, count, i+1, new_sum, current_indices, time_fn, deadline)
        current_indices.pop()
        if result is not None:
            return result
    
    return None
# EVOLVE-BLOCK-END
