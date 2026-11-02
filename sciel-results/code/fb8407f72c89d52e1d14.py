import time
import sys

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 115
    
    # Helper to generate valid rows
    def generate_rows(max_val):
        candidates = []
        
        def dfs_row(current_row, used_diffs):
            if len(current_row) == k + 1:
                candidates.append((tuple(current_row), used_diffs, current_row[-1]))
                return
            
            last = current_row[-1]
            for x in range(last + 1, max_val + 1):
                new_diffs = {x - a for a in current_row}
                if new_diffs & used_diffs:
                    continue
                dfs_row(current_row + [x], used_diffs | new_diffs)
        
        dfs_row([0], set())
        return candidates

    def set_to_bitmask(s):
        mask = 0
        for d in s:
            mask |= (1 << d)
        return mask

    # Initial generation
    M_gen = 150
    all_candidates = generate_rows(M_gen)
    for i in range(len(all_candidates)):
        row, diff_set, max_v = all_candidates[i]
        mask = set_to_bitmask(diff_set)
        all_candidates[i] = (row, mask, max_v)
    all_candidates.sort(key=lambda x: x[2])

    def can_pack(M):
        # Filter
        valid_cands = [c for c in all_candidates if c[2] <= M]
        if len(valid_cands) < n_rows:
            return None

        found = []
        
        # DFS with backtracking
        # To avoid deep recursion issues, we can use iterative or just hope it's shallow enough.
        # n_rows is small (e.g., 7).
        
        def dfs(idx, used_mask, count):
            if count == n_rows:
                return True
            
            # Pruning
            if len(valid_cands) - idx < n_rows - count:
                return False
                
            for i in range(idx, len(valid_cands)):
                row, mask, max_v = valid_cands[i]
                if used_mask & mask == 0:
                    found.append(row)
                    if dfs(i + 1, used_mask | mask, count + 1):
                        return True
                    found.pop()
                    
            return False

        if dfs(0, 0, 0):
            return found
        return None

    best_rows = None
    best_max = float('inf')
    
    low = 0
    high = M_gen
    
    while time.time() - start_time < time_limit - 5:
        if high > M_gen:
            # Extend generation
            M_gen = high + 50
            all_candidates = generate_rows(M_gen)
            for i in range(len(all_candidates)):
                row, diff_set, max_v = all_candidates[i]
                mask = set_to_bitmask(diff_set)
                all_candidates[i] = (row, mask, max_v)
            all_candidates.sort(key=lambda x: x[2])
            
        mid = (low + high) // 2
        res = can_pack(mid)
        
        if res:
            best_rows = res
            best_max = mid
            high = mid - 1
        else:
            low = mid + 1
            
        if low > high:
            break

    if best_rows:
        return [list(r) for r in best_rows]
        
    # Fallback: Greedy
    used_mask = 0
    rows = []
    for row, mask, max_v in all_candidates:
        if len(rows) == n_rows:
            break
        if used_mask & mask == 0:
            rows.append(row)
            used_mask |= mask
            
    if rows:
        return [list(r) for r in rows]
        
    return []
