import time
import sys

def construct(n_rows, k):
    """
    Constructs a set of n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimizes the largest entry.
    
    Approach:
    1. Calculate a lower bound for the scope.
    2. Iterate scope S from lower bound upwards.
    3. For each S, run a backtracking search constrained to max value < S.
    4. Use a heuristic that picks the candidate x that leaves the most options for the next step y.
    """
    start_time = time.time()
    time_limit = 115.0  # Leave buffer for 120s limit
    
    # Total differences required: n_rows * k * (k+1) / 2
    total_diffs_needed = n_rows * k * (k + 1) // 2
    
    # Lower bound on scope:
    # If max entry is M, differences are in 1..M.
    # We need total_diffs_needed distinct differences.
    # So M must be at least total_diffs_needed.
    # Also, for a single row of length k+1 starting at 0, if max is M,
    # the number of distinct differences in that row is k(k+1)/2.
    # A tighter bound: The sum of the k largest differences in a row with max M is at most M + (M-1) + ... ? No.
    # Simple counting bound: M >= total_diffs_needed.
    # For 7 rows, k=5: 7 * 15 = 105.
    
    lower_bound = total_diffs_needed
    
    # We will search for the smallest S such that a solution exists with max entry < S.
    # The "scope" is the max entry. If we constrain max entry < S, the max entry is at most S-1.
    # So we iterate S from lower_bound + 1 upwards.
    # If we find a solution with max entry M, the score is M.
    
    best_result = None
    best_scope = float('inf')
    
    # To speed up, we can parallelize or just run sequential with tight time checks.
    # Given the small search space (7x6), sequential is likely fine if the heuristic is good.
    
    for S in range(lower_bound + 1, lower_bound + 150):
        if time.time() - start_time > time_limit:
            break
            
        # Check if we can find a solution with max entry < S
        # i.e., max entry <= S - 1
        
        solution = _solve_with_bound(n_rows, k, S - 1)
        
        if solution is not None:
            # Verify the solution
            if _is_valid(solution, n_rows, k):
                max_val = max(max(row) for row in solution)
                if max_val < best_scope:
                    best_scope = max_val
                    best_result = solution
                
                # Since we are iterating S upwards, the first valid S gives the minimum max_val?
                # Not necessarily. We constrain max_val < S.
                # If we find a solution with S=113 (max_val <= 112), and then S=114 (max_val <= 113),
                # the one with S=113 is better or equal.
                # However, the search might find a solution with max_val=110 when S=113.
                # If we find ANY solution for a given S, we should check if it improves the global best.
                # But since we start from the lowest S, the first S that yields a solution
                # will have the smallest possible upper bound for the max entry.
                # Actually, if a solution exists with max entry M, it will be found when S > M.
                # The smallest S that allows a solution is M_opt + 1.
                # So the first success gives the optimal scope.
                
                # We can break early if we are confident, but let's just return the best found so far
                # or break if we hit the time limit.
                # Since we iterate S ascending, the first valid S yields the minimum possible max entry
                # (assuming the search is complete for that S).
                # If the search is incomplete (timeout), we might miss it.
                # But for small S, the search space is small.
                
                return best_result
        
    # If no optimal solution found within time, fall back to a greedy construction
    # to ensure we return a valid (though possibly suboptimal) result.
    if best_result is not None:
        return best_result
        
    return _greedy_construct(n_rows, k)


def _is_valid(rows, n_rows, k):
    """Check if the constructed rows are valid."""
    if len(rows) != n_rows:
        return False
    
    used_diffs = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                if row[j] <= row[i]:
                    return False
                d = row[j] - row[i]
                if d in used_diffs:
                    return False
                used_diffs.add(d)
    return True


def _solve_with_bound(n_rows, k, max_val_bound):
    """
    Backtracking search to find a solution where all entries are <= max_val_bound.
    Uses a heuristic: pick x that maximizes the number of valid next steps y.
    """
    start_time = time.time()
    time_limit = 115.0  # Global limit, but we check per call? No, better to pass remaining time or just check global.
    # We'll use a shared time check logic.
    
    used_diffs = set()
    rows = []
    
    # Heuristic: For a given state (row_idx, current_row), 
    # we want to choose the next element x.
    # The "degree" of x is the number of valid y > x such that {y-a for a in current_row+[x]} are all unused.
    
    def get_heuristic_score(current_row, x, used_diffs_local):
        """
        Returns a score for candidate x. Higher is better.
        Score = number of valid next elements y.
        """
        new_row = current_row + [x]
        new_diffs = {y - a for a in new_row}
        
        if new_diffs & used_diffs_local:
            return -1  # Invalid candidate, shouldn't be called for invalid ones
        
        # Count valid next elements y
        valid_next_count = 0
        # y must be > x and <= max_val_bound
        for y in range(x + 1, max_val_bound + 1):
            # Check if adding y to new_row creates any used diffs
            # Diffs involving y: y - a for a in new_row
            # We only need to check if any of these are in used_diffs_local or in the diffs of new_row (which are distinct by construction if y > x and row is increasing? No, diffs within a row are distinct if row is strictly increasing? Yes, a_j - a_i are distinct for fixed row? No, e.g., 0, 2, 4 -> 2, 4, 2. Wait. 4-2=2, 2-0=2. Duplicate diff 2.
            # So we must check all diffs.
            
            # Optimization: 
            # Diffs in new_row are D_new.
            # Diffs involving y are D_y = {y - a for a in new_row}.
            # Total diffs would be D_new | D_y.
            # We need (D_new | D_y) & (used_diffs_local | D_y) == empty?
            # Actually, D_new and D_y might intersect.
            # And we need (D_new | D_y) & used_diffs_local == empty.
            
            # Let's just compute the set for simplicity, as k is small (5).
            # new_row has size <= k+1 = 6.
            # y is one element.
            # Diffs are O(k^2).
            
            potential_diffs = set()
            for a in new_row:
                d = y - a
                potential_diffs.add(d)
            
            # Check against used_diffs_local
            if potential_diffs & used_diffs_local:
                continue
            
            # Check for internal conflicts in the new row formed by adding y
            # The new row is new_row + [y].
            # We need to ensure all diffs in this new row are distinct.
            # We already know diffs in new_row are distinct? No, we assumed that.
            # Actually, we should verify that new_row itself has distinct diffs.
            # But if we only add x if it didn't create internal conflicts with current_row, and current_row was valid...
            # Wait, the problem states "all positive differences a_j - a_i (i < j) inside the rows are distinct over all rows".
            # This implies that within a single row, differences must also be distinct?
            # "distinct over all rows" usually means the global set of all differences from all rows is a set (no duplicates).
            # So yes, differences within a row must be distinct, AND differences across rows must be distinct.
            
            # So, for the candidate y, we need:
            # 1. (Diffs of new_row + [y]) is a set of distinct values.
            # 2. (Diffs of new_row + [y]) & used_diffs_local is empty.
            
            # Let's compute all diffs of new_row + [y].
            full_row = new_row + [y]
            diffs = set()
            conflict = False
            for i in range(len(full_row)):
                for j in range(i + 1, len(full_row)):
                    d = full_row[j] - full_row[i]
                    if d in diffs:
                        conflict = True
                        break
                    diffs.add(d)
                if conflict:
                    break
            
            if conflict:
                continue
            
            if diffs & used_diffs_local:
                continue
            
            valid_next_count += 1
            
        return valid_next_count

    def backtrack(row_idx, current_row):
        nonlocal used_diffs, rows
        
        if time.time() - start_time > time_limit:
            return False
        
        if row_idx == n_rows:
            # All rows constructed
            return True
        
        if len(current_row) == k + 1:
            # Row complete
            rows.append(current_row)
            if backtrack(row_idx + 1, [0]):
                return True
            rows.pop()
            return False
        
        last_val = current_row[-1]
        
        # Generate candidates for x
        # x > last_val and x <= max_val_bound
        candidates = []
        
        for x in range(last_val + 1, max_val_bound + 1):
            # Check if x is valid with current_row
            new_row = current_row + [x]
            new_diffs = set()
            conflict = False
            for i in range(len(new_row)):
                for j in range(i + 1, len(new_row)):
                    d = new_row[j] - new_row[i]
                    if d in new_diffs:
                        conflict = True
                        break
                    new_diffs.add(d)
            if conflict:
                continue
            
            if new_diffs & used_diffs:
                continue
            
            # Calculate heuristic score
            score = get_heuristic_score(current_row, x, used_diffs)
            
            # We want to maximize score. 
            # If score is 0, it means no valid next step exists (dead end unless this is the last element).
            # If len(new_row) == k+1, score doesn't matter as much, but we still want to keep options open?
            # Actually, if len(new_row) == k+1, we move to next row.
            
            candidates.append((score, x, new_diffs))
            
        if not candidates:
            return False
        
        # Sort by score descending, then by x ascending (prefer smaller x to keep scope tight? 
        # Actually, if we are minimizing max_val, smaller x is generally better for future rows 
        # as it leaves more room for large values? No, it uses smaller diffs.
        # Large diffs are "scarce" in the sense that there are fewer of them? 
        # No, there are M-1, M-2... 
        # Smaller diffs are more likely to conflict with many other rows.
        # So using larger diffs (which implies larger x) might be safer?
        # But we want to minimize the final max.
        # Let's stick to: Maximize options (score). Tie-breaker: Smaller x? 
        # If we pick small x, we use small diffs. Small diffs are "busy".
        # If we pick large x, we use large diffs.
        # Let's try: Maximize score. Tie-breaker: Smaller x.
        
        candidates.sort(key=lambda item: (-item[0], item[1]))
        
        for score, x, new_diffs in candidates:
            old_used = used_diffs
            used_diffs = used_diffs | new_diffs
            
            if backtrack(row_idx, current_row + [x]):
                return True
            
            used_diffs = old_used
        
        return False

    if backtrack(0, [0]):
        return [r[:] for r in rows]
    return None


def _greedy_construct(n_rows, k):
    """
    Fallback greedy construction.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            # Check internal conflict
            internal_conflict = False
            temp_diffs = set()
            for i in range(len(row) + 1):
                for j in range(i + 1, len(row) + 1):
                    if i == len(row) and j == len(row):
                        continue
                    a = row[i] if i < len(row) else x
                    b = row[j] if j < len(row) else x
                    if i > j:
                        a, b = b, a
                    d = b - a
                    if d in temp_diffs:
                        internal_conflict = True
                        break
                    temp_diffs.add(d)
                if internal_conflict:
                    break
            
            if internal_conflict:
                continue
                
            if new_diffs & used:
                continue
            
            row.append(x)
            used |= new_diffs
        rows.append(row)
    return rows

if __name__ == "__main__":
    # Test with the specific task parameters
    res = construct(7, 5)
    print(res)
    print("Max:", max(max(r) for r in res))

