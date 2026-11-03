import time

def construct(n_rows, k):
    start_time = time.time()
    # 120s limit. Use 110s for search.
    deadline = start_time + 110
    
    n = n_rows
    k_val = k
    
    # Counting bound
    # Diffs per row = k*(k+1)/2
    diffs_per_row = k_val * (k_val + 1) // 2
    total_diffs = n * diffs_per_row
    
    # Lower bound for M
    M_min = total_diffs
    # Upper bound for search
    M_max = M_min + 20
    
    # Greedy fallback
    def greedy():
        used, rows = set(), []
        for _ in range(n):
            row, x = [0], 0
            while len(row) < k_val + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    # Backtracking search
    # We try M from M_min to M_max
    for M in range(M_min, M_max + 1):
        # Time limit for this M
        m_deadline = time.time() + 10  # 10 seconds per M
        
        used_diffs = set()
        rows = []
        prev_max = 0
        
        def backtrack(r, i, current_row):
            nonlocal used_diffs, rows, prev_max, m_deadline
            
            if time.time() > m_deadline:
                return False
            
            if r == n:
                return True
            
            # Pruning: Check available differences
            # Diffs in current row
            curr_diffs = set()
            for a in range(len(current_row)):
                for b in range(a + 1, len(current_row)):
                    curr_diffs.add(current_row[b] - current_row[a])
            
            # Total unique diffs used so far
            # used_diffs contains diffs from completed rows
            # curr_diffs contains diffs from current row (not yet in used_diffs)
            # They are disjoint by construction (we check before adding to used_diffs)
            # Wait, we check `new_diffs & (used_diffs | curr_diffs)` in the loop.
            # So used_diffs and curr_diffs are disjoint.
            
            total_used = len(used_diffs) + len(curr_diffs)
            available = M - total_used
            
            # Diffs needed
            # Current row needs: diffs_per_row - len(curr_diffs)
            # Remaining rows need: diffs_per_row * (n - r - 1)
            needed = (diffs_per_row - len(curr_diffs)) + diffs_per_row * (n - r - 1)
            
            if available < needed:
                return False
            
            if i == k_val:
                # Row complete
                if current_row[-1] < prev_max:
                    return False
                
                # Add to used_diffs
                used_diffs.update(curr_diffs)
                rows.append(current_row[:])
                prev_max = current_row[-1]
                
                if backtrack(r + 1, 0, [0]):
                    return True
                
                # Backtrack
                rows.pop()
                for d in curr_diffs:
                    used_diffs.discard(d)
                return False
            
            # Choose next value
            start_val = current_row[i - 1] + 1 if i > 0 else 1
            
            # Heuristic: Try smaller values first
            for x in range(start_val, M + 1):
                # Check diffs
                new_diffs = set()
                for y in current_row:
                    new_diffs.add(x - y)
                
                # Conflict check
                if new_diffs & (used_diffs | curr_diffs):
                    continue
                
                # Symmetry check for last element
                if i == k_val and x < prev_max:
                    continue
                
                if backtrack(r, i + 1, current_row + [x]):
                    return True
            
            return False
        
        # Start
        if backtrack(0, 1, [0]):
            return rows

    # Fallback
    return greedy()
