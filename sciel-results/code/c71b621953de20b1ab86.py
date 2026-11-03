"""Improved program for the difference triangle set task."""
import random

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimize the largest entry.
    
    Uses randomized backtracking with multiple restarts.
    """
    best_result = None
    best_scope = float('inf')
    
    num_trials = 30
    
    for trial in range(num_trials):
        result = _solve_backtrack(n_rows, k, trial)
        if result is not None:
            scope = max(max(row) for row in result)
            if scope < best_scope:
                best_scope = scope
                best_result = result
                if scope <= 105:
                    break
    
    if best_result is not None:
        return best_result
    
    return _greedy(n_rows, k)


def _solve_backtrack(n_rows, k, seed_offset=0):
    """
    Backtracking solver. Each row has entries a_0=0, a_1, ..., a_k with a_0 < a_1 < ... < a_k.
    All differences a_j - a_i for i<j must be unique across all rows.
    """
    random.seed(seed_offset * 7919 + 12345)
    
    used_diffs = set()
    rows = []
    
    # Order in which to fill rows
    row_order = list(range(n_rows))
    random.shuffle(row_order)
    
    def backtrack(row_idx, pos, current_row):
        """
        row_idx: 0..n_rows-1, index into row_order
        pos: 0..k, position in current_row (current_row[0] is always 0)
        current_row: list with current_row[0]=0 and pos entries filled
        """
        if pos == k:
            rows.append(current_row[:])
            if row_idx == n_rows - 1:
                return True
            if backtrack(row_idx + 1, 0, [0]):
                return True
            rows.pop()
            return False
        
        min_val = current_row[-1] + 1
        
        # Generate candidate values
        # Focus on smaller values first, but include some larger ones
        candidates = []
        # Dense small range
        for v in range(min_val, min_val + 20):
            candidates.append(v)
        # Sparse larger range
        for v in range(min_val + 20, 115, 4):
            candidates.append(v)
        
        # Filter and score
        valid_candidates = []
        for val in candidates:
            new_diffs = set()
            valid = True
            for a in current_row:
                d = val - a
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            if valid:
                # Score: number of new diffs that are "small" (prefer smaller diffs? or just count)
                valid_candidates.append((val, new_diffs))
        
        # Sort by value (prefer smaller) with some randomization
        valid_candidates.sort(key=lambda x: (x[0], random.random()))
        
        # Take top 10
        top = valid_candidates[:10]
        
        for val, new_diffs in top:
            current_row.append(val)
            used_diffs.update(new_diffs)
            
            if backtrack(row_idx, pos + 1, current_row):
                return True
            
            current_row.pop()
            used_diffs.difference_update(new_diffs)
        
        return False
    
    if backtrack(0, 0, [0]):
        final_rows = [None] * n_rows
        for i, row in enumerate(rows):
            final_rows[row_order[i]] = row
        return final_rows
    
    return None


def _greedy(n_rows, k):
    """Fallback greedy approach."""
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

# EVOLVE-BLOCK-END
