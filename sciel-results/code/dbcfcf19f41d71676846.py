import time
import random
import sys

def construct(n_rows, k):
    """
    Finds a set of n_rows rows of k+1 increasing integers starting at 0,
    such that all differences are distinct across all rows.
    Minimizes the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # 1. Greedy Construction
    def greedy_construct():
        used = set()
        rows = []
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

    rows = greedy_construct()
    best_rows = rows
    best_max = max(max(r) for r in rows)
    
    # Helper to get differences of a row
    def get_diffs(row):
        return {row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))}
    
    # Helper to check if a row is compatible with a set of used diffs
    def is_compatible(row, used_diffs):
        diffs = get_diffs(row)
        return len(diffs & used_diffs) == 0

    # Backtracking search for a replacement row with max <= limit
    def find_replacement(used_diffs, limit, k):
        # We need to find a row of size k+1 starting at 0, max <= limit
        # such that diffs are not in used_diffs
        row = [0]
        
        def backtrack(idx, current_row):
            if time.time() > deadline:
                return None
            if len(current_row) == k + 1:
                return current_row[:]
            
            last = current_row[-1]
            # Try next elements from last + 1 to limit
            # Optimization: start from small values
            for x in range(last + 1, limit + 1):
                new = {x - a for a in current_row}
                if new & used_diffs:
                    continue
                # Additional pruning: check if remaining slots can be filled?
                # Simple heuristic: just try
                current_row.append(x)
                res = backtrack(idx + 1, current_row)
                if res is not None:
                    return res
                current_row.pop()
            return None
        
        return backtrack(1, row)

    # Local Search Loop
    while time.time() < deadline:
        improved = False
        # Sort rows by max element descending to target the largest ones first
        # We need to keep track of indices
        max_indices = sorted(range(n_rows), key=lambda i: max(rows[i]), reverse=True)
        
        for idx in max_indices:
            if time.time() > deadline:
                break
            
            current_max = max(rows[idx])
            if current_max <= best_max:
                continue
                
            # Try to find a replacement with max < current_max
            # We can try reducing by 1, or just try to find any row with max <= current_max - 1
            target_max = current_max - 1
            
            # Get used diffs from all other rows
            other_rows = [rows[i] for i in range(n_rows) if i != idx]
            used_diffs = set()
            for r in other_rows:
                used_diffs.update(get_diffs(r))
            
            # Try to find replacement
            # To speed up, we can randomize the search order or use a time limit for this specific search
            # The backtracking function above doesn't have a strict internal time limit per call,
            # but it checks the global deadline.
            
            new_row = find_replacement(used_diffs, target_max, k)
            
            if new_row is not None:
                rows[idx] = new_row
                new_max = max(new_row)
                if new_max < best_max:
                    best_max = new_max
                    best_rows = [r[:] for r in rows]
                improved = True
                # If we improved, restart the loop to re-evaluate max indices
                break
        
        if not improved:
            # If no improvement in this pass, we might be stuck.
            # We can try a random restart or a different strategy.
            # For now, let's try to "shuffle" by replacing a random row with any valid row 
            # (not necessarily smaller) to escape local minima?
            # Or just break if we've made no progress for a while?
            # Given the deadline, we can try a few random restarts of the greedy part 
            # with different random seeds if we are stuck.
            
            # Let's try one random restart of the greedy construction
            # and see if it yields a better initial solution
            if random.random() < 0.1: # 10% chance to restart
                # Simple randomized greedy
                used = set()
                new_rows = []
                success = True
                for _ in range(n_rows):
                    row, x = [0], 0
                    # Randomize the step size slightly?
                    # Or just shuffle the order of checking x?
                    # Standard greedy is deterministic.
                    # Let's try a randomized backtracking for the first row?
                    # Too complex. Let's just break if stuck.
                    pass
            
            # If we are stuck, we can try to reduce the target_max more aggressively?
            # No, if find_replacement fails for target_max, it will fail for smaller.
            # So we are stuck at this configuration.
            # We can try to modify the "other rows" to free up space.
            # This is complex. Let's just terminate if no improvement for a few seconds.
            
            # Simple heuristic: if no improvement, break to avoid wasting time
            # unless we have a lot of time left.
            if time.time() > start_time + 60:
                break

    return best_rows

# EVOLVE-BLOCK-END
