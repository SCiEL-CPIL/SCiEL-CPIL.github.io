"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within each row are distinct across all rows.
    Minimizes the largest entry.
    """
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0] * (k + 1) for _ in range(n_rows)]
    
    num_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * num_diffs_per_row
    lower_bound = max(k, total_diffs)
    
    # 1. Greedy Construction to get an initial valid solution
    def greedy_construct(m_limit=None):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                if m_limit is not None and x > m_limit:
                    return None
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    # Find a feasible upper bound
    ub = lower_bound
    initial_sol = None
    while True:
        initial_sol = greedy_construct(ub)
        if initial_sol is not None:
            break
        ub += 10
        if ub > 500:
            # Fallback
            initial_sol = greedy_construct(500)
            if initial_sol is None:
                return [[0, 1, 2, 3, 4, 5, 6] for _ in range(n_rows)]
            ub = max(max(r) for r in initial_sol)
            break
            
    best_rows = [r[:] for r in initial_sol]
    best_max = max(max(r) for r in best_rows)
    
    # 2. Local Search to improve (minimize max)
    # We treat this as a force-directed system where we try to "compress" the rows.
    
    def get_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def is_valid(rows):
        used = set()
        for r in rows:
            diffs = get_diffs(r)
            if diffs & used:
                return False
            used |= diffs
        return True

    def max_entry(rows):
        return max(max(r) for r in rows)

    # Local Search: Try to reduce the max entry by moving elements down.
    # We iterate until no improvement or time limit.
    start_time = time.time()
    time_limit = 115.0
    
    # We will try to "push" the largest elements down.
    # Strategy: Identify the row with the max element. Try to reduce its elements.
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        # Sort rows by max element descending to focus on the worst offenders
        row_indices = sorted(range(n_rows), key=lambda i: max(best_rows[i]), reverse=True)
        
        for idx in row_indices:
            if time.time() - start_time > time_limit:
                break
                
            row = best_rows[idx]
            # Try to reduce each element from largest to smallest
            for j in range(k, -1, -1):
                if j == 0:
                    continue # 0 is fixed
                current_val = row[j]
                # Try to reduce current_val
                # It must be > row[j-1]
                # And maintain distinct diffs
                
                # We can try values from current_val - 1 down to row[j-1] + 1
                # But we should check validity efficiently.
                
                # Remove current row's diffs from global used set for testing
                # This is expensive. Instead, we can just test the new row against all other rows.
                
                # Optimization: Precompute diffs of other rows?
                # For k=6, n_rows is small (e.g., 7), so checking is fast.
                
                other_rows = [best_rows[i] for i in range(n_rows) if i != idx]
                other_diffs = set()
                for r in other_rows:
                    other_diffs |= get_diffs(r)
                
                # Try new values
                found = False
                # Try smaller values first to minimize max
                for new_val in range(current_val - 1, row[j-1], -1):
                    new_row = row[:]
                    new_row[j] = new_val
                    
                    # Check if new_row is valid internally (increasing)
                    # It is, since new_val > row[j-1] and new_val < row[j] < row[j+1] (if j<k)
                    # Wait, if j < k, we must ensure new_val < row[j+1].
                    if j < k and new_val >= row[j+1]:
                        continue
                        
                    new_diffs = get_diffs(new_row)
                    
                    # Check against other rows
                    if not (new_diffs & other_diffs):
                        # Also check internal distinctness (already guaranteed by set size? No, set handles it)
                        # We must ensure new_diffs has size k*(k+1)/2
                        if len(new_diffs) == num_diffs_per_row:
                            # Valid move
                            best_rows[idx] = new_row
                            if new_val < best_max:
                                best_max = max_entry(best_rows)
                                improved = True
                            found = True
                            break
                            
                if found:
                    # If we changed the row, other_diffs for subsequent j in this row might change?
                    # Yes, but we are iterating j downwards.
                    # The "other_diffs" for this row index should be updated if we accept a move.
                    # However, we only update best_rows.
                    # The next j (smaller index) will recompute other_diffs?
                    # No, we computed other_diffs once outside the j loop.
                    # This is a bug. If we change row[idx], the diffs of row[idx] change,
                    # but other_diffs (from other rows) does NOT change.
                    # So the check `not (new_diffs & other_diffs)` is still valid for other rows.
                    # The only thing that changes is the internal consistency of row[idx],
                    # which we handle by constructing new_row.
                    # So the logic holds.
                    pass

    return best_rows
# EVOLVE-BLOCK-END
