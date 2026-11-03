"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within each row are distinct across all rows.
    Minimizes the largest entry.
    
    Uses a combination of:
    1. A greedy construction to get an initial upper bound
    2. Local search (simulated annealing / hill climbing) to improve
    """
    deadline = time.time() + 110  # safety margin before 120s kill
    
    def greedy_construct(n_rows, k):
        """Greedy: pick smallest next mark whose diffs are all unused."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows, used
    
    def all_diffs(row):
        """Return set of all positive differences in a row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    def is_valid(rows):
        """Check if all differences are distinct across all rows."""
        seen = set()
        for row in rows:
            for d in all_diffs(row):
                if d in seen:
                    return False
                seen.add(d)
        return True
    
    def max_entry(rows):
        return max(max(row) for row in rows)
    
    # Phase 1: Greedy to get initial solution
    initial_rows, _ = greedy_construct(n_rows, k)
    best_rows = [row[:] for row in initial_rows]
    best_score = max_entry(best_rows)
    
    if time.time() > deadline - 5:
        return best_rows
    
    # Phase 2: Local search - try to reduce the maximum entry
    # Strategy: for each row, try to reconstruct it with a smaller max
    # by using a modified greedy that avoids used diffs from other rows
    
    def reconstruct_row(row_idx, rows, used_by_other, k, max_limit=None):
        """
        Try to reconstruct row row_idx using greedy, avoiding diffs in used_by_other.
        If max_limit is set, only use marks up to max_limit.
        Returns the new row or None if impossible.
        """
        row = [0]
        x = 0
        max_x = max_limit if max_limit is not None else 200
        while len(row) < k + 1:
            x += 1
            if x > max_x:
                return None
            new_diffs = {x - a for a in row}
            if not new_diffs & used_by_other:
                row.append(x)
        return row
    
    def used_by_other(rows, row_idx):
        """Get set of all differences from all rows except row_idx."""
        used = set()
        for i, row in enumerate(rows):
            if i == row_idx:
                continue
            used |= all_diffs(row)
        return used
    
    # Try to improve each row individually
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(n_rows):
            current_max = max(best_rows[i])
            # Try to find a row with smaller max
            for new_max in range(current_max - 1, -1, -1):
                if new_max < k:  # minimum possible
                    break
                other_used = used_by_other(best_rows, i)
                new_row = reconstruct_row(i, best_rows, other_used, k, new_max)
                if new_row is not None:
                    # Verify it's valid (no conflicts with other rows)
                    new_diffs = all_diffs(new_row)
                    if not new_diffs & other_used:
                        best_rows[i] = new_row
                        best_score = max(max(row) for row in best_rows)
                        improved = True
                        break
    
    if time.time() > deadline - 5:
        return best_rows
    
    # Phase 3: More aggressive local search - swap marks within rows
    # Try small perturbations: change one element of a row and repair
    
    def try_perturb(rows, k, deadline):
        """Try to improve by perturbing individual elements."""
        rows = [row[:] for row in rows]
        best = max(max(row) for row in rows)
        
        for _ in range(200):  # limited iterations
            if time.time() > deadline - 5:
                break
            
            # Pick a random row and a random position to change
            ri = random.randint(0, n_rows - 1)
            pos = random.randint(1, k)  # not the first element (which is 0)
            
            old_val = rows[ri][pos]
            # Try nearby values
            for delta in random.sample(range(-5, 6), 11):
                if delta == 0:
                    continue
                new_val = old_val + delta
                if new_val <= rows[ri][pos - 1] or (pos < k and new_val >= rows[ri][pos + 1]):
                    continue
                if new_val < 0:
                    continue
                
                # Create modified row
                test_row = rows[ri][:]
                test_row[pos] = new_val
                
                # Check if this row's diffs conflict with other rows
                other_used = set()
                for j, row in enumerate(rows):
                    if j != ri:
                        other_used |= all_diffs(row)
                
                test_diffs = all_diffs(test_row)
                if test_diffs & other_used:
                    continue
                
                # Also check within-row uniqueness (should be fine since strictly increasing)
                # Accept if it reduces the max
                new_max = max(test_row)
                if new_max < max(rows[ri]):
                    rows[ri] = test_row
                    best = max(max(row) for row in rows)
                    break
        
        return rows, best
    
    best_rows, best_score = try_perturb(best_rows, k, deadline)
    
    if time.time() > deadline - 3:
        return best_rows
    
    # Phase 4: Full row replacement with randomized greedy
    for _ in range(50):
        if time.time() > deadline - 3:
            break
        
        for i in range(n_rows):
            if time.time() > deadline - 3:
                break
            
            other_used = used_by_other(best_rows, i)
            current_max = max(best_rows[i])
            
            # Try randomized greedy to find a better row
            for attempt in range(10):
                row = [0]
                x = 0
                max_x = current_max - 1
                if max_x < k:
                    continue
                success = True
                while len(row) < k + 1:
                    x += 1
                    if x > max_x:
                        success = False
                        break
                    new_diffs = {x - a for a in row}
                    if not new_diffs & other_used:
                        row.append(x)
                    elif random.random() < 0.3:
                        # Sometimes skip ahead
                        x += random.randint(1, 3)
                if success and len(row) == k + 1:
                    new_diffs = all_diffs(row)
                    if not new_diffs & other_used:
                        if max(row) < max(best_rows[i]):
                            best_rows[i] = row
                            best_score = max(max(row) for row in best_rows)
    
    return best_rows
# EVOLVE-BLOCK-END
