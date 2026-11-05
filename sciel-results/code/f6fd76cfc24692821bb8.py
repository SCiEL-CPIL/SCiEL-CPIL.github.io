import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct a difference triangle set: n_rows rows of k+1 increasing integers
    starting at 0, such that all positive differences within rows are distinct
    across all rows. Minimize the largest entry (scope).
    
    Strategy:
    1. Greedy initialization.
    2. Iterative local search: 
       - Identify rows contributing to the current scope.
       - Attempt to reconstruct these rows with a smaller maximum value.
       - Use randomized backtracking for row reconstruction.
       - If stuck, perturb random rows to escape local minima.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # 1. Greedy Initialization
    rows = _greedy_construct(n_rows, k)
    current_scope = _get_scope(rows)
    best_rows = [row[:] for row in rows]
    best_scope = current_scope
    
    rng = random.Random(42)
    
    def get_other_diffs(row_idx):
        """Calculate the set of differences used by all rows except row_idx."""
        other_diffs = set()
        for i in range(n_rows):
            if i == row_idx:
                continue
            row = rows[i]
            for j in range(len(row)):
                for l in range(j + 1, len(row)):
                    other_diffs.add(row[l] - row[j])
        return other_diffs
    
    def try_reconstruct_row(row_idx, max_allowed, time_limit):
        """
        Try to find a new row for row_idx with max value <= max_allowed.
        Returns the new row if found, else None.
        """
        if max_allowed < k: # Minimum scope for k+1 elements is k (0, 1, ..., k)
            return None
            
        other_diffs = get_other_diffs(row_idx)
        
        # Backtracking search for the new row
        # We want to fill k elements (plus the leading 0)
        # row = [0, x1, x2, ..., xk]
        
        def backtrack(row, current_diffs):
            if time.time() > time_limit:
                return None
            
            if len(row) == k + 1:
                return row[:]
            
            last_val = row[-1]
            # Next value must be > last_val and <= max_allowed
            # Also, the new differences (x - a for a in row) must not be in current_diffs or other_diffs
            
            # Heuristic: Try values in random order to escape local minima
            candidates = list(range(last_val + 1, max_allowed + 1))
            rng.shuffle(candidates)
            
            for x in candidates:
                new_diffs = {x - a for a in row}
                # Check conflicts with existing diffs in this row and other rows
                if new_diffs & current_diffs:
                    continue
                if new_diffs & other_diffs:
                    continue
                
                # Forward check: 
                # Remaining elements: rem = k + 1 - len(row) - 1
                # We need to fit rem more elements in [x+1, max_allowed]
                # So max_allowed - x >= rem
                if max_allowed - x < (k + 1 - len(row) - 1):
                    continue
                
                row.append(x)
                current_diffs |= new_diffs
                
                result = backtrack(row, current_diffs)
                if result:
                    return result
                
                row.pop()
                current_diffs -= new_diffs
            
            return None
        
        # Allow 2 seconds per row reconstruction attempt
        result = backtrack([0], set(), time.time() + 2.0)
        return result

    stagnation_count = 0
    max_stagnation = 200
    
    while time.time() < deadline:
        # Identify rows with max value equal to current_scope
        max_rows = [i for i, r in enumerate(rows) if max(r) == current_scope]
        
        if not max_rows:
            # Update scope if it dropped
            new_scope = _get_scope(rows)
            if new_scope < current_scope:
                current_scope = new_scope
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [row[:] for row in rows]
                    stagnation_count = 0
            max_rows = [i for i, r in enumerate(rows) if max(r) == current_scope]
            if not max_rows:
                break
        
        # Try to reduce one of the max rows
        success = False
        if max_rows:
            # Pick a random max row to try to reduce
            target_idx = rng.choice(max_rows)
            # Try to reconstruct with max_allowed = current_scope - 1
            new_row = try_reconstruct_row(target_idx, current_scope - 1, time.time())
            if new_row:
                rows[target_idx] = new_row
                # Update current scope
                current_scope = _get_scope(rows)
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [row[:] for row in rows]
                    stagnation_count = 0
                    success = True
                else:
                    # We reduced a row's max, but global max didn't drop.
                    # This is still a progress in terms of "tightening".
                    # We accept it to allow further reductions.
                    pass
        
        if not success:
            stagnation_count += 1
            
            # If stuck, perturb the system
            if stagnation_count > max_stagnation:
                # Pick a random row and modify a random element
                r_idx = rng.randint(0, n_rows - 1)
                pos = rng.randint(1, k)
                old_val = rows[r_idx][pos]
                
                min_val = rows[r_idx][pos - 1] + 1
                max_val = rows[r_idx][pos + 1] - 1 if pos < k else best_scope + 10
                
                if max_val >= min_val:
                    # Try a few random values
                    for _ in range(50):
                        new_val = rng.randint(min_val, max_val)
                        if new_val == old_val:
                            continue
                        
                        rows[r_idx][pos] = new_val
                        if _is_valid(rows, n_rows, k):
                            break
                        else:
                            rows[r_idx][pos] = old_val
                
                # Reset stagnation counter after perturbation
                stagnation_count = 0
        
        # Periodic check for best update
        current_scope = _get_scope(rows)
        if current_scope < best_scope:
            best_scope = current_scope
            best_rows = [row[:] for row in rows]

    return best_rows

def _get_scope(rows):
    return max(max(row) for row in rows)

def _is_valid(rows, n_rows, k):
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diff = row[j] - row[i]
                if diff <= 0:
                    return False
                if diff in used:
                    return False
                used.add(diff)
    return len(rows) == n_rows

def _greedy_construct(n_rows, k):
    """Greedy construction: build each row with smallest possible marks."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows

# EVOLVE-BLOCK-END
