"""Rewritten program for the difference triangle set task."""
import random
import sys
import time
import math

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within each row are distinct across all rows.
    Minimizes the largest entry.
    
    Uses Simulated Annealing to find a low-conflict configuration, then
    attempts to minimize the maximum value.
    """
    # Parameters for Simulated Annealing
    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer
    
    # Initial solution: Greedy construction to get a valid (but likely suboptimal) start
    # This ensures we start with 0 conflicts.
    initial_rows = _greedy_construct(n_rows, k)
    if not initial_rows:
        return []
        
    # State: list of rows
    rows = [list(r) for r in initial_rows]
    
    # Calculate initial energy (number of duplicate differences)
    # Since greedy is valid, energy should be 0.
    # But we want to minimize the MAX VALUE, not just validity.
    # So, we will use a two-phase approach:
    # Phase 1: Minimize conflicts (if we started invalid) or maintain 0 conflicts while reducing max value.
    # Actually, a better objective for SA is: 
    # Energy = (number of duplicate diffs) * LARGE_PENALTY + max_value
    # This way, the solver tries to keep diffs unique (primary) and reduce max value (secondary).
    
    LARGE_PENALTY = 10000
    
    def get_energy(rows):
        """
        Calculate the energy of the configuration.
        Energy = count_duplicates * LARGE_PENALTY + max_value
        """
        diff_counts = {}
        max_val = 0
        for r in rows:
            if not r: continue
            max_val = max(max_val, r[-1])
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    d = r[j] - r[i]
                    if d in diff_counts:
                        diff_counts[d] += 1
                    else:
                        diff_counts[d] = 1
        
        # Count duplicates: sum(max(0, count - 1)) for each diff
        dup_count = 0
        for count in diff_counts.values():
            if count > 1:
                dup_count += count - 1
                
        return dup_count * LARGE_PENALTY + max_val

    current_energy = get_energy(rows)
    best_rows = [list(r) for r in rows]
    best_energy = current_energy
    best_max_val = max(max(r) for r in rows)
    
    # Annealing parameters
    T = 100.0  # Initial temperature
    T_min = 0.01
    alpha = 0.9995  # Cooling rate
    max_iterations = 100000
    
    # Helper to get a random valid position for an element
    # A position is valid if it maintains increasing order and doesn't immediately
    # create a self-conflict (though global conflicts are handled by energy).
    # For simplicity, we will swap elements or move them to nearby positions.
    
    def random_move(rows):
        """
        Perform a random perturbation on the rows.
        Returns the modified rows.
        """
        # Pick a random row and a random element index (excluding 0 which is fixed)
        r_idx = random.randint(0, len(rows) - 1)
        row = rows[r_idx]
        
        # We can only move elements at index > 0
        if len(row) <= 1:
            return rows
            
        # Strategy: Pick an element and try to move it to a new position
        # To keep it simple and valid (increasing), we can:
        # 1. Pick index i (1 <= i < len(row))
        # 2. Pick a new value v such that row[i-1] < v < row[i+1] (if i+1 exists)
        #    Or if i is last, v > row[i-1].
        #    We also need to ensure v is not 0.
        
        i = random.randint(1, len(row) - 1)
        old_val = row[i]
        
        # Determine valid range for new value
        lower = row[i-1] + 1
        if i < len(row) - 1:
            upper = row[i+1] - 1
        else:
            # For the last element, we can go up to some reasonable limit
            # To avoid huge jumps, let's limit the search space
            upper = max(lower, int(max_val * 1.2) + 10)
            
        if lower > upper:
            # Cannot move in place, try swapping with another element in the same row?
            # Or just return original if no space.
            # Let's try swapping with another element in the same row
            if len(row) > 2:
                j = random.randint(1, len(row) - 1)
                if j != i:
                    # Swap only if it maintains order? 
                    # Swapping breaks order. We must re-sort? 
                    # No, the row must be strictly increasing.
                    # So simple swap is invalid unless we re-sort and check.
                    # Re-sorting changes the "identity" of differences? No, differences depend on values.
                    # If we swap values, the set of values in the row is the same, so internal diffs are the same.
                    # But the row must be sorted. So we just swap and sort.
                    # However, sorting might change which value is at which index, but the set of diffs is invariant to permutation.
                    # So we can just pick a new random value for one element.
                    pass
            
            # If no valid spot, try a different row or element
            return rows

        new_val = random.randint(lower, upper)
        
        # Apply move
        rows[r_idx][i] = new_val
        
        return rows

    # Main SA Loop
    iteration = 0
    while T > T_min and time.time() - start_time < time_limit:
        new_rows = random_move(rows)
        
        # Check if the move was actually applied (random_move returns same object if invalid)
        # To properly handle this, random_move should return a new list or a flag.
        # Let's refactor random_move to return (new_rows, success)
        
        # ... Refactoring random_move inside the loop for clarity ...
        # Actually, let's just implement the move logic directly here for speed.
        
        r_idx = random.randint(0, len(rows) - 1)
        row = rows[r_idx]
        if len(row) <= 1:
            continue
            
        i = random.randint(1, len(row) - 1)
        old_val = row[i]
        
        lower = row[i-1] + 1
        if i < len(row) - 1:
            upper = row[i+1] - 1
        else:
            upper = max(lower, int(best_max_val * 1.2) + 10)
            
        if lower > upper:
            continue
            
        new_val = random.randint(lower, upper)
        
        # Apply
        row[i] = new_val
        
        # Calculate new energy
        new_energy = get_energy(rows)
        
        # Acceptance criterion
        delta_E = new_energy - current_energy
        if delta_E < 0 or random.random() < math.exp(-delta_E / T):
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_rows = [list(r) for r in rows]
                # Extract max val for best
                # We need to check if this is truly valid (0 duplicates) to update best_max_val properly
                # The energy includes penalty. If penalty is 0, it's valid.
                dup_count = current_energy // LARGE_PENALTY
                if dup_count == 0:
                    best_max_val = min(best_max_val, current_energy) # Since penalty is 0, energy == max_val
                else:
                    # If invalid, we don't update best_max_val based on this
                    # But we should track the best valid solution separately
                    pass
        else:
            # Reject: revert
            row[i] = old_val
            
        T *= alpha
        iteration += 1
        
        # Periodic restart or local search? 
        # For now, just continue.
        
        # Check time
        if time.time() - start_time > time_limit:
            break

    # After SA, we have best_rows.
    # We can try to improve the max value further with a local search on the best valid solution.
    # If best_rows is valid (0 dups), try to reduce max value.
    
    # Verify best_rows
    # If best_rows is not valid (due to SA accepting invalid states and cooling down), 
    # we should fall back to the initial greedy or the last valid state.
    # Let's check validity of best_rows.
    
    def is_valid(rows):
        used = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    d = r[j] - r[i]
                    if d in used:
                        return False
                    used.add(d)
        return True
        
    if not is_valid(best_rows):
        # Find the last valid state? 
        # We didn't track it perfectly. 
        # Let's just use the initial greedy if best is invalid.
        # Or, run a quick repair.
        # Given the penalty, SA should strongly prefer valid states.
        # If it's invalid, it's likely due to a bug or extreme cooling.
        # Let's try to repair best_rows by removing duplicates? 
        # Simpler: if invalid, return greedy.
        return initial_rows

    # Now, try to minimize the max value of the valid best_rows using a local search.
    # We can try to shift elements down if possible.
    
    current_max = max(max(r) for r in best_rows)
    
    # Local Search: Try to decrease the max element.
    # Find the row with the max element.
    # Try to move that element to a smaller value.
    
    for _ in range(10000): # Limited local search
        if time.time() - start_time > time_limit:
            break
            
        # Find the global max element
        max_row_idx = -1
        max_col_idx = -1
        global_max = -1
        for i, r in enumerate(best_rows):
            if r:
                for j, val in enumerate(r):
                    if val > global_max:
                        global_max = val
                        max_row_idx = i
                        max_col_idx = j
                        
        if max_row_idx == -1:
            break
            
        row = best_rows[max_row_idx]
        # Try to reduce row[max_col_idx]
        # It must be > row[max_col_idx-1]
        lower_bound = row[max_col_idx-1] + 1 if max_col_idx > 0 else 1
        upper_bound = row[max_col_idx] - 1
        
        if upper_bound < lower_bound:
            # Cannot reduce this specific element. 
            # Try another element in the same row? Or break.
            # Let's try to reduce any element in the row with the current max.
            # If we can't reduce the max, we can try to reduce other elements to allow future reduction?
            # For now, just break if we can't reduce the current max.
            # A more robust LS would try all elements.
            
            # Try to reduce a random element in the row to see if it opens up space?
            # No, that increases max or keeps it same. We want to decrease max.
            # So we must decrease the element that IS the max.
            # If we can't, we are stuck with this max for this row.
            # Maybe another row has a smaller max that we can reduce?
            # But global max is determined by the largest value.
            # If we can't reduce the current global max, we can't lower the score.
            break
            
        new_val = random.randint(lower_bound, upper_bound)
        
        # Check if new_val creates a conflict
        # We need to check diffs involving this element
        # Remove old diffs, add new diffs, check for duplicates
        
        old_val = row[max_col_idx]
        
        # Collect diffs to remove
        diffs_to_remove = set()
        for i in range(len(row)):
            if i == max_col_idx: continue
            d = abs(row[i] - old_val)
            diffs_to_remove.add(d)
            
        # Collect diffs to add
        diffs_to_add = set()
        for i in range(len(row)):
            if i == max_col_idx: continue
            d = abs(row[i] - new_val)
            diffs_to_add.add(d)
            
        # Check if any diff in diffs_to_add is already used by OTHER elements in the same row (not the one being moved)
        # Or by other rows.
        
        # Efficient check:
        # 1. Check against global used set (excluding the row's current contribution)
        # This is complex. Let's just use a full validation for simplicity in LS, as k is small.
        
        # Temporarily apply
        row[max_col_idx] = new_val
        
        if is_valid(best_rows):
            # Accepted
            current_max = max(max(r) for r in best_rows)
            # Continue loop
        else:
            # Revert
            row[max_col_idx] = old_val
            # Try a different random value?
            # For now, just break or try next iteration with new random
            # Let's try a few times before breaking
            # Actually, if we can't find a valid smaller value, we should break.
            # But random might find one.
            # Let's limit tries per max element
            # If we fail 10 times, break.
            # Simple approach: if one try fails, break.
            break

    return best_rows


def _greedy_construct(n_rows, k):
    """Original greedy construction."""
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
    return rows


def _counting_bound(n_rows, k):
    """
    Simple counting bound: total distinct differences needed is n_rows * k*(k+1)/2.
    The number of possible differences up to m is m(m+1)/2.
    So we need m(m+1)/2 >= n_rows * k*(k+1)/2
    => m(m+1) >= n_rows * k*(k+1)
    """
    target = n_rows * k * (k + 1)
    m = 1
    while m * (m + 1) < target:
        m += 1
    return m
# EVOLVE-BLOCK-END
