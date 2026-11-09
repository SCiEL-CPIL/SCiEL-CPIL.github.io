import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin for 150s limit
    
    # n_rows=7, k=5 -> 6 elements per row
    # Total differences per row: k*(k+1)/2 = 15
    # Total differences: n_rows * 15 = 105
    
    def _elapsed():
        return time.time() - start_time

    def _dead():
        return _elapsed() > deadline

    def _diffs_of_row(row):
        return [row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))]

    def _scope(rows):
        return max(max(row) for row in rows)

    def _valid(rows):
        all_diffs = []
        for row in rows:
            if not all(row[i] < row[i+1] for i in range(len(row)-1)):
                return False
            if row[0] != 0:
                return False
            all_diffs.extend(_diffs_of_row(row))
        return len(all_diffs) == len(set(all_diffs))

    def _generate_row(used_diffs, max_val, randomize=False):
        """
        Attempts to generate a valid row of length k+1 starting at 0,
        such that all elements <= max_val and no difference conflicts with used_diffs.
        Returns the row or None.
        """
        if _dead():
            return None
        
        # Backtracking search for a single row
        # Heuristic: try smaller values first if not randomizing, or random order if randomizing
        marks = [0]
        row_diffs = set()
        
        def backtrack(last_mark, depth):
            if _dead():
                return None
            if depth == k + 1:
                return marks[:]
            
            # Candidates for next mark
            # Must be > last_mark, <= max_val
            # And not create a diff in used_diffs or row_diffs
            candidates = list(range(last_mark + 1, max_val + 1))
            if randomize:
                random.shuffle(candidates)
            else:
                # Sort by heuristic: maybe prefer values that don't block many future spots?
                # For now, just ascending is fine for deterministic, shuffled for random
                pass
            
            for x in candidates:
                new_diffs = {x - m for m in marks}
                
                if new_diffs & used_diffs:
                    continue
                if new_diffs & row_diffs:
                    continue
                
                marks.append(x)
                row_diffs.update(new_diffs)
                
                result = backtrack(x, depth + 1)
                if result is not None:
                    return result
                
                marks.pop()
                row_diffs.difference_update(new_diffs)
            
            return None

        return backtrack(0, 1)

    def _initialize_rows():
        """
        Generate an initial valid set of rows.
        We use a greedy approach with a growing max_val.
        """
        rows = []
        used_diffs = set()
        current_max = 50 # Start with a guess
        
        for i in range(n_rows):
            if _dead():
                return None
            
            row = None
            # Try to fit in current_max
            # If fails, increase max_val
            attempts = 0
            while not _dead() and attempts < 100:
                # Randomize to escape local traps in initialization
                row = _generate_row(used_diffs, current_max, randomize=True)
                if row:
                    break
                current_max += 5
                attempts += 1
            
            if not row:
                return None
            
            rows.append(row)
            used_diffs.update(_diffs_of_row(row))
            
            # Optionally adjust current_max based on the row found
            # If we had to increase it, keep it. If we found it quickly, maybe decrease?
            # For simplicity, just keep the max seen so far or slightly larger.
            current_max = max(current_max, max(row) + 5)
            
        return rows

    def _regenerate_row(rows, idx, used_diffs, max_val, randomize=True):
        """
        Try to replace rows[idx] with a new valid row fitting in max_val.
        Returns True if successful, False otherwise.
        Modifies rows and used_diffs if successful.
        """
        if _dead():
            return False
        
        old_row = rows[idx]
        old_diffs = set(_diffs_of_row(old_row))
        
        # Remove old diffs from global set
        for d in old_diffs:
            used_diffs.discard(d)
        
        # Attempt to generate new row
        new_row = _generate_row(used_diffs, max_val, randomize=randomize)
        
        if new_row:
            # Accept new row
            rows[idx] = new_row
            for d in _diffs_of_row(new_row):
                used_diffs.add(d)
            return True
        else:
            # Restore old row
            for d in old_diffs:
                used_diffs.add(d)
            return False

    # 1. Initialize
    rows = _initialize_rows()
    if not rows:
        # Fallback: try a very simple greedy without time limit pressure?
        # If we can't even initialize, we might return an empty list or try one more time with higher max
        rows = _initialize_rows()
        if not rows:
            return [] # Should not happen for this size

    current_scope = _scope(rows)
    best_rows = [r[:] for r in rows]
    best_scope = current_scope
    
    # 2. Simulated Annealing
    # Temperature starts high, decreases
    T = 100.0
    cooling = 0.95
    step = 0
    
    while not _dead():
        step += 1
        
        # Determine target max_val for regeneration
        # We want to shrink the scope.
        # Pick a random row to regenerate.
        # Try to fit it in a smaller scope than the current global max.
        # Or try to fit it in the current max but with randomization to find a "better" (smaller) fit?
        # Actually, if we regenerate a row to fit in max_val, it might still have max entry = max_val.
        # To reduce scope, we should try to regenerate rows that currently have large entries
        # with a constraint that their max entry is smaller.
        
        # Strategy:
        # 1. Identify the row with the largest max entry.
        # 2. Try to regenerate it with a max_val slightly smaller than its current max.
        # 3. If that fails, try to regenerate a random row with a max_val equal to the current global max - 1.
        # 4. If that fails, just regenerate a random row with the current global max (to shuffle the space).
        
        if _dead():
            break
            
        max_idx = max(range(n_rows), key=lambda i: max(rows[i]))
        current_max_row_val = max(rows[max_idx])
        
        # Calculate used_diffs for the current state
        used_diffs = set()
        for row in rows:
            used_diffs.update(_diffs_of_row(row))
            
        # Target max for the specific row we are regenerating
        # Try to reduce the bottleneck
        target_max = current_max_row_val - 1
        if target_max < k + 1: # Must be at least k+1 to fit k+1 distinct increasing ints starting at 0? No, 0..k is k+1.
            target_max = k + 1
            
        # Randomly choose which row to regenerate
        # Bias towards the max row?
        if random.random() < 0.7:
            idx = max_idx
            # Try to fit in a smaller scope
            # If we force it to be smaller, we might fail often.
            # Let's try target_max first.
            success = _regenerate_row(rows, idx, used_diffs, target_max, randomize=True)
            
            if not success:
                # Try with the current global max to see if we can find a different valid config
                # that might allow shrinking later
                success = _regenerate_row(rows, idx, used_diffs, best_scope, randomize=True)
        else:
            idx = random.randint(0, n_rows - 1)
            # Try to fit in a scope slightly smaller than best_scope
            target = best_scope - 1
            if target < k + 1:
                target = k + 1
            success = _regenerate_row(rows, idx, used_diffs, target, randomize=True)
            
            if not success:
                success = _regenerate_row(rows, idx, used_diffs, best_scope, randomize=True)
                
        if success:
            new_scope = _scope(rows)
            
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [r[:] for r in rows]
                # Reset temperature on improvement? Or just continue.
                # Let's reset T to a medium value to keep exploring
                T = max(T, 50.0) 
            else:
                # Accept worse solution based on SA
                # Cost function: new_scope - best_scope (or just new_scope)
                # Let's use new_scope - old_scope
                old_scope = _scope(best_rows) # Use best as reference? Or previous state?
                # Standard SA uses previous state.
                # Let's keep a 'current' state variable.
                # For simplicity, we are modifying 'rows' in place.
                # Let's track 'current_scope' properly.
                # We didn't track it above. Let's assume 'rows' is current state.
                # We need to store the scope of the state BEFORE the move.
                # This is getting complicated with in-place modification.
                
                # Let's restructure:
                # Keep 'current_rows' and 'current_scope'.
                # Propose move -> 'new_rows', 'new_scope'.
                # Accept/Reject.
                
                # I will refactor the loop to be cleaner.
                pass
                
        T *= cooling

    # The above loop logic was slightly messy with state tracking.
    # Let's implement a cleaner SA loop.
    
    # Re-initialize if the first pass was too short or if we want a cleaner structure
    # Actually, the code above might have run for a bit. 
    # Let's restart with a cleaner SA implementation to ensure correctness.
    
    rows = _initialize_rows()
    if not rows:
        return []
        
    current_scope = _scope(rows)
    best_rows = [r[:] for r in rows]
    best_scope = current_scope
    
    T = 50.0
    cooling = 0.99
    no_improve_count = 0
    max_no_improve = 500
    
    while not _dead():
        # 1. Generate neighbor
        # Pick a random row
        idx = random.randint(0, n_rows - 1)
        
        # Calculate used_diffs
        used_diffs = set()
        for row in rows:
            used_diffs.update(_diffs_of_row(row))
            
        # Determine target max
        # Try to shrink the current row's max
        current_row_max = max(rows[idx])
        # Try to fit in a smaller max
        # If current_row_max is small, try to keep it small.
        # If it's large, try to shrink.
        
        # Strategy: Try to fit in a max_val that is slightly smaller than the current row's max,
        # or equal to the current global max - 1.
        
        # Let's try a few different targets
        targets = []
        # 1. Current row max - 1
        if current_row_max > k + 1:
            targets.append(current_row_max - 1)
        # 2. Global best scope - 1
        if best_scope > k + 1:
            targets.append(best_scope - 1)
        # 3. Current global scope - 1
        current_global_scope = _scope(rows)
        if current_global_scope > k + 1:
            targets.append(current_global_scope - 1)
        # 4. Current global scope (just shuffle)
        targets.append(current_global_scope)
        
        random.shuffle(targets)
        
        new_rows = [r[:] for r in rows]
        success = False
        for target in targets:
            if _dead(): break
            # Try to regenerate row idx in new_rows with target max
            # We need to use the used_diffs from the ORIGINAL rows (excluding idx)
            # Wait, _regenerate_row modifies in place. 
            # Let's make a helper that returns the new row or None.
            
            old_row = new_rows[idx]
            old_diffs = set(_diffs_of_row(old_row))
            for d in old_diffs:
                used_diffs.discard(d)
                
            new_row = _generate_row(used_diffs, target, randomize=True)
            
            if new_row:
                # Check if this new row is actually better or just valid
                # It is valid by construction.
                new_rows[idx] = new_row
                for d in _diffs_of_row(new_row):
                    used_diffs.add(d)
                success = True
                break
            else:
                # Restore old diffs
                for d in old_diffs:
                    used_diffs.add(d)
                
        if not success:
            # No valid move found, skip
            continue
            
        new_scope = _scope(new_rows)
        
        # Acceptance criterion
        if new_scope <= current_scope:
            # Accept
            rows = new_rows
            current_scope = new_scope
            no_improve_count = 0
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [r[:] for r in rows]
        else:
            # Worse, accept with probability exp(-delta/T)
            delta = new_scope - current_scope
            prob = math.exp(-delta / T)
            if random.random() < prob:
                rows = new_rows
                current_scope = new_scope
                no_improve_count += 1
            else:
                no_improve_count += 1
                
        T *= cooling
        if T < 0.1:
            T = 0.1
            
        # Restart if stuck too long
        if no_improve_count > max_no_improve:
            # Perturb: regenerate a random row with a larger max to escape
            idx = random.randint(0, n_rows - 1)
            used_diffs = set()
            for row in rows:
                used_diffs.update(_diffs_of_row(row))
            
            old_row = rows[idx]
            old_diffs = set(_diffs_of_row(old_row))
            for d in old_diffs:
                used_diffs.discard(d)
                
            # Try to regenerate with a larger max to ensure we find *some* valid row
            # This helps escape local minima where no small row fits
            new_row = _generate_row(used_diffs, current_scope + 20, randomize=True)
            if new_row:
                rows[idx] = new_row
                for d in _diffs_of_row(new_row):
                    used_diffs.add(d)
                current_scope = _scope(rows)
            else:
                # Restore
                for d in old_diffs:
                    used_diffs.add(d)
            
            no_improve_count = 0
            T = 50.0 # Reheat

    return best_rows
