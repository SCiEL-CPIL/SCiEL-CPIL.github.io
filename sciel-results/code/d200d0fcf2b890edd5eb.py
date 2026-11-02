import time
import random
import sys

def construct(n_rows, k):
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    best_scope = float('inf')
    best_rows = None
    
    # Each row has k+1 elements, so C(k+1, 2) differences
    diffs_per_row = (k + 1) * k // 2
    
    def get_row_diffs(row):
        """Return list of differences for a single row."""
        return [row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row))]

    def is_valid_rows(rows):
        """Check if all differences across all rows are unique."""
        all_diffs = []
        for row in rows:
            all_diffs.extend(get_row_diffs(row))
        return len(all_diffs) == len(set(all_diffs))

    def scope(rows):
        return max(max(row) for row in rows)

    def greedy_row(used_diffs, max_scope_hint=None):
        """
        Generate a single valid row given the currently used differences.
        Returns the row or None if it can't be constructed within reasonable bounds.
        """
        row = [0]
        current_max = 0
        limit = max_scope_hint if max_scope_hint else 10000
        
        while len(row) < k + 1:
            # Try to find the next smallest valid number
            next_val = row[-1] + 1
            found = False
            while next_val < limit:
                new_diffs = [next_val - x for x in row]
                # Check if any new diff is already used or duplicate within new_diffs
                if len(set(new_diffs)) == len(new_diffs) and not any(d in used_diffs for d in new_diffs):
                    row.append(next_val)
                    # Update local used_diffs? No, we pass the global one, 
                    # but for generation we just need to ensure no conflict with EXISTING rows.
                    # The new diffs are not yet in used_diffs, so we just check against used_diffs.
                    # However, we must ensure new_diffs don't conflict with each other (handled by set check)
                    # and don't conflict with used_diffs (handled by any check).
                    # We do NOT add to used_diffs here because this is just generation.
                    # The caller will add them.
                    found = True
                    break
                next_val += 1
            
            if not found:
                return None
            
            current_max = row[-1]
            
        return row

    def greedy_construct(used_diffs=None):
        """
        Construct a full set of rows using greedy method.
        """
        if used_diffs is None:
            used_diffs = set()
        
        rows = []
        for _ in range(n_rows):
            row = greedy_row(used_diffs)
            if row is None:
                return None
            rows.append(row)
            # Add differences of this row to used_diffs
            used_diffs.update(get_row_diffs(row))
            
        return rows

    # Initial solution
    initial_rows = greedy_construct()
    if initial_rows is None:
        # Fallback: very loose greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = [x - a for a in row]
                if not any(d in used for d in new):
                    row.append(x)
                    used.update(new)
            rows.append(row)
        initial_rows = rows
        
    best_rows = [row[:] for row in initial_rows]
    best_scope = scope(best_rows)
    
    # Simulated Annealing
    def simulated_annealing(rows, time_limit):
        start_time = time.time()
        current_rows = [row[:] for row in rows]
        current_scope = scope(current_rows)
        best_local_rows = [row[:] for row in current_rows]
        best_local_scope = current_scope
        
        # Initialize used_diffs set for fast lookup
        used_diffs = set()
        for r in current_rows:
            used_diffs.update(get_row_diffs(r))
            
        temperature = max(current_scope * 0.1, 10.0)
        min_temp = 1.0
        cooling_rate = 0.9995
        
        while time.time() - start_time < time_limit:
            if temperature < min_temp:
                break
                
            # Choose a random move
            move_type = random.random()
            
            if move_type < 0.4:
                # Move 1: Regenerate a random row
                r_idx = random.randint(0, n_rows - 1)
                old_row = current_rows[r_idx]
                
                # Remove old row's diffs from used_diffs
                old_diffs = get_row_diffs(old_row)
                for d in old_diffs:
                    used_diffs.discard(d)
                
                # Generate new row
                new_row = greedy_row(used_diffs, max_scope_hint=best_local_scope * 1.5)
                
                if new_row is not None:
                    new_diffs = get_row_diffs(new_row)
                    # Check for internal duplicates (greedy_row ensures this, but double check)
                    if len(set(new_diffs)) == len(new_diffs):
                        # Check for conflicts with existing used_diffs
                        # greedy_row already checked against used_diffs, so this should be safe
                        # But let's verify no new_diff is in used_diffs
                        conflict = False
                        for d in new_diffs:
                            if d in used_diffs:
                                conflict = True
                                break
                                
                        if not conflict:
                            # Accept the move
                            current_rows[r_idx] = new_row
                            used_diffs.update(new_diffs)
                            new_scope = scope(current_rows)
                            
                            delta = new_scope - current_scope
                            if delta < 0 or random.random() < pow(2.71828, -delta / temperature):
                                current_scope = new_scope
                                if new_scope < best_local_scope:
                                    best_local_scope = new_scope
                                    best_local_rows = [row[:] for row in current_rows]
                            else:
                                # Reject move, revert
                                current_rows[r_idx] = old_row
                                used_diffs.update(old_diffs)
                        else:
                            # Revert
                            current_rows[r_idx] = old_row
                            used_diffs.update(old_diffs)
                    else:
                        current_rows[r_idx] = old_row
                        used_diffs.update(old_diffs)
                else:
                    # Couldn't generate, revert (nothing changed)
                    current_rows[r_idx] = old_row
                    used_diffs.update(old_diffs)
                    
            elif move_type < 0.7:
                # Move 2: Perturb an entry in a random row
                r_idx = random.randint(0, n_rows - 1)
                row = current_rows[r_idx]
                if len(row) <= 1:
                    continue
                # Pick an index to change (not the first one)
                i_idx = random.randint(1, len(row) - 1)
                old_val = row[i_idx]
                
                # Determine bounds for new value
                min_val = row[i_idx - 1] + 1
                max_val = (row[i_idx + 1] - 1) if i_idx < len(row) - 1 else best_local_scope + 10
                
                if min_val >= max_val:
                    continue
                
                new_val = random.randint(min_val, max_val)
                if new_val == old_val:
                    continue
                
                # Calculate new diffs for this row
                # Remove old diffs
                old_diffs = get_row_diffs(row)
                for d in old_diffs:
                    used_diffs.discard(d)
                
                # Apply change
                row[i_idx] = new_val
                new_diffs = get_row_diffs(row)
                
                # Check validity
                valid = True
                if len(set(new_diffs)) != len(new_diffs):
                    valid = False
                else:
                    for d in new_diffs:
                        if d in used_diffs:
                            valid = False
                            break
                
                if valid:
                    used_diffs.update(new_diffs)
                    new_scope = scope(current_rows)
                    delta = new_scope - current_scope
                    if delta < 0 or random.random() < pow(2.71828, -delta / temperature):
                        current_scope = new_scope
                        if new_scope < best_local_scope:
                            best_local_scope = new_scope
                            best_local_rows = [row[:] for row in current_rows]
                    else:
                        # Revert
                        row[i_idx] = old_val
                        used_diffs.update(old_diffs)
                else:
                    # Revert
                    row[i_idx] = old_val
                    used_diffs.update(old_diffs)
                    
            else:
                # Move 3: Swap two rows (heuristic, might help symmetry breaking in search)
                r1 = random.randint(0, n_rows - 1)
                r2 = random.randint(0, n_rows - 1)
                if r1 != r2:
                    current_rows[r1], current_rows[r2] = current_rows[r2], current_rows[r1]
                    # Scope doesn't change with swap, but it's a valid move to explore
                    # We don't need to update used_diffs as it's a set
                    # We accept this move always as it doesn't change energy
                    # But it doesn't change best_local_scope either
                    pass

            temperature *= cooling_rate
            
        return best_local_rows, best_local_scope

    # Run multiple SA instances
    num_sa_runs = 0
    while time.time() < deadline - 10.0:
        # Run SA for a chunk of time
        time_per_run = min(20.0, deadline - 10.0 - time.time())
        if time_per_run < 2.0:
            break
            
        rows, sc = simulated_annealing(best_rows, time_per_run)
        
        if sc < best_scope:
            best_scope = sc
            best_rows = rows
            num_sa_runs += 1
        else:
            # If no improvement, maybe restart from a fresh greedy solution?
            # Or just keep using the current best as start point.
            # Let's try restarting from a new greedy solution occasionally to escape deep local minima
            if num_sa_runs % 3 == 2:
                new_greedy = greedy_construct()
                if new_greedy:
                    gs = scope(new_greedy)
                    if gs < best_scope:
                        best_rows = new_greedy
                        best_scope = gs
                    else:
                        # Start SA from the better of the two
                        pass

    return best_rows

