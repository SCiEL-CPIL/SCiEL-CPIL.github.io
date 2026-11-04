import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Total differences needed
    diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * diffs_per_row
    lower_bound = total_diffs
    
    best_scope = float('inf')
    best_solution = None
    solutions = []
    
    # Fast validity check using bit operations
    def check_and_get_diffs(row, x, used_diffs):
        """Check if x can be added to row given used_diffs. Return new_diffs bitset or None."""
        new_diffs = 0
        for y in row:
            d = x - y
            if d <= 0:
                return None
            bit = 1 << d
            if used_diffs & bit:
                return None
            new_diffs |= bit
        return new_diffs
    
    # Lower bound estimate: given current row with 'len(row)' elements and 'current_max',
    # estimate minimum possible final scope
    def estimate_min_scope(row, current_max, used_diffs, row_idx):
        """Rough lower bound on final scope from this state."""
        if row_idx >= n_rows:
            return current_max
        
        # Remaining rows
        remaining_rows = n_rows - row_idx
        # Current row needs (k+1 - len(row)) more elements
        elems_left_in_row = (k + 1) - len(row)
        
        # Each remaining element must create at least one new difference
        # Simple heuristic: we need at least 'elems_left_in_row' more distinct diffs in this row
        # and full diffs for remaining_rows-1 complete rows
        # This is very rough; just use current_max as a baseline
        return current_max
    
    def backtrack(row_idx, current_row, used_diffs, current_max):
        nonlocal best_scope, best_solution, solutions
        
        if time.time() > deadline:
            return False
        
        if row_idx == n_rows:
            scope = current_max
            if scope < best_scope:
                best_scope = scope
                best_solution = [r[:] for r in solutions]
            return True
        
        if len(current_row) == 0:
            current_row = [0]
            current_max = 0
        
        if len(current_row) < k + 1:
            last_val = current_row[-1]
            start_val = last_val + 1
            
            # Determine max candidate to search
            if best_scope != float('inf'):
                max_candidate = best_scope - 1
            else:
                # Heuristic: don't go too far initially
                max_candidate = min(last_val + 150, lower_bound + 80)
            
            # Pruning: if start_val > max_candidate, no valid move
            if start_val > max_candidate:
                return False
            
            # Try values in increasing order for deterministic search
            for x in range(start_val, max_candidate + 1):
                new_diffs = check_and_get_diffs(current_row, x, used_diffs)
                if new_diffs is None:
                    continue
                
                current_row.append(x)
                new_max = max(x, current_max)
                
                # Prune if we exceed best known scope
                if new_max < best_scope or best_solution is None:
                    if backtrack(row_idx, current_row, used_diffs | new_diffs, new_max):
                        current_row.pop()
                        return True
                
                current_row.pop()
            
            return False
        else:
            solutions.append(current_row[:])
            return backtrack(row_idx + 1, [], used_diffs, current_max)
    
    # Run initial deterministic backtracking
    solutions.clear()
    backtrack(0, [], 0, 0)
    
    # If time remains, try randomized restarts with different initial biases
    if best_solution is not None and time.time() < deadline - 5.0:
        # Try several randomized restarts with different random seeds
        for restart in range(20):
            if time.time() > deadline - 2.0:
                break
            
            # Randomize the order of candidate values slightly
            # We'll use a randomized greedy to find a starting point, then backtrack from there
            # Actually, let's just run backtracking again but with a randomized initial row construction
            
            # Strategy: Build first row greedily with some randomness, then backtrack the rest
            solutions.clear()
            used_diffs = 0
            current_row = [0]
            
            # Build first row with slight randomization
            for pos in range(k):
                last_val = current_row[-1]
                # Try a range of candidates, pick randomly from valid ones
                candidates = []
                max_try = min(last_val + 50, lower_bound + 50)
                for x in range(last_val + 1, max_try + 1):
                    new_diffs = check_and_get_diffs(current_row, x, used_diffs)
                    if new_diffs is not None:
                        candidates.append((x, new_diffs))
                
                if not candidates:
                    # Fallback to deterministic
                    for x in range(last_val + 1, lower_bound + 100):
                        new_diffs = check_and_get_diffs(current_row, x, used_diffs)
                        if new_diffs is not None:
                            candidates.append((x, new_diffs))
                            break
                
                if not candidates:
                    # Can't complete this row, break
                    break
                
                # Pick randomly from top candidates (weighted toward smaller values)
                # Use a softmax-like selection
                weights = [1.0 / (1.0 + i * 0.1) for i, (x, _) in enumerate(candidates)]
                total_w = sum(weights)
                r = random.random() * total_w
                cumulative = 0
                chosen_idx = 0
                for i, w in enumerate(weights):
                    cumulative += w
                    if r <= cumulative:
                        chosen_idx = i
                        break
                
                x, new_diffs = candidates[chosen_idx]
                current_row.append(x)
                used_diffs |= new_diffs
            
            # Now backtrack the remaining rows from this partial solution
            # We need to restore the state and continue
            # Actually, it's cleaner to just run full backtracking with a time limit per restart
            
            # Let's instead run a fresh backtracking with a tighter time limit
            solutions.clear()
            t_restart = time.time()
            restart_deadline = t_restart + min(5.0, deadline - t_restart - 1.0)
            
            def bt_restart(row_idx, current_row, used_diffs, current_max):
                nonlocal best_scope, best_solution, solutions, restart_deadline
                
                if time.time() > restart_deadline:
                    return False
                
                if row_idx == n_rows:
                    scope = current_max
                    if scope < best_scope:
                        best_scope = scope
                        best_solution = [r[:] for r in solutions]
                    return True
                
                if len(current_row) == 0:
                    current_row = [0]
                    current_max = 0
                
                if len(current_row) < k + 1:
                    last_val = current_row[-1]
                    start_val = last_val + 1
                    
                    if best_scope != float('inf'):
                        max_candidate = best_scope - 1
                    else:
                        max_candidate = min(last_val + 150, lower_bound + 80)
                    
                    if start_val > max_candidate:
                        return False
                    
                    # Randomize order slightly: shuffle the candidate range
                    # For efficiency, just try in order but with a random offset in tie-breaking
                    for x in range(start_val, max_candidate + 1):
                        new_diffs = check_and_get_diffs(current_row, x, used_diffs)
                        if new_diffs is None:
                            continue
                        
                        current_row.append(x)
                        new_max = max(x, current_max)
                        
                        if new_max < best_scope or best_solution is None:
                            if bt_restart(row_idx, current_row, used_diffs | new_diffs, new_max):
                                current_row.pop()
                                return True
                        
                        current_row.pop()
                    
                    return False
                else:
                    solutions.append(current_row[:])
                    return bt_restart(row_idx + 1, [], used_diffs, current_max)
            
            bt_restart(0, [], 0, 0)
    
    # Fallback if no solution found
    if best_solution is None:
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                valid = True
                new_diffs = 0
                for y in row:
                    d = x - y
                    if (used >> d) & 1:
                        valid = False
                        break
                    new_diffs |= (1 << d)
                if valid:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        best_solution = rows
        best_scope = max(max(r) for r in rows)
    
    return best_solution
