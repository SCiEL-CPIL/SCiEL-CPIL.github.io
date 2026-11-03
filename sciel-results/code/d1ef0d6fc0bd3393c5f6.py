import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 100.0  # Leave margin for 120s limit
    
    best_scope = float('inf')
    best_rows = None
    
    # Pre-compute nothing, just run randomized search
    for attempt in range(100000):
        if time.time() - start_time > time_limit:
            break
            
        used = set()
        rows = []
        
        # Heuristic: Try to build rows in a randomized order of "difficulty"? 
        # Or just standard order but with randomized choices.
        # Standard order is usually fine for greedy.
        
        success = True
        for i in range(n_rows):
            row = [0]
            # Backtracking stack for this row
            # We need to be able to undo choices in the current row
            # Simple backtracking within the row construction
            
            # Instead of full backtracking tree which might be deep, 
            # let's use a randomized greedy with "lookahead" or just restarts.
            # Full backtracking for 7 rows of 6 elements might be feasible if pruned well.
            
            # Let's try a simpler randomized greedy with restarts first.
            # If that fails to improve significantly, we can add backtracking.
            
            # Randomized Greedy Step:
            # For each position in the row, pick a random valid candidate from the smallest few.
            
            current_row = [0]
            dead_end = False
            
            for pos in range(1, k + 1):
                candidates = []
                # Upper bound for search: current best scope or a reasonable limit
                # If we have a best_scope, we don't need to exceed it.
                # If not, we can search up to, say, 200.
                max_val = best_scope if best_scope < float('inf') else 200
                
                # Find all valid x
                # To speed up, we can iterate x from current_row[-1] + 1 to max_val
                valid_xs = []
                for x in range(current_row[-1] + 1, max_val + 1):
                    diffs = {x - a for a in current_row}
                    if not (diffs & used):
                        valid_xs.append(x)
                    
                    # Early termination if we found enough candidates? 
                    # No, we need the smallest ones for greedy.
                    # But for randomized, we just need a set of "good" candidates.
                    
                    # Optimization: if we have many, stop early? 
                    # No, we want the smallest ones to minimize scope.
                    
                if not valid_xs:
                    dead_end = True
                    break
                
                # Select a candidate
                # Strategy: Pick from the smallest 10% or top 5 candidates randomly
                # This encourages small scope but allows exploration.
                # Sort is already done by range iteration.
                
                # Take the first few valid candidates
                # The number of candidates can be large. 
                # Let's take the smallest 5.
                top_candidates = valid_xs[:5]
                x = random.choice(top_candidates)
                
                current_row.append(x)
                used.update({x - a for a in current_row[:-1]}) # Exclude x-x=0
                
            if not dead_end:
                rows.append(current_row)
            else:
                success = False
                break
        
        if success:
            scope = max(max(r) for r in rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rows]
                
                # If we hit the theoretical lower bound or a very good bound, we can stop?
                # Lower bound is 105. If we hit 105, we can stop.
                if best_scope <= 105:
                    break

    if best_rows is None:
        # Fallback to deterministic greedy if random search fails completely
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    return best_rows
