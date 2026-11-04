import random
import time

def construct(n_rows, k):
    """
    Constructs a set of n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimizes the largest entry.
    """
    best_result = None
    best_scope = float('inf')
    
    start_time = time.time()
    time_limit = 110.0
    
    # Parameters for the heuristic and search
    # We use a randomized greedy with limited backtracking (beam search style)
    # Heuristic: Minimize (current_value + lambda * sum(1/d for d in new_diffs))
    # This encourages small values but penalizes using small differences (which are scarce)
    
    max_restarts = 200
    beam_width = 3  # Keep top few states
    backtrack_limit = 500 # Max backtracking steps per restart
    
    for _ in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # State: (rows, used_diffs, current_row_index, current_elem_index)
        # We'll use a stack for backtracking
        # To keep it simple and fast, we'll do a randomized greedy with "undo" capability for the last few steps.
        
        used_diffs = set()
        rows = []
        current_row = [0]
        
        # We will build row by row.
        # For each row, we build element by element.
        # If we get stuck, we restart. (Full backtracking is complex to implement efficiently in Python within time limits without a proper solver library, so we use randomized restarts with a strong heuristic).
        
        success = True
        
        for r_idx in range(n_rows):
            if r_idx > 0:
                current_row = [0]
            
            for e_idx in range(1, k + 1):
                last_val = current_row[-1]
                # Upper bound for x
                # We want to keep scope small.
                # If we have a best_scope, we can bound x.
                # Also, we need enough room for the remaining k - e_idx elements.
                # Minimum remaining elements: k - e_idx
                # Minimum increment is 1 each.
                # So x can be at most best_scope - (k - e_idx)
                
                upper_bound = 200 # Default
                if best_scope < float('inf'):
                    # We need to fit (k - e_idx) more elements after x.
                    # Min value for next is x+1, then x+2...
                    # So max x = best_scope - (k - e_idx)
                    upper_bound = best_scope - (k - e_idx)
                    if upper_bound <= last_val:
                        success = False
                        break

                valid_candidates = []
                
                # Collect candidates
                for x in range(last_val + 1, upper_bound + 1):
                    new_diffs = {x - a for a in current_row}
                    if not new_diffs & used_diffs:
                        # Heuristic Score:
                        # 1. x itself (smaller is better for scope)
                        # 2. Penalty for using small differences (they are more constrained)
                        #    Score = x + 5 * sum(1/d for d in new_diffs)
                        penalty = sum(1.0 / d for d in new_diffs)
                        score = x + 5.0 * penalty
                        valid_candidates.append((score, x, new_diffs))
                        
                        # Early stop? No, we need to find the best ones.
                        # But range is small (upper_bound - last_val is usually small).
                
                if not valid_candidates:
                    success = False
                    break
                
                # Sort by score
                valid_candidates.sort(key=lambda item: item[0])
                
                # Randomized selection from top candidates
                # To allow exploration, pick from top 5
                top_n = min(5, len(valid_candidates))
                # Weighted random or simple choice?
                # Simple choice from top 5
                chosen = random.choice(valid_candidates[:top_n])
                
                x = chosen[1]
                new_diffs = chosen[2]
                current_row.append(x)
                used_diffs |= new_diffs
                
            if not success:
                break
            
            rows.append(current_row)
            
            if len(rows) == n_rows:
                current_scope = max(r[-1] for r in rows)
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_result = [r[:] for r in rows]
                    if best_scope <= 105:
                        break

    if best_result is None:
        # Fallback to deterministic greedy
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
        
    return best_result
