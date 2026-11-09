import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_rows = None
    best_max = float('inf')
    
    # Initial greedy solution to set a good upper bound
    used_diffs = set()
    greedy_rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used_diffs:
                row.append(x)
                used_diffs |= new_diffs
        greedy_rows.append(row)
    best_max = max(max(r) for r in greedy_rows)
    best_rows = greedy_rows

    # Randomized search to improve
    # We will try to find a solution with max < best_max
    # We use backtracking for each row, with randomization
    
    # To speed up, we can pre-compute some things? No, state is dynamic.
    
    def backtrack_row(used_diffs, k, max_allowed, row, idx, random_state):
        if idx == k:
            return row
        
        last = row[-1]
        # Candidates must be > last
        # And diffs must be disjoint from used_diffs
        # And last element must be <= max_allowed
        
        candidates = []
        # We don't know the exact range, but we can start from last+1
        # and go up to max_allowed
        # To be efficient, we can iterate and filter
        
        # Optimization: If we have a current best, we can stop searching if we exceed it
        # But this is per-row, and the row's max is the last element.
        
        # Let's generate candidates up to a limit
        # A rough upper bound for the next element:
        # If we need 'rem' more elements, and min diff is 1, 
        # we need space. But diffs must be distinct.
        # Simple bound: last + rem * 1 is too tight.
        # Let's just iterate from last+1 to max_allowed.
        
        # To randomize, we can collect valid candidates and pick one
        # But backtracking requires trying them in order.
        # We can shuffle the candidate list.
        
        valid_candidates = []
        # We can limit the search space by not going too far
        # If we are too far, we might exceed max_allowed
        # Let's iterate
        for x in range(last + 1, max_allowed + 1):
            new_diffs = {x - a for a in row}
            if not new_diffs & used_diffs:
                valid_candidates.append(x)
                
        if not valid_candidates:
            return None
            
        # Shuffle to explore different paths
        random_state.shuffle(valid_candidates)
        
        for x in valid_candidates:
            # Pruning: if x is already too large? 
            # We are minimizing the max, so any x <= max_allowed is potentially useful.
            
            new_row = row + [x]
            new_diffs = {x - a for a in row}
            
            # Update used_diffs for recursion
            # We need to pass a copy or use a set with add/remove
            # Since n_rows and k are small, copying the set is fine.
            # Or we can use a list of sets? No, just copy.
            # Actually, copying a set of ~100 ints is fast.
            
            res = backtrack_row(used_diffs | new_diffs, k, max_allowed, new_row, idx + 1, random_state)
            if res is not None:
                return res
                
        return None

    # Main loop
    while time.time() < deadline:
        random_state = random.Random()
        # We try to find a solution with max < best_max
        # If we find one, we update best_max and continue to find even better
        # If we don't find one, we might want to try a slightly larger bound?
        # No, we want to minimize. So we try to beat the current best.
        
        # To make it more likely to find a good solution, we can start with a slightly higher bound
        # and then tighten? Or just use best_max - 1.
        # If we use best_max - 1, we force an improvement.
        # If we can't find one, we might be stuck.
        # So we can try with best_max (to find alternative solutions) or best_max - 1.
        # Let's try best_max - 1 to force improvement.
        # If we fail many times, we can relax.
        
        target_max = best_max - 1
        
        # Construct rows one by one
        # Note: The choice of Row 1 affects Row 2.
        # So we need to backtrack over ALL rows?
        # That would be expensive.
        # Instead, we can use the "randomized greedy with backtracking for current row" approach.
        # We build Row 1. Then Row 2.
        # If Row 2 fails, we backtrack Row 1?
        # This is complex.
        
        # Alternative: Just use the randomized greedy (window) which is faster.
        # Let's switch back to Randomized Greedy with Window, it's simpler and often effective.
        
        used_diffs = set()
        rows = []
        success = True
        for i in range(n_rows):
            row = [0]
            # We want to keep the last element small
            # So we limit the search for x
            # How much can we grow?
            # If we have 'rem' elements left, and we are at 'last',
            # we need to fit 'rem' more elements.
            # The minimum possible last element is roughly last + rem * (avg_diff).
            # Let's just use the greedy logic but with randomization.
            
            while len(row) < k + 1:
                last = row[-1]
                # Find valid candidates
                # We don't know the upper bound for this specific row yet,
                # but we know the global target_max.
                # So x must be <= target_max.
                
                valid_candidates = []
                # We can limit the range of x to avoid searching too far
                # If we are close to target_max, we have few options.
                
                # Heuristic: The next element should not be too large
                # Let's iterate from last+1 to target_max
                # This might be slow if target_max is large (134).
                # 134 iterations per step, 7 steps, 4 rows -> 134*7*4 = ~3700 checks.
                # Each check is O(k) for diffs. 3700 * 7 = ~26k ops. Very fast.
                
                for x in range(last + 1, target_max + 1):
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used_diffs:
                        valid_candidates.append(x)
                        
                if not valid_candidates:
                    success = False
                    break
                    
                # Pick randomly from the first few candidates to diversify
                # But we want small x. So pick from the first W candidates.
                W = 3
                if len(valid_candidates) > W:
                    # Pick randomly from the first W
                    x = random_state.choice(valid_candidates[:W])
                else:
                    x = random_state.choice(valid_candidates)
                    
                row.append(x)
                used_diffs |= {x - a for a in row[:-1]} # Note: row[:-1] are the previous elements
                
            if success:
                rows.append(row)
                
        if success:
            current_max = max(max(r) for r in rows)
            if current_max < best_max:
                best_max = current_max
                best_rows = rows
            # If we found a solution with max == target_max (which is best_max - 1),
            # we updated best_max.
            # We can continue to try to improve further.

    return best_rows
