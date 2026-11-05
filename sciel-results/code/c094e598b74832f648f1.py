import time
import random
import math
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Lower bound on the maximum entry (scope)
    total_diffs_needed = n_rows * (k + 1) * k // 2
    # Solve M(M+1)/2 >= total_diffs_needed
    lower_bound = math.ceil((-1 + math.sqrt(1 + 8 * total_diffs_needed)) / 2.0)
    
    # Initial greedy construction to get a valid upper bound
    def greedy_construct():
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used_diffs):
                    row.append(x)
                    used_diffs.update(new_diffs)
            rows.append(row)
        return rows, max(max(r) for r in rows)

    initial_rows, initial_max = greedy_construct()
    best_rows = [r[:] for r in initial_rows]
    best_max = initial_max
    
    # Bitset for fast difference checking
    # We assume max_scope won't be too large (e.g., < 10000) for bitset to be efficient in Python
    # If it's larger, we might fall back to set, but for (4,7) it's ~134, so bitset is fine.
    # Let's use an integer bitmask for differences.
    # diff d corresponds to bit d.
    
    def get_row_diffs_mask(row):
        mask = 0
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                mask |= (1 << (row[j] - row[i]))
        return mask

    def backtrack_solve(target_max):
        """
        Attempts to find n_rows rows with max entry <= target_max.
        Returns list of rows or None.
        """
        used_mask = 0
        rows = []
        
        def bt(idx):
            nonlocal used_mask, rows
            if time.time() > deadline:
                return False
            
            if idx == n_rows:
                return True
            
            # We need to find a row of length k+1, starting with 0, increasing, max <= target_max
            # such that its diff mask is disjoint from used_mask.
            
            # Heuristic: Try to construct the row.
            # We can use a recursive helper to build the row element by element.
            # To add diversity and avoid getting stuck in the same local minima,
            # we can randomize the choice of the next element from valid candidates.
            
            # However, doing this purely recursively for each row generation inside bt
            # can be slow. Let's try a specific approach:
            # Generate candidate rows using a randomized greedy method with pruning.
            
            # We'll try a few random greedy constructions for this row.
            # If any works, we commit and recurse.
            
            # To ensure we explore different rows, we shuffle the priority of candidates
            # or pick randomly from a set of valid next steps.
            
            # Let's implement a helper that generates ONE valid row given used_mask.
            # If it fails, we try again with different random seeds/choices.
            
            # Actually, to be robust, let's integrate the row construction into the backtracking
            # by choosing the next element of the current row step-by-step.
            # State: current_row, current_mask, next_pos
            
            # But standard backtracking is over rows.
            # Let's stick to: "Try to find a row".
            # We will try up to `attempts` random greedy rows.
            
            attempts = 20
            for _ in range(attempts):
                row = [0]
                row_mask = 0
                success = True
                for pos in range(1, k + 1):
                    # Find valid next elements
                    # x > row[-1], x <= target_max
                    # new_diffs = {x - a for a in row}
                    # new_mask = sum(1 << d for d in new_diffs)
                    # if (row_mask | new_mask) & used_mask == 0:
                    #    valid
                    
                    # Optimization: Pre-compute potential diffs?
                    # For small k, iterating x is fine.
                    
                    valid_xs = []
                    # Randomize start or just iterate?
                    # Iterating from row[-1]+1 to target_max is O(M).
                    # M is ~100-200. k is 7.
                    # This is fast.
                    
                    for x in range(row[-1] + 1, target_max + 1):
                        new_mask = 0
                        # Check diffs with existing elements in row
                        valid = True
                        for a in row:
                            d = x - a
                            bit = 1 << d
                            if (bit & (row_mask | used_mask)):
                                valid = False
                                break
                            new_mask |= bit
                        if valid:
                            valid_xs.append((x, new_mask))
                    
                    if not valid_xs:
                        success = False
                        break
                    
                    # Heuristic:
                    # If we want to keep the row "tight", pick smallest x.
                    # If we want diversity, pick randomly.
                    # Let's pick randomly from the first half of valid_xs to balance tightness and diversity.
                    # Or just random.choice(valid_xs).
                    # random.choice is better for exploring different row structures.
                    x, new_mask = random.choice(valid_xs)
                    row.append(x)
                    row_mask |= new_mask
                
                if success:
                    # Check if the whole row is valid (redundant if we checked step by step, but safe)
                    if not (row_mask & used_mask):
                        # Commit
                        used_mask |= row_mask
                        rows.append(row)
                        
                        if bt(idx + 1):
                            return True
                        
                        # Undo
                        rows.pop()
                        used_mask ^= row_mask # XOR works if no overlap, which is guaranteed
                        # Note: used_mask |= row_mask then used_mask &= ~row_mask is safer if we used OR.
                        # But since we know row_mask is disjoint from previous used_mask,
                        # and we just added it, XOR removes exactly those bits.
            
            return False
        
        return bt(0) if bt(0) else None

    # Main search loop
    # Try to improve upon initial_max
    # We search downwards from initial_max - 1 to lower_bound
    
    current_target = initial_max - 1
    
    # To avoid getting stuck, we can limit time per target
    # But simple loop is okay if bt is fast.
    
    # We'll try targets in decreasing order.
    # If we find a solution, we update best and continue down.
    
    # Optimization: If initial_max is already close to lower_bound, we might not need to search much.
    
    # Let's add a time check inside the main loop too.
    
    # We'll use a simple loop.
    # To prevent infinite loops in bt, we rely on the deadline check inside bt.
    
    # However, if bt is very slow for a target, we might timeout.
    # The deadline check in bt handles this.
    
    # One issue: bt returns False on timeout. We should treat this as "no solution found in time"
    # and move to the next target (lower).
    
    while current_target >= lower_bound and time.time() < deadline:
        # Give each target a reasonable time slice?
        # The deadline check inside bt will stop it.
        # But if it stops early, we move on.
        
        res = backtrack_solve(current_target)
        
        if res is not None:
            # Verify
            all_diffs = set()
            valid = True
            for r in res:
                for i in range(len(r)):
                    for j in range(i+1, len(r)):
                        d = r[j] - r[i]
                        if d in all_diffs:
                            valid = False
                            break
                        all_diffs.add(d)
                if not valid:
                    break
            
            if valid:
                best_rows = [r[:] for r in res]
                best_max = max(max(r) for r in res)
                # If we found a solution with max < current_target, we can jump down.
                # But usually the max entry of the found solution is <= current_target.
                # It might be strictly less.
                # Let's update current_target to best_max - 1 to be efficient.
                current_target = best_max - 1
            else:
                # Should not happen if logic is correct
                current_target -= 1
        else:
            current_target -= 1

    return best_rows
