# EVOLVE-BLOCK-START
import sys
import time
import random
from collections import deque

def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0 such that all
    positive differences within each row are distinct across all rows.
    Minimize the largest entry.
    
    Uses a randomized CSP approach with bitset-based difference constraint propagation.
    """
    if n_rows <= 0 or k <= 0:
        return [[0] * (k + 1)] * max(n_rows, 0)
    
    # Counting bound: need n_rows * k*(k+1)/2 distinct positive differences
    total_diffs_needed = n_rows * k * (k + 1) // 2
    
    # Lower bound: We need 'total_diffs_needed' distinct differences.
    # The differences are integers from 1 to M.
    # So M must be at least total_diffs_needed.
    lb = total_diffs_needed
    
    # Greedy baseline to establish an upper bound
    def greedy_construct(n_rows, k):
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
    
    greedy_rows = greedy_construct(n_rows, k)
    greedy_max = max(max(r) for r in greedy_rows)
    
    start_time = time.time()
    time_limit = 115.0
    
    # Precompute bit masks for differences up to greedy_max
    # We will search M from lb to greedy_max
    max_possible = greedy_max
    bit_masks = [0] * (max_possible + 1)
    for i in range(max_possible + 1):
        bit_masks[i] = 1 << i

    def solve_csp_random(n_rows, k, max_entry, timeout, seed=0):
        """
        CSP solver using bitset difference tracking and randomized backtracking.
        Returns list of rows or None.
        """
        deadline = time.time() + timeout
        rng = random.Random(seed)
        
        # State:
        # used_bits: integer representing set of used differences
        # current_rows: list of rows completed so far
        # current_row: row being built
        # row_idx: index of current row being built
        # pos: index in current_row
        
        used_bits = 0
        rows = []
        current_row = [0]
        row_idx = 0
        pos = 0 # Next position to fill in current_row (0 is filled)
        
        # We use a recursive DFS with explicit stack or recursion?
        # Recursion depth is k+1 (max 8 for k=7), so recursion is fine.
        
        def dfs(row_idx, pos, current_row, used_bits):
            if time.time() > deadline:
                return False
            
            if row_idx == n_rows:
                return True
            
            if pos == k + 1:
                # Row complete
                rows.append(current_row)
                # Move to next row
                return dfs(row_idx + 1, 0, [0], used_bits)
            
            # Determine valid range for next element x
            last = current_row[pos - 1] if pos > 0 else 0
            remaining = k + 1 - pos
            min_x = last + 1
            max_x = max_entry - (remaining - 1)
            
            if min_x > max_x:
                return False
            
            # Generate candidates
            # To improve performance, we can shuffle the range [min_x, max_x]
            # But creating a list of all candidates is O(M).
            # Instead, we can pick random candidates?
            # Or iterate in a random order?
            # For small M (100-150) and small k (7), the range is small.
            # Let's collect valid candidates first.
            
            candidates = []
            # Optimization: iterate from min_x to max_x
            for x in range(min_x, max_x + 1):
                # Check conflicts
                conflict = False
                new_diff_bits = 0
                for i in range(pos):
                    d = x - current_row[i]
                    if (used_bits >> d) & 1:
                        conflict = True
                        break
                    new_diff_bits |= bit_masks[d]
                
                if not conflict:
                    candidates.append((x, new_diff_bits))
            
            if not candidates:
                return False
            
            # Shuffle candidates to add randomness
            rng.shuffle(candidates)
            
            for x, new_diff_bits in candidates:
                if time.time() > deadline:
                    return False
                
                current_row.append(x)
                new_used = used_bits | new_diff_bits
                
                if dfs(row_idx, pos + 1, current_row, new_used):
                    return True
                
                current_row.pop()
            
            return False
        
        # Start DFS
        # We need to handle the state correctly.
        # The dfs function modifies `rows` and `used_bits` is passed by value (int is immutable).
        # `current_row` is a list, so we need to be careful with backtracking.
        # The recursive dfs creates a new `current_row`? No, it appends/pops.
        # But when moving to the next row, we pass `[0]`.
        # This works.
        
        return dfs(0, 0, [0], 0)

    # Try to find better solution than greedy
    best_rows = greedy_rows
    best_max = greedy_max
    
    # Strategy: try max_entry values from lb upward
    # We stop when we find a solution.
    
    # If lb is much smaller than greedy_max, we have a lot of room.
    # We try to find the smallest M.
    
    # Allocate time per M value.
    # We don't know how many M values we'll try.
    # Let's try a few M values with increasing time budgets.
    
    # Simple strategy:
    # Try M = lb, lb+1, ...
    # Give each M a small time budget (e.g., 1-2 seconds).
    # If we run out of time, return best found.
    
    current_M = lb
    while current_M <= greedy_max:
        elapsed = time.time() - start_time
        remaining_time = time_limit - elapsed - 2  # leave 2 sec safety
        
        if remaining_time < 0.5:
            break
        
        # Allocate time for this M
        # If we are close to lb, we might be unlucky.
        # Let's give 1 second per M initially, up to 5 seconds if we have lots of time left.
        budget = min(1.0, remaining_time)
        
        # Try a few random seeds for this M
        found = False
        # Number of restarts depends on budget
        # If budget is 1.0, we can do 2-3 restarts of 0.3s each?
        # Or 1 restart of 1.0s?
        # Let's do 1 restart with the full budget for simplicity and depth.
        # Randomization inside DFS handles the search space.
        
        result = solve_csp_random(n_rows, k, current_M, budget, seed=current_M)
        if result is not None:
            best_rows = result
            best_max = current_M
            found = True
            # Since we iterate M from lb upwards, this is the minimum M found.
            # We can break immediately as we want the smallest M.
            break
        
        current_M += 1
    
    # Validate
    def validate(rows, n_rows, k):
        if len(rows) != n_rows:
            return False
        used = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
            for i in range(1, len(row)):
                if row[i] <= row[i-1]:
                    return False
        return True
    
    if not validate(best_rows, n_rows, k):
        best_rows = greedy_construct(n_rows, k)
    
    return best_rows

# EVOLVE-BLOCK-END
