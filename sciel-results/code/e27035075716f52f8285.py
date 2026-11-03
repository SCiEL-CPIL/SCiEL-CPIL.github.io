import time
import random
from itertools import combinations

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140 # 10s safety margin
    
    # Initial greedy solution
    rows = []
    used = set()
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
        
    best_rows = [r[:] for r in rows]
    best_max = max(max(r) for r in best_rows)
    
    # Local Search
    while time.time() < deadline:
        # Pick random row to re-optimize
        i = random.randint(0, n_rows - 1)
        
        # Calculate used diffs from other rows
        other_used = set()
        for j in range(n_rows):
            if j == i: continue
            r = rows[j]
            for a in range(len(r)):
                for b in range(a+1, len(r)):
                    other_used.add(r[b] - r[a])
                    
        # Try to find a better row for i
        # We want max element < best_max eventually
        # Start from lower bound
        # Lower bound for 1 row of order k+1 is k(k+1)/2? No, minimal Golomb length.
        # We can just start from k (trivial lower bound) or current rows[i][-1]
        
        # To improve global max, we need new_max < best_max
        # So we limit search to M < best_max
        max_allowed = best_max - 1 if best_max > k else k
        
        # If we can't go lower, maybe we can find a valid one with same max but freeing up space?
        # No, objective is minimize max.
        
        # Try to find ANY valid row with max <= max_allowed
        # If found, it improves or maintains.
        # To strictly improve, we need max < current max of that row?
        # Actually, if we replace row i with one having max < best_max, global max might drop.
        
        # Let's try to find a row with max < best_max
        # If best_max is the global min, we are stuck.
        
        # We iterate M from k to max_allowed
        # But M=k is impossible for k>1.
        # Minimal M for Golomb ruler of order k+1 is roughly k^2/2?
        
        # Let's just try to find a row with max < rows[i][-1] first?
        # Or just < best_max.
        
        # Optimization: Start M from a heuristic lower bound
        # Counting bound for 1 row: M(M+1)/2 >= k(k+1)/2 => M ~ k
        # Actually M must be at least k(k+1)/2? No.
        
        # Let's try M from k to max_allowed
        # This might be slow if max_allowed is large.
        # We can randomize the start or use a smaller range first.
        
        found = False
        # Try to find a row with max < best_max
        # We can limit the search space by trying smaller M first.
        
        # To speed up, we can try a few random M values near the lower bound?
        # No, we want the smallest M.
        
        # Backtracking for single row
        # We'll implement a fast backtracker
        
        # If we find one, update
        new_row = _find_golomb_row(k, max_allowed, other_used, deadline)
        if new_row:
            # Check if it improves best_max
            new_max = new_row[-1]
            if new_max < best_max:
                rows[i] = new_row
                best_max = new_max
                best_rows = [r[:] for r in rows]
            elif new_max < rows[i][-1]:
                # Improves this row, might help later
                rows[i] = new_row
            else:
                # No improvement, maybe swap?
                pass
                
    return best_rows

def _find_golomb_row(k, M_limit, used_diffs, deadline):
    # Find a Golomb ruler of order k+1 with max element <= M_limit
    # that avoids used_diffs
    # Returns row or None
    
    # Backtracking
    # row[0] = 0
    # row[1..k] to be determined
    
    # To speed up, we can fix the last element?
    # No, we iterate last element from k to M_limit
    
    # Pruning:
    # The differences must be unique and not in used_diffs.
    
    # We'll use a recursive helper
    row = [0]
    
    def backtrack(pos, last_val):
        if time.time() > deadline:
            return False
        if pos == k + 1:
            return True
        # Next value x
        # x > last_val
        # x <= M_limit
        # Diffs with previous elements must be valid
        
        # Heuristic: Try smaller x first?
        # To find a valid ruler, smaller x is usually denser, harder?
        # Actually, we just need ONE.
        
        # We can try x from last_val + 1 to M_limit
        # But we can prune if x - row[i] is in used_diffs or already in current row diffs.
        
        # Current diffs in row
        curr_diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                curr_diffs.add(row[j] - row[i])
                
        # This is O(k^2) per step. k is small (7).
        
        for x in range(last_val + 1, M_limit + 1):
            # Check diffs
            valid = True
            for a in row:
                d = x - a
                if d in used_diffs or d in curr_diffs:
                    valid = False
                    break
            if valid:
                row.append(x)
                if backtrack(pos + 1, x):
                    return True
                row.pop()
        return False

    # We can optimize by trying different last elements?
    # The loop above tries all x for the next position.
    # This explores the whole space.
    
    # To find a solution with SMALL max element, we should iterate M_limit?
    # No, M_limit is fixed in this call.
    # We want to know if a solution exists with max <= M_limit.
    
    # But we want the SMALLEST max.
    # So in the main loop, we should call this with increasing M_limit?
    # Or we can modify this to return the row with smallest max element found.
    
    # Let's change strategy:
    # In main loop, we know best_max.
    # We want to find a row with max < best_max.
    # So we call _find_golomb_row with M_limit = best_max - 1.
    # If it finds one, great.
    # If not, we might be stuck.
    
    # To make it more robust, we can try to find a row with max < rows[i][-1]
    # even if it's not < best_max, to improve local state.
    
    # But the function above returns the FIRST solution found in DFS.
    # DFS with increasing x tends to find small max elements?
    # Not necessarily.
    
    # Let's stick to this. It's a start.
    
    if backtrack(1, 0):
        return row[:]
    return None
