import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145 # 5s safety margin

    def time_left():
        return deadline - time.time()

    def row_diffs(row):
        return {row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))}

    def is_valid(rows):
        seen = set()
        for r in rows:
            for d in row_diffs(r):
                if d in seen:
                    return False
                seen.add(d)
        return True

    def scope(rows):
        return max(max(r) for r in rows)

    # 1. Greedy Initialization
    def greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            row_d = set()
            while len(row) < k + 1:
                x += 1
                new_d = {x - a for a in row}
                if not (new_d & used) and not (new_d & row_d):
                    row.append(x)
                    row_d |= new_d
                    used |= new_d
            rows.append(row)
        return rows

    best_rows = greedy()
    best_scope = scope(best_rows)
    
    # 2. Row-wise DFS Optimizer
    def find_row(used_other, max_val, node_limit=100000):
        """
        Find a row of length k+1 starting at 0, max element <= max_val,
        disjoint from used_other.
        Returns list or None.
        """
        if max_val < k: # Need at least 1..k
            return None
            
        used_local = set()
        row = [0]
        nodes = 0
        
        def dfs(last_val, pos):
            nonlocal nodes
            nodes += 1
            if nodes > node_limit:
                return False
            
            if pos == k + 1:
                return True
            
            # Pruning: min possible final value
            # We need (k+1 - pos) more elements.
            # Min gap is 1.
            min_final = last_val + (k + 1 - pos)
            if min_final > max_val:
                return False
            
            # Try next values
            # Optimization: Only try values that generate new diffs not in used_other
            # and not in used_local
            
            # Heuristic: Try smaller values first
            for x in range(last_val + 1, max_val + 1):
                new_d = {x - a for a in row}
                
                # Check conflicts
                if new_d & used_other or new_d & used_local:
                    continue
                
                # Accept
                row.append(x)
                used_local.update(new_d)
                
                if dfs(x, pos + 1):
                    return True
                
                # Backtrack
                row.pop()
                used_local.difference_update(new_d)
                
            return False
            
        if dfs(0, 0):
            return list(row)
        return None

    # 3. Iterative Improvement
    improved = True
    while improved and time_left() > 5:
        improved = False
        
        # Sort rows by scope descending to attack the bottleneck first
        order = sorted(range(n_rows), key=lambda i: -max(best_rows[i]))
        
        for i in order:
            if time_left() < 5:
                break
                
            current_scope_i = max(best_rows[i])
            # We want to reduce this row's scope
            # Target: current_scope_i - 1
            # If we can't reduce by 1, maybe we can't reduce at all?
            # Let's try to find a row with max < current_scope_i
            
            used_other = set()
            for j in range(n_rows):
                if j != i:
                    used_other |= row_diffs(best_rows[j])
            
            # Try to find a better row
            # We can try a few different max_val limits if the first fails?
            # No, if max_val = current_scope_i - 1 fails, lower limits will also fail (subset of search space).
            # So just try max_val = current_scope_i - 1.
            
            new_row = find_row(used_other, current_scope_i - 1)
            
            if new_row:
                best_rows[i] = new_row
                best_scope = scope(best_rows)
                improved = True
                # Restart the loop to re-order and check new bottlenecks
                break 

    # 4. Final Polish: Try to reduce by 1 again in case of ties or subtle interactions
    # Also, if we get stuck, try a "kick" on the worst row
    if time_left() > 10:
        # Kick strategy: Replace a random row with a valid one (possibly larger) 
        # then try to reduce it.
        # This helps escape local minima where no single row can be reduced.
        
        for _ in range(5): # 5 kick attempts
            if time_left() < 5:
                break
            
            # Pick the row with the largest scope
            i = max(range(n_rows), key=lambda j: max(best_rows[j]))
            
            # Generate a random valid row (greedy-like but with jitter)
            used_other = set()
            for j in range(n_rows):
                if j != i:
                    used_other |= row_diffs(best_rows[j])
            
            # Try to generate a row that is "different"
            # We can just use greedy on the available space
            # But greedy is deterministic. Let's add randomness.
            
            # Simple random restart for this row
            found = False
            for attempt in range(20):
                row = [0]
                used_local = set()
                x = 0
                # Jitter: skip some valid candidates
                while len(row) < k + 1:
                    x += 1
                    new_d = {x - a for a in row}
                    if not (new_d & used_other) and not (new_d & used_local):
                        # 10% chance to skip
                        if random.random() < 0.1 and x < 200:
                            continue
                        row.append(x)
                        used_local |= new_d
                
                if len(row) == k + 1:
                    # Check if this row is valid (it is by construction)
                    # Check if it improves scope?
                    # If it's worse, we don't adopt it unless we plan to polish it down.
                    # Let's adopt it and then run the reduction loop again.
                    best_rows[i] = row
                    found = True
                    break
            
            if found:
                # Run the reduction loop again
                improved = True
                while improved and time_left() > 5:
                    improved = False
                    order = sorted(range(n_rows), key=lambda j: -max(best_rows[j]))
                    for j in order:
                        if time_left() < 5:
                            break
                        current_scope_j = max(best_rows[j])
                        used_other = set()
                        for m in range(n_rows):
                            if m != j:
                                used_other |= row_diffs(best_rows[m])
                        
                        new_row = find_row(used_other, current_scope_j - 1)
                        if new_row:
                            best_rows[j] = new_row
                            best_scope = scope(best_rows)
                            improved = True
                            break

    return best_rows
