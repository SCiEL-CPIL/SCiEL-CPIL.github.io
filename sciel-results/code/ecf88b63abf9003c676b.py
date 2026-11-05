import time
import random
import sys
import math
import heapq

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill

    # Each row has k+1 elements starting at 0.
    # Number of differences per row = (k+1)*k/2
    diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * diffs_per_row
    lower_bound = total_diffs

    # Maximum possible scope estimate
    max_scope_estimate = total_diffs + k

    def get_all_diffs(row):
        """Get all differences in a row as a set."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def is_valid_solution(rows):
        """Check if the solution is valid (all differences distinct across all rows)."""
        used_diffs = set()
        for row in rows:
            for d in get_all_diffs(row):
                if d in used_diffs:
                    return False
                used_diffs.add(d)
        return True

    def greedy_construct():
        """
        Pure greedy construction: build rows one by one, each element greedily.
        Returns list of rows or None if stuck.
        """
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                last_val = row[-1]
                found = False
                for x in range(last_val + 1, max_scope_estimate + 1):
                    valid = True
                    for y in row:
                        d = x - y
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        row.append(x)
                        for y in row[:-1]:
                            used_diffs.add(x - y)
                        found = True
                        break
                if not found:
                    return None
            rows.append(row)
        return rows

    def interval_packing_greedy():
        """
        Dynamic interval packing: represent each row as k gaps (consecutive differences).
        Use a priority queue to select the next gap that minimizes collision probability.
        Maintain a global histogram of used differences.
        """
        # used_diffs[d] = True if difference d is already used
        used_diffs = [False] * (max_scope_estimate + 1)
        rows = []
        
        for row_idx in range(n_rows):
            row = [0]
            # We need to choose k gaps: g1, g2, ..., gk
            # where row[i] = sum(g1, ..., gi)
            # The differences are all partial sums and their differences
            
            # For each position, we track the set of differences that would be created
            # if we choose a particular gap length
            
            # Priority queue: (collision_count, gap_length, position)
            # We'll build the row by choosing gaps one at a time
            
            current_pos = 0
            for gap_idx in range(k):
                # For each possible gap length, compute how many new differences it creates
                # and how many of those are already used
                
                best_gap = None
                best_collision = float('inf')
                best_new_diffs = None
                
                # The gap length must be at least 1
                # The maximum gap is limited by remaining scope
                max_possible_gap = max_scope_estimate - current_pos
                
                # Try gap lengths in increasing order, but use a heuristic to skip many
                # We'll sample a subset and also try small gaps
                candidates = set()
                # Always try small gaps
                for g in range(1, min(20, max_possible_gap + 1)):
                    candidates.add(g)
                # Try some random larger gaps
                for _ in range(30):
                    g = random.randint(1, max_possible_gap)
                    candidates.add(g)
                # Try gaps that are "nice" (multiples of small numbers)
                for base in [2, 3, 5, 7, 11]:
                    for mult in range(1, max_possible_gap // base + 1):
                        candidates.add(base * mult)
                
                for g in candidates:
                    if g < 1 or g > max_possible_gap:
                        continue
                    
                    new_val = current_pos + g
                    
                    # Compute all new differences this gap creates
                    # The new element is at position new_val
                    # Differences with all previous elements in this row
                    new_diffs = set()
                    valid = True
                    for y in row:
                        d = new_val - y
                        if d > max_scope_estimate:
                            valid = False
                            break
                        if used_diffs[d]:
                            valid = False
                            break
                        new_diffs.add(d)
                    
                    if not valid:
                        continue
                    
                    # Count how many of these differences are "close" to other used differences
                    # This is a heuristic for future collision risk
                    collision_score = 0
                    for d in new_diffs:
                        # Check if d+1 or d-1 are used (proximity)
                        if d + 1 <= max_scope_estimate and used_diffs[d + 1]:
                            collision_score += 1
                        if d - 1 >= 1 and used_diffs[d - 1]:
                            collision_score += 1
                    
                    # Prefer gaps that create fewer collisions
                    if collision_score < best_collision:
                        best_collision = collision_score
                        best_gap = g
                        best_new_diffs = new_diffs
                        # If collision is 0, we can stop early
                        if collision_score == 0:
                            break
                
                if best_gap is None:
                    # Fallback: try all gap lengths
                    for g in range(1, max_possible_gap + 1):
                        new_val = current_pos + g
                        new_diffs = set()
                        valid = True
                        for y in row:
                            d = new_val - y
                            if d > max_scope_estimate:
                                valid = False
                                break
                            if used_diffs[d]:
                                valid = False
                                break
                            new_diffs.add(d)
                        if valid:
                            best_gap = g
                            best_new_diffs = new_diffs
                            break
                
                if best_gap is None:
                    return None
                
                # Commit to this gap
                current_pos += best_gap
                row.append(current_pos)
                for d in best_new_diffs:
                    used_diffs[d] = True
            
            rows.append(row)
        
        return rows

    def local_search(initial_rows, time_limit):
        """
        Local search with hill climbing and occasional random restarts.
        Focus on reducing the maximum element.
        """
        rows = [r[:] for r in initial_rows]
        
        def compute_used_diffs(rows):
            used = set()
            for row in rows:
                for i in range(len(row)):
                    for j in range(i+1, len(row)):
                        used.add(row[j] - row[i])
            return used
        
        used_diffs = compute_used_diffs(rows)
        current_scope = max(max(r) for r in rows)
        best_rows = [r[:] for r in rows]
        best_scope = current_scope
        
        no_improvement_count = 0
        
        while time.time() < start_time + time_limit:
            # Pick a random row and position to modify
            row_idx = random.randint(0, n_rows - 1)
            pos = random.randint(1, k)
            
            old_val = rows[row_idx][pos]
            min_val = rows[row_idx][pos-1] + 1
            max_val = rows[row_idx][pos+1] - 1 if pos < k else best_scope
            
            if min_val > max_val:
                continue
            
            # Try to find a new value that reduces scope
            # Prefer values that are smaller
            new_val = random.randint(min_val, max_val)
            
            if new_val == old_val:
                continue
            
            # Check validity
            old_diffs = set()
            for i in range(k + 1):
                if i != pos:
                    d = abs(old_val - rows[row_idx][i])
                    old_diffs.add(d)
            
            for d in old_diffs:
                used_diffs.discard(d)
            
            new_diffs = set()
            valid = True
            for i in range(k + 1):
                if i != pos:
                    if i < pos:
                        d = new_val - rows[row_idx][i]
                    else:
                        d = rows[row_idx][i] - new_val
                    if d in used_diffs or d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
            
            if not valid:
                used_diffs.update(old_diffs)
                continue
            
            # Compute new scope
            new_scope = max(max(rows[row_idx][:pos] + [new_val] + rows[row_idx][pos+1:]),
                           max(max(r) for r in rows if r is not rows[row_idx])) if n_rows > 1 else max(new_val)
            
            if new_scope < current_scope:
                rows[row_idx][pos] = new_val
                used_diffs.update(new_diffs)
                current_scope = new_scope
                no_improvement_count = 0
                
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [r[:] for r in rows]
            else:
                used_diffs.update(old_diffs)
                no_improvement_count += 1
            
            # If stuck, do a random perturbation
            if no_improvement_count > 200:
                # Reset and try a different approach
                no_improvement_count = 0
                # Randomly change a few positions
                for _ in range(2):
                    r_idx = random.randint(0, n_rows - 1)
                    p_idx = random.randint(1, k)
                    min_v = rows[r_idx][p_idx-1] + 1
                    max_v = rows[r_idx][p_idx+1] - 1 if p_idx < k else best_scope
                    if min_v <= max_v:
                        new_v = random.randint(min_v, max_v)
                        old_v = rows[r_idx][p_idx]
                        if new_v != old_v:
                            old_d = set()
                            for i in range(k + 1):
                                if i != p_idx:
                                    d = abs(old_v - rows[r_idx][i])
                                    old_d.add(d)
                            for d in old_d:
                                used_diffs.discard(d)
                            
                            new_d = set()
                            valid_perturb = True
                            for i in range(k + 1):
                                if i != p_idx:
                                    if i < p_idx:
                                        d = new_v - rows[r_idx][i]
                                    else:
                                        d = rows[r_idx][i] - new_v
                                    if d in used_diffs or d in new_d:
                                        valid_perturb = False
                                        break
                                    new_d.add(d)
                            
                            if valid_perturb:
                                rows[r_idx][p_idx] = new_v
                                used_diffs.update(new_d)
                                current_scope = max(max(r) for r in rows)
                            else:
                                used_diffs.update(old_d)
        
        return best_rows, best_scope

    # Step 1: Try interval packing greedy
    best_rows = None
    best_scope = float('inf')
    
    # Try interval packing greedy multiple times
    for attempt in range(3):
        if time.time() > deadline - 30:
            break
        rows = interval_packing_greedy()
        if rows is not None and len(rows) == n_rows:
            scope = max(max(r) for r in rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rows]
    
    # Step 2: If interval packing failed, try pure greedy
    if best_rows is None:
        rows = greedy_construct()
        if rows is not None:
            best_rows = rows
            best_scope = max(max(r) for r in rows)
    
    # Step 3: If still no solution, use last resort
    if best_rows is None:
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                valid = True
                for y in row:
                    if x - y in used_diffs:
                        valid = False
                        break
                if valid:
                    row.append(x)
                    for y in row[:-1]:
                        used_diffs.add(x - y)
            rows.append(row)
        return rows
    
    # Step 4: Local search to improve
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        best_rows, best_scope = local_search(best_rows, remaining_time - 5)
    
    return best_rows
