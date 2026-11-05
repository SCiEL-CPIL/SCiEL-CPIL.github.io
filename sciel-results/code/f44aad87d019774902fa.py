import time
import random
import copy

def construct(n_rows, k):
    deadline = time.time() + 140  # safety margin before 150s kill
    
    def time_up():
        return time.time() > deadline
    
    # Phase 1: Fast greedy to get a valid solution quickly
    def greedy_construct():
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
        return rows, used
    
    rows, used = greedy_construct()
    best_rows = [r[:] for r in rows]
    best_max = max(max(r) for r in best_rows)
    
    # Phase 2: Local search to reduce maximum entry
    def compute_used(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        return used
    
    def is_valid(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True
    
    def get_max_entry(rows):
        return max(max(r) for r in rows)
    
    # Local search: try to reduce max entry by compacting rows
    # Strategy: for each row, try to move the last element (or others) to smaller values
    def local_search(rows, used, max_iters=50000):
        rows = [r[:] for r in rows]
        used = set(used)
        current_max = get_max_entry(rows)
        best_local = [r[:] for r in rows]
        best_local_max = current_max
        iterations = 0
        
        while iterations < max_iters and not time_up():
            iterations += 1
            improved = False
            
            # Try to compact: for each row, try moving elements to smaller positions
            for ri in range(n_rows):
                row = rows[ri]
                # Try to move the last element to a smaller value
                last_idx = len(row) - 1
                old_last = row[last_idx]
                
                # Try values from old_last - 1 down to row[last_idx-1] + 1
                for new_val in range(old_last - 1, row[last_idx - 1], -1):
                    if new_val <= row[last_idx - 1]:
                        break
                    # Check if new_val creates valid differences
                    # Remove old differences involving last element
                    old_diffs = {old_last - row[i] for i in range(last_idx)}
                    # Check new differences
                    new_diffs = {new_val - row[i] for i in range(last_idx)}
                    
                    # Check if new_diffs conflict with used minus old_diffs
                    conflict = False
                    for d in new_diffs:
                        if d in used and d not in old_diffs:
                            conflict = True
                            break
                    
                    if not conflict:
                        # Apply the move
                        used -= old_diffs
                        row[last_idx] = new_val
                        used |= new_diffs
                        current_max = min(current_max, new_val)
                        if current_max < best_local_max:
                            best_local_max = current_max
                            best_local = [r[:] for r in rows]
                        improved = True
                        break
                
                if time_up():
                    break
            
            if not improved:
                # Try random perturbations: swap elements or try different positions
                # Pick a random row and try to move a random element
                if random.random() < 0.3:
                    ri = random.randrange(n_rows)
                    row = rows[ri]
                    # Pick a non-first element to try moving
                    if len(row) > 1:
                        ei = random.randrange(1, len(row))
                        old_val = row[ei]
                        # Try a random smaller value
                        min_allowed = row[ei - 1] + 1
                        if old_val > min_allowed:
                            new_val = random.randrange(min_allowed, old_val)
                            # Compute old and new differences
                            old_diffs = set()
                            new_diffs = set()
                            for i in range(len(row)):
                                if i == ei:
                                    continue
                                if i < ei:
                                    old_diffs.add(old_val - row[i])
                                    new_diffs.add(new_val - row[i])
                                else:
                                    old_diffs.add(row[i] - old_val)
                                    new_diffs.add(row[i] - new_val)
                            
                            # Check conflicts
                            conflict = False
                            for d in new_diffs:
                                if d in used and d not in old_diffs:
                                    conflict = True
                                    break
                            
                            if not conflict:
                                used -= old_diffs
                                row[ei] = new_val
                                used |= new_diffs
                                current_max = get_max_entry(rows)
                                if current_max < best_local_max:
                                    best_local_max = current_max
                                    best_local = [r[:] for r in rows]
                                improved = True
                        if not improved and time_up():
                            break
            
            if time_up():
                break
        
        return best_local, best_local_max
    
    # Run local search with increasing effort
    search_budget = 10000
    while not time_up():
        rows_copy = [r[:] for r in rows]
        used_copy = compute_used(rows_copy)
        improved_rows, improved_max = local_search(rows_copy, used_copy, max_iters=search_budget)
        
        if improved_max < best_max:
            best_max = improved_max
            best_rows = [r[:] for r in improved_rows]
            # Update working state
            rows = [r[:] for r in improved_rows]
            used = compute_used(rows)
            # Increase budget slightly for further improvement
            search_budget = min(search_budget * 2, 100000)
        else:
            # No improvement, try with random restarts
            # Shuffle the order of rows and retry
            random.shuffle(rows)
            used = compute_used(rows)
            rows_copy = [r[:] for r in rows]
            improved_rows, improved_max = local_search(rows_copy, used_copy, max_iters=search_budget)
            if improved_max < best_max:
                best_max = improved_max
                best_rows = [r[:] for r in improved_rows]
                rows = [r[:] for r in improved_rows]
                used = compute_used(rows)
        
        if time_up():
            break
    
    # Verify the best solution is valid
    if not is_valid(best_rows):
        # Fallback to greedy
        best_rows, _ = greedy_construct()
    
    return best_rows
