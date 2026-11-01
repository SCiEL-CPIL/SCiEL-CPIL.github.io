import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = 140.0  # Keep a margin from the 150s limit
    
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    def get_scope(rows):
        return max(max(r) for r in rows)
    
    def compute_row_diffs(row):
        """Compute all positive differences in a row as a set."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    def check_valid(rows):
        """Check if all differences across all rows are distinct."""
        all_diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False
                    all_diffs.add(d)
        return True
    
    def greedy_rows(aggressiveness=1.0, seed=None):
        """Generate rows using randomized greedy with aggressiveness parameter."""
        if seed is not None:
            random.seed(seed)
        
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k + 1:
                # Try candidates from current_max+1 upward
                # aggressiveness controls how aggressively we search for smaller values
                best_x = None
                for x in range(current_max + 1, current_max + int(50 * aggressiveness) + 1):
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used_diffs:
                        best_x = x
                        break
                    # If aggressiveness is low, accept slightly larger jumps
                    if aggressiveness < 1.0 and x > current_max + 10:
                        # Allow some conflicts to be resolved later
                        pass
                
                if best_x is None:
                    # Fallback: find smallest valid x
                    x = current_max + 1
                    while True:
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used_diffs:
                            break
                        x += 1
                    best_x = x
                
                row.append(best_x)
                current_max = best_x
                used_diffs |= {best_x - a for a in row[:-1]}
            rows.append(row)
        return rows
    
    def local_search(rows, max_iters=50000):
        """
        Iterative local search using targeted moves:
        1. Try to reduce the maximum element
        2. Try 2-opt swaps within rows
        3. Try moving individual elements
        """
        current_rows = [r[:] for r in rows]
        row_diffs = [compute_row_diffs(r) for r in current_rows]
        global_used = set()
        for rd in row_diffs:
            global_used |= rd
        
        current_scope = get_scope(current_rows)
        best_rows = [r[:] for r in current_rows]
        best_scope = current_scope
        
        no_improve_count = 0
        max_no_improve = 5000
        
        for it in range(max_iters):
            if time.time() > deadline - 3.0:
                break
            if no_improve_count > max_no_improve:
                break
            
            # Pick a random row
            row_idx = random.randint(0, n_rows - 1)
            row = current_rows[row_idx]
            rd = row_diffs[row_idx]
            
            # Choose move type
            move_type = random.random()
            
            if move_type < 0.4:
                # Try to reduce the last element (if it's the max)
                if row[-1] == current_scope:
                    pos = len(row) - 1
                    old_val = row[pos]
                    
                    # Try to decrease
                    for delta in range(1, min(15, old_val)):
                        new_val = old_val - delta
                        if new_val <= row[pos - 1]:
                            break
                        
                        # Compute new diffs for this position
                        new_diffs = set()
                        for j in range(len(row)):
                            if j != pos:
                                if j < pos:
                                    new_diffs.add(new_val - row[j])
                                else:
                                    new_diffs.add(row[j] - new_val)
                        
                        # Check conflicts
                        forbidden = global_used - rd
                        if new_diffs & forbidden:
                            continue
                        
                        # Accept!
                        new_row = row[:]
                        new_row[pos] = new_val
                        new_rd = (rd - {old_val - row[j] for j in range(pos)} - 
                                  {row[j] - old_val for j in range(pos + 1, len(row))}) | new_diffs
                        
                        current_rows[row_idx] = new_row
                        row_diffs[row_idx] = new_rd
                        global_used = (global_used - rd) | new_rd
                        current_scope = get_scope(current_rows)
                        
                        if current_scope < best_scope:
                            best_scope = current_scope
                            best_rows = [r[:] for r in current_rows]
                        no_improve_count = 0
                        break
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1
                    
            elif move_type < 0.7:
                # 2-opt: swap two elements in the row (maintaining order)
                if len(row) >= 3:
                    # Pick two positions to potentially reorder
                    p1 = random.randint(1, len(row) - 2)
                    p2 = random.randint(p1 + 1, len(row) - 1)
                    
                    # Try swapping row[p1] and row[p2]
                    # But we need to maintain strict increase, so try:
                    # Remove both, reinsert in different positions
                    
                    old_vals = [row[p1], row[p2]]
                    remaining = row[:p1] + row[p1+1:p2] + row[p2+1:]
                    
                    # Try inserting old_vals[0] and old_vals[1] in swapped order
                    # or at different positions
                    candidates = []
                    
                    # Try: insert old_vals[1] first, then old_vals[0]
                    for ins_pos1 in range(p1, min(p1 + 3, len(remaining) + 1)):
                        for ins_pos2 in range(ins_pos1, min(ins_pos2 + 3, len(remaining) + 1)):
                            # Build new row
                            new_row = remaining[:ins_pos1] + [old_vals[1]] + remaining[ins_pos1:ins_pos2] + [old_vals[0]] + remaining[ins_pos2:]
                            # Check if strictly increasing
                            valid = True
                            for i in range(len(new_row) - 1):
                                if new_row[i] >= new_row[i+1]:
                                    valid = False
                                    break
                            if valid and len(new_row) == len(row):
                                candidates.append(new_row)
                    
                    if not candidates:
                        no_improve_count += 1
                        continue
                    
                    # Try a random candidate
                    new_row = random.choice(candidates)
                    new_rd = compute_row_diffs(new_row)
                    
                    # Check conflicts
                    forbidden = global_used - rd
                    if new_rd & forbidden:
                        no_improve_count += 1
                        continue
                    
                    # Accept
                    current_rows[row_idx] = new_row
                    row_diffs[row_idx] = new_rd
                    global_used = (global_used - rd) | new_rd
                    current_scope = get_scope(current_rows)
                    
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best_rows = [r[:] for r in current_rows]
                    no_improve_count = 0
                    
            else:
                # Move a single element
                pos = random.randint(1, len(row) - 1)
                old_val = row[pos]
                
                # Try to move to a nearby position
                delta = random.randint(-6, 6)
                new_val = old_val + delta
                
                # Ensure strictly increasing
                if pos > 0 and new_val <= row[pos - 1]:
                    no_improve_count += 1
                    continue
                if pos < len(row) - 1 and new_val >= row[pos + 1]:
                    no_improve_count += 1
                    continue
                
                # Compute new diffs
                new_diffs = set()
                for j in range(len(row)):
                    if j != pos:
                        if j < pos:
                            new_diffs.add(new_val - row[j])
                        else:
                            new_diffs.add(row[j] - new_val)
                
                # Old diffs
                old_diffs = set()
                for j in range(len(row)):
                    if j != pos:
                        if j < pos:
                            old_diffs.add(old_val - row[j])
                        else:
                            old_diffs.add(row[j] - old_val)
                
                # Check conflicts
                forbidden = global_used - old_diffs
                if new_diffs & forbidden:
                    no_improve_count += 1
                    continue
                
                # Accept
                new_row = row[:]
                new_row[pos] = new_val
                new_rd = (rd - old_diffs) | new_diffs
                
                current_rows[row_idx] = new_row
                row_diffs[row_idx] = new_rd
                global_used = (global_used - old_diffs) | new_diffs
                current_scope = get_scope(current_rows)
                
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [r[:] for r in current_rows]
                no_improve_count = 0
        
        return best_rows, best_scope
    
    # Phase 1: Generate diverse initial solutions using randomized greedy
    initial_solutions = []
    
    # Base greedy
    base_rows = greedy_rows(aggressiveness=1.0, seed=42)
    initial_solutions.append(base_rows)
    
    # Variants with different aggressiveness
    for aggr in [0.5, 0.7, 1.5, 2.0]:
        if time.time() > deadline - 60.0:
            break
        rows = greedy_rows(aggressiveness=aggr, seed=random.randint(0, 10000))
        initial_solutions.append(rows)
    
    # More randomized restarts
    for _ in range(5):
        if time.time() > deadline - 40.0:
            break
        rows = greedy_rows(aggressiveness=random.uniform(0.5, 2.0), seed=random.randint(0, 100000))
        initial_solutions.append(rows)
    
    # Phase 2: Local search on each initial solution
    best_rows = None
    best_scope = float('inf')
    
    for sol in initial_solutions:
        if time.time() > deadline - 10.0:
            break
        
        # Validate
        if not check_valid(sol):
            continue
        
        scope = get_scope(sol)
        
        # Run local search
        opt_rows, opt_scope = local_search(sol, max_iters=20000)
        
        if opt_scope < best_scope:
            best_scope = opt_scope
            best_rows = [r[:] for r in opt_rows]
    
    # Phase 3: Final polish if time remains
    if best_rows is None:
        # Fallback to base greedy
        best_rows = base_rows
        best_scope = get_scope(base_rows)
    elif time.time() < deadline - 5.0:
        opt_rows, opt_scope = local_search(best_rows, max_iters=30000)
        if opt_scope < best_scope:
            best_scope = opt_scope
            best_rows = [r[:] for r in opt_rows]
    
    return best_rows
