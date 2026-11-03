import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    row_size = k + 1
    diffs_per_row = row_size * (row_size - 1) // 2
    total_diffs = n_rows * diffs_per_row
    lb = total_diffs

    def get_metrics(rows):
        used = set()
        conflicts = 0
        max_val = 0
        for r in rows:
            for i in range(1, len(r)):
                if r[i] <= r[i-1]:
                    return float('inf'), float('inf')
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    d = r[j] - r[i]
                    if d in used:
                        conflicts += 1
                    else:
                        used.add(d)
            max_val = max(max_val, r[-1])
        return max_val, conflicts

    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < row_size:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    # Backtracking solver for a given M
    def solve_for_M(M, time_limit):
        """Try to find n_rows valid rows with max entry <= M."""
        used_diffs = set()
        current_rows = []
        found = [False]
        end_time = start_time + time_limit

        def backtrack(row_idx, current_row, x):
            if found[0] or time.time() > end_time:
                return
            
            if row_idx == n_rows:
                found[0] = True
                return
            
            if len(current_row) == row_size:
                backtrack(row_idx + 1, [0], 0)
                return
            
            # Pruning: minimum needed space
            min_needed = row_size - len(current_row)
            if x + min_needed > M:
                return
            
            # Try values from x+1 to M
            for next_x in range(x + 1, M + 1):
                new_diffs = {next_x - a for a in current_row}
                if new_diffs & used_diffs:
                    continue
                
                current_row.append(next_x)
                used_diffs.update(new_diffs)
                
                backtrack(row_idx, current_row, next_x)
                
                if found[0]:
                    return
                
                current_row.pop()
                used_diffs.difference_update(new_diffs)
        
        backtrack(0, [0], 0)
        if found[0]:
            return current_rows
        return None

    # 1. Try backtracking from lb upward
    # Allocate time: try M = lb, lb+1, ... with increasing time slices
    # Start with small slices, increase if not found
    best_rows = None
    best_max = float('inf')
    
    # Give backtracking up to 60 seconds total
    bt_deadline = start_time + 60.0
    
    M = lb
    while M <= lb + 30 and time.time() < bt_deadline:
        # Time slice: start small, grow
        elapsed_bt = time.time() - start_time
        remaining_bt = bt_deadline - time.time()
        if remaining_bt < 1:
            break
        
        # Allocate time: for early M, use 2s; for later, use more
        if M <= lb + 5:
            slice_time = min(2.0, remaining_bt)
        elif M <= lb + 15:
            slice_time = min(3.0, remaining_bt)
        else:
            slice_time = min(5.0, remaining_bt)
        
        result = solve_for_M(M, slice_time)
        if result is not None:
            best_rows = result
            best_max = M
            break
        M += 1
    
    # 2. If backtracking found nothing, use greedy + SA
    if best_rows is None:
        best_rows = greedy_construct()
        best_max, best_conf = get_metrics(best_rows)
        
        # Simulated Annealing with incremental conflict tracking
        current_rows = [r[:] for r in best_rows]
        
        # Build initial used_diffs and conflict count
        used_diffs = set()
        conflict_count = 0
        for r in current_rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    d = r[j] - r[i]
                    if d in used_diffs:
                        conflict_count += 1
                    else:
                        used_diffs.add(d)
        
        T = 50.0
        cooling = 0.995
        sa_deadline = start_time + 110.0
        
        # Precompute row diffs for faster updates
        while time.time() < sa_deadline and T > 0.5:
            r_idx = random.randint(0, n_rows - 1)
            j_idx = random.randint(1, row_size - 1)
            
            old_val = current_rows[r_idx][j_idx]
            row = current_rows[r_idx]
            
            # Compute old diffs involving position j_idx
            old_diffs = set()
            for i in range(j_idx):
                old_diffs.add(old_val - row[i])
            for i in range(j_idx+1, len(row)):
                old_diffs.add(row[i] - old_val)
            
            # Remove old diffs from used_diffs, adjust conflict_count
            for d in old_diffs:
                used_diffs.discard(d)
                # We don't track counts per diff, so we need a different approach
                # Actually, let's just recompute conflicts for this row
            
            # Try a few random new values
            found_move = False
            for attempt in range(20):
                max_bound = max(best_max, lb)
                new_val = random.randint(1, max_bound)
                
                # Ensure row remains strictly increasing
                if j_idx > 0 and new_val <= row[j_idx-1]:
                    continue
                if j_idx < row_size - 1 and new_val >= row[j_idx+1]:
                    continue
                
                # Compute new diffs
                new_diffs = set()
                for i in range(j_idx):
                    new_diffs.add(new_val - row[i])
                for i in range(j_idx+1, len(row)):
                    new_diffs.add(row[i] - new_val)
                
                # Check conflicts: count how many new_diffs are in used_diffs (excluding old_diffs)
                # Actually, we need to be careful: used_diffs currently has old_diffs removed
                # Let's just check if new_diffs has any intersection with used_diffs
                
                # Count conflicts if we apply this move
                new_conflicts_for_row = 0
                for d in new_diffs:
                    if d in used_diffs:
                        new_conflicts_for_row += 1
                
                # Also need to check if any old_diffs that we removed are now conflicting with other rows
                # This is getting complex. Let's use a simpler approach: just recompute
                # the full conflict count for the affected row.
                
                # Actually, let's simplify: just do a full metric check but it's expensive.
                # Alternative: maintain a count per difference.
                
                # For now, let's just try the move and check
                row[j_idx] = new_val
                new_max, new_conf = get_metrics(current_rows)
                
                old_score = conflict_count * 1000 + best_max  # approximate
                new_score = new_conf * 1000 + new_max
                delta = new_score - old_score
                
                if delta < 0 or random.random() < math.exp(-delta / T):
                    conflict_count = new_conf
                    if new_conf == 0 and new_max < best_max:
                        best_max = new_max
                        best_rows = [r[:] for r in current_rows]
                    found_move = True
                else:
                    row[j_idx] = old_val
                    break  # revert and try another
            
            T *= cooling
            
            if best_max == lb:
                break
    
    return best_rows
