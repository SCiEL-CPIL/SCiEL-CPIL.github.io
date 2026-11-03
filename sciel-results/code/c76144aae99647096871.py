"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within each row are distinct across all rows.
    Minimizes the largest entry.
    """
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0] * (k + 1) for _ in range(n_rows)]
    
    num_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * num_diffs_per_row

    def get_diffs(row):
        """Calculate the set of positive differences for a row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def max_entry(rows):
        return max(max(r) for r in rows)

    def sum_entries(rows):
        return sum(sum(r) for r in rows)

    def greedy_row(used_diffs, max_val=None, randomize=False, skip_prob=0.1):
        """
        Build a row greedily.
        If randomize is True, it may skip valid candidates to explore.
        """
        row = [0]
        local_diffs = set()
        x = 0
        limit = max_val if max_val else 200
        
        while len(row) < k + 1:
            x += 1
            if max_val and x > max_val:
                return None, None
            
            new_diffs = {x - a for a in row}
            
            # Check against global and local
            if new_diffs & used_diffs:
                continue
            if new_diffs & local_diffs:
                continue
            
            # Acceptance probability
            if randomize and random.random() < skip_prob:
                continue
                
            row.append(x)
            local_diffs |= new_diffs
                
        return row, local_diffs

    # 1. Initial Construction
    best_rows = None
    best_max = float('inf')
    
    for strategy in range(10):
        used = set()
        rows = []
        valid = True
        # Use different randomization levels
        skip_prob = 0.0 if strategy == 0 else 0.05 * strategy
        for i in range(n_rows):
            row, row_diffs = greedy_row(used, randomize=(strategy > 0), skip_prob=skip_prob)
            if row is None:
                valid = False
                break
            rows.append(row)
            used |= row_diffs
            
        if valid:
            cur_max = max_entry(rows)
            if cur_max < best_max:
                best_max = cur_max
                best_rows = [r[:] for r in rows]

    if best_rows is None:
        # Fallback: very loose greedy
        used = set()
        rows = []
        for i in range(n_rows):
            row, row_diffs = greedy_row(used, max_val=1000)
            rows.append(row)
            used |= row_diffs
        return rows

    # 2. Local Search / Simulated Annealing with "Force-Directed" logic
    # We treat the rows as points in a space. We want to minimize the "bounding box" (max entry).
    # We use secondary costs (sum of entries) to guide the search when max is stuck.
    
    current_rows = [r[:] for r in best_rows]
    current_max = best_max
    current_sum = sum_entries(current_rows)
    
    # Pre-compute diffs
    row_diffs_list = [get_diffs(r) for r in current_rows]
    used_set = set()
    for d in row_diffs_list:
        used_set |= d
        
    start_time = time.time()
    time_limit = 110
    
    no_improve_count = 0
    max_no_improve = 200
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Strategy A: Try to shrink the row with the current max
        # Find the index of the row with the max entry
        max_idx = max(range(n_rows), key=lambda i: current_rows[i][-1])
        
        # Try to rebuild that row with max_val = current_max - 1
        old_diffs = row_diffs_list[max_idx]
        used_without_max = used_set - old_diffs
        
        new_row, new_diffs = greedy_row(used_without_max, max_val=current_max - 1)
        
        if new_row is not None:
            current_rows[max_idx] = new_row
            row_diffs_list[max_idx] = new_diffs
            used_set = used_without_max | new_diffs
            current_max = max_entry(current_rows)
            current_sum = sum_entries(current_rows)
            best_max = current_max
            best_rows = [r[:] for r in current_rows]
            improved = True
            no_improve_count = 0
            continue

        # Strategy B: Random Perturbation of any row to find a better "layout"
        # Pick a random row
        i = random.randint(0, n_rows - 1)
        
        old_diffs = row_diffs_list[i]
        used_without_i = used_set - old_diffs
        
        # Try to build a new row for index i
        # We allow it to go up to current_max, but we prefer smaller sums
        new_row, new_diffs = greedy_row(used_without_i, max_val=current_max, randomize=True, skip_prob=0.2)
        
        if new_row is not None and new_row != current_rows[i]:
            new_max = max_entry([r if j != i else new_row for j, r in enumerate(current_rows)])
            new_sum = sum_entries([r if j != i else new_row for j, r in enumerate(current_rows)])
            
            # Acceptance Criteria:
            # 1. If new_max < current_max: Always accept (Global Improvement)
            # 2. If new_max == current_max and new_sum < current_sum: Accept with high prob (Secondary Improvement)
            # 3. If new_max == current_max and new_sum == current_sum: Accept with low prob (Random Walk)
            
            accept = False
            if new_max < current_max:
                accept = True
            elif new_max == current_max:
                if new_sum < current_sum:
                    accept = True
                elif new_sum == current_sum:
                    if random.random() < 0.1:
                        accept = True
                # If new_sum > current_sum, reject to keep packing tight
                
            if accept:
                current_rows[i] = new_row
                row_diffs_list[i] = new_diffs
                used_set = used_without_i | new_diffs
                current_max = new_max
                current_sum = new_sum
                
                if new_max < best_max:
                    best_max = new_max
                    best_rows = [r[:] for r in current_rows]
                    improved = True
                    no_improve_count = 0
                else:
                    # It's a secondary improvement or random walk
                    # We don't update best_rows unless max improves, 
                    # but we do update current state to explore from here.
                    pass

        if not improved:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # If stuck, try a more aggressive restart of the worst row?
                max_idx = max(range(n_rows), key=lambda i: current_rows[i][-1])
                old_diffs = row_diffs_list[max_idx]
                used_without_max = used_set - old_diffs
                new_row, new_diffs = greedy_row(used_without_max, max_val=current_max, randomize=True, skip_prob=0.5)
                if new_row is not None:
                    current_rows[max_idx] = new_row
                    row_diffs_list[max_idx] = new_diffs
                    used_set = used_without_max | new_diffs
                    current_sum = sum_entries(current_rows)
                    # Reset no_improve to give it a chance
                    no_improve_count = 0
                else:
                    break
        else:
            no_improve_count = 0
            
    return best_rows
# EVOLVE-BLOCK-END
