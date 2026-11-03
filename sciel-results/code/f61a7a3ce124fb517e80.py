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

    def is_valid(rows):
        """Check if all differences across all rows are distinct."""
        used = set()
        for r in rows:
            d = get_diffs(r)
            if d & used:
                return False
            used |= d
        return True

    def max_entry(rows):
        return max(max(r) for r in rows)

    def greedy_row(used_diffs, max_val=None, randomize=False, seed=0):
        """
        Build a row greedily.
        If randomize is True, it may skip valid candidates to explore.
        """
        row = [0]
        local_diffs = set()
        x = 0
        attempts = 0
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
            if randomize:
                # Prefer smaller x, but allow some exploration
                # Probability decreases as x increases relative to current max
                # Simple: accept with prob 0.8, else try next
                if random.random() < 0.8:
                    row.append(x)
                    local_diffs |= new_diffs
                else:
                    continue
            else:
                row.append(x)
                local_diffs |= new_diffs
                
        return row, local_diffs

    # 1. Initial Construction
    # Try a few different greedy strategies to get a good starting point
    best_rows = None
    best_max = float('inf')
    
    for strategy in range(5):
        used = set()
        rows = []
        valid = True
        for i in range(n_rows):
            # Bias: later rows might need to be larger if used space is crowded
            # But we want to minimize max, so we just greedily fill
            row, row_diffs = greedy_row(used, randomize=(strategy > 0), seed=strategy*100+i)
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

    # 2. Local Search / Simulated Annealing
    # We want to reduce best_max.
    # State: best_rows
    # We will perform iterative improvement.
    
    current_rows = [r[:] for r in best_rows]
    current_max = best_max
    used_set = set()
    for r in current_rows:
        used_set |= get_diffs(r)
        
    # Pre-compute diffs for each row for speed
    row_diffs_list = [get_diffs(r) for r in current_rows]
    
    no_improve_count = 0
    max_no_improve = 100
    
    # Time limit check
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Try to improve by rebuilding each row
        for i in range(n_rows):
            if time.time() - start_time > time_limit:
                break
            
            # Remove current row's diffs
            old_diffs = row_diffs_list[i]
            used_without_i = used_set - old_diffs
            
            # Try to rebuild row i with a smaller max
            # We know the current max is current_max.
            # Try to build a row with max < current_max.
            # If that fails, try with max = current_max but different structure?
            # Or just try to build the tightest possible row given the constraints.
            
            # Strategy: Greedy rebuild with tight bound
            # If we can build a row with max < current_max, we improve global max.
            # If not, maybe we can build a row that uses "cheaper" differences,
            # freeing up space for other rows to be smaller?
            
            # Let's try to build the tightest row possible.
            new_row, new_diffs = greedy_row(used_without_i, max_val=current_max - 1)
            
            if new_row is not None:
                # We found a row that fits in a smaller max!
                current_rows[i] = new_row
                row_diffs_list[i] = new_diffs
                used_set = used_without_i | new_diffs
                current_max = max_entry(current_rows)
                best_max = current_max
                best_rows = [r[:] for r in current_rows]
                improved = True
                no_improve_count = 0
                continue # Restart loop as max changed
            
            # If we couldn't reduce the max, try to find an alternative row
            # that might allow OTHER rows to be reduced later.
            # This is harder. Let's try a randomized rebuild within the current max.
            new_row, new_diffs = greedy_row(used_without_i, max_val=current_max, randomize=True, seed=random.randint(0, 10000))
            
            if new_row is not None:
                # Check if this change allows us to improve other rows later?
                # For now, just accept if it's valid (it is by construction)
                # and maybe it has a different structure.
                # To avoid oscillation, only accept if it's "different" or we are stuck.
                if new_row != current_rows[i]:
                    # Accept the change to explore neighborhood
                    current_rows[i] = new_row
                    row_diffs_list[i] = new_diffs
                    used_set = used_without_i | new_diffs
                    # Note: current_max might stay same or decrease if this row was the bottleneck
                    # But we checked max_val=current_max, so it won't increase.
                    if max_entry(current_rows) < best_max:
                        best_max = max_entry(current_rows)
                        best_rows = [r[:] for r in current_rows]
                        improved = True
                        no_improve_count = 0
                    else:
                        # No global improvement, but state changed.
                        # This is a random walk. We need a mechanism to revert if it gets stuck.
                        # For simplicity, let's only accept if it leads to immediate global improvement
                        # OR if we are in a "stuck" state.
                        # Actually, random walks without acceptance criteria can diverge.
                        # Let's revert this unless it improved global max.
                        current_rows[i] = best_rows[i] # Revert to best
                        row_diffs_list[i] = get_diffs(best_rows[i])
                        used_set = set()
                        for r in best_rows:
                            used_set |= get_diffs(r)
                        row_diffs_list = [get_diffs(r) for r in best_rows]
                        current_max = best_max

        if not improved:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
        else:
            no_improve_count = 0
            
    return best_rows
# EVOLVE-BLOCK-END
