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

    def greedy_row(used_diffs, max_val=None):
        """
        Build a row greedily to fit within max_val, avoiding used_diffs.
        Returns (row, diffs) or (None, None) if not possible.
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
            
            row.append(x)
            local_diffs |= new_diffs
                
        return row, local_diffs

    def randomized_greedy(used_diffs, max_val, rand_seed):
        """
        Build a row with some randomness to explore different valid structures.
        """
        row = [0]
        local_diffs = set()
        x = 0
        random.seed(rand_seed)
        
        while len(row) < k + 1:
            x += 1
            if max_val and x > max_val:
                return None, None
            
            new_diffs = {x - a for a in row}
            
            if new_diffs & used_diffs:
                continue
            if new_diffs & local_diffs:
                continue
            
            # With probability 0.7, accept this x. Otherwise skip to next x.
            # This allows finding different valid rows within the same max bound.
            if random.random() < 0.7:
                row.append(x)
                local_diffs |= new_diffs
                
        return row, local_diffs

    # 1. Initial Construction: Greedy
    best_rows = None
    best_max = float('inf')
    
    # Lower bound on max entry is at least total_diffs
    lower_bound = total_diffs
    
    # Try to construct with increasing max limits
    for margin in range(0, 1000, 20):
        used = set()
        rows = []
        valid = True
        current_max = 0
        for i in range(n_rows):
            row, row_diffs = greedy_row(used, max_val=lower_bound + margin)
            if row is None:
                valid = False
                break
            rows.append(row)
            used |= row_diffs
            if max(row) > current_max:
                current_max = max(row)
        
        if valid:
            if current_max < best_max:
                best_max = current_max
                best_rows = [r[:] for r in rows]
                if current_max == lower_bound:
                    break

    if best_rows is None:
        # Fallback: very loose greedy
        used = set()
        rows = []
        for i in range(n_rows):
            row, row_diffs = greedy_row(used, max_val=2000)
            rows.append(row)
            used |= row_diffs
        return rows

    # 2. Simulated Annealing
    current_rows = [r[:] for r in best_rows]
    current_max = best_max
    
    # Pre-compute diffs for speed
    row_diffs_list = [get_diffs(r) for r in current_rows]
    used_set = set()
    for d in row_diffs_list:
        used_set |= d

    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    # SA Parameters
    T = 10.0  # Initial temperature
    cooling_rate = 0.995
    min_T = 0.01
    
    while time.time() - start_time < time_limit and T > min_T:
        # Pick a random row to modify
        i = random.randint(0, n_rows - 1)
        
        # Remove current row
        old_diffs = row_diffs_list[i]
        used_without_i = used_set - old_diffs
        
        # Generate a new row
        new_row = None
        new_diffs = None
        
        # 1. Try to improve: max < current_max
        new_row, new_diffs = greedy_row(used_without_i, max_val=current_max - 1)
        
        if new_row is None:
            # 2. Try to maintain/explore: max = current_max with randomness
            new_row, new_diffs = randomized_greedy(used_without_i, current_max, random.randint(0, 100000))
            
            if new_row is None or new_row == current_rows[i]:
                T *= cooling_rate
                continue

        new_row_max = max(new_row)
        
        # Determine acceptance
        accept = False
        if new_row_max < current_max:
            accept = True
        elif new_row_max == current_max:
            # Accept with some probability to explore isomorphs
            if random.random() < 0.3:
                accept = True
        else:
            # Accept worse solution with probability exp(-delta/T)
            delta = new_row_max - current_max
            if delta < 50:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    accept = True
        
        if accept:
            # Update state
            current_rows[i] = new_row
            row_diffs_list[i] = new_diffs
            used_set = used_without_i | new_diffs
            current_max = max(max(r) for r in current_rows)
            
            # Track best
            if current_max < best_max:
                best_max = current_max
                best_rows = [r[:] for r in current_rows]
        
        T *= cooling_rate
        
    return best_rows
# EVOLVE-BLOCK-END
