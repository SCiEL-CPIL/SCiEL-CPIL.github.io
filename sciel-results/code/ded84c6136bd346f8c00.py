import time
import random
import sys

def construct(n_rows, k):
    """
    Constructs a set of n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimizes the largest entry.
    """
    start_time = time.time()
    deadline = 115.0  # Safe margin before 120s kill
    
    # Total number of differences
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Greedy initialization
    def greedy_solve(max_scope):
        used_diffs = set()
        rows = []
        
        for _ in range(n_rows):
            current_row = [0]
            for _ in range(k):
                last_val = current_row[-1]
                found = False
                for x in range(last_val + 1, max_scope + 1):
                    new_diffs = {x - a for a in current_row}
                    if not (new_diffs & used_diffs):
                        current_row.append(x)
                        used_diffs.update(new_diffs)
                        found = True
                        break
                if not found:
                    return None
            rows.append(current_row)
        return rows

    # Initialize with a valid solution using a generous scope
    initial_rows = None
    scope_try = 200
    while initial_rows is None and scope_try <= 500:
        initial_rows = greedy_solve(scope_try)
        if initial_rows is None:
            scope_try += 10
    
    if initial_rows is None:
        # Fallback: extremely loose greedy
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            current_row = [0]
            for _ in range(k):
                last_val = current_row[-1]
                x = last_val + 1
                while True:
                    new_diffs = {x - a for a in current_row}
                    if not (new_diffs & used_diffs):
                        current_row.append(x)
                        used_diffs.update(new_diffs)
                        break
                    x += 1
            rows.append(current_row)
        initial_rows = rows

    # State representation
    # rows: list of lists
    # diff_count: dict mapping diff value to count
    # unique_diffs: int
    # sum_elements: int
    
    def calculate_state_metrics(rows):
        diff_count = {}
        unique_diffs = 0
        sum_elements = 0
        for r in range(n_rows):
            row = rows[r]
            for i in range(len(row)):
                sum_elements += row[i]
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d not in diff_count:
                        diff_count[d] = 1
                        unique_diffs += 1
                    else:
                        diff_count[d] += 1
        return diff_count, unique_diffs, sum_elements

    diff_count, unique_diffs, sum_elements = calculate_state_metrics(initial_rows)
    current_rows = [row[:] for row in initial_rows]
    
    def get_cost():
        dupes = total_diffs - unique_diffs
        # We want to minimize sum_elements, but primarily avoid dupes.
        # If dupes > 0, cost is very high.
        return sum_elements + 10000 * dupes

    current_cost = get_cost()
    best_cost = current_cost
    best_rows = [row[:] for row in current_rows]
    
    # Simulated Annealing
    T = 50.0
    cooling = 0.9999
    min_T = 0.01
    
    # Helper to get diffs for a specific index in a row
    def get_diffs_for_index(row, idx):
        diffs = []
        # Diffs ending at idx: row[idx] - row[i] for i < idx
        for i in range(idx):
            diffs.append(row[idx] - row[i])
        # Diffs starting at idx: row[j] - row[idx] for j > idx
        for j in range(idx + 1, len(row)):
            diffs.append(row[j] - row[idx])
        return diffs

    # Main SA loop
    steps = 0
    last_improve_step = 0
    no_improve_count = 0
    
    while time.time() - start_time < deadline:
        # Generate a random move
        r_idx = random.randint(0, n_rows - 1)
        # Choose a non-zero index to change
        j_idx = random.randint(1, k) # Indices 1..k
        
        row = current_rows[r_idx]
        old_val = row[j_idx]
        
        # Determine valid range for new value
        lower_bound = row[j_idx - 1] + 1
        # Upper bound: if j_idx < k, it must be < row[j_idx + 1]
        # If j_idx == k, it can be anything > lower_bound, but we should keep it reasonable.
        # Let's allow it to go up to current max + 10 to allow "swaps" that temporarily increase sum.
        if j_idx < k:
            upper_bound = row[j_idx + 1] - 1
        else:
            # Allow some expansion, but not too much
            current_max = max(max(r) for r in current_rows)
            upper_bound = current_max + 20
            
        if lower_bound > upper_bound:
            # Invalid move configuration, skip
            continue
            
        new_val = random.randint(lower_bound, upper_bound)
        
        if new_val == old_val:
            continue
            
        # Calculate delta cost
        # 1. Remove old diffs
        old_diffs = get_diffs_for_index(row, j_idx)
        old_unique_diffs = unique_diffs
        old_sum_elements = sum_elements
        
        # Update diff_count and unique_diffs for removal
        for d in old_diffs:
            diff_count[d] -= 1
            if diff_count[d] == 0:
                unique_diffs -= 1
                # Clean up dict to save memory? Optional.
                # del diff_count[d]
                
        # Update sum
        sum_elements -= old_val
        sum_elements += new_val
        
        # 2. Add new diffs
        # Temporarily set new val to calculate new diffs
        row[j_idx] = new_val
        new_diffs = get_diffs_for_index(row, j_idx)
        
        for d in new_diffs:
            if d not in diff_count:
                diff_count[d] = 1
                unique_diffs += 1
            else:
                diff_count[d] += 1
        
        new_cost = get_cost()
        delta_cost = new_cost - current_cost
        
        # Acceptance
        accept = False
        if delta_cost < 0:
            accept = True
        else:
            # Boltzmann factor
            # Avoid overflow
            if T > 0:
                prob = 1.0 / (1.0 + delta_cost / T) # Simple approximation or exp
                # Using exp for proper SA
                # exp(-delta/T)
                if delta_cost / T < 700: # Avoid underflow/overflow issues
                    accept = random.random() < (2.718281828 ** (-delta_cost / T))
                else:
                    accept = False
        
        if accept:
            current_cost = new_cost
            # State is already updated in row, diff_count, unique_diffs, sum_elements
            if current_cost < best_cost:
                best_cost = current_cost
                best_rows = [row[:] for row in current_rows]
                last_improve_step = steps
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            # Revert changes
            # 1. Remove new diffs
            for d in new_diffs:
                diff_count[d] -= 1
                if diff_count[d] == 0:
                    unique_diffs -= 1
                    # del diff_count[d]
            
            # 2. Restore old diffs
            row[j_idx] = old_val
            for d in old_diffs:
                if d not in diff_count:
                    diff_count[d] = 1
                    unique_diffs += 1
                else:
                    diff_count[d] += 1
            
            # 3. Restore sum
            sum_elements -= new_val
            sum_elements += old_val
            
            current_cost = old_sum_elements + 10000 * (total_diffs - unique_diffs)
            no_improve_count += 1

        steps += 1
        
        # Cooling
        if T > min_T:
            T *= cooling
        
        # Restart if stuck
        if no_improve_count > 10000:
            T = 50.0
            no_improve_count = 0
            # Optional: randomize a bit to escape local minima
            # For now, just reset temperature.

    return best_rows
