import time
import random
import sys
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill

    # Each row has k+1 elements starting at 0.
    # Number of differences per row = (k+1)*k/2
    diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * diffs_per_row
    lower_bound = total_diffs

    def greedy_construct(scope_limit=None):
        """
        Greedy construction: build rows one by one, each row greedily.
        Returns list of rows or None if stuck.
        """
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                found = False
                last_val = row[-1]
                max_val = scope_limit if scope_limit is not None else 10000
                for x in range(last_val + 1, max_val + 1):
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

    def simulated_annealing(initial_rows, time_limit):
        """
        Simulated annealing to reduce scope while maintaining validity.
        Uses a boolean array for O(1) difference lookups.
        """
        rows = [r[:] for r in initial_rows]
        
        # Estimate max possible difference for array size
        # Conservative upper bound
        max_possible = max(max(r) for r in rows) + 100
        # Use a boolean array for used differences
        used_arr = bytearray(max_possible + 1)
        
        # Initialize used_arr
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d <= max_possible:
                        used_arr[d] = 1
        
        def compute_scope(rows):
            return max(max(r) for r in rows)
        
        current_scope = compute_scope(rows)
        best_rows = [r[:] for r in rows]
        best_scope = current_scope
        
        # Temperature schedule
        t_start = time.time()
        initial_temp = max(1.0, current_scope / 5.0)
        cooling_rate = 0.9995
        
        while time.time() - t_start < time_limit:
            # Pick a random row and a random position to modify
            row_idx = random.randint(0, n_rows - 1)
            
            # Bias toward the last element to encourage scope reduction
            if random.random() < 0.4:
                pos = k  # Last element
            else:
                pos = random.randint(1, k)
            
            old_val = rows[row_idx][pos]
            
            # Try to change this value
            min_val = rows[row_idx][pos-1] + 1
            max_val = rows[row_idx][pos+1] - 1 if pos < k else best_scope + 50
            
            if min_val > max_val:
                continue

            # For the last element, bias toward smaller values to reduce scope
            if pos == k and random.random() < 0.6:
                # Try values in [min_val, old_val] to shrink
                if old_val > min_val:
                    new_val = random.randint(min_val, old_val)
                else:
                    continue
            else:
                new_val = random.randint(min_val, max_val)
            
            if new_val == old_val:
                continue

            # Check if changing this value maintains validity
            # Collect old diffs involving position pos
            old_diffs = []
            for i in range(k + 1):
                if i != pos:
                    if i < pos:
                        d = old_val - rows[row_idx][i]
                    else:
                        d = rows[row_idx][i] - old_val
                    old_diffs.append(d)

            # Temporarily remove old_diffs from used_arr
            for d in old_diffs:
                if d <= max_possible:
                    used_arr[d] = 0

            # Try adding new diffs
            new_diffs = []
            valid = True
            for i in range(k + 1):
                if i != pos:
                    if i < pos:
                        d = new_val - rows[row_idx][i]
                    else:
                        d = rows[row_idx][i] - new_val
                    if d > max_possible:
                        valid = False
                        break
                    if used_arr[d] == 1:
                        valid = False
                        break
                    new_diffs.append(d)

            if not valid:
                # Restore old_diffs
                for d in old_diffs:
                    if d <= max_possible:
                        used_arr[d] = 1
                continue

            # Compute new scope
            new_row_last = max(rows[row_idx][:pos] + [new_val] + rows[row_idx][pos+1:])
            other_max = max(max(r) for r in rows if r is not rows[row_idx]) if n_rows > 1 else 0
            new_scope = max(other_max, new_row_last)

            # Accept or reject using smooth surrogate
            delta = current_scope - new_scope
            if delta > 0 or random.random() < math.exp(delta / max(initial_temp, 0.1)):
                # Accept
                rows[row_idx][pos] = new_val
                for d in new_diffs:
                    if d <= max_possible:
                        used_arr[d] = 1
                current_scope = new_scope

                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [r[:] for r in rows]
            else:
                # Reject: restore old_diffs
                for d in old_diffs:
                    if d <= max_possible:
                        used_arr[d] = 1

            # Cool down
            initial_temp *= cooling_rate

        return best_rows, best_scope

    def local_polish(rows, time_limit):
        """
        Final high-precision polish: try to reduce the last element of each row
        by scanning downward from current value.
        """
        rows = [r[:] for r in rows]
        
        # Build used differences set
        used_set = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    used_set.add(row[j] - row[i])
        
        def compute_scope(rows):
            return max(max(r) for r in rows)
        
        current_scope = compute_scope(rows)
        best_rows = [r[:] for r in rows]
        best_scope = current_scope
        
        t_start = time.time()
        
        while time.time() - t_start < time_limit:
            improved = False
            for row_idx in range(n_rows):
                if time.time() - t_start >= time_limit:
                    break
                
                # Try to reduce the last element
                row = rows[row_idx]
                last_val = row[k]
                
                # Scan downward from last_val - 1
                for new_val in range(last_val - 1, row[k-1], -1):
                    # Check if we can set row[k] to new_val
                    # Remove old diffs involving position k
                    old_diffs = set()
                    for i in range(k):
                        d = last_val - row[i]
                        old_diffs.add(d)
                    
                    # Check new diffs
                    valid = True
                    for i in range(k):
                        d = new_val - row[i]
                        if d in used_set or d in old_diffs:
                            valid = False
                            break
                    
                    if valid:
                        # Apply change
                        for d in old_diffs:
                            used_set.discard(d)
                        for i in range(k):
                            d = new_val - row[i]
                            used_set.add(d)
                        
                        row[k] = new_val
                        current_scope = compute_scope(rows)
                        improved = True
                        
                        if current_scope < best_scope:
                            best_scope = current_scope
                            best_rows = [r[:] for r in rows]
                        
                        # Continue trying to reduce further
                        last_val = new_val
                    else:
                        break
            
            if not improved:
                break
        
        return best_rows, best_scope

    # Step 1: Greedy construction
    rows = greedy_construct()
    if rows is None:
        # Fallback: try with a larger scope
        rows = greedy_construct(scope_limit=5000)
        if rows is None:
            # Last resort: simple backtracking for small cases
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

    current_scope = max(max(r) for r in rows)

    # Step 2: Simulated annealing to improve
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        # Split time between SA and polish
        sa_time = remaining_time * 0.7
        polish_time = remaining_time * 0.3 - 5
        
        best_rows, best_scope = simulated_annealing(rows, sa_time)
        
        # Step 3: Local polish
        remaining = deadline - time.time()
        if remaining > 5:
            final_rows, final_scope = local_polish(best_rows, min(polish_time, remaining - 3))
            return final_rows
        
        return best_rows

    return rows
