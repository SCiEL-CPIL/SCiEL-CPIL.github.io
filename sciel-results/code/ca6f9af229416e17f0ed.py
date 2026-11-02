import random
import time
import math

def construct(n_rows, k):
    """
    Constructs a difference triangle set with n_rows rows of k+1 integers.
    Minimizes the largest entry.
    """
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    def get_max_val(rows):
        return max(max(row) for row in rows) if rows else 0

    def get_all_diffs(rows):
        row_diffs = []
        used_diffs = set()
        for row in rows:
            diffs = set()
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    diffs.add(d)
            row_diffs.append(diffs)
            used_diffs.update(diffs)
        return row_diffs, used_diffs

    def is_valid_change(rows, row_diffs, used_diffs, row_idx, new_row):
        if new_row[0] != 0:
            return False
        for i in range(1, len(new_row)):
            if new_row[i] <= new_row[i-1]:
                return False
        
        new_diffs = set()
        for i in range(len(new_row)):
            for j in range(i+1, len(new_row)):
                d = new_row[j] - new_row[i]
                new_diffs.add(d)
                
        old_diffs = row_diffs[row_idx]
        
        # Check conflict
        for d in new_diffs:
            if d in used_diffs and d not in old_diffs:
                return False
        
        return True

    def generate_initial_solution():
        used_diffs = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used_diffs):
                        row.append(x)
                        used_diffs.update(new_diffs)
                        break
                    x += 1
            rows.append(row)
            
        return rows

    def simulated_annealing(rows):
        row_diffs, used_diffs = get_all_diffs(rows)
        current_max = get_max_val(rows)
        best_rows = [r[:] for r in rows]
        best_max = current_max
        
        initial_temp = 5.0
        cooling_rate = 0.99
        min_temp = 0.1
        max_iter = 10000
        
        temp = initial_temp
        rows = [r[:] for r in rows]
        
        for i in range(max_iter):
            if time.time() - start_time > time_limit - 2:
                break
                
            # Identify max elements to bias search
            max_val = current_max
            max_elements = []
            for r_idx, row in enumerate(rows):
                for e_idx, val in enumerate(row):
                    if val == max_val:
                        max_elements.append((r_idx, e_idx))
            
            # Pick a target
            if max_elements and random.random() < 0.7:
                r_idx, elem_idx = random.choice(max_elements)
            else:
                r_idx = random.randint(0, n_rows - 1)
                elem_idx = random.randint(1, k) # 0 is fixed
                
            current_row = rows[r_idx]
            
            # Determine bounds
            lower_bound = current_row[elem_idx - 1] + 1
            upper_bound = current_row[elem_idx + 1] - 1 if elem_idx < k else current_row[elem_idx] + 20
            
            # Try to reduce if possible
            if elem_idx == k or current_row[elem_idx] == max_val:
                # Bias towards smaller values
                delta = random.randint(-15, 5)
            else:
                delta = random.randint(-5, 15)
                
            new_val = current_row[elem_idx] + delta
            
            if new_val < lower_bound:
                new_val = lower_bound
            if new_val > upper_bound:
                new_val = upper_bound
                
            if new_val == current_row[elem_idx]:
                continue
                
            new_row = current_row[:]
            new_row[elem_idx] = new_val
            
            if is_valid_change(rows, row_diffs, used_diffs, r_idx, new_row):
                new_max = get_max_val(rows)
                
                if new_max < current_max:
                    current_max = new_max
                    if new_max < best_max:
                        best_max = new_max
                        best_rows = [r[:] for r in rows]
                        
                    old_diffs = row_diffs[r_idx]
                    new_diffs = set()
                    for i in range(len(new_row)):
                        for j in range(i+1, len(new_row)):
                            new_diffs.add(new_row[j] - new_row[i])
                            
                    used_diffs -= old_diffs
                    used_diffs |= new_diffs
                    rows[r_idx] = new_row
                    row_diffs[r_idx] = new_diffs
                else:
                    if random.random() < math.exp(-(new_max - current_max) / temp):
                        current_max = new_max
                        
                        old_diffs = row_diffs[r_idx]
                        new_diffs = set()
                        for i in range(len(new_row)):
                            for j in range(i+1, len(new_row)):
                                new_diffs.add(new_row[j] - new_row[i])
                            
                        used_diffs -= old_diffs
                        used_diffs |= new_diffs
                        rows[r_idx] = new_row
                        row_diffs[r_idx] = new_diffs
            
            temp *= cooling_rate
            if temp < min_temp:
                temp = min_temp
                
        return best_rows, best_max

    def solve():
        best_rows = None
        best_score = float('inf')
        
        # Run multiple iterations
        # Time budget per iteration depends on N,K
        # For small N,K, we can do many.
        
        iterations = 0
        while time.time() - start_time < time_limit - 5:
            rows = generate_initial_solution()
            improved_rows, score = simulated_annealing(rows)
            
            if score < best_score:
                best_score = score
                best_rows = improved_rows
            iterations += 1
            # If we've done a reasonable number of iterations, break
            if iterations > 50:
                break
                
        if best_rows is None:
            best_rows = generate_initial_solution()
            
        return best_rows

    return solve()
