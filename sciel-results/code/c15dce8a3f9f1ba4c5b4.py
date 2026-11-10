import time
import random
import copy

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    def greedy(n_rows, k):
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
        return rows
    
    def get_max_entry(rows):
        return max(max(row) for row in rows)
    
    def get_all_diffs(rows):
        diffs = []
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    diffs.append(row[j] - row[i])
        return diffs
    
    def is_valid(rows):
        diffs = get_all_diffs(rows)
        return len(diffs) == len(set(diffs))
    
    def count_conflicts(rows):
        diffs = get_all_diffs(rows)
        return len(diffs) - len(set(diffs))
    
    # Initialize with greedy
    best_rows = greedy(n_rows, k)
    best_score = get_max_entry(best_rows)
    
    if not is_valid(best_rows):
        return best_rows
    
    # Local search: try to reduce the maximum entry
    # Strategy: pick a row, try to reduce its last element, then fix conflicts
    best_rows_copy = [row[:] for row in best_rows]
    
    def try_improve(rows, max_time):
        """Try to improve the solution by reducing entries."""
        current = [row[:] for row in rows]
        current_score = get_max_entry(current)
        best = [row[:] for row in current]
        best_s = current_score
        temp = 10.0
        cooling = 0.999
        
        while time.time() < max_time:
            if temp < 0.1:
                break
            
            # Pick a random row and a random position to try reducing
            row_idx = random.randint(0, n_rows - 1)
            # Try to reduce the last few elements
            pos = random.randint(2, len(current[row_idx]) - 1)
            
            old_val = current[row_idx][pos]
            # Try a smaller value
            delta = random.randint(1, min(5, old_val - current[row_idx][pos-1] - 1)) if old_val > current[row_idx][pos-1] + 1 else 0
            if delta == 0:
                continue
            new_val = old_val - delta
            
            if new_val <= current[row_idx][pos - 1]:
                continue
            
            # Apply the change
            current[row_idx][pos] = new_val
            
            # Check validity
            if is_valid(current):
                new_score = get_max_entry(current)
                if new_score < best_s:
                    best = [row[:] for row in current]
                    best_s = new_score
                # Accept if better or with probability
                if new_score <= current_score:
                    current_score = new_score
                else:
                    import math
                    if random.random() < math.exp(-(new_score - current_score) / temp):
                        current_score = new_score
                    else:
                        current[row_idx][pos] = old_val
            else:
                current[row_idx][pos] = old_val
            
            temp *= cooling
        
        return best, best_s
    
    # Run improvement for a while
    while time.time() < deadline:
        improved, improved_score = try_improve(best_rows, min(deadline, time.time() + 10))
        if improved_score < best_score:
            best_rows = improved
            best_score = improved_score
        else:
            # No improvement, break
            break
    
    return best_rows
