import time
import sys
import random
from collections import Counter

def construct(n_rows, k):
    """
    Construct a difference triangle set with n_rows rows, each of k+1 elements.
    Returns a list of rows.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Each row has k+1 elements starting at 0.
    # Number of differences per row = (k+1)*k/2
    diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * diffs_per_row
    
    def get_scope(rows):
        """Get the maximum entry across all rows."""
        return max(max(r) for r in rows)
    
    def count_conflicts(rows):
        """Count the number of duplicate differences across all rows."""
        diffs = []
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    diffs.append(r[j] - r[i])
        counts = Counter(diffs)
        conflicts = 0
        for count in counts.values():
            if count > 1:
                conflicts += count - 1
        return conflicts
    
    def is_valid(rows):
        diffs = []
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    diffs.append(r[j] - r[i])
        return len(diffs) == len(set(diffs))
    
    # Initial solution: Greedy
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
        return rows
    
    current_rows = greedy_construct()
    best_rows = [r[:] for r in current_rows]
    best_scope = get_scope(best_rows)
    
    # Compute initial conflict count
    initial_conflicts = count_conflicts(current_rows)
    
    # Phase 1: Eliminate conflicts using SA
    # Objective: minimize conflicts, then scope
    if initial_conflicts > 0:
        temp = 500.0
        cooling = 0.9995
        min_temp = 1.0
        
        while time.time() < deadline - 20.0 and temp > min_temp:
            # Generate neighbor: change one element in one row
            new_rows = [r[:] for r in current_rows]
            row_idx = random.randint(0, n_rows - 1)
            elem_idx = random.randint(1, k)
            
            low = new_rows[row_idx][elem_idx - 1] + 1
            high = new_rows[row_idx][elem_idx + 1] - 1 if elem_idx < k else best_scope + 50
            
            if low > high:
                continue
                
            new_val = random.randint(low, high)
            if new_val == new_rows[row_idx][elem_idx]:
                continue
            
            old_val = new_rows[row_idx][elem_idx]
            new_rows[row_idx][elem_idx] = new_val
            
            # Check strictly increasing
            row = new_rows[row_idx]
            valid = True
            for i in range(k):
                if row[i] >= row[i + 1]:
                    valid = False
                    break
            if not valid:
                new_rows = [r[:] for r in current_rows]
                continue
            
            new_conflicts = count_conflicts(new_rows)
            old_conflicts = count_conflicts(current_rows)
            
            delta = new_conflicts - old_conflicts
            if delta < 0 or random.random() < pow(2.718, -delta / temp):
                current_rows = new_rows
                if count_conflicts(current_rows) < initial_conflicts:
                    initial_conflicts = count_conflicts(current_rows)
                
                cur_scope = get_scope(current_rows)
                if cur_scope < best_scope:
                    best_scope = cur_scope
                    best_rows = [r[:] for r in current_rows]
            
            temp *= cooling
    
    # Phase 2: Minimize scope while keeping conflicts at 0
    cur_conflicts = count_conflicts(current_rows)
    if cur_conflicts == 0:
        # We have a valid solution, now minimize scope
        temp = 200.0
        cooling = 0.999
        min_temp = 0.5
        phase2_deadline = deadline - 5.0
        
        while time.time() < phase2_deadline and temp > min_temp:
            # Generate neighbor
            new_rows = [r[:] for r in current_rows]
            row_idx = random.randint(0, n_rows - 1)
            elem_idx = random.randint(1, k)
            
            low = new_rows[row_idx][elem_idx - 1] + 1
            high = new_rows[row_idx][elem_idx + 1] - 1 if elem_idx < k else best_scope + 10
            
            if low > high:
                continue
                
            new_val = random.randint(low, high)
            if new_val == new_rows[row_idx][elem_idx]:
                continue
            
            new_rows[row_idx][elem_idx] = new_val
            
            # Check strictly increasing
            row = new_rows[row_idx]
            valid = True
            for i in range(k):
                if row[i] >= row[i + 1]:
                    valid = False
                    break
            if not valid:
                new_rows = [r[:] for r in current_rows]
                continue
            
            new_conflicts = count_conflicts(new_rows)
            if new_conflicts > 0:
                # Reject if it introduces conflicts
                continue
            
            new_scope = get_scope(new_rows)
            cur_scope = get_scope(current_rows)
            
            delta = new_scope - cur_scope
            if delta < 0 or random.random() < pow(2.718, -delta / temp):
                current_rows = new_rows
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [r[:] for r in current_rows]
            
            temp *= cooling
    
    # Final validation
    if not is_valid(best_rows):
        # Fallback: return greedy if it was valid, otherwise best effort
        if is_valid(current_rows):
            return current_rows
        return best_rows
    
    return best_rows
