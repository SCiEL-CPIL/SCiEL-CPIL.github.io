import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    # Deadline guard
    start_time = time.time()
    timeout = 115  # Leave 5s margin for 120s limit
    
    # Helper: compute all positive differences in a row
    def row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    # Helper: check if a set of diffs conflicts with used
    def no_conflict(diffs, used):
        return len(diffs & used) == 0
    
    # Greedy construction to get initial valid solution
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row if x > a}
                    if no_conflict(new_diffs, used) and len(new_diffs) == len(row):
                        row.append(x)
                        used |= new_diffs
                        break
                    x += 1
            rows.append(row)
        return rows
    
    # Get initial solution
    best_rows = greedy_construct()
    best_max = max(max(r) for r in best_rows)
    
    # Simulated annealing
    def get_score(rows):
        return max(max(r) for r in rows)
    
    def is_valid(rows):
        all_diffs = set()
        for row in rows:
            rd = row_diffs(row)
            if len(rd & all_diffs) > 0:
                return False
            all_diffs |= rd
        return True
    
    # Move: try to decrease a specific entry in a specific row
    def try_decrease_entry(rows, row_idx, pos, target_val):
        """Try to decrease rows[row_idx][pos] to target_val while maintaining validity."""
        if target_val <= rows[row_idx][pos-1] or target_val >= rows[row_idx][pos+1] if pos+1 < len(rows[row_idx]) else target_val > rows[row_idx][pos]:
            return None
        
        new_rows = [r[:] for r in rows]
        new_rows[row_idx][pos] = target_val
        
        # Check if the row is still strictly increasing
        for i in range(len(new_rows[row_idx]) - 1):
            if new_rows[row_idx][i] >= new_rows[row_idx][i+1]:
                return None
        
        # Check validity
        if not is_valid(new_rows):
            return None
        
        return new_rows
    
    # Move: rebuild a row with a target max
    def rebuild_row(rows, row_idx, target_max, used_other):
        """Try to rebuild rows[row_idx] with max entry <= target_max."""
        # Remove this row's diffs from consideration
        old_diffs = row_diffs(rows[row_idx])
        
        # Try greedy construction with constraint
        row = [0]
        while len(row) < k + 1:
            found = False
            for x in range(row[-1] + 1, target_max + 1):
                new_diffs = {x - a for a in row}
                # Check against other rows' diffs
                if new_diffs & used_other:
                    continue
                # Check internal uniqueness (should be guaranteed by construction)
                if len(new_diffs) == len(row):
                    row.append(x)
                    found = True
                    break
            if not found:
                return None
        
        if max(row) > target_max:
            return None
        
        # Verify full validity
        new_rows = rows[:]
        new_rows[row_idx] = row
        if not is_valid(new_rows):
            return None
        
        return new_rows
    
    # Simulated annealing loop
    current_rows = best_rows
    current_max = best_max
    best_rows_sa = best_rows
    best_max_sa = best_max
    
    temp = 10.0
    cooling = 0.999
    
    while time.time() - start_time < timeout:
        if temp < 0.1:
            break
        
        # Choose a move
        move_type = random.random()
        
        if move_type < 0.5:
            # Try to decrease an entry
            row_idx = random.randint(0, n_rows - 1)
            pos = random.randint(1, k)  # Can't change position 0 (it's 0)
            
            current_val = current_rows[row_idx][pos]
            if current_val == 0:
                continue
            
            # Target: decrease by 1, or to some smaller value
            target = current_val - 1
            if target <= current_rows[row_idx][pos-1]:
                continue
            
            new_rows = try_decrease_entry(current_rows, row_idx, pos, target)
            if new_rows is None:
                continue
        else:
            # Try to rebuild a row with lower max
            row_idx = random.randint(0, n_rows - 1)
            current_row_max = max(current_rows[row_idx])
            target_max = current_row_max - 1
            if target_max < k:  # Minimum possible max for k+1 elements starting at 0
                target_max = k
            
            # Compute used diffs from other rows
            used_other = set()
            for i in range(n_rows):
                if i != row_idx:
                    used_other |= row_diffs(current_rows[i])
            
            new_rows = rebuild_row(current_rows, row_idx, target_max, used_other)
            if new_rows is None:
                continue
        
        # Evaluate new solution
        new_max = get_score(new_rows)
        delta = new_max - current_max
        
        # Acceptance criterion
        if delta < 0 or random.random() < pow(2.718281828, -delta / temp):
            current_rows = new_rows
            current_max = new_max
            
            if current_max < best_max_sa:
                best_rows_sa = current_rows
                best_max_sa = current_max
        
        temp *= cooling
    
    return best_rows_sa
# EVOLVE-BLOCK-END
