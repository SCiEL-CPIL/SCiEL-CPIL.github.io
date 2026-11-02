import random
import time
import copy

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 115.0  # Leave buffer for overhead
    
    # 1. Initial Solution: Greedy
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
    
    # Precompute global used diffs
    used_diffs = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                used_diffs.add(row[j] - row[i])
                
    best_rows = copy.deepcopy(rows)
    best_max = max(max(r) for r in rows)
    current_max = best_max
    
    # SA Parameters
    T = 10.0
    T_min = 0.01
    cooling = 0.99995
    random.seed(42)
    
    # Helper to calculate energy (max value)
    def get_max_val(rows):
        return max(max(r) for r in rows)
    
    # Helper to check if a move is valid
    def is_valid_move(rows, ri, ei, new_val, used_diffs):
        row = rows[ri]
        old_val = row[ei]
        if new_val == old_val:
            return False
        
        # Check bounds
        if ei > 0 and new_val <= row[ei-1]:
            return False
        if ei < k and new_val >= row[ei+1]:
            return False
            
        # Check for conflicts
        # Generate new diffs
        new_diffs = set()
        for j in range(len(row)):
            if j != ei:
                new_diffs.add(abs(new_val - row[j]))
                
        # Generate old diffs to exclude from conflict check
        old_diffs = set()
        for j in range(len(row)):
            if j != ei:
                old_diffs.add(abs(old_val - row[j]))
                
        # If any new diff is in used_diffs AND not in old_diffs, it's a conflict
        for d in new_diffs:
            if d in used_diffs and d not in old_diffs:
                return False
                
        return True

    while time.time() - start_time < time_limit:
        # Pick random row and element
        ri = random.randint(0, n_rows - 1)
        ei = random.randint(1, k)
        
        row = rows[ri]
        old_val = row[ei]
        
        # Determine valid range for new value
        lower = row[ei-1] + 1
        upper = row[ei+1] - 1 if ei < k else current_max + 50
        
        if lower >= upper:
            continue
            
        # Choose a new value randomly in the range
        # Bias towards smaller values to encourage compression
        if random.random() < 0.7 and lower < old_val:
            # Try to shrink
            new_val = random.randint(lower, old_val - 1)
        elif upper > old_val:
            # Try to expand or shift right
            new_val = random.randint(old_val + 1, upper)
        else:
            # Fallback
            new_val = random.randint(lower, upper)
            
        if not is_valid_move(rows, ri, ei, new_val, used_diffs):
            continue
            
        # Apply move temporarily
        row[ei] = new_val
        
        # Update used_diffs
        # Remove old diffs
        for j in range(len(row)):
            if j != ei:
                used_diffs.remove(abs(old_val - row[j]))
                
        # Add new diffs
        for j in range(len(row)):
            if j != ei:
                used_diffs.add(abs(new_val - row[j]))
        
        new_max = get_max_val(rows)
        
        # Acceptance
        delta_e = new_max - current_max
        if delta_e < 0 or random.random() < pow(2.71828, -delta_e / T):
            current_max = new_max
            if new_max < best_max:
                best_max = new_max
                best_rows = copy.deepcopy(rows)
        else:
            # Reject move: revert
            row[ei] = old_val
            
            # Revert used_diffs
            for j in range(len(row)):
                if j != ei:
                    used_diffs.remove(abs(new_val - row[j]))
            for j in range(len(row)):
                if j != ei:
                    used_diffs.add(abs(old_val - row[j]))
            
        T *= cooling
        if T < T_min:
            T = T_min

    return best_rows
