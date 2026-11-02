import random
import copy
import math

def construct(n_rows, k):
    # Build initial solution using greedy
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
    
    # Compute initial max
    max_val = max(max(r) for r in rows)
    
    # Precompute used_diffs
    used_diffs = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                used_diffs.add(row[j] - row[i])
    
    best_rows = copy.deepcopy(rows)
    best_max = max_val
    
    # SA parameters
    T = 50.0
    cooling = 0.999
    iterations = 50000
    
    random.seed(42)
    
    for it in range(iterations):
        # Pick a random row to regenerate
        ri = random.randint(0, n_rows - 1)
        
        # Save old row and remove its diffs
        old_row = rows[ri][:]
        
        old_diffs = set()
        for i in range(len(old_row)):
            for j in range(i+1, len(old_row)):
                old_diffs.add(old_row[j] - old_row[i])
        
        # Remove old diffs from global set
        for d in old_diffs:
            used_diffs.discard(d)
            
        # Try to reconstruct the row
        new_row = [0]
        x = 0
        success = True
        
        # Limit search space to avoid infinite loops or extremely large numbers
        # If we can't find a solution within best_max + margin, fail
        limit = best_max + 20
        
        while len(new_row) < k + 1:
            x += 1
            if x > limit:
                success = False
                break
                
            # Check if x is valid
            valid = True
            for a in new_row:
                d = x - a
                if d in used_diffs:
                    valid = False
                    break
            if valid:
                new_row.append(x)
        
        if not success:
            # Revert: add old diffs back
            used_diffs |= old_diffs
            T *= cooling
            continue
            
        # Calculate new max
        new_max = max(new_row)
        
        # Calculate new diffs
        new_diffs = set()
        for i in range(len(new_row)):
            for j in range(i+1, len(new_row)):
                new_diffs.add(new_row[j] - new_row[i])
        
        # Acceptance criterion
        other_max = max(max(r) for r in rows if r != ri) if n_rows > 1 else 0
        cur_global_max = max(max(r) for r in rows)
        new_global_max = max(new_max, other_max)
        
        delta_E = new_global_max - cur_global_max
        
        accept = False
        if delta_E <= 0:
            accept = True
        else:
            if T > 0:
                if delta_E == 0:
                    p = 1.0
                else:
                    p = math.exp(-delta_E / T)
                if random.random() < p:
                    accept = True
            
        if accept:
            # Apply move
            rows[ri] = new_row
            used_diffs |= new_diffs
            
            new_global_max = max(max(r) for r in rows)
            if new_global_max < best_max:
                best_max = new_global_max
                best_rows = copy.deepcopy(rows)
        else:
            # Revert: add old diffs back
            used_diffs |= old_diffs
            
        T *= cooling
        
    return best_rows
