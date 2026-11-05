import time
import random
import copy
import math

def construct(n_rows, k):
    start_time = time.time()
    # Evaluator kills at 150s, task says 120s. Use 115s to be safe.
    deadline = start_time + 115
    
    # 1. Greedy Initialization
    def greedy_init():
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                # Check differences with existing elements in row
                diffs = {x - a for a in row}
                if not (diffs & used_diffs):
                    row.append(x)
                    used_diffs.update(diffs)
            rows.append(row)
        return rows

    curr_rows = greedy_init()
    
    # 2. Compute Initial State
    def get_diff_counts(rows):
        counts = {}
        for r in rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    d = r[j] - r[i]
                    counts[d] = counts.get(d, 0) + 1
        return counts

    def get_penalty(counts):
        pen = 0
        for c in counts.values():
            if c > 1:
                pen += c - 1
        return pen

    def get_M(rows):
        return max(max(r) for r in rows)

    curr_counts = get_diff_counts(curr_rows)
    curr_penalty = get_penalty(curr_counts)
    curr_M = get_M(curr_rows)
    
    best_rows = copy.deepcopy(curr_rows)
    best_M = curr_M
    
    # Track row maxes for efficient M calculation
    row_maxes = [max(r) for r in curr_rows]
    
    # SA Parameters
    T = 30.0
    cooling = 0.9995
    lambda_pen = 50.0
    
    curr_energy = curr_M + lambda_pen * curr_penalty
    
    # 3. SA Loop
    while time.time() < deadline:
        # Generate Move
        r_idx = random.randint(0, n_rows - 1)
        row = curr_rows[r_idx]
        
        # Pick an index to change. Index 0 is fixed at 0.
        j_idx = random.randint(1, k)
        
        old_val = row[j_idx]
        
        # Determine valid range for new value
        low = row[j_idx - 1] + 1
        # Allow expansion slightly beyond current M to escape local minima,
        # but energy penalty discourages large M.
        high = row[j_idx + 1] - 1 if j_idx < k else curr_M + 5
        
        if high < low:
            continue
            
        new_val = random.randint(low, high)
        
        if new_val == old_val:
            continue
            
        # Remove old diffs and update penalty incrementally
        for i in range(j_idx):
            d = old_val - row[i]
            c = curr_counts[d]
            if c > 1:
                curr_penalty -= 1
            curr_counts[d] -= 1
            if curr_counts[d] == 0:
                del curr_counts[d]
                
        for h in range(j_idx + 1, k + 1):
            d = row[h] - old_val
            c = curr_counts[d]
            if c > 1:
                curr_penalty -= 1
            curr_counts[d] -= 1
            if curr_counts[d] == 0:
                del curr_counts[d]
        
        # Tentatively set new value
        row[j_idx] = new_val
        
        # Add new diffs and update penalty incrementally
        for i in range(j_idx):
            d = new_val - row[i]
            c = curr_counts.get(d, 0)
            if c >= 1:
                curr_penalty += 1
            curr_counts[d] = c + 1
            
        for h in range(j_idx + 1, k + 1):
            d = row[h] - new_val
            c = curr_counts.get(d, 0)
            if c >= 1:
                curr_penalty += 1
            curr_counts[d] = c + 1
            
        # Calculate new M efficiently using row_maxes
        old_row_max = row_maxes[r_idx]
        new_row_max = max(row)
        if new_row_max > old_row_max:
            new_M = max(new_row_max, max(row_maxes[:r_idx] + row_maxes[r_idx+1:]) if n_rows > 1 else 0)
        else:
            # Row max decreased or stayed same; M might decrease if this was the max row
            other_max = max(row_maxes[:r_idx] + row_maxes[r_idx+1:]) if n_rows > 1 else 0
            new_M = max(new_row_max, other_max)
        
        new_energy = new_M + lambda_pen * curr_penalty
        
        delta_E = new_energy - curr_energy
        
        # Acceptance criterion
        if delta_E < 0 or (T > 0.01 and random.random() < math.exp(-delta_E / T)):
            curr_energy = new_energy
            curr_M = new_M
            row_maxes[r_idx] = new_row_max
            
            if curr_penalty == 0:
                if curr_M < best_M:
                    best_M = curr_M
                    best_rows = copy.deepcopy(curr_rows)
        else:
            # Revert
            row[j_idx] = old_val
            
            # Remove new diffs (which were added)
            for i in range(j_idx):
                d = new_val - row[i]
                c = curr_counts[d]
                if c > 1:
                    curr_penalty -= 1
                curr_counts[d] -= 1
                if curr_counts[d] == 0:
                    del curr_counts[d]
                    
            for h in range(j_idx + 1, k + 1):
                d = row[h] - new_val
                c = curr_counts[d]
                if c > 1:
                    curr_penalty -= 1
                curr_counts[d] -= 1
                if curr_counts[d] == 0:
                    del curr_counts[d]
            
            # Add old diffs (which were removed)
            for i in range(j_idx):
                d = old_val - row[i]
                c = curr_counts.get(d, 0)
                if c >= 1:
                    curr_penalty += 1
                curr_counts[d] = c + 1
                
            for h in range(j_idx + 1, k + 1):
                d = row[h] - old_val
                c = curr_counts.get(d, 0)
                if c >= 1:
                    curr_penalty += 1
                curr_counts[d] = c + 1

            # M and row_maxes unchanged on revert

        # Cool down
        T *= cooling
        if T < 0.5:
            T = 0.5

    # 4. Final polish: try to reduce M by shifting rows
    # If we have a valid solution, try to minimize M further
    if best_rows and get_diff_counts(best_rows) and all(c == 1 for c in get_diff_counts(best_rows).values()):
        # Try small perturbations to reduce M
        polish_deadline = start_time + 118
        while time.time() < polish_deadline:
            r_idx = random.randint(0, n_rows - 1)
            row = best_rows[r_idx]
            j_idx = random.randint(1, k)
            old_val = row[j_idx]
            
            low = row[j_idx - 1] + 1
            high = row[j_idx + 1] - 1 if j_idx < k else best_M
            
            if high < low:
                continue
                
            # Try values that might reduce M
            candidates = []
            if j_idx == k:
                # Last element: try to reduce
                for v in range(low, old_val):
                    candidates.append(v)
            else:
                # Try small changes
                for v in [old_val - 1, old_val + 1]:
                    if low <= v <= high:
                        candidates.append(v)
            
            if not candidates:
                continue
                
            new_val = random.choice(candidates)
            if new_val == old_val:
                continue
            
            # Check if this change maintains validity
            test_row = row[:]
            test_row[j_idx] = new_val
            
            # Verify strict increase
            if any(test_row[i] >= test_row[i-1] for i in range(1, len(test_row))):
                continue
                
            # Check all differences are unique
            all_diffs = set()
            valid = True
            for r in best_rows:
                if r is row:
                    r = test_row
                for i in range(len(r)):
                    for j in range(i+1, len(r)):
                        d = r[j] - r[i]
                        if d in all_diffs:
                            valid = False
                            break
                        all_diffs.add(d)
                if not valid:
                    break
            
            if valid:
                new_M = max(max(r) for r in best_rows)
                if new_M < best_M:
                    row[j_idx] = new_val
                    best_M = new_M

    return best_rows
