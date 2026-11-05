import random
import time

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer
    
    best_solution = None
    best_max = float('inf')
    
    def is_valid_solution(rows):
        diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d in diffs:
                        return False
                    diffs.add(d)
        return True

    def local_search(rows, time_budget):
        """
        Local search to minimize the maximum value by packing rows tightly.
        """
        rows = [r[:] for r in rows]
        
        # Build global used_diffs
        used_diffs = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    used_diffs.add(r[j] - r[i])
        
        # We will perform passes. In each pass, we try to reduce every element to its minimum valid value.
        # Order of elements in a pass: randomize to avoid bias.
        
        max_no_improvement_passes = 5
        
        no_improvement_passes = 0
        
        while time.time() - start_time < time_budget:
            if no_improvement_passes >= max_no_improvement_passes:
                break
            
            # Create a list of all (row_idx, elem_idx) to process
            moves = []
            for r_idx in range(n_rows):
                for e_idx in range(1, k + 1): # 1 to k inclusive (indices 1..k)
                    moves.append((r_idx, e_idx))
            
            random.shuffle(moves)
            
            improved_in_pass = False
            
            for r_idx, e_idx in moves:
                if time.time() - start_time > time_budget:
                    break
                
                row = rows[r_idx]
                current_val = row[e_idx]
                prev_val = row[e_idx - 1]
                
                # If we can't reduce below current_val, skip
                # min possible is prev_val + 1
                if prev_val + 1 >= current_val:
                    continue
                
                # Remove current contributions
                curr_contribs = set()
                for i in range(e_idx):
                    curr_contribs.add(current_val - row[i])
                for j in range(e_idx + 1, len(row)):
                    curr_contribs.add(row[j] - current_val)
                
                for d in curr_contribs:
                    used_diffs.discard(d)
                
                # Search for smallest valid v in [prev_val + 1, current_val - 1]
                found_val = None
                min_v = prev_val + 1
                max_v = current_val - 1
                
                # Optimization: If the range is large, we might not find a valid one quickly.
                # But typically the "gap" is small.
                
                for v in range(min_v, max_v + 1):
                    # Check validity
                    new_diffs = set()
                    valid = True
                    
                    # Left diffs
                    for i in range(e_idx):
                        d = v - row[i]
                        if d in used_diffs or d in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    
                    if not valid:
                        continue
                        
                    # Right diffs
                    for j in range(e_idx + 1, len(row)):
                        d = row[j] - v
                        if d in used_diffs or d in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    
                    if valid:
                        found_val = v
                        break
                
                if found_val is not None:
                    # Update row and used_diffs
                    row[e_idx] = found_val
                    for d in new_diffs:
                        used_diffs.add(d)
                    improved_in_pass = True
                else:
                    # Restore
                    used_diffs.update(curr_contribs)
            
            if not improved_in_pass:
                no_improvement_passes += 1
            else:
                no_improvement_passes = 0
        
        return rows

    def greedy_construct(randomize=True):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                last = row[-1]
                valid_candidates = []
                x = last + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        valid_candidates.append(x)
                        if not randomize or len(valid_candidates) >= 5:
                            break
                    x += 1
                    # No hard cap, but if x gets huge, it's likely to fail or be slow.
                    # However, for k=8, n_rows=4, values are < 200 usually.
                    if x > 500:
                        break
                
                if not valid_candidates:
                    return None
                
                if randomize:
                    x = random.choice(valid_candidates)
                else:
                    x = valid_candidates[0]
                
                row.append(x)
                used |= {x - a for a in row[:-1]}
            rows.append(row)
        return rows

    # Run multiple iterations
    num_iterations = 200
    for i in range(num_iterations):
        if time.time() - start_time > time_limit:
            break
            
        random.seed(i)
        rows = greedy_construct(randomize=True)
        if rows is None:
            continue
        
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed
        if remaining < 1.0:
            break
            
        rows = local_search(rows, time_limit)
        
        current_max = max(row[-1] for row in rows)
        if current_max < best_max:
            best_max = current_max
            best_solution = [r.copy() for r in rows]
            
    if best_solution is None:
        return greedy_construct(randomize=False)
        
    return best_solution
