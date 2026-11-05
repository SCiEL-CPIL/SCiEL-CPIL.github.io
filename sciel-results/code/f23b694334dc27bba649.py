import time
import random
import sys

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    """
    Constructs a set of n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimizes the largest entry.
    """
    start_time = time.time()
    time_limit = 115.0

    # Total differences per row
    diffs_per_row = k * (k + 1) // 2
    
    def greedy_construct():
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            for _ in range(k):
                last = row[-1]
                found = False
                # Search for next element
                # Upper bound heuristic: we need space for remaining k - len(row) + 1 elements
                # Max possible value is roughly (k - len(row) + 1) * (current_max_diff_needed)
                # Just use a generous bound
                upper_bound = 200 
                for x in range(last + 1, upper_bound):
                    new_diffs = {x - a for a in row}
                    if new_diffs & used_diffs:
                        continue
                    row.append(x)
                    used_diffs |= new_diffs
                    found = True
                    break
                if not found:
                    # Try larger range
                    for x in range(upper_bound, 500):
                        new_diffs = {x - a for a in row}
                        if new_diffs & used_diffs:
                            continue
                        row.append(x)
                        used_diffs |= new_diffs
                        found = True
                        break
                if not found:
                    return None
            rows.append(row)
        return rows

    best_rows = greedy_construct()
    if best_rows is None:
        # Fallback: very loose construction
        used_diffs = set()
        best_rows = []
        for _ in range(n_rows):
            row = [0]
            for _ in range(k):
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used_diffs):
                        break
                    x += 1
                row.append(x)
                used_diffs |= new_diffs
            best_rows.append(row)

    def get_scope(rows):
        return max(r[-1] for r in rows)

    def is_valid_config(rows):
        used = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def compress_rows(rows):
        improved = True
        while improved:
            improved = False
            tasks = [(r, p) for r in range(n_rows) for p in range(1, k + 1)]
            random.shuffle(tasks)
            
            for r_idx, p_idx in tasks:
                row = rows[r_idx]
                current_val = row[p_idx]
                lower_bound = row[p_idx - 1] + 1
                
                if lower_bound >= current_val:
                    continue

                # Calculate occupied diffs excluding those involving (r_idx, p_idx)
                occupied = set()
                
                # Diffs from other rows
                for i in range(n_rows):
                    if i == r_idx: continue
                    r = rows[i]
                    for a in range(len(r)):
                        for b in range(a + 1, len(r)):
                            occupied.add(r[b] - r[a])
                
                # Diffs from current row not involving p_idx
                r = rows[r_idx]
                for a in range(len(r)):
                    if a == p_idx: continue
                    for b in range(a + 1, len(r)):
                        if b == p_idx: continue
                        occupied.add(r[b] - r[a])
                
                # Try to find smallest valid v
                found_val = None
                for v in range(lower_bound, current_val):
                    valid = True
                    for a in range(p_idx):
                        d = v - r[a]
                        if d in occupied:
                            valid = False
                            break
                    if not valid: continue
                    
                    for b in range(p_idx + 1, len(r)):
                        d = r[b] - v
                        if d in occupied:
                            valid = False
                            break
                    if not valid: continue
                    
                    found_val = v
                    break
                
                if found_val is not None and found_val < current_val:
                    row[p_idx] = found_val
                    improved = True
        
        return rows

    current_scope = get_scope(best_rows)
    no_improve_count = 0
    max_no_improve = 200

    while time.time() - start_time < time_limit:
        new_rows = [r[:] for r in best_rows]
        compress_rows(new_rows)
        
        new_scope = get_scope(new_rows)
        
        if new_scope < current_scope:
            best_rows = new_rows
            current_scope = new_scope
            no_improve_count = 0
        else:
            no_improve_count += 1
            
            if no_improve_count >= max_no_improve:
                # Try perturbation
                perturbed_rows = [r[:] for r in best_rows]
                # Increase a random element to break symmetry/stuck state
                for _ in range(10): # Try a few random perturbations
                    r_idx = random.randint(0, n_rows - 1)
                    p_idx = random.randint(1, k)
                    old_val = perturbed_rows[r_idx][p_idx]
                    # Increase by 1 to 5
                    inc = random.randint(1, 5)
                    new_val = old_val + inc
                    if new_val > current_scope + 10:
                        continue
                    perturbed_rows[r_idx][p_idx] = new_val
                    
                    if is_valid_config(perturbed_rows):
                        compress_rows(perturbed_rows)
                        p_scope = get_scope(perturbed_rows)
                        if p_scope < current_scope:
                            best_rows = perturbed_rows
                            current_scope = p_scope
                            no_improve_count = 0
                            break
                    # Reset perturbation if not used
                    perturbed_rows[r_idx][p_idx] = old_val
                else:
                    # If no perturbation worked, reset counter to allow more time?
                    # Or just break?
                    # Let's just reset counter to try again with different shuffles
                    no_improve_count = 0
                    # Small sleep to avoid busy loop if time is plenty?
                    # No, just continue.
                    
    return best_rows
