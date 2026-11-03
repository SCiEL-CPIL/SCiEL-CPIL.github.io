import time
import random
import math
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    total_diffs = n_rows * k * (k + 1) // 2
    lb = total_diffs + 1
    
    # Helper to validate and get scope
    def get_scope(rows):
        return max(max(r) for r in rows)

    def is_valid(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def compute_used_set(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        return used

    def greedy_construct(seed):
        """
        Fast greedy construction to get a valid starting point.
        """
        rng = random.Random(seed)
        used = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                last = row[-1]
                found = False
                # Try to find a number close to last to keep it compact
                # But must avoid used diffs
                # Search range: last + 1 to last + 100 (dynamic)
                search_limit = last + 50 + rng.randint(0, 50)
                
                # To speed up, we can pre-compute forbidden diffs for this row
                # For x, diffs are x - a for a in row.
                # We want x such that all x-a are not in used.
                
                candidates = range(last + 1, search_limit)
                # Randomize order slightly to escape deterministic traps
                if rng.random() < 0.5:
                    cand_list = list(candidates)
                    rng.shuffle(cand_list)
                else:
                    cand_list = candidates

                for x in cand_list:
                    new_diffs = [x - a for a in row]
                    if any(d in used for d in new_diffs):
                        continue
                    row.append(x)
                    used.update(new_diffs)
                    found = True
                    break
                
                if not found:
                    # Expand search
                    for x in range(last + 1, last + 200):
                        new_diffs = [x - a for a in row]
                        if any(d in used for d in new_diffs):
                            continue
                        row.append(x)
                        used.update(new_diffs)
                        found = True
                        break
                    if not found:
                        return None
            rows.append(row)
        return rows

    # 1. Get initial valid solution
    best_rows = None
    best_scope = float('inf')
    
    for i in range(50):
        if time.time() > deadline - 2:
            break
        rows = greedy_construct(i)
        if rows:
            scope = get_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
                
    if best_rows is None:
        # Fallback: very wide search
        rows = greedy_construct(9999)
        return rows if rows else [[0]] * n_rows

    current_rows = [list(r) for r in best_rows]
    current_scope = best_scope
    used_set = compute_used_set(current_rows)

    # 2. Simulated Annealing
    # We will try to move elements to reduce scope.
    # Move: pick a row, pick an index > 0, change its value.
    # To maintain validity, we check if new diffs are in used_set.
    
    T = 100.0
    T_min = 0.1
    cooling = 0.995
    
    # Pre-calculate max possible scope for pruning? No, just use time.
    
    while T > T_min and time.time() < deadline - 5:
        # Pick a random row
        r_idx = random.randint(0, n_rows - 1)
        row = current_rows[r_idx]
        
        # Pick a random element to move (index 1 to k)
        # Bias towards larger indices (larger values) as they contribute more to scope
        # Or just uniform? Let's try uniform first, then bias.
        e_idx = random.randint(1, len(row) - 1)
        
        old_val = row[e_idx]
        
        # Determine search range for new value
        # Must be > row[e_idx-1] and < row[e_idx+1] (if exists)
        min_val = row[e_idx - 1] + 1
        max_val = row[e_idx + 1] - 1 if e_idx < len(row) - 1 else old_val + 10 # Allow slight increase
        
        # We want to try values smaller than old_val primarily
        # Sample a few candidates
        candidates = []
        
        # Try a few random values in the valid range
        # Focus on reducing scope
        if old_val > min_val:
            # Try values close to min_val
            for _ in range(5):
                v = random.randint(min_val, old_val)
                candidates.append(v)
            # Try values close to old_val (for exploration)
            for _ in range(2):
                v = random.randint(max(1, old_val - 10), old_val + 5)
                if min_val <= v <= max_val:
                    candidates.append(v)
        
        # Remove duplicates
        candidates = list(set(candidates))
        
        found_improvement = False
        best_candidate = None
        best_new_scope = current_scope
        
        for new_val in candidates:
            if new_val == old_val:
                continue
            
            # Temporarily remove old_val and its diffs
            # This is inefficient. Better to check validity directly.
            
            # New row would be:
            # row[:e_idx] + [new_val] + row[e_idx+1:]
            # Check if new_val is strictly between neighbors
            if new_val <= row[e_idx - 1]:
                continue
            if e_idx < len(row) - 1 and new_val >= row[e_idx + 1]:
                continue
            
            # Calculate new diffs for this row
            new_row = row[:e_idx] + [new_val] + row[e_idx+1:]
            
            # Check if diffs within this row are unique
            row_diffs = set()
            valid_row = True
            for i in range(len(new_row)):
                for j in range(i + 1, len(new_row)):
                    d = new_row[j] - new_row[i]
                    if d in row_diffs:
                        valid_row = False
                        break
                    row_diffs.add(d)
            if not valid_row:
                continue
            
            # Check global validity
            # Old diffs that are removed:
            # Diffs involving old_val at e_idx
            removed_diffs = set()
            for i in range(len(row)):
                if i == e_idx:
                    continue
                d = abs(row[i] - old_val)
                removed_diffs.add(d)
            
            # New diffs that are added:
            added_diffs = set()
            for i in range(len(new_row)):
                if i == e_idx:
                    continue
                d = abs(new_row[i] - new_val)
                added_diffs.add(d)
            
            # Diffs that remain unchanged are fine.
            # We need to check if any added_diff is in (used_set - removed_diffs)
            # And if any added_diff is in added_diffs (duplicate within row, already checked)
            
            # Actually, simpler:
            # new_used = used_set - removed_diffs
            # if added_diffs & new_used: invalid
            # else: valid
            
            new_used = used_set - removed_diffs
            if added_diffs & new_used:
                continue
            
            # Valid move!
            new_scope = max(max(r) for r in current_rows) # Wait, only this row changed
            # Update scope only for this row
            new_row_max = max(new_row)
            # Other rows are unchanged
            other_maxes = [max(current_rows[i]) for i in range(n_rows) if i != r_idx]
            final_scope = max(new_row_max, max(other_maxes) if other_maxes else 0)
            
            if final_scope < best_new_scope:
                best_new_scope = final_scope
                best_candidate = (new_val, new_row, removed_diffs, added_diffs)
        
        if best_candidate:
            new_val, new_row, removed_diffs, added_diffs = best_candidate
            delta_scope = best_new_scope - current_scope
            
            # Acceptance criterion
            accept = False
            if delta_scope < 0:
                accept = True
            else:
                # Simulated Annealing: accept with probability exp(-delta/T)
                # Delta is in terms of scope (integer).
                # Scale T appropriately.
                prob = math.exp(-delta_scope / T)
                if random.random() < prob:
                    accept = True
            
            if accept:
                # Apply move
                current_rows[r_idx] = new_row
                used_set = (used_set - removed_diffs) | added_diffs
                current_scope = best_new_scope
                
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [list(r) for r in current_rows]
                    
                # Restart cooling if we found a new best?
                # T = T * 0.9 (reheat slightly)
                # T = max(T * 0.9, 10.0) 
                
        T *= cooling

    # 3. Final Local Search (Hill Climbing) to polish
    if time.time() < deadline - 2:
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for r_idx in range(n_rows):
                row = current_rows[r_idx]
                for e_idx in range(len(row) - 1, 0, -1):
                    old_val = row[e_idx]
                    if old_val <= 1:
                        continue
                    
                    min_val = row[e_idx - 1] + 1
                    max_val = row[e_idx + 1] - 1 if e_idx < len(row) - 1 else old_val
                    
                    # Try to find a smaller valid value
                    # Iterate from max_val down to min_val
                    # To speed up, we can check candidates in descending order
                    # But we want the SMALLEST valid value to maximize compaction
                    # So iterate from min_val UP to old_val?
                    # Yes, we want to push it as far left as possible.
                    
                    found = False
                    for new_val in range(min_val, old_val):
                        new_row = row[:e_idx] + [new_val] + row[e_idx+1:]
                        
                        # Check row validity
                        row_diffs = set()
                        valid_row = True
                        for i in range(len(new_row)):
                            for j in range(i + 1, len(new_row)):
                                d = new_row[j] - new_row[i]
                                if d in row_diffs:
                                    valid_row = False
                                    break
                                row_diffs.add(d)
                        if not valid_row:
                            continue
                        
                        removed_diffs = set()
                        for i in range(len(row)):
                            if i == e_idx:
                                continue
                            d = abs(row[i] - old_val)
                            removed_diffs.add(d)
                        
                        added_diffs = set()
                        for i in range(len(new_row)):
                            if i == e_idx:
                                continue
                            d = abs(new_row[i] - new_val)
                            added_diffs.add(d)
                        
                        new_used = used_set - removed_diffs
                        if added_diffs & new_used:
                            continue
                        
                        # Valid
                        current_rows[r_idx] = new_row
                        used_set = (used_set - removed_diffs) | added_diffs
                        new_scope = max(max(r) for r in current_rows)
                        if new_scope < current_scope:
                            current_scope = new_scope
                            improved = True
                        if new_scope < best_scope:
                            best_scope = new_scope
                            best_rows = [list(r) for r in current_rows]
                        found = True
                        break # Found the smallest valid value for this element, move to next
    
    return best_rows

if __name__ == "__main__":
    result = construct(4, 7)
    if result:
        print(f"Scope: {max(max(r) for r in result)}")
        print(f"Rows: {result}")
        # Validate
        used = set()
        valid = True
        for row in result:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        valid = False
                        print(f"Duplicate diff {d}")
                    used.add(d)
        print(f"Valid: {valid}")
    else:
        print("No valid construction found")
