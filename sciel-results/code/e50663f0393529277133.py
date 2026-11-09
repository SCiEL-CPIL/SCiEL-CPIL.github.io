import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115
    
    total_diffs = n_rows * (k + 1) * k // 2
    
    def is_valid(rows):
        used = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    if d in used:
                        return False
                    used.add(d)
        return True
    
    def get_scope(rows):
        return max(max(r) for r in rows)
    
    def get_all_diffs(rows):
        diffs = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    diffs.add(r[j] - r[i])
        return diffs
    
    # Interleaved greedy initialization
    def interleaved_greedy(M_limit, seed_val=0):
        rng = random.Random(seed_val)
        used_diffs = set()
        rows = [[0] for _ in range(n_rows)]
        
        # Track position for each row
        positions = [0] * n_rows  # next position to fill (1..k)
        
        # Round-robin: for each round, try to fill next element in each row
        for round_num in range(k):
            for r_idx in range(n_rows):
                if positions[r_idx] > k:
                    continue
                
                row = rows[r_idx]
                last_val = row[-1]
                remaining = k - positions[r_idx]  # how many more to fill after this one
                
                # Find smallest valid value
                found = False
                for val in range(last_val + 1, M_limit - remaining + 1):
                    new_diffs = {val - x for x in row}
                    if new_diffs & used_diffs:
                        continue
                    # Check internal conflicts
                    internal_ok = True
                    for i in range(len(row)):
                        for j in range(i + 1, len(row)):
                            if row[j] - row[i] in new_diffs:
                                internal_ok = False
                                break
                        if not internal_ok:
                            break
                    if not internal_ok:
                        continue
                    
                    row.append(val)
                    used_diffs.update(new_diffs)
                    positions[r_idx] += 1
                    found = True
                    break
                
                if not found:
                    # Try with larger M
                    return None
        
        return rows
    
    # Local search: try to reduce scope by modifying elements
    def local_search(rows, M_target, seed_val=0, max_iters=50000):
        rng = random.Random(seed_val)
        current_rows = [r[:] for r in rows]
        current_scope = get_scope(current_rows)
        best_rows = [r[:] for r in current_rows]
        best_scope = current_scope
        
        used_diffs = get_all_diffs(current_rows)
        
        for it in range(max_iters):
            if time.time() > deadline - 15:
                break
            
            # Choose a random row and position to modify
            r_idx = rng.randint(0, n_rows - 1)
            # Bias toward last element (controls scope)
            if rng.random() < 0.7:
                pos = k  # last element
            else:
                pos = rng.randint(1, k)
            
            row = current_rows[r_idx]
            old_val = row[pos]
            
            # Determine valid range for new value
            if pos == 0:
                continue  # First element is always 0
            
            min_val = row[pos - 1] + 1
            max_val = min(M_target, row[pos + 1] - 1 if pos < k else M_target)
            
            if min_val >= max_val:
                continue
            
            # Try to find a smaller value (to reduce scope) or any valid value
            # Focus on reducing: try values below old_val first
            candidates = []
            # Try values that might reduce scope
            if old_val == current_scope:
                # Try to reduce this element
                for v in range(min_val, old_val):
                    candidates.append(v)
            # Also try random valid values
            for _ in range(5):
                v = rng.randint(min_val, max_val)
                if v not in candidates:
                    candidates.append(v)
            
            rng.shuffle(candidates)
            
            moved = False
            for new_val in candidates:
                if new_val == old_val:
                    continue
                
                # Remove old diffs
                old_diffs = set()
                for i in range(len(row)):
                    if i != pos:
                        d = abs(row[i] - old_val)
                        old_diffs.add(d)
                
                # Check if removing old_diffs is safe (they should only be from this row)
                # Actually, we need to check that no other row uses these diffs
                # For simplicity, let's just try the move and validate
                
                # Tentatively set new value
                row[pos] = new_val
                
                # Check if row is still increasing
                valid = True
                for i in range(1, len(row)):
                    if row[i] <= row[i-1]:
                        valid = False
                        break
                if not valid:
                    row[pos] = old_val
                    continue
                
                # Compute new diffs for this row
                new_row_diffs = set()
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        new_row_diffs.add(row[j] - row[i])
                
                # Check against other rows' diffs
                other_diffs = set()
                for other_r in range(n_rows):
                    if other_r == r_idx:
                        continue
                    other_row = current_rows[other_r]
                    for i in range(len(other_row)):
                        for j in range(i + 1, len(other_row)):
                            other_diffs.add(other_row[j] - other_row[i])
                
                if new_row_diffs & other_diffs:
                    row[pos] = old_val
                    continue
                
                # Move is valid
                new_scope = get_scope(current_rows)
                if new_scope < current_scope or (new_scope == current_scope and rng.random() < 0.1):
                    current_scope = new_scope
                    moved = True
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [r[:] for r in current_rows]
                    break
                else:
                    row[pos] = old_val
            
            if not moved and rng.random() < 0.01:
                # Occasionally reset to best
                current_rows = [r[:] for r in best_rows]
                current_scope = best_scope
        
        return best_rows, best_scope
    
    # Simulated annealing
    def simulated_annealing(rows, M_target, seed_val=0, max_iters=100000):
        rng = random.Random(seed_val)
        current_rows = [r[:] for r in rows]
        current_scope = get_scope(current_rows)
        best_rows = [r[:] for r in current_rows]
        best_scope = current_scope
        
        T = 10.0
        cooling = 0.9995
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            T *= cooling
            if T < 0.01:
                T = 0.01
            
            r_idx = rng.randint(0, n_rows - 1)
            # Bias toward last element
            if rng.random() < 0.7:
                pos = k
            else:
                pos = rng.randint(1, k)
            
            row = current_rows[r_idx]
            old_val = row[pos]
            
            min_val = row[pos - 1] + 1
            max_val = min(M_target, row[pos + 1] - 1 if pos < k else M_target)
            
            if min_val >= max_val:
                continue
            
            new_val = rng.randint(min_val, max_val)
            if new_val == old_val:
                continue
            
            row[pos] = new_val
            
            # Check increasing
            valid = True
            for i in range(1, len(row)):
                if row[i] <= row[i-1]:
                    valid = False
                    break
            if not valid:
                row[pos] = old_val
                continue
            
            # Check conflicts
            new_row_diffs = set()
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    new_row_diffs.add(row[j] - row[i])
            
            other_diffs = set()
            for other_r in range(n_rows):
                if other_r == r_idx:
                    continue
                other_row = current_rows[other_r]
                for i in range(len(other_row)):
                    for j in range(i + 1, len(other_row)):
                        other_diffs.add(other_row[j] - other_row[i])
            
            if new_row_diffs & other_diffs:
                row[pos] = old_val
                continue
            
            new_scope = get_scope(current_rows)
            delta = new_scope - current_scope
            
            if delta < 0 or rng.random() < pow(2.718281828, -delta / T):
                current_scope = new_scope
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [r[:] for r in current_rows]
            else:
                row[pos] = old_val
        
        return best_rows, best_scope
    
    # Final polish: try to reduce each element to minimum valid value
    def final_polish(rows):
        rows = [r[:] for r in rows]
        improved = True
        while improved:
            if time.time() > deadline - 5:
                break
            improved = False
            for r_idx in range(n_rows):
                row = rows[r_idx]
                # Try to reduce each element from right to left
                for pos in range(k, 0, -1):
                    old_val = row[pos]
                    min_val = row[pos - 1] + 1
                    max_val = row[pos + 1] - 1 if pos < k else old_val
                    
                    if min_val >= old_val:
                        continue
                    
                    for new_val in range(min_val, old_val):
                        row[pos] = new_val
                        
                        # Check increasing
                        valid = True
                        for i in range(1, len(row)):
                            if row[i] <= row[i-1]:
                                valid = False
                                break
                        if not valid:
                            continue
                        
                        # Check conflicts
                        new_row_diffs = set()
                        for i in range(len(row)):
                            for j in range(i + 1, len(row)):
                                new_row_diffs.add(row[j] - row[i])
                        
                        other_diffs = set()
                        for other_r in range(n_rows):
                            if other_r == r_idx:
                                continue
                            other_row = rows[other_r]
                            for i in range(len(other_row)):
                                for j in range(i + 1, len(other_row)):
                                    other_diffs.add(other_row[j] - other_row[i])
                        
                        if new_row_diffs & other_diffs:
                            continue
                        
                        # Valid move
                        if new_val < old_val:
                            improved = True
                            break
                        else:
                            row[pos] = old_val
                    else:
                        row[pos] = old_val
        
        return rows
    
    # Main search
    best_rows = None
    best_scope = float('inf')
    
    # Try multiple seeds for interleaved greedy
    M_start = total_diffs
    M_max = total_diffs + 20
    
    for M in range(M_start, M_max + 1):
        if time.time() > deadline - 30:
            break
        
        found = False
        for seed in range(5):
            if time.time() > deadline - 30:
                break
            
            rows = interleaved_greedy(M, seed)
            if rows is not None and is_valid(rows):
                scope = get_scope(rows)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = [r[:] for r in rows]
                found = True
        
        if found and M == total_diffs:
            break
    
    if best_rows is None:
        # Fallback: use larger M
        M_fallback = total_diffs + 50
        for seed in range(10):
            if time.time() > deadline - 20:
                break
            rows = interleaved_greedy(M_fallback, seed)
            if rows is not None and is_valid(rows):
                scope = get_scope(rows)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = [r[:] for r in rows]
                break
    
    if best_rows is None:
        # Emergency fallback
        M_huge = 500
        rows = interleaved_greedy(M_huge, 0)
        if rows is not None:
            best_rows = rows
            best_scope = get_scope(rows)
        else:
            # Should not happen
            best_rows = [[0] + list(range(1, k + 1))] * n_rows
    
    # Local search to improve
    if time.time() < deadline - 20:
        improved_rows, improved_scope = local_search(best_rows, best_scope, seed=123)
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved_rows
    
    # Simulated annealing
    if time.time() < deadline - 15:
        sa_rows, sa_scope = simulated_annealing(best_rows, best_scope, seed=456)
        if sa_scope < best_scope:
            best_scope = sa_scope
            best_rows = sa_rows
    
    # Final polish
    if time.time() < deadline - 5:
        polished_rows = final_polish(best_rows)
        if is_valid(polished_rows):
            polished_scope = get_scope(polished_rows)
            if polished_scope < best_scope:
                best_scope = polished_scope
                best_rows = polished_rows
    
    return best_rows
