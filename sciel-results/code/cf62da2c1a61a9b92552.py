import random
import time
import sys

def construct(n_rows, k):
    """
    Constructs a difference triangle set.
    n_rows: number of rows
    k: parameter such that each row has k+1 elements.
    """
    start_time = time.time()
    time_limit = 145.0  # Safety margin against 150s kill
    
    k_plus_1 = k + 1  # Number of elements per row
    
    def get_score(rows):
        if not rows:
            return float('inf')
        return max(max(row) for row in rows)
    
    def is_valid(rows):
        """Checks if the rows form a valid difference triangle set."""
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    diff = row[j] - row[i]
                    if diff in used:
                        return False
                    used.add(diff)
        return True
    
    def greedy_solution(seed=0):
        """Generate a solution using randomized greedy with adaptive candidate selection."""
        random.seed(seed)
        used = set()
        rows = []
        
        for row_idx in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k_plus_1:
                # Find valid candidates with a wider search
                candidates = []
                x = current_max + 1
                # Adaptive limit: wider for later rows and elements
                limit = x + 600 + row_idx * 150 + (len(row) - 1) * 50
                while x <= limit:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        candidates.append(x)
                        if len(candidates) >= 30:
                            break
                    x += 1
                
                if not candidates:
                    x = current_max + 1
                    while True:
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used):
                            candidates.append(x)
                            break
                        x += 1
                
                # Pick randomly from first few candidates, biased toward smaller
                pick_from = min(len(candidates), 8)
                # Weight toward smaller values
                weights = [1.0 / (i + 1) for i in range(pick_from)]
                total_w = sum(weights)
                r = random.random() * total_w
                cumulative = 0
                chosen = candidates[0]
                for i in range(pick_from):
                    cumulative += weights[i]
                    if r <= cumulative:
                        chosen = candidates[i]
                        break
                
                row.append(chosen)
                used.update({chosen - a for a in row[:-1]})
                current_max = chosen
            
            rows.append(row)
        
        return rows
    
    def local_search(rows, max_time=5.0):
        """Try to improve the solution by reducing the maximum entry."""
        rows = [list(r) for r in rows]
        best_rows = [list(r) for r in rows]
        best_score = get_score(rows)
        
        start = time.time()
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    used.add(row[j] - row[i])
        
        improved = True
        while improved and time.time() - start < max_time:
            improved = False
            
            # Find the element with the largest value
            max_val = -1
            max_r = -1
            max_i = -1
            for r in range(n_rows):
                for i in range(len(rows[r])):
                    if rows[r][i] > max_val:
                        max_val = rows[r][i]
                        max_r = r
                        max_i = i
            
            if max_i == 0:
                break
            
            row = rows[max_r]
            current_val = row[max_i]
            
            # Determine valid range
            min_val = row[max_i - 1] + 1
            max_allowed = row[max_i + 1] - 1 if max_i < len(row) - 1 else current_val - 1
            
            if max_allowed < min_val:
                break
            
            # Remove current differences from used
            old_diffs = set()
            for a in row:
                if a != current_val:
                    d = abs(a - current_val)
                    old_diffs.add(d)
            used -= old_diffs
            
            new_val_found = None
            new_diffs_set = None
            # Try smallest valid replacement first
            for v in range(min_val, current_val):
                valid = True
                new_diffs = set()
                for a in row:
                    if a != current_val:
                        d = abs(a - v)
                        if d in used:
                            valid = False
                            break
                        new_diffs.add(d)
                if valid:
                    new_val_found = v
                    new_diffs_set = new_diffs
                    break
            
            if new_val_found is not None:
                row[max_i] = new_val_found
                used |= new_diffs_set
                improved = True
                score = get_score(rows)
                if score < best_score:
                    best_score = score
                    best_rows = [list(r) for r in rows]
            else:
                used |= old_diffs
                # Try to move other elements to free up space
                # Try swapping with a smaller element in the same row
                found_swap = False
                for i in range(len(row) - 1):
                    if i == max_i:
                        continue
                    v = row[i]
                    if v == 0:
                        continue
                    # Try to move element i to a different position
                    old_diffs_i = set()
                    for a in row:
                        if a != v:
                            d = abs(a - v)
                            old_diffs_i.add(d)
                    used -= old_diffs_i
                    
                    new_pos_found = None
                    for new_pos in range(i + 1, len(row)):
                        if new_pos == max_i:
                            continue
                        min_v = row[new_pos - 1] + 1
                        max_v = row[new_pos + 1] - 1 if new_pos < len(row) - 1 else v
                        if max_v < min_v:
                            continue
                        for nv in range(min_v, v):
                            valid = True
                            nd = set()
                            for a in row:
                                if a != v:
                                    d = abs(a - nv)
                                    if d in used:
                                        valid = False
                                        break
                                    nd.add(d)
                            if valid:
                                new_pos_found = (new_pos, nv, nd)
                                break
                        if new_pos_found:
                            break
                    
                    if new_pos_found:
                        new_pos, nv, nd = new_pos_found
                        # Remove v from position i
                        row.pop(i)
                        # Insert nv at new_pos
                        row.insert(new_pos, nv)
                        used |= nd
                        found_swap = True
                        improved = True
                        score = get_score(rows)
                        if score < best_score:
                            best_score = score
                            best_rows = [list(r) for r in rows]
                        break
                    else:
                        used |= old_diffs_i
                
                if not found_swap:
                    break
        
        return best_rows
    
    def rebuild_tight(rows, seed=0):
        """Rebuild rows more tightly using current solution as a guide."""
        random.seed(seed)
        used = set()
        new_rows = []
        
        # Sort rows by their max value to process in order
        row_order = sorted(range(n_rows), key=lambda r: max(rows[r]))
        
        for r_idx in row_order:
            template = rows[r_idx]
            row = [0]
            current_max = 0
            
            for pos in range(1, len(template)):
                target = template[pos]
                # Try to find a value close to target but valid
                best_val = None
                best_dist = float('inf')
                
                # Search in a window around target
                search_start = max(1, target - 50)
                search_end = target + 100
                
                for x in range(search_start, search_end + 1):
                    if x <= current_max:
                        continue
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        dist = abs(x - target)
                        if dist < best_dist:
                            best_dist = dist
                            best_val = x
                
                if best_val is None:
                    # Fallback: search from current_max+1
                    x = current_max + 1
                    while True:
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used):
                            best_val = x
                            break
                        x += 1
                
                row.append(best_val)
                used.update({best_val - a for a in row[:-1]})
                current_max = best_val
            
            new_rows.append(row)
        
        return new_rows
    
    # Main search loop
    best_rows = None
    best_score = float('inf')
    
    # Try multiple seeds with different strategies
    for seed in range(300):
        if time.time() - start_time > time_limit - 15:
            break
        
        # Alternate between pure greedy and rebuild+local search
        if seed % 3 == 0:
            rows = greedy_solution(seed)
        elif seed % 3 == 1:
            rows = greedy_solution(seed * 7 + 1)
            rows = local_search(rows, max_time=1.5)
        else:
            if best_rows is not None:
                rows = rebuild_tight(best_rows, seed)
            else:
                rows = greedy_solution(seed * 13 + 2)
        
        if rows is None:
            continue
        
        # Apply local search
        remaining = time_limit - 10 - (time.time() - start_time)
        if remaining > 2:
            rows = local_search(rows, max_time=min(2.0, remaining))
        
        score = get_score(rows)
        if score < best_score:
            best_score = score
            best_rows = [list(r) for r in rows]
            
            # Early exit if we hit the theoretical bound
            if score <= 112:
                break
    
    if best_rows is None:
        # Fallback to simple greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k_plus_1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
    
    # Final polish pass if time remains
    remaining = time_limit - 5 - (time.time() - start_time)
    if remaining > 3:
        best_rows = local_search(best_rows, max_time=remaining)
    
    return best_rows
