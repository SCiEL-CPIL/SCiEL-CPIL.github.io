import random
import time

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 115.0

    best_rows = None
    best_scope = float('inf')
    expected_diffs = (k + 1) * k // 2

    def get_diffs(row):
        return {row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row))}

    def is_valid_row(row):
        if len(row) != k + 1 or row[0] != 0:
            return False
        for i in range(1, len(row)):
            if row[i] <= row[i - 1]:
                return False
        return len(get_diffs(row)) == expected_diffs

    def greedy_construct(used_diffs, max_scope=None):
        """Construct a single row given used_diffs and max_scope."""
        if max_scope is None:
            max_scope = 200
        row = [0]
        while len(row) < k + 1:
            last = row[-1]
            found = False
            for x in range(last + 1, max_scope + 1):
                new_diffs = {x - a for a in row}
                if not (new_diffs & used_diffs) and len(new_diffs) == len(row):
                    row.append(x)
                    used_diffs.update(new_diffs)
                    found = True
                    break
            if not found:
                return None
        return row

    def full_greedy(max_scope=None):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = greedy_construct(used, max_scope)
            if row is None:
                return None
            rows.append(row)
        return rows

    def local_search(rows, max_time):
        current_rows = [list(r) for r in rows]
        current_scope = max(max(r) for r in current_rows)
        row_diff_sets = [get_diffs(r) for r in current_rows]
        
        start_ls = time.time()
        improved = True
        while improved and time.time() - start_ls < max_time and time.time() - start_time < time_limit:
            improved = False
            # Randomize row order
            order = list(range(n_rows))
            random.shuffle(order)
            
            for i in order:
                if time.time() - start_ls >= max_time or time.time() - start_time >= time_limit:
                    break
                
                row = current_rows[i]
                # Try to improve by reducing max element
                # 1. Single element reduction
                for pos in range(len(row) - 1, 0, -1):
                    if time.time() - start_ls >= max_time or time.time() - start_time >= time_limit:
                        break
                    current_val = row[pos]
                    min_val = row[pos - 1] + 1
                    max_val = current_val - 1
                    if max_val < min_val:
                        continue
                    
                    other_diffs = set()
                    for j in range(n_rows):
                        if j != i:
                            other_diffs.update(row_diff_sets[j])
                            
                    found = False
                    for v in range(min_val, max_val + 1):
                        if v <= row[pos - 1]:
                            continue
                        if pos < len(row) - 1 and v >= row[pos + 1]:
                            continue
                        
                        candidate_row = list(row)
                        candidate_row[pos] = v
                        cand_diffs = get_diffs(candidate_row)
                        
                        if len(cand_diffs) != expected_diffs:
                            continue
                        if cand_diffs & other_diffs:
                            continue
                        
                        current_rows[i] = candidate_row
                        row_diff_sets[i] = cand_diffs
                        new_scope = max(max(r) for r in current_rows)
                        if new_scope < current_scope:
                            current_scope = new_scope
                            improved = True
                        found = True
                        break
                    if found:
                        break
                        
                # 2. If no improvement, try to rebuild the row with a lower max scope
                # This is expensive, so do it only if stuck
                if not improved and time.time() - start_ls < max_time - 1:
                    # Calculate max scope allowed for this row
                    # We want to keep global scope low, so try to reduce this row's max
                    target_max = max(row) - 1
                    if target_max >= row[-2] + 1:
                        other_diffs = set()
                        for j in range(n_rows):
                            if j != i:
                                other_diffs.update(row_diff_sets[j])
                        
                        new_row = greedy_construct(other_diffs, target_max)
                        if new_row is not None:
                            # Check if it's valid globally (greedy_construct ensures no conflict with other_diffs)
                            # But we need to ensure it doesn't conflict with itself (handled by greedy)
                            # And it must be strictly increasing (handled by greedy)
                            # Accept if it reduces scope or keeps it same
                            new_scope = max(max(r) for r in current_rows if current_rows.index(r) != i) # This is O(N)
                            # Simpler: just compute new global scope
                            temp_rows = [current_rows[j] if j != i else new_row for j in range(n_rows)]
                            new_global_scope = max(max(r) for r in temp_rows)
                            if new_global_scope <= current_scope:
                                current_rows[i] = new_row
                                row_diff_sets[i] = get_diffs(new_row)
                                if new_global_scope < current_scope:
                                    current_scope = new_global_scope
                                    improved = True

        return current_rows, current_scope

    # Main loop
    # Start with a high scope
    current_max_scope = 150
    
    while time.time() - start_time < time_limit:
        # Try greedy construction
        rows = full_greedy(current_max_scope)
        
        if rows is not None:
            scope = max(max(r) for r in rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = rows
                
                # Local search
                if time.time() - start_time < time_limit - 10:
                    ls_time = min(30.0, time_limit - (time.time() - start_time))
                    improved_rows, improved_scope = local_search(best_rows, ls_time)
                    if improved_scope < best_scope:
                        best_scope = improved_scope
                        best_rows = improved_rows
                
                # If we found a solution, we can try to lower the bound
                if best_scope > 105:
                    current_max_scope = best_scope - 1
                else:
                    current_max_scope = 105 # Don't go below theoretical min
            
        # If greedy failed, we might need to increase scope or try random restarts
        # But greedy with max_scope=150 should work.
        # If it fails, it means no solution exists with max_scope? Unlikely.
        # So we just retry.
        
        if best_rows is not None and best_scope <= 105:
            break

    if best_rows is None:
        # Fallback
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used) and len(new_diffs) == len(row):
                        break
                    x += 1
                row.append(x)
                used.update({x - a for a in row[:-1]})
            rows.append(row)
        best_rows = rows

    return best_rows
