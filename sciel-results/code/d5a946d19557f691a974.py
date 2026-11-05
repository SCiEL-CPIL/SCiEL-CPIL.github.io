import time
import random
import math
import sys

def _scope(rows):
    return max(r[-1] for r in rows)

def _is_valid(rows):
    all_diffs = []
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i+1, n):
                all_diffs.append(row[j] - row[i])
    return len(all_diffs) == len(set(all_diffs))

def _backtrack_solve(M, n_rows, num_elem, deadline):
    """Try to find n_rows valid rows with scope ≤ M using backtracking."""
    nodes = 0
    max_nodes = 5000000  # Limit
    
    def backtrack(row_idx, pos, current_row, used_mask):
        nonlocal nodes
        nodes += 1
        if nodes > max_nodes or time.time() > deadline:
            return None
        
        if row_idx == n_rows:
            return [r[:] for r in current_rows]
        
        if pos == num_elem:
            current_rows.append(current_row[:])
            result = backtrack(row_idx + 1, 1, [0], used_mask)
            if result is not None:
                return result
            current_rows.pop()
            return None
        
        start_val = current_row[pos - 1] + 1
        for val in range(start_val, M + 1):
            new_diffs = []
            valid = True
            for i in range(pos):
                d = val - current_row[i]
                if d in new_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if not valid:
                continue
            
            conflict = False
            for d in new_diffs:
                if (used_mask >> d) & 1:
                    conflict = True
                    break
            if conflict:
                continue
            
            new_mask = used_mask
            for d in new_diffs:
                new_mask |= (1 << d)
            
            current_row.append(val)
            result = backtrack(row_idx, pos + 1, current_row, new_mask)
            if result is not None:
                return result
            current_row.pop()
        
        return None
    
    current_rows = []
    return backtrack(0, 1, [0], 0)

def _greedy_construct(n_rows, num_elem, scope_hint):
    rows = [[0] for _ in range(n_rows)]
    used_mask = 0
    rng = random.Random(42)
    
    for col in range(1, num_elem):
        order = list(range(n_rows))
        rng.shuffle(order)
        for r in order:
            row = rows[r]
            last_val = row[-1]
            found = False
            for val in range(last_val + 1, scope_hint + 1):
                new_diffs = [val - prev for prev in row]
                if len(new_diffs) != len(set(new_diffs)):
                    continue
                conflict = False
                for d in new_diffs:
                    if (used_mask >> d) & 1:
                        conflict = True
                        break
                if conflict:
                    continue
                row.append(val)
                for d in new_diffs:
                    used_mask |= (1 << d)
                found = True
                break
            if not found:
                return None
    return rows

def _local_search(rows, n_rows, num_elem, deadline):
    """Try to reduce scope by moving elements down."""
    current = [r[:] for r in rows]
    current_scope = _scope(current)
    best = [r[:] for r in current]
    best_scope = current_scope
    
    row_masks = []
    for r in current:
        mask = 0
        n = len(r)
        for i in range(n):
            for j in range(i+1, n):
                mask |= (1 << (r[j] - r[i]))
        row_masks.append(mask)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for r in range(n_rows):
            for pos in range(num_elem - 1, 0, -1):
                row = current[r]
                old_val = row[pos]
                low = row[pos - 1] + 1
                high = row[pos + 1] - 1 if pos < num_elem - 1 else old_val - 1
                
                if low >= old_val:
                    continue
                
                for new_val in range(old_val - 1, low - 1, -1):
                    row[pos] = new_val
                    # Check internal validity
                    diffs = []
                    n = len(row)
                    for i in range(n):
                        for j in range(i+1, n):
                            diffs.append(row[j] - row[i])
                    if len(diffs) != len(set(diffs)):
                        continue
                    # Check global
                    new_mask = 0
                    for d in diffs:
                        new_mask |= (1 << d)
                    other_mask = 0
                    for r2 in range(n_rows):
                        if r2 != r:
                            other_mask |= row_masks[r2]
                    if new_mask & other_mask:
                        continue
                    # Accept
                    row_masks[r] = new_mask
                    new_scope = _scope(current)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best = [r[:] for r in current]
                    improved = True
                    break
                else:
                    row[pos] = old_val
    
    return best

def _greedy_fallback(n_rows, num_elem):
    rows = []
    used = set()
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < num_elem:
            x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0
    
    num_elem = k + 1
    diffs_per_row = num_elem * (num_elem - 1) // 2
    total_diffs = n_rows * diffs_per_row
    min_scope = total_diffs
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Backtracking
    for M in range(min_scope, min_scope + 15):
        if time.time() > deadline - 40:
            break
        rows = _backtrack_solve(M, n_rows, num_elem, deadline - 40)
        if rows is not None and _is_valid(rows):
            best_rows = rows
            best_scope = M
            break
    
    # Phase 2: Greedy + local search
    remaining = deadline - time.time()
    if remaining > 15:
        greedy = _greedy_construct(n_rows, num_elem, min_scope + 15)
        if greedy is not None:
            ls = _local_search(greedy, n_rows, num_elem, deadline - 5)
            s = _scope(ls)
            if s < best_scope:
                best_rows = ls
                best_scope = s
    
    if best_rows is None:
        best_rows = _greedy_fallback(n_rows, num_elem)
    
    return best_rows
