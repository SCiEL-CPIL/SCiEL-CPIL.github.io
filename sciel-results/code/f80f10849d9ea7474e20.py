import time
import random
import sys
import math

# EVOLVE-BLOCK-START

def _diffs_of_row(row):
    """Return frozenset of positive differences for a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return frozenset(diffs)

def _total_scope(rows):
    """Return the maximum entry across all rows."""
    if not rows:
        return 0
    return max(max(r) for r in rows)

def _greedy_construct(n_rows, k, rng):
    """
    Fast greedy construction to find an initial valid solution.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        new_diffs = set()
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not (new & used) and not (new & new_diffs):
                row.append(x)
                new_diffs |= new
                if x > 200: # Safety break
                    break
        if len(row) < k + 1:
            return None
        rows.append(row)
        used |= new_diffs
    return rows

def _generate_row_with_max(used_diffs, k, max_val, deadline):
    """
    Generate a row of length k+1 starting at 0 and ending at max_val
    that is compatible with used_diffs.
    Returns None if no such row exists.
    """
    if max_val < k: # Minimum max val for k+1 increasing ints starting at 0 is k
        return None
        
    row = [0] + [None] * (k - 1) + [max_val]
    
    def fill(pos, curr_diffs):
        if time.time() > deadline:
            return False
        
        if pos == k:
            # We are about to set row[k] = max_val
            # Check diffs with max_val
            new_diffs = {max_val - a for a in row[:k]}
            if new_diffs & used_diffs or new_diffs & curr_diffs:
                return False
            # Valid row found
            return True
        
        # We are filling row[pos]
        min_val = row[pos - 1] + 1
        # We need to leave room for (k - pos) more elements up to max_val
        # So val <= max_val - (k - pos)
        upper_bound = max_val - (k - pos)
        
        for val in range(min_val, upper_bound + 1):
            new_diffs = {val - a for a in row[:pos]}
            if new_diffs & used_diffs or new_diffs & curr_diffs:
                continue
            
            row[pos] = val
            curr_diffs.update(new_diffs)
            
            if fill(pos + 1, curr_diffs):
                return True
            
            curr_diffs.difference_update(new_diffs)
            row[pos] = None
            
        return False
    
    if fill(1, set()):
        return row[:]
    return None

def _search_rows(n_rows, k, deadline):
    """
    Depth-first search to find n_rows rows with minimal scope.
    """
    best_scope = [float('inf')]
    best_rows = [None]
    
    rng = random.Random(42)
    init_rows = _greedy_construct(n_rows, k, rng)
    if init_rows:
        best_scope[0] = _total_scope(init_rows)
        best_rows[0] = init_rows
    
    used_diffs = set()
    current_rows = []
    
    def dfs(idx, current_max):
        if time.time() > deadline:
            return
        
        if idx == n_rows:
            if current_max < best_scope[0]:
                best_scope[0] = current_max
                best_rows[0] = [r[:] for r in current_rows]
            return
        
        if current_max >= best_scope[0]:
            return
        
        min_max = current_max if current_rows else 0
        
        # Iterate possible max values for the new row
        for max_val in range(min_max, best_scope[0]):
            if time.time() > deadline:
                return
            
            row = _generate_row_with_max(used_diffs, k, max_val, deadline)
            if row is None:
                continue
            
            new_diffs = _diffs_of_row(row)
            if new_diffs & used_diffs:
                continue
                
            used_diffs.update(new_diffs)
            current_rows.append(row)
            
            next_max = max(current_max, max_val)
            
            dfs(idx + 1, next_max)
            
            current_rows.pop()
            used_diffs.difference_update(new_diffs)
            
            if best_scope[0] <= max_val:
                break

    dfs(0, 0)
    
    if best_rows[0] is None:
        return _greedy_construct(n_rows, k, rng)
        
    return best_rows[0]

def _local_search_scope(rows, k, deadline):
    """
    Local search to reduce the scope.
    """
    rows = [r[:] for r in rows]
    used = set()
    for r in rows:
        used |= _diffs_of_row(r)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(len(rows)):
            if time.time() > deadline:
                break
            row = rows[idx]
            for pos in range(len(row) - 1, 0, -1):
                if time.time() > deadline:
                    break
                old_val = row[pos]
                for new_val in range(old_val - 1, row[pos - 1], -1):
                    old_diffs = set()
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        if i < pos:
                            old_diffs.add(old_val - row[i])
                        else:
                            old_diffs.add(row[i] - old_val)
                    new_diffs = set()
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        if i < pos:
                            new_diffs.add(new_val - row[i])
                        else:
                            new_diffs.add(row[i] - new_val)
                    remaining = used - old_diffs
                    if not (new_diffs & remaining):
                        row[pos] = new_val
                        used = (used - old_diffs) | new_diffs
                        improved = True
                        break
    return rows

def construct(n_rows, k):
    """
    Construct n_rows difference triangles of k+1 integers each.
    Minimizes the scope (largest entry).
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    rows = _search_rows(n_rows, k, deadline)
    
    if time.time() < deadline - 5:
        rows = _local_search_scope(rows, k, deadline)
        
    return rows

# EVOLVE-BLOCK-END
