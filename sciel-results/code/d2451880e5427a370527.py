"""Difference triangle set (7,5) - row-by-row DFS with iterative local optimization.

Combines bitmask-based difference checking with flexible per-row DFS search
and iterative local improvement passes to minimize the scope.
"""

import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 112.0  # safety margin before 120s kill
    row_len = k + 1
    
    # Fast greedy initialization
    rows = _greedy_construct(n_rows, k)
    best_rows = [r[:] for r in rows]
    best_scope = _scope(best_rows)
    
    # Iterative local optimization
    rng = random.Random(42)
    
    while time.time() < deadline - 5:
        improved = False
        
        # Order rows by descending scope, then random tiebreak
        order = list(range(n_rows))
        order.sort(key=lambda i: (-rows[i][-1], rng.random()))
        
        # Compute used differences
        used = set()
        for r in rows:
            used |= _row_diffs(r)
        
        for idx in order:
            if time.time() > deadline - 3:
                break
            
            # Remove this row's differences from used set
            r = rows[idx]
            d = _row_diffs(r)
            used -= d
            
            cur_max = r[-1]
            
            # Try to find a better row (smaller max)
            res = _find_row(used, cur_max, 200, 150000, deadline, rng)
            if res is not None:
                nr, nd = res
                rows[idx] = nr
                used |= nd
                improved = True
            else:
                # Keep original row
                used |= d
        
        # If no improvement, try a perturbation
        if not improved:
            if time.time() > deadline - 3:
                break
            pi = rng.randrange(n_rows)
            d = _row_diffs(rows[pi])
            used -= d
            rows[pi] = _greedy_row(used, row_len)
            used |= _row_diffs(rows[pi])
            continue
        
        # Update best if improved
        s = _scope(rows)
        if s < best_scope:
            best_scope = s
            best_rows = [r[:] for r in rows]
    
    return best_rows


def _scope(rows):
    return max(r[-1] for r in rows)


def _row_diffs(row):
    """Compute all pairwise differences for a single row."""
    n = len(row)
    diffs = set()
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            diffs.add(row[j] - ri)
    return diffs


def _greedy_construct(n_rows, k):
    """Greedy construction: build each row by picking the smallest next mark
    that doesn't conflict with any previously used difference."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k + 1)
        rows.append(row)
        used |= _row_diffs(row)
    return rows


def _greedy_row(used, row_len):
    """Build a single row greedily."""
    row = [0]
    x = 0
    while len(row) < row_len:
        x += 1
        new_diffs = {x - a for a in row}
        if not (new_diffs & used):
            row.append(x)
            used |= new_diffs
    return row


def _find_row(used, best_max_init, S_cap, node_budget, dl, rng):
    """Search for a row with max mark < best_max_init, avoiding used differences.
    
    Returns (row, diffs) or None.
    """
    best = [None]
    best_max = [best_max_init]
    nodes = [0]
    
    def dfs(row, rdiffs, last):
        if time.time() > dl or nodes[0] >= node_budget:
            return
        nodes[0] += 1
        
        if len(row) == 6:  # target row length
            m = row[-1]
            if m < best_max[0]:
                best_max[0] = m
                best[0] = row[:]
            return
        
        hi = min(best_max[0] - 1, S_cap)
        lo = last + 1
        if lo > hi:
            return
        
        for x in range(lo, hi + 1):
            new = {x - a for a in row}
            if new & used or new & rdiffs:
                continue
            row.append(x)
            rdiffs |= new
            dfs(row, rdiffs, x)
            rdiffs -= new
            row.pop()
            if nodes[0] >= node_budget or time.time() > dl:
                return
    
    dfs([0], set(), 0)
    
    if best[0] is None:
        return None
    return best[0], _row_diffs(best[0])


def _verify_solution(rows, n_rows, k):
    """Verify that the solution is valid."""
    if len(rows) != n_rows:
        return False
    all_diffs = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row) - 1):
            if row[i + 1] <= row[i]:
                return False
        diffs = _row_diffs(row)
        if diffs & all_diffs:
            return False
        all_diffs |= diffs
    return True
