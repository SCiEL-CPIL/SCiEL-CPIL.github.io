import time
import random
import sys
from collections import defaultdict

def _greedy_row(used_diffs, M, k):
    """Greedy construction of a single row given used differences."""
    row = [0]
    current_diffs = set()
    while len(row) < k + 1:
        last = row[-1]
        found = False
        for x in range(last + 1, M + 1):
            new_diffs = set()
            valid = True
            for a in row:
                d = x - a
                if d in used_diffs or d in current_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            if valid:
                row.append(x)
                current_diffs |= new_diffs
                found = True
                break
        if not found:
            return None
    return row

def _generate_sidon_rows(M, k, max_rows=None):
    """Generate all valid Sidon rows (internally distinct differences) up to M."""
    rows = []
    if max_rows is None:
        max_rows = 500
    
    def dfs(row, current_diffs):
        if len(row) == k + 1:
            rows.append(list(row))
            return True
        if len(rows) >= max_rows:
            return True
        
        last = row[-1]
        for x in range(last + 1, M + 1):
            new_diffs = set()
            valid = True
            for a in row:
                d = x - a
                if d in current_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            if valid:
                row.append(x)
                current_diffs |= new_diffs
                dfs(row, current_diffs)
                row.pop()
                current_diffs -= new_diffs
                if len(rows) >= max_rows:
                    return True
        return False
    
    dfs([0], set())
    return rows

def _row_diffs(row):
    """Get all differences in a row."""
    diffs = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.add(row[j] - row[i])
    return diffs

def _conflict_graph(rows):
    """Build conflict graph: two rows conflict if they share any difference."""
    n = len(rows)
    row_diffs = [_row_diffs(r) for r in rows]
    conflicts = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if row_diffs[i] & row_diffs[j]:
                conflicts[i].add(j)
                conflicts[j].add(i)
    return conflicts, row_diffs

def _lp_relaxation_bound(conflicts, n_needed):
    """
    Compute LP relaxation bound for minimum weight independent set.
    Uses a simple fractional packing approach.
    Returns a lower bound on the number of rows needed.
    """
    n = len(conflicts)
    if n == 0:
        return 0
    
    # Simple bound: use clique cover approximation
    # For each node, count its non-neighbors + 1
    max_degree = max(len(c) for c in conflicts)
    # A very rough bound
    return max(1, n - max_degree)

def _branch_and_bound(conflicts, row_diffs, n_needed, deadline, max_depth=20):
    """
    Branch and bound to find independent set of size n_needed.
    Returns list of indices if found, else None.
    """
    n = len(conflicts)
    best_solution = None
    best_size = 0
    
    def dfs(selected, available, depth):
        nonlocal best_solution, best_size
        
        if time.time() > deadline:
            return False
        
        if len(selected) == n_needed:
            best_solution = list(selected)
            return True
        
        if not available:
            return False
        
        if depth > max_depth:
            return False
        
        # Lower bound: if we can't reach n_needed even with all available
        if len(selected) + len(available) < n_needed:
            return False
        
        # Choose next row: prioritize rows with fewer conflicts
        candidates = sorted(available, key=lambda i: len(conflicts[i] & available))
        
        for i in candidates:
            if i not in available:
                continue
            
            new_selected = selected + [i]
            # New available: remove i and all its conflicts
            new_available = set(available)
            new_available.discard(i)
            for c in conflicts[i]:
                new_available.discard(c)
            
            if dfs(new_selected, new_available, depth + 1):
                return True
        
        return False
    
    initial_available = set(range(n))
    dfs([], initial_available, 0)
    
    return best_solution

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences are distinct across all rows.
    Minimizes the largest entry.
    """
    deadline = time.time() + 110.0
    
    # Lower bound on M
    total_diffs_needed = n_rows * k * (k + 1) // 2
    M_lower = int((2 * total_diffs_needed) ** 0.5)
    while M_lower * (M_lower + 1) // 2 < total_diffs_needed:
        M_lower += 1
    
    # Try increasing M values
    for M in range(M_lower, M_lower + 30):
        if time.time() > deadline - 5.0:
            break
        
        # Generate candidate Sidon rows
        candidate_rows = _generate_sidon_rows(M, k, max_rows=300)
        
        if len(candidate_rows) < n_rows:
            continue
        
        # Build conflict graph
        conflicts, row_diffs = _conflict_graph(candidate_rows)
        
        # Try to find independent set of size n_rows
        indices = _branch_and_bound(conflicts, row_diffs, n_rows, deadline - 5.0, max_depth=15)
        
        if indices is not None:
            return [candidate_rows[i] for i in indices]
    
    # Fallback: greedy construction
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows
