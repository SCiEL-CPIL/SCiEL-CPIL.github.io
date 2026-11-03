import time
import random
import math
from collections import Counter
from itertools import combinations

def _compute_row_diffs(row):
    """Compute all pairwise differences for a single row."""
    n = len(row)
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _is_valid(rows, k=None):
    """Check if all differences are distinct across all rows."""
    seen = set()
    for row in rows:
        if k is not None and len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            if i > 0 and row[i] <= row[i-1]:
                return False
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in seen:
                    return False
                seen.add(d)
    return True

def _get_scope(rows):
    """Return the largest entry in any row."""
    if not rows:
        return 0
    return max(max(row) for row in rows)

def _generate_candidate_rows(k, max_val):
    """
    Generate a list of candidate rows (Sidon sets of size k+1 starting at 0)
    with max entry <= max_val.
    """
    candidates = []
    
    def backtrack(current_row, last_val, used_diffs):
        if len(current_row) == k + 1:
            candidates.append(list(current_row))
            return
        
        for next_val in range(last_val + 1, max_val + 1):
            new_diffs = {next_val - a for a in current_row}
            if not (new_diffs & used_diffs):
                current_row.append(next_val)
                backtrack(current_row, next_val, used_diffs | new_diffs)
                current_row.pop()
                
    backtrack([0], 0, set())
    return candidates

def _find_clique_min_scope(candidates, k, n_rows, deadline):
    """
    Find a set of n_rows candidates from the list such that their difference sets
    are mutually disjoint, minimizing the maximum entry in the selected set.
    """
    if not candidates:
        return None, float('inf')
    
    # Precompute difference sets and max entries
    row_data = []
    for r in candidates:
        diffs = _compute_row_diffs(r)
        max_entry = max(r)
        row_data.append((r, diffs, max_entry))
    
    # Sort by max entry to prioritize smaller scopes
    row_data.sort(key=lambda x: x[2])
    
    best_scope = float('inf')
    best_clique = None
    
    n_cands = len(row_data)
    if n_cands == 0:
        return None, float('inf')
    
    # Greedy heuristic to get an initial upper bound
    def greedy_clique():
        used_diffs = set()
        selected = []
        remaining = list(range(n_cands))
        
        for _ in range(n_rows):
            if not remaining:
                return None, float('inf')
            
            best_idx = -1
            best_score = float('inf')
            for idx in remaining:
                r, d, m = row_data[idx]
                if d & used_diffs:
                    continue
                if m < best_score:
                    best_score = m
                    best_idx = idx
            
            if best_idx == -1:
                return None, float('inf')
                
            selected.append(best_idx)
            used_diffs |= row_data[best_idx][1]
            remaining.remove(best_idx)
            
        return selected, max(row_data[i][2] for i in selected)
        
    init_clique, init_scope = greedy_clique()
    if init_clique:
        best_scope = init_scope
        best_clique = init_clique
        
    # Backtracking search
    # We only consider candidates with max_entry < best_scope
    active_indices = [i for i in range(n_cands) if row_data[i][2] < best_scope]
    
    if not active_indices:
        return best_clique, best_scope
        
    def search(current_clique, current_used_diffs, current_max, start_idx):
        nonlocal best_scope, best_clique
        
        if time.time() > deadline:
            return
            
        if len(current_clique) == n_rows:
            if current_max < best_scope:
                best_scope = current_max
                best_clique = list(current_clique)
            return
            
        if current_max >= best_scope:
            return
            
        for i in range(start_idx, len(active_indices)):
            idx = active_indices[i]
            r, d, m = row_data[idx]
            
            if d & current_used_diffs:
                continue
                
            new_max = max(current_max, m)
            if new_max >= best_scope:
                break
                
            current_clique.append(idx)
            search(current_clique, current_used_diffs | d, new_max, i + 1)
            current_clique.pop()
            
    search([], set(), 0, 0)
    
    if best_clique:
        return [row_data[i][0] for i in best_clique], best_scope
        
    return None, float('inf')

def construct(n_rows, k):
    deadline = time.time() + 145.0
    
    # Generate candidates up to a reasonable max value
    # Best known is 112. Let's try up to 120.
    max_val = 120
    candidates = _generate_candidate_rows(k, max_val)
    
    if not candidates:
        # Fallback
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
        
    best_rows, best_scope = _find_clique_min_scope(candidates, k, n_rows, deadline)
    
    if best_rows is None:
        # Fallback if clique search fails
        used = set()
        rows = []
        cands_sorted = sorted(candidates, key=max)
        for r in cands_sorted:
            if len(rows) == n_rows:
                break
            d = _compute_row_diffs(r)
            if not (d & used):
                rows.append(r)
                used |= d
                
        if len(rows) < n_rows:
            # Final fallback: greedy construction
            used = set()
            rows = []
            for _ in range(n_rows):
                row = [0]
                x = 0
                while len(row) < k + 1:
                    x += 1
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        row.append(x)
                        used |= new_diffs
                rows.append(row)
        return rows
        
    return best_rows
