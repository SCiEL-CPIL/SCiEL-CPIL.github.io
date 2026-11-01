import time
import random
import sys
from collections import defaultdict

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    deadline = time.time() + 115  # Safety margin before 120s limit
    
    # Generate initial solution using greedy construction
    best_solution = _greedy_construct(n_rows, k)
    if not best_solution:
        return []
    
    best_score = max(max(r) for r in best_solution)
    
    # Precompute all differences for the current solution
    all_diffs = set()
    for r in best_solution:
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                all_diffs.add(r[j] - r[i])
    
    # Iterated Local Search with valid perturbations
    while time.time() < deadline - 15:
        # Perturb the best solution using valid moves
        candidate = [r[:] for r in best_solution]
        
        # Perform several random valid moves (perturbation)
        for _ in range(random.randint(3, 8)):
            r_idx = random.randint(0, n_rows - 1)
            pos = random.randint(1, k)  # Don't touch position 0
            
            # Find a random valid new value for candidate[r_idx][pos]
            new_val = _find_random_valid_value(candidate, r_idx, pos, n_rows, k)
            if new_val is not None and new_val != candidate[r_idx][pos]:
                candidate[r_idx][pos] = new_val
        
        # Local search to minimize scope
        improved = _local_search_minimize(candidate, n_rows, k, deadline)
        
        if improved:
            curr_score = max(max(r) for r in improved)
            if curr_score < best_score:
                best_score = curr_score
                best_solution = improved
    
    return best_solution

def _find_random_valid_value(rows, r_idx, pos, n_rows, k):
    """Find a random valid value for rows[r_idx][pos] different from current."""
    row = rows[r_idx]
    old_val = row[pos]
    
    # Calculate forbidden diffs (excluding those involving old_val)
    all_diffs = set()
    for r in rows:
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                all_diffs.add(r[j] - r[i])
    
    diffs_involving_old = set()
    for i in range(len(row)):
        if i == pos:
            continue
        if i < pos:
            diffs_involving_old.add(old_val - row[i])
        else:
            diffs_involving_old.add(row[i] - old_val)
    
    forbidden_diffs = all_diffs - diffs_involving_old
    
    # Search for a new value
    lower_bound = row[pos - 1] + 1
    upper_bound = 200  # Arbitrary limit
    if pos + 1 < len(row):
        upper_bound = min(upper_bound, row[pos + 1] - 1)
    
    # Collect valid candidates
    valid_candidates = []
    for new_val in range(lower_bound, upper_bound + 1):
        if new_val == old_val:
            continue
        
        # Check validity
        valid = True
        new_diffs = set()
        for i in range(len(row)):
            if i == pos:
                continue
            if i < pos:
                d = new_val - row[i]
            else:
                d = row[i] - new_val
                
            if d in forbidden_diffs or d in new_diffs:
                valid = False
                break
            new_diffs.add(d)
            
        if valid:
            valid_candidates.append(new_val)
            
    if not valid_candidates:
        return None
        
    # Pick randomly
    return random.choice(valid_candidates)

def _local_search_minimize(rows, n_rows, k, deadline):
    """Greedy local search to minimize max entry."""
    best_rows = [r[:] for r in rows]
    
    while time.time() < deadline - 5:
        progress = False
        indices = [(r_idx, pos) for r_idx in range(n_rows) for pos in range(1, k + 1)]
        random.shuffle(indices)
        
        for r_idx, pos in indices:
            if time.time() > deadline - 5:
                break
                
            row = best_rows[r_idx]
            old_val = row[pos]
            
            # Try to find the smallest valid value < old_val
            new_val = _find_smallest_valid_value(best_rows, r_idx, pos, old_val, n_rows, k)
            
            if new_val is not None and new_val < old_val:
                best_rows[r_idx][pos] = new_val
                progress = True
                
        if not progress:
            break
            
    return best_rows

def _find_smallest_valid_value(rows, r_idx, pos, old_val, n_rows, k):
    """Find the smallest valid value for rows[r_idx][pos] that is < old_val."""
    row = rows[r_idx]
    
    # Calculate forbidden diffs
    all_diffs = set()
    for r in rows:
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                all_diffs.add(r[j] - r[i])
    
    diffs_involving_old = set()
    for i in range(len(row)):
        if i == pos:
            continue
        if i < pos:
            diffs_involving_old.add(old_val - row[i])
        else:
            diffs_involving_old.add(row[i] - old_val)
            
    forbidden_diffs = all_diffs - diffs_involving_old
    
    lower_bound = row[pos - 1] + 1
    upper_bound = old_val - 1
    if pos + 1 < len(row):
        upper_bound = min(upper_bound, row[pos + 1] - 1)
        
    if upper_bound < lower_bound:
        return None
        
    # Search for smallest valid value
    for new_val in range(lower_bound, upper_bound + 1):
        valid = True
        new_diffs = set()
        for i in range(len(row)):
            if i == pos:
                continue
            if i < pos:
                d = new_val - row[i]
            else:
                d = row[i] - new_val
                
            if d in forbidden_diffs or d in new_diffs:
                valid = False
                break
            new_diffs.add(d)
            
        if valid:
            return new_val
            
    return None

def _greedy_construct(n_rows, k):
    """Fast greedy baseline construction."""
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        current_max = 0
        while len(row) < k + 1:
            found = False
            for x in range(current_max + 1, current_max + 1000):
                new_diffs = []
                valid = True
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if valid:
                    row.append(x)
                    used_diffs.update(new_diffs)
                    current_max = x
                    found = True
                    break
            if not found:
                return None
        rows.append(row)
    return rows
