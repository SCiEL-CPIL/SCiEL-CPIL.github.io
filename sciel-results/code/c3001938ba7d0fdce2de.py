"""
Program to find a tournament on n vertices where every set of 4 vertices has a common dominator.
Uses the known Paley tournament on n=67 as a starting point, then attempts local search
to find a valid tournament on n=66 (or smaller) within the time budget.
If no smaller valid tournament is found, returns the Paley 67 solution.
"""

import time
import random
import itertools
import numpy as np

def _paley_orientation(n):
    """Generate Paley tournament orientation for n vertices (n must be a prime)."""
    squares = {x * x % n for x in range(1, n)}
    orientation = []
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % n in squares:
                orientation.append(0)
            else:
                orientation.append(1)
    return orientation

def _orientation_to_adj(n, orientation):
    """Convert flat orientation list to adjacency matrix (adj[i][j] = 1 if i->j)."""
    adj = np.zeros((n, n), dtype=np.uint8)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            bit = orientation[idx]
            idx += 1
            if bit == 0:
                adj[i][j] = 1
            else:
                adj[j][i] = 1
    return adj

def _adj_to_orientation(n, adj):
    """Convert adjacency matrix to flat orientation list."""
    orientation = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j] == 1:
                orientation.append(0)
            else:
                orientation.append(1)
    return orientation

def _count_violations(adj):
    """Count number of 4-subsets that do NOT have a common dominator."""
    n = adj.shape[0]
    violations = 0
    # For each 4-subset, check if any vertex dominates all others in the subset
    for combo in itertools.combinations(range(n), 4):
        a, b, c, d = combo
        # Check if a dominates b,c,d
        if adj[a][b] and adj[a][c] and adj[a][d]:
            continue
        # Check if b dominates a,c,d
        if adj[b][a] and adj[b][c] and adj[b][d]:
            continue
        # Check if c dominates a,b,d
        if adj[c][a] and adj[c][b] and adj[c][d]:
            continue
        # Check if d dominates a,b,c
        if adj[d][a] and adj[d][b] and adj[d][c]:
            continue
        violations += 1
    return violations

def _count_violations_fast(adj):
    """Faster violation count using numpy operations."""
    n = adj.shape[0]
    violations = 0
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    # Check if any of a,b,c,d dominates the other three
                    if (adj[a][b] and adj[a][c] and adj[a][d] or
                        adj[b][a] and adj[b][c] and adj[b][d] or
                        adj[c][a] and adj[c][b] and adj[c][d] or
                        adj[d][a] and adj[d][b] and adj[d][c]):
                        continue
                    violations += 1
    return violations

def _local_search(adj, max_time=5.0):
    """Attempt to reduce violations via single-edge flips. Returns best adj found."""
    n = adj.shape[0]
    best_adj = adj.copy()
    best_violations = _count_violations_fast(best_adj)
    start_time = time.time()
    
    while time.time() - start_time < max_time:
        if best_violations == 0:
            break
        
        # Try flipping each edge and see if it reduces violations
        improved = False
        random.seed(int(time.time() * 1000) % 100000)
        edge_order = list(range(n * (n - 1) // 2))
        random.shuffle(edge_order)
        
        for idx in edge_order:
            if time.time() - start_time > max_time:
                break
            
            # Convert flat index to (i, j)
            i = 0
            count = 0
            found = False
            for a in range(n):
                for b in range(a + 1, n):
                    if count == idx:
                        i, j = a, b
                        found = True
                        break
                    count += 1
                if found:
                    break
            if not found:
                continue
            
            # Flip edge (i, j)
            old_val = best_adj[i][j]
            best_adj[i][j] = 1 - old_val
            best_adj[j][i] = 1 - best_adj[i][j]
            
            new_violations = _count_violations_fast(best_adj)
            
            if new_violations < best_violations:
                best_violations = new_violations
                improved = True
            else:
                # Revert
                best_adj[i][j] = old_val
                best_adj[j][i] = 1 - old_val
        
        if not improved:
            break
    
    return best_adj, best_violations

def construct():
    """
    Returns the smallest known valid tournament for the Schütte k=4 problem.
    Tries to find a valid tournament on n=66 via local search; if unsuccessful,
    returns the Paley tournament on n=67.
    """
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Start with Paley 67 (known valid)
    p_known = 67
    orientation_67 = _paley_orientation(p_known)
    adj_67 = _orientation_to_adj(p_known, orientation_67)
    
    # Verify Paley 67 is valid
    v67 = _count_violations_fast(adj_67)
    if v67 != 0:
        # Should not happen, but fallback
        return {'n': p_known, 'orientation': orientation_67}
    
    # Try to find a valid tournament on n=66
    # Initialize with a random tournament
    n_try = 66
    random.seed(42)
    orientation_try = [random.randint(0, 1) for _ in range(n_try * (n_try - 1) // 2)]
    adj_try = _orientation_to_adj(n_try, orientation_try)
    
    # Local search on n=66
    time_left = deadline - time.time()
    max_ls_time = min(time_left * 0.5, 30.0)
    adj_best, v_best = _local_search(adj_try, max_time=max_ls_time)
    
    if v_best == 0:
        orientation_best = _adj_to_orientation(n_try, adj_best)
        return {'n': n_try, 'orientation': orientation_best}
    
    # Try multiple random restarts on n=66
    time_left = deadline - time.time()
    if time_left > 5.0:
        num_restarts = min(10, int(time_left // 3))
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
            random.seed(42 + restart * 1000)
            orientation_try = [random.randint(0, 1) for _ in range(n_try * (n_try - 1) // 2)]
            adj_try = _orientation_to_adj(n_try, orientation_try)
            
            max_ls_time = min(deadline - time.time() - 5, 10.0)
            if max_ls_time <= 0:
                break
            adj_best, v_best = _local_search(adj_try, max_time=max_ls_time)
            
            if v_best == 0:
                orientation_best = _adj_to_orientation(n_try, adj_best)
                return {'n': n_try, 'orientation': orientation_best}
    
    # If no smaller valid tournament found, return Paley 67
    return {'n': p_known, 'orientation': orientation_67}
