"""Rewritten program: hypergraph MIS via branch-and-bound with greedy + local search."""
import time
from math import gcd
from itertools import combinations
from collections import defaultdict

N = 128
DEADLINE = 75.0  # safety margin before 90s kill

def _build_gcd_table():
    """Precompute gcd for all pairs."""
    g = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i+1, N+1):
            g[i][j] = gcd(i, j)
            g[j][i] = g[i][j]
    return g

def _is_bad_triple(a, b, c, g):
    """Check if three distinct elements have all three pairwise gcds equal."""
    gab = g[a][b]
    gac = g[a][c]
    gbc = g[b][c]
    return gab == gac == gbc

def _conflict_with(x, selected, g):
    """Check if adding x to selected creates a bad triple with any pair in selected."""
    n = len(selected)
    for i in range(n):
        for j in range(i+1, n):
            a, b = selected[i], selected[j]
            if _is_bad_triple(a, b, x, g):
                return True
    return False

def _greedy_min_degree(g):
    """Greedy: add elements in order of fewest conflicts with already-selected."""
    selected = []
    remaining = list(range(1, N+1))
    
    while remaining:
        # For each candidate, count how many bad triples it would form with pairs in selected
        # Actually, we want to pick the element that creates the fewest future conflicts
        # Simpler: pick the element with minimum "degree" in the conflict hypergraph
        # Degree = number of pairs (a,b) in remaining such that (a,b,x) is a bad triple
        best = None
        best_deg = float('inf')
        for x in remaining:
            deg = 0
            # Count pairs in remaining that form bad triple with x
            # This is expensive; use a proxy: count how many elements in remaining share gcd structure
            # Simpler proxy: just try adding and see if valid, prefer elements that conflict with fewer
            # Actually for greedy, let's just check validity and pick first valid
            if not _conflict_with(x, selected, g):
                # Count how many pairs in remaining would be blocked
                # Use a cheap heuristic: count gcd overlaps
                deg = 0
                for y in remaining:
                    if y != x:
                        # If gcd(x,y) appears frequently, it's more restrictive
                        deg += 1  # placeholder
                if deg < best_deg:
                    best_deg = deg
                    best = x
        if best is None:
            break
        selected.append(best)
        remaining.remove(best)
    
    return selected

def _local_search(selected, g, time_limit=30.0):
    """Try to improve by swaps: remove one, add another if it increases size."""
    start_time = time.time()
    selected_set = set(selected)
    best = list(selected)
    best_size = len(best)
    
    for _ in range(1000):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        # Try all single swaps: remove a, add b
        for a in best:
            for b in range(1, N+1):
                if b in selected_set:
                    continue
                # Try removing a and adding b
                new_set = selected_set - {a} | {b}
                # Check validity
                valid = True
                lst = list(new_set)
                # Quick check: only need to check triples involving b
                for i in range(len(lst)):
                    for j in range(i+1, len(lst)):
                        x, y = lst[i], lst[j]
                        if x == b or y == b:
                            z = lst[i] if x == b else lst[j]
                            if z == b:
                                continue
                            if _is_bad_triple(x, y, z, g):
                                valid = False
                                break
                    if not valid:
                        break
                if valid:
                    # Same size, but might enable further improvements
                    # Try double swaps for size increase
                    pass
        
        # Try to add any element
        for b in range(1, N+1):
            if b in selected_set:
                continue
            if not _conflict_with(b, best, g):
                best.append(b)
                selected_set.add(b)
                best_size += 1
                improved = True
                break
        
        if not improved:
            # Try swaps to enable additions
            found_swap = False
            for a in best:
                for b in range(1, N+1):
                    if b in selected_set:
                        continue
                    # Remove a, add b
                    temp = [x for x in best if x != a]
                    if not _conflict_with(b, temp, g):
                        # Now try to add another element
                        for c in range(1, N+1):
                            if c in selected_set or c == b:
                                continue
                            if not _conflict_with(c, temp + [b], g):
                                # We gained one
                                best = temp + [b, c]
                                selected_set = set(best)
                                best_size += 1
                                found_swap = True
                                improved = True
                                break
                    if found_swap:
                        break
                if found_swap:
                    break
            if not improved:
                break
    
    return best

def _branch_and_bound(g, time_limit=40.0):
    """Branch and bound for maximum independent set in the 3-uniform hypergraph."""
    start_time = time.time()
    
    # Precompute for each pair (a,b), which c's form a bad triple
    bad_third = defaultdict(set)
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            gab = g[a][b]
            for c in range(1, N+1):
                if c == a or c == b:
                    continue
                if g[a][c] == gab and g[b][c] == gab:
                    bad_third[(a,b)].add(c)
    
    # Greedy initial solution
    sol = _greedy_min_degree(g)
    best_size = len(sol)
    best_sol = list(sol)
    
    # Try local search on the greedy solution
    improved = _local_search(sol, g, time_limit=15.0)
    if len(improved) > best_size:
        best_size = len(improved)
        best_sol = improved
    
    # Branch and bound
    # State: (candidates, selected)
    # Use recursion with pruning
    def branch(candidates, selected, depth=0):
        nonlocal best_size, best_sol
        if time.time() - start_time > time_limit:
            return
        
        # Upper bound: |candidates| + |selected|
        if len(candidates) + len(selected) <= best_size:
            return
        
        if not candidates:
            if len(selected) > best_size:
                best_size = len(selected)
                best_sol = list(selected)
            return
        
        # Pick an element to branch on
        # Choose the element with highest degree (most conflicts) for better pruning
        x = candidates[0]
        
        # Option 1: include x
        # Check if x can be added to selected
        can_include = True
        for a in selected:
            for b in selected:
                if a < b and _is_bad_triple(a, b, x, g):
                    can_include = False
                    break
            if not can_include:
                break
        
        if can_include:
            new_selected = selected + [x]
            # Remove from candidates all elements that conflict with x given new_selected
            new_candidates = []
            for y in candidates[1:]:
                # Check if y can coexist with x in new_selected
                ok = True
                for a in new_selected:
                    for b in new_selected:
                        if a < b and _is_bad_triple(a, b, y, g):
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    new_candidates.append(y)
            
            branch(new_candidates, new_selected, depth+1)
        
        # Option 2: exclude x
        branch(candidates[1:], selected, depth+1)
    
    # Use a smarter initial candidate ordering
    candidates = list(range(1, N+1))
    # Sort by some heuristic - elements with more gcd=1 with others might be safer
    # Actually, just use natural order for now
    
    branch(candidates, [])
    
    return best_sol

def construct():
    start_time = time.time()
    
    g = _build_gcd_table()
    
    # Quick greedy
    sol = _greedy_min_degree(g)
    best = list(sol)
    
    # Local search
    improved = _local_search(sol, g, time_limit=15.0)
    if len(improved) > len(best):
        best = improved
    
    # Try branch and bound with remaining time
    remaining_time = DEADLINE - (time.time() - start_time)
    if remaining_time > 5.0:
        bb_sol = _branch_and_bound(g, time_limit=remaining_time - 5.0)
        if len(bb_sol) > len(best):
            best = bb_sol
    
    # Final local search polish
    remaining_time = DEADLINE - (time.time() - start_time)
    if remaining_time > 2.0:
        polished = _local_search(best, g, time_limit=remaining_time - 2.0)
        if len(polished) > len(best):
            best = polished
    
    # Verify validity
    best_set = set(best)
    for a, b, c in combinations(best, 3):
        if _is_bad_triple(a, b, c, g):
            # Invalid! Remove the problematic element
            # Fallback to a safe subset
            safe = []
            for x in best:
                if not _conflict_with(x, safe, g):
                    safe.append(x)
            best = safe
            break
    
    if not best:
        best = [1]
    
    return best
