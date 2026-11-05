"""Minimal maximal no-isosceles set via Branch-and-Bound with greedy initialization."""
import itertools
import numpy as np
import time

N = 16


def _dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _is_isosceles(a, b, c):
    d1 = _dist2(a, b)
    d2 = _dist2(a, c)
    d3 = _dist2(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3


def construct():
    all_points = [(x, y) for x in range(N) for y in range(N)]
    n_points = len(all_points)
    n = n_points
    
    # Precompute iso_pair bitmasks
    iso_pair = [[0] * n for _ in range(n)]
    
    for i in range(n):
        pi = all_points[i]
        for j in range(i + 1, n):
            pj = all_points[j]
            mask = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                pk = all_points[k]
                if _is_isosceles(pi, pj, pk):
                    mask |= (1 << k)
            iso_pair[i][j] = mask
            iso_pair[j][i] = mask
            
    full_mask = (1 << n) - 1
    
    def greedy_solve():
        """Standard greedy to get a good initial upper bound."""
        chosen = []
        chosen_mask = 0
        blocked_mask = 0
        
        while True:
            all_non_chosen_mask = full_mask & ~chosen_mask
            if (all_non_chosen_mask & ~blocked_mask) == 0:
                break
            
            candidate_mask = all_non_chosen_mask & ~blocked_mask
            if candidate_mask == 0:
                break
            
            best_c = -1
            best_score = -1
            
            # Iterate through candidates
            c = 0
            while c < n:
                if (candidate_mask >> c) & 1:
                    new_blocked = 0
                    for a in chosen:
                        new_blocked |= iso_pair[a][c]
                    new_points = new_blocked & ~blocked_mask & ~chosen_mask
                    score = bin(new_points).count('1')
                    if score > best_score:
                        best_score = score
                        best_c = c
                c += 1
            
            if best_c < 0:
                break
            
            chosen.append(best_c)
            chosen_mask |= (1 << best_c)
            
            for a in chosen[:-1]:
                blocked_mask |= iso_pair[a][best_c]
            
            blocked_mask &= ~chosen_mask
            
        return [all_points[i] for i in chosen], len(chosen)

    # Get initial solution
    initial_solution, initial_size = greedy_solve()
    best_size = initial_size
    best_set = initial_solution
    
    # Branch and Bound / Limited Discrepancy Search
    # We try to find a solution smaller than best_size.
    # We limit the branching factor to keep it fast.
    
    def get_candidates(chosen, chosen_mask, blocked_mask):
        """Get candidates sorted by blocking power."""
        all_non_chosen_mask = full_mask & ~chosen_mask
        candidate_mask = all_non_chosen_mask & ~blocked_mask
        
        if candidate_mask == 0:
            return []
        
        candidates = []
        c = 0
        while c < n:
            if (candidate_mask >> c) & 1:
                new_blocked = 0
                for a in chosen:
                    new_blocked |= iso_pair[a][c]
                new_points = new_blocked & ~blocked_mask & ~chosen_mask
                score = bin(new_points).count('1')
                candidates.append((score, c))
            c += 1
        
        # Sort by score descending
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [c for score, c in candidates]

    def dfs(chosen, chosen_mask, blocked_mask, depth, limit):
        nonlocal best_size, best_set
        
        # Pruning: if current size is already >= best_size, no improvement possible
        if len(chosen) >= best_size:
            return
        
        # Check if maximal
        all_non_chosen_mask = full_mask & ~chosen_mask
        if (all_non_chosen_mask & ~blocked_mask) == 0:
            # Found a maximal set
            if len(chosen) < best_size:
                best_size = len(chosen)
                best_set = [all_points[i] for i in chosen]
            return
        
        # Get candidates
        candidates = get_candidates(chosen, chosen_mask, blocked_mask)
        
        if not candidates:
            return
        
        # Explore top 'limit' candidates
        # Limit is small to prevent exponential blowup
        for c in candidates[:limit]:
            new_chosen = chosen + [c]
            new_chosen_mask = chosen_mask | (1 << c)
            
            new_blocked = 0
            for a in chosen:
                new_blocked |= iso_pair[a][c]
            
            new_blocked_mask = blocked_mask | new_blocked
            new_blocked_mask &= ~new_chosen_mask
            
            dfs(new_chosen, new_chosen_mask, new_blocked_mask, depth + 1, limit)

    # Run B&B with a time limit
    start_time = time.time()
    time_limit = 280  # Leave some margin for overhead
    
    # Start with empty set
    # We use a limited discrepancy search approach:
    # First, follow the greedy path (limit=1) to verify/confirm.
    # Then, allow branching (limit=2, 3) to explore alternatives.
    
    # To make it more effective, we can restart with different limits or randomizations.
    # But simple B&B with limit=3 should be decent.
    
    # Optimization: We can also try starting with different "seeds" if time permits,
    # but for now, we rely on the B&B from the empty set with good ordering.
    
    # We'll run multiple passes with increasing limit if time allows, or just one pass with limit=4.
    # Let's try limit=4 for the first pass.
    
    dfs([], 0, 0, 0, 4)
    
    # If time remains, we can try to improve further with a different heuristic or restart.
    # For now, return the best found.
    
    if best_set:
        return best_set
    else:
        return initial_solution

