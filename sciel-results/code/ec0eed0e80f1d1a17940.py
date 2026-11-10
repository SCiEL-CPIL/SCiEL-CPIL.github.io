import time
import random
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Use more of the 150s budget
    
    N = 7
    points = [(x, y) for x in range(N) for y in range(N)]
    n_points = len(points)
    
    # Precompute all maximal lines (sets of collinear points with >= 2 points)
    all_lines = set()
    for i in range(n_points):
        for j in range(i + 1, n_points):
            p1 = points[i]
            p2 = points[j]
            x1, y1 = p1
            x2, y2 = p2
            dx = x2 - x1
            dy = y2 - y1
            
            pts_on_line = set([i, j])
            for k in range(n_points):
                if k == i or k == j:
                    continue
                x3, y3 = points[k]
                if dx * (y3 - y1) == dy * (x3 - x1):
                    pts_on_line.add(k)
            
            all_lines.add(frozenset(pts_on_line))
    
    # Separate lines by size
    forbidden_lines = [l for l in all_lines if len(l) >= 5]  # Cannot select all
    target_lines = [l for l in all_lines if len(l) == 4]     # Want to maximize
    longer_lines = [l for l in all_lines if len(l) > 4]      # Count if exactly 4 selected
    
    def is_valid(selected_indices):
        """Check if no 5 points are collinear."""
        sel_frozen = frozenset(selected_indices)
        for fl in forbidden_lines:
            if fl.issubset(sel_frozen):
                return False
        return True
    
    def get_score(selected_indices):
        """Count lines with exactly 4 of the selected points."""
        sel_frozen = frozenset(selected_indices)
        score = 0
        for tl in target_lines:
            if tl.issubset(sel_frozen):
                score += 1
        for ll in longer_lines:
            if len(ll & sel_frozen) == 4:
                score += 1
        return score

    best_score = -1
    best_indices = None

    def hill_climb(initial_set):
        nonlocal best_score, best_indices
        
        current = set(initial_set)
        current_score = get_score(current)
        
        if not is_valid(current):
            return -1, set()
        
        if current_score > best_score:
            best_score = current_score
            best_indices = set(current)
            
        # Hill climbing with 2-for-2 swaps
        # We try to swap 2 points out with 2 points in
        # To keep it fast, we sample random pairs
        max_iterations = 5000 
        no_improve_count = 0
        stagnation_limit = 200
        
        for it in range(max_iterations):
            if time.time() > deadline:
                break
                
            if no_improve_count > stagnation_limit:
                break
                
            # Generate a random 2-for-2 swap
            # Pick 2 points to remove
            to_remove = random.sample(list(current), 2)
            candidates_in = [p for p in range(n_points) if p not in current]
            if len(candidates_in) < 2:
                break
            to_add = random.sample(candidates_in, 2)
            
            new_set = set(current)
            new_set.difference_update(to_remove)
            new_set.update(to_add)
            
            if not is_valid(new_set):
                no_improve_count += 1
                continue
                
            new_score = get_score(new_set)
            
            if new_score > current_score:
                current = new_set
                current_score = new_score
                no_improve_count = 0
                if current_score > best_score:
                    best_score = current_score
                    best_indices = set(current)
            else:
                no_improve_count += 1
                
        return current_score, current

    # Multiple restarts
    n_restarts = 50
    for r in range(n_restarts):
        if time.time() > deadline:
            break
            
        # Diverse initial configurations
        # Start with a random valid set
        initial = None
        for _ in range(100):
            cand = set(random.sample(range(n_points), 16))
            if is_valid(cand):
                initial = cand
                break
                
        if initial is None:
            # Fallback to a simple grid pattern if random fails
            initial = set(range(16))
            if not is_valid(initial):
                continue
                
        _, current_best = hill_climb(initial)
        
        # If hill climb got stuck, try a single-point swap polish
        if current_best:
            # Quick polish
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for p_out in list(current_best):
                    if time.time() > deadline:
                        break
                    for p_in in range(n_points):
                        if p_in in current_best:
                            continue
                        new_set = set(current_best)
                        new_set.remove(p_out)
                        new_set.add(p_in)
                        if not is_valid(new_set):
                            continue
                        new_score = get_score(new_set)
                        cur_score = get_score(current_best)
                        if new_score > cur_score:
                            current_best = new_set
                            improved = True
                            if new_score > best_score:
                                best_score = new_score
                                best_indices = new_set
                            break

    # Fallback if nothing found
    if best_indices is None:
        best_indices = set(range(16))
        if not is_valid(best_indices):
            for _ in range(1000):
                candidate = set(random.sample(range(n_points), 16))
                if is_valid(candidate):
                    best_indices = candidate
                    break
            if best_indices is None:
                best_indices = set(range(16))

    return [[points[i][0], points[i][1]] for i in sorted(best_indices)]
