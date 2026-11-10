import time
import random
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 40s safety margin before 150s kill
    
    N = 5
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
        for fl in forbidden_lines:
            if fl.issubset(frozenset(selected_indices)):
                return False
        return True
    
    def get_score(selected_indices):
        """Count lines with exactly 4 of the selected points."""
        score = 0
        sel = frozenset(selected_indices)
        for tl in target_lines:
            if tl.issubset(sel):
                score += 1
        for ll in longer_lines:
            if len(ll & sel) == 4:
                score += 1
        return score
    
    def get_score_fast(selected_indices):
        """Faster scoring using precomputed counts."""
        sel = frozenset(selected_indices)
        score = 0
        for tl in target_lines:
            if tl.issubset(sel):
                score += 1
        for ll in longer_lines:
            if len(ll & sel) == 4:
                score += 1
        return score
    
    # Simulated annealing with diverse restarts
    best_score = -1
    best_indices = None
    
    def simulated_annealing(initial_set, iterations, start_temp, end_temp):
        nonlocal best_score, best_indices
        
        current = set(initial_set)
        current_score = get_score(current)
        
        if not is_valid(current):
            return
        
        if current_score > best_score:
            best_score = current_score
            best_indices = set(current)
        
        for it in range(iterations):
            if time.time() > deadline:
                return
            
            # Pick a random point to swap
            p_in = random.choice(list(current))
            candidates_out = [p for p in range(n_points) if p not in current]
            if not candidates_out:
                continue
            p_out = random.choice(candidates_out)
            
            new_set = set(current)
            new_set.remove(p_in)
            new_set.add(p_out)
            
            if not is_valid(new_set):
                continue
            
            new_score = get_score(new_set)
            
            # Temperature for simulated annealing
            t = start_temp * (end_temp / start_temp) ** (it / max(iterations - 1, 1))
            
            if new_score >= current_score or random.random() < (2.718281828 ** ((new_score - current_score) / max(t, 1e-10))):
                current = new_set
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_indices = set(current)
        
        return
    
    # Multiple restarts with diverse initial configurations
    n_restarts = 20
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Diverse initial configurations
        if r == 0:
            # Start with a structured configuration
            initial = set(range(16))
        elif r == 1:
            # Start with rows 0-3
            initial = set(range(0, 16))
        elif r == 2:
            # Start with a checkerboard pattern
            initial = set()
            for x in range(N):
                for y in range(N):
                    if (x + y) % 2 == 0:
                        initial.add(x * N + y)
            if len(initial) < 16:
                remaining = [p for p in range(n_points) if p not in initial]
                initial.update(random.sample(remaining, 16 - len(initial)))
            initial = set(list(initial)[:16])
        else:
            # Random initial configuration
            initial = set(random.sample(range(n_points), 16))
            # Try to make it valid
            attempts = 0
            while not is_valid(initial) and attempts < 50:
                initial = set(random.sample(range(n_points), 16))
                attempts += 1
            if not is_valid(initial):
                continue
        
        # Simulated annealing
        iterations = max(500, int((deadline - time.time()) * 20))
        simulated_annealing(initial, iterations, 10.0, 0.1)
    
    # Final exhaustive local polish: try all single-point swaps
    if best_indices is not None and time.time() < deadline - 5:
        current = set(best_indices)
        current_score = get_score(current)
        
        # Multiple passes of exhaustive local search
        for _ in range(3):
            if time.time() > deadline - 2:
                break
            
            improved = False
            for p_in in list(current):
                if time.time() > deadline - 2:
                    break
                for p_out in range(n_points):
                    if p_out in current:
                        continue
                    
                    new_set = set(current)
                    new_set.remove(p_in)
                    new_set.add(p_out)
                    
                    if not is_valid(new_set):
                        continue
                    
                    new_score = get_score(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_indices = set(current)
            
            if not improved:
                break
    
    # Fallback if nothing found
    if best_indices is None:
        # Use a known valid configuration
        best_indices = set(range(16))
        if not is_valid(best_indices):
            # Find any valid configuration
            for _ in range(1000):
                candidate = set(random.sample(range(n_points), 16))
                if is_valid(candidate):
                    best_indices = candidate
                    break
            if best_indices is None:
                best_indices = set(range(16))
    
    return [[points[i][0], points[i][1]] for i in sorted(best_indices)]
