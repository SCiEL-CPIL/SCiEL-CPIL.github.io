import time
import random
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # 1. Generate Pool
    # Use a grid to ensure many collinearities are possible.
    # Coords: -50, -40, ..., 50 (11 values)
    coords = list(range(-50, 51, 10))
    pool = []
    point_to_idx = {}
    for x in coords:
        for y in coords:
            idx = len(pool)
            pool.append((x, y))
            point_to_idx[(x, y)] = idx
    
    n_pool = len(pool)
    
    # 2. Precompute Lines
    # A line is defined by a normalized (a, b, c) such that ax + by + c = 0.
    # We only care about lines that have at least 3 points in the pool? 
    # Actually, for the constraint "no 4 collinear", we need to know all lines 
    # that could potentially have 4 points from the selected subset.
    # Since the max line size in a 11x11 grid is 11 (diagonals/rows/cols), 
    # we must track all lines that have >= 3 points in the pool to detect violations 
    # if we select 4 of them.
    # However, for scoring "triples", we only count lines with exactly 3 points in the subset.
    
    line_map = {} # key: (a,b,c) -> set of point indices
    for i in range(n_pool):
        for j in range(i + 1, n_pool):
            x1, y1 = pool[i]
            x2, y2 = pool[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy))
            if g == 0: continue
            dx //= g
            dy //= g
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            a, b = dy, -dx
            c = a * x1 + b * y1
            key = (a, b, c)
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
            
    # Filter lines to only those with >= 3 points in the pool.
    # Lines with < 3 points in the pool can never form a triple or a 4-tuple in the subset.
    relevant_lines = {}
    line_id_counter = 0
    for key, pts in line_map.items():
        if len(pts) >= 3:
            line_id = line_id_counter
            line_id_counter += 1
            relevant_lines[line_id] = pts
            
    n_lines = line_id_counter
    
    # point_to_lines: list of line IDs for each point
    point_to_lines = [[] for _ in range(n_pool)]
    for lid, pts in relevant_lines.items():
        for p in pts:
            point_to_lines[p].append(lid)
            
    # 3. Objective Function
    # State: subset (list of 20 indices)
    # We maintain line_counts: array of size n_lines
    # triple_count: number of lines with count == 3
    # covered_count: number of points that are in at least one line with count >= 2? 
    # Actually, "covered" usually means part of a triple. Let's stick to maximizing triples.
    # But to minimize the max no-3-collinear subset, covering all points in triples is ideal.
    # If a point is not in any triple, it can always be added to a no-3-collinear set 
    # unless it creates a collinearity with two others that are already in the set.
    # Max triples is a good proxy.
    
    def calculate_score(subset, line_counts):
        # line_counts is updated externally or calculated here?
        # For local search, it's better to maintain state.
        # Here we just compute from scratch for initialization.
        counts = [0] * n_lines
        for p in subset:
            for lid in point_to_lines[p]:
                counts[lid] += 1
        
        triples = 0
        invalid = False
        for c in counts:
            if c > 3:
                invalid = True
                break
            if c == 3:
                triples += 1
        return triples, invalid, counts

    # 4. Initialization
    # Greedy: Pick points one by one to maximize triples.
    # This is O(20 * n_pool * (cost to eval)).
    
    def greedy_init():
        subset = []
        line_counts = [0] * n_lines
        best_triples = 0
        
        for _ in range(20):
            best_point = -1
            best_score = -1
            best_new_counts = None
            
            for p in range(n_pool):
                if p in subset:
                    continue
                    
                # Tentatively add p
                new_counts = line_counts[:] # Copy is expensive, but n_lines is small (~100-200?)
                # Let's check size of relevant_lines.
                # 11x11 grid. Lines:
                # Rows: 11. Cols: 11. Diags: ~20. Slopes 1/2, 2/1, etc.
                # Total lines with >=3 pts is probably < 100.
                
                valid = True
                for lid in point_to_lines[p]:
                    new_counts[lid] += 1
                    if new_counts[lid] > 3:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                triples = sum(1 for c in new_counts if c == 3)
                if triples > best_score:
                    best_score = triples
                    best_point = p
                    best_new_counts = new_counts
            
            if best_point != -1:
                subset.append(best_point)
                line_counts = best_new_counts
                best_triples = best_score
            else:
                # If no point can be added without violating constraints or improving score?
                # Just pick any valid point.
                for p in range(n_pool):
                    if p in subset: continue
                    valid = True
                    for lid in point_to_lines[p]:
                        if line_counts[lid] + 1 > 3:
                            valid = False
                            break
                    if valid:
                        subset.append(p)
                        for lid in point_to_lines[p]:
                            line_counts[lid] += 1
                        break
                if len(subset) < 20:
                    # Force fill if stuck (shouldn't happen with this pool)
                    for p in range(n_pool):
                        if p not in subset:
                            subset.append(p)
                            break
                            
        return subset, line_counts

    subset, line_counts = greedy_init()
    
    # 5. Local Search
    # Move: Swap p_in with p_out.
    # Delta calculation:
    # Removing p_in decreases counts of its lines.
    # Adding p_out increases counts of its lines.
    
    def get_score_delta(p_in, p_out, line_counts):
        # Returns (delta_triples, is_valid, new_line_counts)
        # We need to check validity first.
        
        # Check if adding p_out causes any line to exceed 3, considering p_in is removed.
        # Actually, we should simulate the change.
        
        # Lines affected: union of lines(p_in) and lines(p_out)
        lines_p_in = set(point_to_lines[p_in])
        lines_p_out = set(point_to_lines[p_out])
        affected_lines = lines_p_in | lines_p_out
        
        new_counts = line_counts[:]
        valid = True
        old_triples = 0
        new_triples = 0
        
        # Calculate old triples for affected lines
        for lid in affected_lines:
            if line_counts[lid] == 3:
                old_triples += 1
                
        # Apply changes
        for lid in lines_p_in:
            new_counts[lid] -= 1
        for lid in lines_p_out:
            new_counts[lid] += 1
            if new_counts[lid] > 3:
                valid = False
                break
        
        if not valid:
            return 0, False, None
            
        # Calculate new triples for affected lines
        for lid in affected_lines:
            if new_counts[lid] == 3:
                new_triples += 1
                
        delta = new_triples - old_triples
        return delta, True, new_counts

    current_triples, _, _ = calculate_score(subset, line_counts)
    best_subset = subset[:]
    best_line_counts = line_counts[:]
    best_score = current_triples
    
    # Local Search Loop
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        best_swap = None
        best_delta = 0
        best_new_counts = None
        
        p_in_list = subset[:]
        
        for p_in in p_in_list:
            if time.time() > deadline - 5:
                break
            
            # Try swapping p_in with every p_out not in subset
            # Optimization: Only consider p_out that shares lines with p_in or adds new triples?
            # For now, try all. 20 * (121-20) = 2020 swaps.
            # Each swap check is O(num_affected_lines) ~ 20-50.
            # Total ops ~ 100k. Very fast.
            
            for p_out in range(n_pool):
                if p_out in subset:
                    continue
                    
                delta, valid, new_counts = get_score_delta(p_in, p_out, line_counts)
                
                if valid and delta > best_delta:
                    best_delta = delta
                    best_swap = (p_in, p_out)
                    best_new_counts = new_counts
                    improved = True
        
        if improved:
            p_in, p_out = best_swap
            # Apply swap
            subset.remove(p_in)
            subset.append(p_out)
            line_counts = best_new_counts
            current_triples += best_delta
            
            if current_triples > best_score:
                best_score = current_triples
                best_subset = subset[:]
                best_line_counts = line_counts[:]
        else:
            # No improving swap found.
            # Try random restarts or accept neutral moves?
            # To escape local optima, we can try a random swap that is valid, even if delta <= 0.
            # Or restart with a different greedy initialization.
            
            # Let's try a few random valid swaps to escape.
            escaped = False
            for _ in range(50):
                p_in = random.choice(subset)
                p_out = random.choice([p for p in range(n_pool) if p not in subset])
                delta, valid, new_counts = get_score_delta(p_in, p_out, line_counts)
                if valid and delta >= 0: # Accept neutral or improving
                    subset.remove(p_in)
                    subset.append(p_out)
                    line_counts = new_counts
                    current_triples += delta
                    escaped = True
                    break
            
            if not escaped:
                # Try to restart with a slightly perturbed greedy init
                # Or just break if we've been stuck for a while?
                # Given the time budget, let's try a few more iterations of random search.
                
                # Random restart:
                # Create a new subset by taking best_subset and randomizing 5 points.
                new_subset = best_subset[:]
                for _ in range(5):
                    idx = random.randint(0, 19)
                    new_subset[idx] = random.choice([p for p in range(n_pool) if p not in new_subset])
                
                # Check validity of new_subset
                valid, _, new_lc = calculate_score(new_subset, None)
                if not valid:
                    continue
                
                new_triples, _, _ = calculate_score(new_subset, None)
                if new_triples > current_triples:
                    subset = new_subset
                    line_counts = new_lc
                    current_triples = new_triples
                    improved = True # Continue search
                else:
                    # Accept if equal to explore?
                    if new_triples == current_triples:
                         subset = new_subset
                         line_counts = new_lc
                         improved = True

    # Final Result
    final_points = [pool[i] for i in best_subset]
    return [[p[0], p[1]] for p in final_points]
