import math
import time
import random

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 150.0
    
    # Lattice point representation: (a, b) for a + b*omega
    # Distance squared between (a1, b1) and (a2, b2) is (a1-a2)^2 + (a1-a2)(b1-b2) + (b1-b2)^2
    def dist_sq(p1, p2):
        da = p1[0] - p2[0]
        db = p1[1] - p2[1]
        return da*da + da*db + db*db
    
    # Generate candidate points in a spiral around the origin
    # We need enough candidates to fill 32 points. A hexagon of radius 5 has 91 points.
    # Let's generate points with |a|,|b| <= 10 to be safe.
    candidates = []
    for a in range(-10, 11):
        for b in range(-10, 11):
            candidates.append((a, b))
    
    # Sort candidates by distance from origin to ensure compactness
    candidates.sort(key=lambda p: p[0]*p[0] + p[0]*p[1] + p[1]*p[1])
    
    # Greedy selection
    selected = []
    for p in candidates:
        if len(selected) == 32:
            break
        # Check if p is at distance >= 1 from all selected points
        # Distance >= 1 means dist_sq >= 1
        valid = True
        for sp in selected:
            if dist_sq(p, sp) < 1:
                valid = False
                break
        if valid:
            selected.append(p)
    
    if len(selected) < 32:
        # Fallback: should not happen with radius 10
        # Fill with arbitrary non-conflicting points if needed, but unlikely
        pass

    def count_edges(points):
        edges = 0
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                if dist_sq(points[i], points[j]) == 1:
                    edges += 1
        return edges

    current_edges = count_edges(selected)
    
    # If we already have 76, we are done
    if current_edges == 76:
        return [[a, b] for a, b in selected]

    # Local Search: Try to improve edge count
    # We can try swapping a selected point with an unselected candidate
    # Or moving a selected point to an empty spot
    # Given the small size, we can do a simple hill climbing
    
    # Create a set of selected points for O(1) lookup
    selected_set = set(selected)
    
    # Generate a list of potential moves: (remove_point, add_point)
    # We will iterate through candidates not in selected_set
    # and try to swap them with each selected point
    
    best_edges = current_edges
    best_config = [tuple(p) for p in selected]
    
    # Limit iterations to respect deadline
    max_iterations = 10000
    iterations = 0
    
    while iterations < max_iterations:
        if time.time() > deadline - 5.0:
            break
            
        iterations += 1
        improved = False
        
        # Try swapping each selected point with a random candidate
        # To be efficient, we can pre-filter candidates that are "close" to the cluster
        # But for 32 points, checking all 441 candidates is fast enough
        
        # Pick a random selected point to remove
        idx_to_remove = random.randint(0, 31)
        p_remove = best_config[idx_to_remove]
        
        # Try adding a random candidate
        # Bias towards candidates close to the center of mass or existing points
        # Simple random from candidates not in best_config
        if random.random() < 0.8:
            # Try a candidate from the original sorted list (compact)
            # Filter out those already in best_config
            available = [p for p in candidates if p not in best_config]
            if not available:
                continue
            # Pick a few random ones to test
            for _ in range(10):
                p_add = random.choice(available)
                # Check if p_add is valid (dist >= 1 from all others)
                valid = True
                temp_config = list(best_config)
                temp_config[idx_to_remove] = p_add
                
                # Check conflicts for p_add
                for i, sp in enumerate(temp_config):
                    if i == idx_to_remove:
                        continue
                    if dist_sq(p_add, sp) < 1:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Calculate new edge count
                # Optimization: Only recalculate edges involving p_add and p_remove
                # Edges lost: edges from p_remove to others
                # Edges gained: edges from p_add to others
                edges_lost = 0
                edges_gained = 0
                for i, sp in enumerate(best_config):
                    if i == idx_to_remove:
                        continue
                    if dist_sq(p_remove, sp) == 1:
                        edges_lost += 1
                    if dist_sq(p_add, sp) == 1:
                        edges_gained += 1
                
                new_edges = best_edges - edges_lost + edges_gained
                
                if new_edges > best_edges:
                    best_edges = new_edges
                    best_config[idx_to_remove] = p_add
                    improved = True
                    if best_edges == 76:
                        return [[a, b] for a, b in best_config]
                    break # Restart search with new best
        
        # If no improvement after many tries, might be stuck, but with 76 being the max, 
        # and greedy start being good, we likely reach it quickly.
        if not improved and iterations % 1000 == 0:
             # Perturb if stuck? For now, random swaps are enough.
             pass

    return [[a, b] for a, b in best_config]
