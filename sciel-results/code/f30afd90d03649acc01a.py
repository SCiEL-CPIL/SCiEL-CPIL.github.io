import math
import random
import time

def construct():
    n = 32
    # Generate candidate points
    # A hexagon of radius 4 has 61 points. We need 32.
    # So we need points with hex distance <= 4.
    candidates = []
    for a in range(-5, 6):
        for b in range(-5, 6):
            d = (abs(a) + abs(b) + abs(a + b)) // 2
            if d <= 4:
                candidates.append((a, b))
    
    # Initial selection: Spiral / Distance sorted
    # Sort by distance, then angle
    candidates.sort(key=lambda p: ((abs(p[0]) + abs(p[1]) + abs(p[0] + p[1])) // 2, math.atan2(p[1], p[0])))
    
    selected_set = set(candidates[:n])
    
    # Helper to count edges
    def count_edges(point_set):
        edges = 0
        points = list(point_set)
        for i in range(len(points)):
            a1, b1 = points[i]
            for j in range(i + 1, len(points)):
                a2, b2 = points[j]
                # Distance squared in lattice coordinates
                # d^2 = (a1-a2)^2 + (b1-b2)^2 + (a1-a2)(b1-b2)
                da = a1 - a2
                db = b1 - b2
                dist_sq = da*da + db*db + da*db
                if dist_sq == 1:
                    edges += 1
        return edges

    current_edges = count_edges(selected_set)
    
    # Hill Climbing
    start_time = time.time()
    deadline = start_time + 100 # Leave 50s buffer
    
    while time.time() < deadline:
        improved = False
        # Try swapping a point in selected_set with a point not in selected_set
        # To speed up, only consider neighbors of the current set
        # Or just random swaps from the candidate pool
        # Candidate pool is small (61 points)
        
        # Create a list of points to try swapping OUT
        # and a list of points to try swapping IN
        out_candidates = list(selected_set)
        in_candidates = [p for p in candidates if p not in selected_set]
        
        # Randomize order for diversity
        random.shuffle(out_candidates)
        random.shuffle(in_candidates)
        
        for p_out in out_candidates:
            for p_in in in_candidates:
                # Calculate change in edges
                # Edges lost: edges from p_out to (selected_set - {p_out})
                # Edges gained: edges from p_in to selected_set
                # Note: p_in is not in selected_set, so no self edge.
                
                edges_lost = 0
                for p in selected_set:
                    if p == p_out: continue
                    da = p_out[0] - p[0]
                    db = p_out[1] - p[1]
                    if da*da + db*db + da*db == 1:
                        edges_lost += 1
                        
                edges_gained = 0
                for p in selected_set:
                    da = p_in[0] - p[0]
                    db = p_in[1] - p[1]
                    if da*da + db*db + da*db == 1:
                        edges_gained += 1
                        
                if edges_gained > edges_lost:
                    # Accept swap
                    selected_set.remove(p_out)
                    selected_set.add(p_in)
                    current_edges += (edges_gained - edges_lost)
                    improved = True
                    break # Restart loops with new set
            if improved:
                break
        
        if not improved:
            # Try a random restart or larger jump?
            # Or just break if no improvement in a full pass
            # To avoid getting stuck in local minimum, we can try a random swap even if neutral or slightly negative?
            # But for max edges, greedy is usually good.
            # Let's try one random neutral/negative swap to escape local minima?
            # Or just break.
            break
            
    return [list(p) for p in selected_set]
