import time
import random
import math

def construct():
    t0 = time.time()
    deadline = 130.0  # Safer margin (evaluator kills at 150s)
    
    # Neighbors in triangular lattice (axial coordinates)
    neighbors = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    
    def count_edges(pts):
        pset = set(pts)
        edges = 0
        for a, b in pts:
            for da, db in neighbors:
                if (a + da, b + db) in pset:
                    edges += 1
        return edges // 2

    def edge_delta_remove_add(pts_set, remove_pt, add_pt):
        lost = 0
        for da, db in neighbors:
            if (remove_pt[0] + da, remove_pt[1] + db) in pts_set:
                lost += 1
        gained = 0
        for da, db in neighbors:
            nb = (add_pt[0] + da, add_pt[1] + db)
            if nb in pts_set and nb != remove_pt:
                gained += 1
        return gained - lost

    def hex_dist(a, b):
        return (abs(a) + abs(b) + abs(a + b)) // 2
    
    # Candidate pool: all points within hex_dist <= 8
    all_candidates = []
    for a in range(-8, 9):
        for b in range(-8, 9):
            if hex_dist(a, b) <= 8:
                all_candidates.append((a, b))
    
    best_points = None
    best_edges = 0
    
    # Method 1: Deterministic hexagonal spiral construction
    # Generate points in order of increasing hex_dist, breaking ties by angle
    def spiral_order(n):
        pts = [(0, 0)]
        dist = 1
        while len(pts) < n:
            # For each ring at distance `dist`, generate 6*dist points
            # Ring points: (dist, 0), (dist-1, 1), ..., (0, dist), (-1, dist), ..., (-dist, 0), (-dist+1, -1), ..., (0, -dist), (1, -dist), ..., (dist, 0)
            for side in range(6):
                for k in range(dist):
                    if side == 0:
                        pt = (dist - k, k)
                    elif side == 1:
                        pt = (dist - k, k - dist)
                    elif side == 2:
                        pt = (k - dist, dist - k)
                    elif side == 3:
                        pt = (k - dist, k)
                    elif side == 4:
                        pt = (dist - k, k)
                    else:
                        pt = (k, dist - k)
                    # Fix: let me use a cleaner spiral generation
                    pass
            dist += 1
        return pts
    
    # Better spiral: use hexagonal spiral with proper ordering
    def generate_hex_spiral(n):
        """Generate n points in hexagonal spiral order starting from origin."""
        if n <= 0:
            return []
        pts = [(0, 0)]
        if n == 1:
            return pts
        
        # Hex spiral: ring r has 6*r points
        # Directions in axial coords: E=(1,0), NE=(0,1), NW=(-1,1), W=(-1,0), SW=(0,-1), SE=(1,-1)
        dirs = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
        
        ring = 1
        while len(pts) < n:
            # Start of ring: go `ring` steps in direction 0
            cur = (0, 0)
            for step in range(ring):
                cur = (cur[0] + dirs[0][0], cur[1] + dirs[0][1])
            # Now traverse 6 sides
            for side in range(6):
                for step in range(ring):
                    if len(pts) >= n:
                        return pts
                    pts.append(cur)
                    cur = (cur[0] + dirs[side][0], cur[1] + dirs[side][1])
            ring += 1
        return pts
    
    spiral_pts = generate_hex_spiral(32)
    spiral_edges = count_edges(spiral_pts)
    if spiral_edges > best_edges:
        best_edges = spiral_edges
        best_points = list(spiral_pts)
    
    if best_edges >= 76:
        return [list(p) for p in best_points]
    
    # Method 2: Greedy with multiple tie-breaking strategies
    for tie_break in ['dist', 'lex', 'angle']:
        current = [(0, 0)]
        current_set = {(0, 0)}
        
        for _ in range(31):
            best_pt = None
            best_score = -1
            best_dist = 999
            best_lex = (999, 999)
            best_angle = 999
            for pt in all_candidates:
                if pt in current_set:
                    continue
                # Count edges this point would add
                score = 0
                for da, db in neighbors:
                    if (pt[0] + da, pt[1] + db) in current_set:
                        score += 1
                d = hex_dist(pt[0], pt[1])
                lex = (abs(pt[0]), abs(pt[1]))
                # Angle from origin for tie-breaking
                angle = math.atan2(pt[1] * 0.866 + pt[0] * 0.5, pt[0] - pt[1] * 0.5)
                
                if tie_break == 'dist':
                    if score > best_score or (score == best_score and d < best_dist):
                        best_score = score
                        best_dist = d
                        best_pt = pt
                elif tie_break == 'lex':
                    if score > best_score or (score == best_score and lex < best_lex):
                        best_score = score
                        best_lex = lex
                        best_pt = pt
                else:  # angle
                    if score > best_score or (score == best_score and abs(angle) < best_angle):
                        best_score = score
                        best_angle = abs(angle)
                        best_pt = pt
            if best_pt:
                current.append(best_pt)
                current_set.add(best_pt)
        
        edges = count_edges(current)
        if edges > best_edges:
            best_edges = edges
            best_points = list(current)
        
        if best_edges >= 76:
            return [list(p) for p in best_points]
    
    # Method 3: Aggressive local search with multiple restarts
    random.seed(42)
    
    for restart in range(50):
        # Initialize from best known or random perturbation
        if restart == 0:
            current = list(best_points)
        else:
            # Start from best and make some random swaps
            current = list(best_points)
            for _ in range(random.randint(1, 5)):
                i = random.randint(0, 31)
                j = random.randint(0, 31)
                current[i], current[j] = current[j], current[i]
        
        current_set = set(current)
        cur_edges = count_edges(current)
        
        if cur_edges > best_edges:
            best_edges = cur_edges
            best_points = list(current)
            if best_edges >= 76:
                return [list(p) for p in best_points]
        
        # Simulated annealing from this starting point
        T = 1.5
        T_min = 0.01
        cooling = 0.9995
        no_improve = 0
        max_no_improve = 5000
        
        while (time.time() - t0) < deadline and no_improve < max_no_improve:
            if T < T_min:
                T = T_min
            
            i = random.randint(0, 31)
            old = current[i]
            
            # Try to find a better replacement
            found = False
            for _ in range(10):  # Try up to 10 candidates
                j = random.randint(0, len(all_candidates) - 1)
                new = all_candidates[j]
                
                if new in current_set:
                    continue
                
                temp_set = current_set - {old}
                delta = edge_delta_remove_add(temp_set, old, new)
                
                if delta > 0:
                    current[i] = new
                    current_set.discard(old)
                    current_set.add(new)
                    cur_edges += delta
                    no_improve = 0
                    found = True
                    
                    if cur_edges > best_edges:
                        best_edges = cur_edges
                        best_points = list(current)
                        if best_edges >= 76:
                            return [list(p) for p in best_points]
                    break
                else:
                    # Accept with probability exp(delta / T)
                    if delta >= -4 and random.random() < (math.exp(delta / T)):
                        current[i] = new
                        current_set.discard(old)
                        current_set.add(new)
                        cur_edges += delta
                        no_improve += 1
                        found = True
                        break
            
            if not found:
                no_improve += 1
            
            T *= cooling
        
        if (time.time() - t0) > deadline - 5:
            break
    
    return [list(p) for p in best_points]
