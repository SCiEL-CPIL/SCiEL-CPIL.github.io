import math
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 115.0

    def dist_sq(p, q):
        da = p[0] - q[0]
        db = p[1] - q[1]
        return da*da + da*db + db*db

    def count_edges(points):
        n = len(points)
        edges = 0
        for i in range(n):
            for j in range(i+1, n):
                if dist_sq(points[i], points[j]) == 1:
                    edges += 1
        return edges

    def gen_hex(a, b, c):
        points = set()
        for y in range(-c, a+b):
            x_min = max(-a, y-c, -b-y)
            x_max = min(b, a-y, c-y)
            for x in range(x_min, x_max+1):
                points.add((x,y))
        return list(points)

    def greedy_construct(seed_points, n_points=32):
        """Build a set of n_points by greedily adding points that maximize edges."""
        if not seed_points:
            return []
        points = [list(p) for p in seed_points[:min(n_points, len(seed_points))]]
        if len(points) >= n_points:
            return points[:n_points]
        
        point_set = set(tuple(p) for p in points)
        
        # Generate candidate positions near existing points
        candidates = set()
        for p in points:
            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1),
                          (2,0), (-2,0), (0,2), (0,-2), (2,-2), (-2,2),
                          (1,1), (-1,-1), (2,-1), (-2,1), (1,-2), (-1,2)]:
                np = (p[0]+dx, p[1]+dy)
                if abs(np[0]) <= 64 and abs(np[1]) <= 64:
                    candidates.add(np)
        
        # Add some spiral positions as fallback
        for r in range(1, 10):
            for t in range(6):
                angle = t * math.pi / 3
                x = round(r * math.cos(angle))
                y = round(r * math.sin(angle) / math.sin(math.pi/3))
                # Convert to axial coordinates
                # Actually let's just add a grid
                pass
        
        # Add grid positions
        for x in range(-8, 9):
            for y in range(-8, 9):
                if abs(x) <= 64 and abs(y) <= 64:
                    candidates.add((x, y))
        
        candidates = list(candidates)
        random.shuffle(candidates)
        
        while len(points) < n_points:
            best_score = -1
            best_point = None
            for cand in candidates:
                if cand in point_set:
                    continue
                # Check validity: no point at distance 0
                valid = True
                for p in points:
                    if dist_sq(cand, p) == 0:
                        valid = False
                        break
                    if dist_sq(cand, p) < 1:
                        valid = False
                        break
                if not valid:
                    continue
                # Count edges to existing points
                score = 0
                for p in points:
                    if dist_sq(cand, p) == 1:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_point = cand
            
            if best_point is None:
                # No valid candidate found, try wider search
                expanded = set()
                for p in points:
                    for dx in range(-3, 4):
                        for dy in range(-3, 4):
                            np = (p[0]+dx, p[1]+dy)
                            if abs(np[0]) <= 64 and abs(np[1]) <= 64:
                                expanded.add(np)
                candidates = [c for c in expanded if c not in point_set]
                random.shuffle(candidates)
                continue
            
            points.append(list(best_point))
            point_set.add(tuple(best_point))
            candidates.remove(best_point)
        
        return points

    def local_search(points, max_iter=5000):
        """Improve the point set via local search."""
        points = [list(p) for p in points]
        n = len(points)
        current_edges = count_edges(points)
        best_points = [list(p) for p in points]
        best_edges = current_edges
        
        for it in range(max_iter):
            if time.time() - start_time > DEADLINE - 5:
                break
            improved = False
            for i in range(n):
                if time.time() - start_time > DEADLINE - 5:
                    break
                ox, oy = points[i][0], points[i][1]
                # Try moving to all 6 neighbors and some further positions
                for dx, dy in [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1),
                               (2,0), (-2,0), (0,2), (0,-2), (2,-2), (-2,2),
                               (1,1), (-1,-1), (2,-1), (-2,1), (1,-2), (-1,2)]:
                    nx, ny = ox + dx, oy + dy
                    if abs(nx) > 64 or abs(ny) > 64:
                        continue
                    # Check if occupied
                    occupied = False
                    for k in range(n):
                        if k != i and points[k][0] == nx and points[k][1] == ny:
                            occupied = True
                            break
                    if occupied:
                        continue
                    # Check distance constraint
                    valid = True
                    for k in range(n):
                        if k == i: continue
                        if dist_sq([nx, ny], points[k]) < 1:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Try the move
                    points[i][0] = nx
                    points[i][1] = ny
                    new_edges = count_edges(points)
                    if new_edges > current_edges:
                        current_edges = new_edges
                        improved = True
                        if current_edges > best_edges:
                            best_edges = current_edges
                            best_points = [list(p) for p in points]
                    else:
                        points[i][0] = ox
                        points[i][1] = oy
                
                if best_edges == 76:
                    break
            if best_edges == 76:
                break
            if not improved:
                # Random perturbation
                if random.random() < 0.1:
                    idx = random.randint(0, n-1)
                    ox, oy = points[idx][0], points[idx][1]
                    for _ in range(20):
                        dx = random.randint(-3, 3)
                        dy = random.randint(-3, 3)
                        nx, ny = ox + dx, oy + dy
                        if abs(nx) > 64 or abs(ny) > 64:
                            continue
                        occupied = any(k != idx and points[k][0]==nx and points[k][1]==ny for k in range(n))
                        if occupied:
                            continue
                        valid = all(dist_sq([nx, ny], points[k]) >= 1 for k in range(n) if k != idx)
                        if not valid:
                            continue
                        points[idx][0] = nx
                        points[idx][1] = ny
                        new_edges = count_edges(points)
                        if new_edges > current_edges:
                            current_edges = new_edges
                            if current_edges > best_edges:
                                best_edges = current_edges
                                best_points = [list(p) for p in points]
                        else:
                            points[idx][0] = ox
                            points[idx][1] = oy
                        break
        
        return best_points, best_edges

    # Strategy: Try multiple initialization strategies
    best_overall = None
    best_overall_edges = 0
    
    # Seed 1: Hexagon (4,4,4) with 2 points removed
    hex34 = gen_hex(4, 4, 4)
    if len(hex34) >= 32:
        # Try all pairs to remove
        for i in range(len(hex34)):
            if time.time() - start_time > DEADLINE - 20:
                break
            for j in range(i+1, len(hex34)):
                if time.time() - start_time > DEADLINE - 20:
                    break
                pts = [p for k, p in enumerate(hex34) if k != i and k != j]
                if len(pts) != 32:
                    continue
                e = count_edges(pts)
                if e > best_overall_edges:
                    best_overall_edges = e
                    best_overall = [list(p) for p in pts]
                if best_overall_edges == 76:
                    break
            if best_overall_edges == 76:
                break
    
    # Seed 2: Greedy construction from a small hexagon
    for seed_hex in [(2,2,2), (3,3,3), (1,3,3), (2,3,3)]:
        if time.time() - start_time > DEADLINE - 30:
            break
        seed_pts = gen_hex(*seed_hex)
        for seed_val in [42, 123, 789]:
            if time.time() - start_time > DEADLINE - 30:
                break
            random.seed(seed_val)
            greedy_pts = greedy_construct(seed_pts, 32)
            if len(greedy_pts) == 32:
                e = count_edges(greedy_pts)
                if e > best_overall_edges:
                    best_overall_edges = e
                    best_overall = greedy_pts
                if best_overall_edges == 76:
                    break
        if best_overall_edges == 76:
            break
    
    # If we have a good starting point, do local search
    if best_overall is not None and best_overall_edges < 76:
        if time.time() - start_time < DEADLINE - 10:
            improved_pts, improved_edges = local_search(best_overall)
            if improved_edges > best_overall_edges:
                best_overall = improved_pts
                best_overall_edges = improved_edges
    
    # If still not optimal, try more greedy constructions with different seeds
    if best_overall_edges < 76 and time.time() - start_time < DEADLINE - 15:
        for seed_val in [1, 2, 3, 4, 5, 10, 20, 50, 100, 200]:
            if time.time() - start_time > DEADLINE - 10:
                break
            random.seed(seed_val)
            # Start from a random small cluster
            init_pts = []
            for _ in range(5):
                x = random.randint(-3, 3)
                y = random.randint(-3, 3)
                if (x, y) not in [tuple(p) for p in init_pts]:
                    valid = all(dist_sq((x,y), p) >= 1 for p in init_pts)
                    if valid:
                        init_pts.append([x, y])
            greedy_pts = greedy_construct(init_pts, 32)
            if len(greedy_pts) == 32:
                e = count_edges(greedy_pts)
                if e > best_overall_edges:
                    best_overall_edges = e
                    best_overall = greedy_pts
                if best_overall_edges == 76:
                    break
            # Local search on this
            if e < 76 and time.time() - start_time < DEADLINE - 5:
                improved_pts, improved_edges = local_search(greedy_pts)
                if improved_edges > best_overall_edges:
                    best_overall = improved_pts
                    best_overall_edges = improved_edges
                if best_overall_edges == 76:
                    break
    
    # Final fallback
    if best_overall is None:
        best_overall = [[i, j] for i in range(4) for j in range(8)]
        best_overall_edges = count_edges(best_overall)
    
    return best_overall
