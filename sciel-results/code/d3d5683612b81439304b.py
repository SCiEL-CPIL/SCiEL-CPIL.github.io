import time
import random
import numpy as np
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 55

    def _score(points):
        """Return (max_multiplicity, best_area) for a point set."""
        n = len(points)
        if n < 3:
            return 0, 0
        areas = {}
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i + 1, n):
                x2, y2 = points[j]
                for k in range(j + 1, n):
                    x3, y3 = points[k]
                    det = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
                    if det > 0:
                        areas[det] = areas.get(det, 0) + 1
        if not areas:
            return 0, 0
        max_mult = max(areas.values())
        # Pick the smallest area with max multiplicity for stability
        max_area = min(a for a, m in areas.items() if m == max_mult)
        return max_mult, max_area

    def _score_fast(points):
        """Faster scoring using numpy for the inner loop."""
        n = len(points)
        if n < 3:
            return 0, 0
        pts = np.array(points, dtype=np.int64)
        areas = {}
        for i in range(n):
            x1, y1 = pts[i]
            for j in range(i + 1, n):
                x2, y2 = pts[j]
                for k in range(j + 1, n):
                    x3, y3 = pts[k]
                    det = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
                    if det > 0:
                        areas[det] = areas.get(det, 0) + 1
        if not areas:
            return 0, 0
        max_mult = max(areas.values())
        max_area = min(a for a, m in areas.items() if m == max_mult)
        return max_mult, max_area

    def _generate_lattice_initializer(seed_val):
        """Generate points on a 2D lattice (arithmetic progression) to get many equal-area triangles."""
        random.seed(seed_val)
        # Choose lattice parameters
        # Points: (a*i + b*j, c*i + d*j) for i in range(m), j in range(p), m*p=30
        factorizations = [(1,30),(2,15),(3,10),(5,6),(6,5),(10,3),(15,2),(30,1)]
        m, p = random.choice(factorizations)
        
        # Random lattice basis vectors, small to stay in bounds
        a = random.randint(-4, 4)
        b = random.randint(-4, 4)
        c = random.randint(-4, 4)
        d = random.randint(-4, 4)
        
        # Ensure non-degenerate lattice
        det = a * d - b * c
        if det == 0:
            return None
        
        pts = []
        for i in range(m):
            for j in range(p):
                x = a * i + b * j
                y = c * i + d * j
                pts.append((x, y))
        
        # Check distinctness
        if len(set(pts)) != 30:
            return None
        
        # Center
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        cx = (min(xs) + max(xs)) // 2
        cy = (min(ys) + max(ys)) // 2
        pts = [(x - cx, y - cy) for x, y in pts]
        
        # Check bounds
        if all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in pts):
            return pts
        return None

    def _generate_sheared_grid(seed_val):
        """Generate sheared grid points."""
        random.seed(seed_val)
        factorizations = [(1,30),(2,15),(3,10),(5,6),(6,5),(10,3),(15,2),(30,1)]
        m, p = random.choice(factorizations)
        s = random.randint(-5, 5)  # shear parameter
        
        pts = []
        for i in range(m):
            for j in range(p):
                x = i + s * j
                y = j
                pts.append((x, y))
        
        if len(set(pts)) != 30:
            return None
        
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        cx = (min(xs) + max(xs)) // 2
        cy = (min(ys) + max(ys)) // 2
        pts = [(x - cx, y - cy) for x, y in pts]
        
        if all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in pts):
            return pts
        return None

    def _generate_random_cluster(seed_val):
        """Generate random points with some structure."""
        random.seed(seed_val)
        # Pick a center and spread
        cx = random.randint(-10, 10)
        cy = random.randint(-10, 10)
        spread = random.randint(3, 12)
        
        pts = set()
        attempts = 0
        while len(pts) < 30 and attempts < 500:
            x = cx + random.randint(-spread, spread)
            y = cy + random.randint(-spread, spread)
            if -30 <= x <= 30 and -30 <= y <= 30:
                pts.add((x, y))
            attempts += 1
        
        if len(pts) != 30:
            return None
        return list(pts)

    def _simulated_annealing(initial_points, max_time):
        """Run simulated annealing on the point set."""
        current_points = list(initial_points)
        current_score, current_area = _score(current_points)
        best_points = list(current_points)
        best_score = current_score
        best_area = current_area
        
        n = 30
        # Temperature schedule
        T_start = 10.0
        T_end = 0.01
        alpha = 0.995  # cooling rate
        
        T = T_start
        no_improve = 0
        max_no_improve = 2000
        
        while time.time() < max_time and no_improve < max_no_improve:
            # Pick a random point to move
            idx = random.randint(0, n - 1)
            old_x, old_y = current_points[idx]
            
            # Random move: small perturbation
            step = random.choice([1, 1, 1, 2, 2, 3, 4, 5])
            dx = random.randint(-step, step)
            dy = random.randint(-step, step)
            if dx == 0 and dy == 0:
                continue
            
            new_x = old_x + dx
            new_y = old_y + dy
            
            if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
                continue
            
            new_pos = (new_x, new_y)
            
            # Check collision
            collision = False
            for i in range(n):
                if i != idx and current_points[i] == new_pos:
                    collision = True
                    break
            if collision:
                continue
            
            # Try the move
            current_points[idx] = new_pos
            new_score, new_area = _score(current_points)
            
            delta = new_score - current_score
            
            if delta >= 0 or random.random() < np.exp(delta / T):
                current_score = new_score
                current_area = new_area
                if new_score > best_score:
                    best_score = new_score
                    best_area = new_area
                    best_points = list(current_points)
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Revert
                current_points[idx] = (old_x, old_y)
                no_improve += 1
            
            T *= alpha
            if T < T_end:
                T = T_end
        
        return best_points, best_score, best_area

    # Generate diverse initial candidates
    candidates = []
    
    # Lattice initializers (key insight: lattice points create many equal-area triangles)
    for seed in range(50):
        pts = _generate_lattice_initializer(seed)
        if pts:
            candidates.append(pts)
    
    # Sheared grids
    for seed in range(50):
        pts = _generate_sheared_grid(seed)
        if pts:
            candidates.append(pts)
    
    # Random clusters
    for seed in range(30):
        pts = _generate_random_cluster(seed)
        if pts:
            candidates.append(pts)
    
    # Standard rectangular grids
    for w, h in [(6,5),(5,6),(10,3),(3,10),(15,2),(2,15)]:
        pts = [(x, y) for x in range(w) for y in range(h)]
        if len(set(pts)) == 30:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            cx = (min(xs) + max(xs)) // 2
            cy = (min(ys) + max(ys)) // 2
            pts_c = [(x - cx, y - cy) for x, y in pts]
            if all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in pts_c):
                candidates.append(pts_c)
    
    # Evaluate all candidates
    scored_candidates = []
    for pts in candidates:
        score, area = _score(pts)
        scored_candidates.append((score, area, pts))
    
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    if not scored_candidates:
        # Fallback: simple grid
        return [(i % 6, i // 6) for i in range(30)]
    
    current_best_score = scored_candidates[0][0]
    current_best_points = scored_candidates[0][2]
    
    # Run simulated annealing on top candidates
    # Use remaining time for SA
    elapsed = time.time() - start_time
    remaining = deadline - start_time - elapsed
    
    if remaining > 10:
        # Run SA on top 5 candidates
        for i, (sc, ar, pts) in enumerate(scored_candidates[:5]):
            if time.time() > deadline - 3:
                break
            
            # Allocate time per SA run
            sa_time = min(remaining / (5 - i) - 1, 15)
            if sa_time < 2:
                break
            
            sa_pts, sa_score, sa_area = _simulated_annealing(pts, time.time() + sa_time)
            
            if sa_score > current_best_score:
                current_best_score = sa_score
                current_best_points = sa_pts
    
    return current_best_points
