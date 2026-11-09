"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import Counter

# EVOLVE-BLOCK-START

def _doubled_area(p, q, r):
    """Compute doubled area (absolute determinant) of triangle p,q,r."""
    return abs((q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0]))

def _compute_areas(points):
    """Compute all positive doubled areas for all triples."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = _doubled_area(points[i], points[j], points[k])
                if a > 0:
                    areas.append(a)
    return areas

def _score(points):
    """Compute the largest multiplicity of a single positive doubled area."""
    areas = _compute_areas(points)
    if not areas:
        return 0
    counts = Counter(areas)
    return max(counts.values())

def _incremental_score(points, target_area=None):
    """Compute score, optionally focusing on a target area."""
    n = len(points)
    if target_area is None:
        areas = []
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    a = _doubled_area(points[i], points[j], points[k])
                    if a > 0:
                        areas.append(a)
        if not areas:
            return 0
        counts = Counter(areas)
        return max(counts.values())
    else:
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    a = _doubled_area(points[i], points[j], points[k])
                    if a == target_area:
                        count += 1
        return count

def _random_point_in_bounds():
    return (random.randint(-30, 30), random.randint(-30, 30))

def _is_valid(points):
    """Check if points are 30 distinct integer pairs in [-30,30]^2."""
    if len(points) != 30:
        return False
    seen = set()
    for p in points:
        if not (isinstance(p, tuple) and len(p) == 2):
            return False
        x, y = p
        if not (isinstance(x, int) and isinstance(y, int)):
            return False
        if not (-30 <= x <= 30 and -30 <= y <= 30):
            return False
        if p in seen:
            return False
        seen.add(p)
    return True

def _hill_climb(points, time_limit, rng):
    """Local search: move one point at a time to improve score."""
    n = len(points)
    points = list(points)
    current_score = _score(points)
    
    # Precompute areas for faster evaluation
    def get_areas(pts):
        areas = []
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    a = _doubled_area(pts[i], pts[j], pts[k])
                    if a > 0:
                        areas.append(a)
        return areas
    
    start_time = time.time()
    no_improve = 0
    max_no_improve = 200
    
    while time.time() - start_time < time_limit:
        # Try random moves
        found_improvement = False
        for attempt in range(50):
            i = rng.randint(0, n-1)
            new_x = rng.randint(-30, 30)
            new_y = rng.randint(-30, 30)
            new_point = (new_x, new_y)
            
            # Check distinctness
            if any(new_point == points[j] for j in range(n) if j != i):
                continue
            
            old_point = points[i]
            points[i] = new_point
            new_score = _score(points)
            
            if new_score > current_score:
                current_score = new_score
                found_improvement = True
                no_improve = 0
                break
            else:
                points[i] = old_point
        
        if not found_improvement:
            no_improve += 1
            if no_improve > max_no_improve:
                break
    
    return points, current_score

def _multi_start(time_budget):
    """Multiple random restarts with hill climbing."""
    rng = random.Random(42)
    best_points = None
    best_score = 0
    
    start_time = time.time()
    per_restart = 5.0
    
    while time.time() - start_time < time_budget - 2.0:
        # Generate random initial configuration
        points = set()
        while len(points) < 30:
            points.add(_random_point_in_bounds())
        points = list(points)
        
        # Hill climb
        improved_points, score = _hill_climb(points, per_restart, rng)
        
        if score > best_score:
            best_score = score
            best_points = improved_points
    
    return best_points, best_score

def _structured_seed(seed_type=0):
    """Generate structured seeds that tend to have many equal areas."""
    if seed_type == 0:
        # Grid-like
        return [(x, y) for x in range(-4, 4) for y in range(-4, 4)][:30]
    elif seed_type == 1:
        # Sparse grid
        pts = []
        for x in range(-6, 7, 2):
            for y in range(-6, 7, 2):
                if len(pts) < 30:
                    pts.append((x, y))
        return pts
    elif seed_type == 2:
        # Concentric pattern
        pts = []
        for r in range(0, 8):
            for angle in range(0, 360, 360 // max(1, r)):
                import math
                x = round(r * math.cos(math.radians(angle)))
                y = round(r * math.sin(math.radians(angle)))
                if -30 <= x <= 30 and -30 <= y <= 30 and (x, y) not in pts:
                    pts.append((x, y))
                if len(pts) >= 30:
                    break
            if len(pts) >= 30:
                break
        while len(pts) < 30:
            p = _random_point_in_bounds()
            if p not in pts:
                pts.append(p)
        return pts[:30]
    else:
        # Random
        pts = set()
        while len(pts) < 30:
            pts.add(_random_point_in_bounds())
        return list(pts)

def construct():
    """Construct 30 points to maximize the largest multiplicity of a doubled area."""
    start_time = time.time()
    total_budget = 85.0  # Safety margin before 90s kill
    
    rng = random.Random(123)
    
    best_points = None
    best_score = 0
    
    # Try structured seeds first
    for seed_type in range(4):
        if time.time() - start_time > total_budget - 10:
            break
        points = _structured_seed(seed_type)
        if not _is_valid(points):
            continue
        score = _score(points)
        if score > best_score:
            best_score = score
            best_points = list(points)
    
    # Multi-start local search
    remaining = total_budget - (time.time() - start_time)
    if remaining > 5.0 and best_points is not None:
        improved_points, improved_score = _multi_start(remaining - 3.0)
        if improved_score > best_score:
            best_score = improved_score
            best_points = improved_points
    
    # Final polish: focused search on the best area
    if best_points is not None:
        remaining = total_budget - (time.time() - start_time)
        if remaining > 3.0:
            areas = _compute_areas(best_points)
            if areas:
                counts = Counter(areas)
                target_area = counts.most_common(1)[0][0]
                
                points = list(best_points)
                n = len(points)
                polish_start = time.time()
                
                while time.time() - polish_start < remaining - 1.0:
                    improved = False
                    for i in range(n):
                        if time.time() - polish_start > remaining - 1.0:
                            break
                        for dx in range(-2, 3):
                            for dy in range(-2, 3):
                                if dx == 0 and dy == 0:
                                    continue
                                nx, ny = points[i][0] + dx, points[i][1] + dy
                                if not (-30 <= nx <= 30 and -30 <= ny <= 30):
                                    continue
                                new_point = (nx, ny)
                                if any(new_point == points[j] for j in range(n) if j != i):
                                    continue
                                
                                old_point = points[i]
                                points[i] = new_point
                                new_score = _score(points)
                                if new_score > best_score:
                                    best_score = new_score
                                    best_points = list(points)
                                    improved = True
                                else:
                                    points[i] = old_point
                                if time.time() - polish_start > remaining - 1.0:
                                    break
                            if time.time() - polish_start > remaining - 1.0:
                                break
                        if time.time() - polish_start > remaining - 1.0:
                            break
                    if not improved:
                        break
    
    if best_points is None:
        # Fallback
        best_points = [(x, y) for x in range(6) for y in range(5)]
    
    return best_points

# EVOLVE-BLOCK-END
