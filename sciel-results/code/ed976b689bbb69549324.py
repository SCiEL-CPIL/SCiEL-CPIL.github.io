"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations
from collections import Counter

import numpy as np

# EVOLVE-BLOCK-START

def _compute_all_areas(points):
    """Compute all positive doubled areas for all triples of points.
    Returns a list of areas (with multiplicity)."""
    n = len(points)
    pts = np.array(points, dtype=np.int64)
    areas = []
    for i in range(n):
        pi = pts[i]
        for j in range(i + 1, n):
            v1 = pts[j] - pi
            for k in range(j + 1, n):
                v2 = pts[k] - pi
                det = abs(v1[0] * v2[1] - v1[1] * v2[0])
                if det > 0:
                    areas.append(det)
    return areas

def _score(points):
    """Return the maximum multiplicity of any single positive doubled area."""
    areas = _compute_all_areas(points)
    if not areas:
        return 0
    counter = Counter(areas)
    return max(counter.values())

def _score_and_counter(points):
    """Return (max_multiplicity, Counter) for a configuration."""
    areas = _compute_all_areas(points)
    if not areas:
        return 0, Counter()
    counter = Counter(areas)
    return max(counter.values()), counter

def _try_move(points, i, nx, ny):
    """Try moving point i to (nx,ny). Return (new_score, new_points) or (None, None)."""
    if nx < -30 or nx > 30 or ny < -30 or ny > 30:
        return None, None
    if (nx, ny) in points:
        return None, None
    new_points = list(points)
    new_points[i] = (nx, ny)
    if len(set(new_points)) != len(points):
        return None, None
    s = _score(new_points)
    return s, new_points

def _hill_climb(points, deadline, max_rounds=200):
    """Hill-climb: try moving each point to nearby positions. Accept if score improves."""
    best_points = list(points)
    best_score = _score(best_points)
    
    n = len(best_points)
    improved = True
    rounds = 0
    
    while improved and rounds < max_rounds and time.time() < deadline - 3:
        improved = False
        rounds += 1
        
        for i in range(n):
            if time.time() > deadline - 3:
                break
            x, y = best_points[i]
            
            # Try a neighborhood of radius 3
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if nx < -30 or nx > 30 or ny < -30 or ny > 30:
                        continue
                    if (nx, ny) in best_points:
                        continue
                    
                    new_points = list(best_points)
                    new_points[i] = (nx, ny)
                    if len(set(new_points)) != n:
                        continue
                    
                    new_score = _score(new_points)
                    if new_score > best_score:
                        best_score = new_score
                        best_points = new_points
                        improved = True
    
    return best_points, best_score

def _simulated_annealing(points, deadline, max_iter=5000):
    """Simulated annealing to escape local optima."""
    best_points = list(points)
    best_score = _score(best_points)
    current_points = list(best_points)
    current_score = best_score
    
    n = len(current_points)
    temp = 5.0
    cooling = 0.997
    
    for it in range(max_iter):
        if time.time() > deadline - 3:
            break
        
        # Random move: pick a point and a random nearby position
        i = random.randint(0, n - 1)
        x, y = current_points[i]
        dx = random.randint(-4, 4)
        dy = random.randint(-4, 4)
        nx, ny = x + dx, y + dy
        if nx < -30 or nx > 30 or ny < -30 or ny > 30:
            continue
        if (nx, ny) in current_points:
            continue
        
        new_points = list(current_points)
        new_points[i] = (nx, ny)
        if len(set(new_points)) != n:
            continue
        
        new_score = _score(new_points)
        delta = new_score - current_score
        
        if delta > 0 or random.random() < np.exp(delta / max(temp, 0.1)):
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = list(current_points)
        
        temp *= cooling
    
    return best_points, best_score

def _generate_grid(w, h, ox=0, oy=0, shear=0):
    """Generate a w x h grid with optional offset and shear."""
    pts = []
    for xi in range(w):
        for yi in range(h):
            x = xi + ox + shear * yi
            y = yi + oy
            if -30 <= x <= 30 and -30 <= y <= 30:
                pts.append((x, y))
    return pts[:30] if len(pts) >= 30 else pts

def _generate_lattice(a1x, a1y, a2x, a2y, ox, oy):
    """Generate 30 points on a 2D lattice with basis vectors."""
    pts = []
    seen = set()
    # Enumerate lattice points in a bounding box
    for i in range(-10, 11):
        for j in range(-10, 11):
            x = ox + i * a1x + j * a2x
            y = oy + i * a1y + j * a2y
            if -30 <= x <= 30 and -30 <= y <= 30:
                p = (x, y)
                if p not in seen:
                    seen.add(p)
                    pts.append(p)
    random.shuffle(pts)
    return pts[:30]

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    # Phase 1: Generate diverse initial configurations
    candidates = []
    
    # Standard grids
    for w, h in [(6,5), (5,6), (3,10), (10,3), (2,15), (15,2)]:
        for ox in range(-5, 6, 3):
            for oy in range(-5, 6, 3):
                pts = _generate_grid(w, h, ox, oy)
                if len(pts) == 30:
                    candidates.append(pts)
    
    # Sheared grids
    for w, h in [(6,5), (5,6)]:
        for shear in [-2, -1, 1, 2]:
            for ox in range(-5, 6, 3):
                for oy in range(-5, 6, 3):
                    pts = _generate_grid(w, h, ox, oy, shear)
                    if len(pts) == 30:
                        candidates.append(pts)
    
    # Lattice-based configurations
    random.seed(42)
    for _ in range(20):
        a1x = random.randint(1, 4)
        a1y = random.randint(0, 2)
        a2x = random.randint(0, 2)
        a2y = random.randint(1, 4)
        # Ensure non-degenerate basis
        det = a1x * a2y - a1y * a2x
        if abs(det) < 1:
            continue
        ox = random.randint(-10, 10)
        oy = random.randint(-10, 10)
        pts = _generate_lattice(a1x, a1y, a2x, a2y, ox, oy)
        if len(pts) == 30:
            candidates.append(pts)
    
    # Random configurations
    for _ in range(10):
        pts = []
        seen = set()
        while len(pts) < 30:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            if (x, y) not in seen:
                seen.add((x, y))
                pts.append((x, y))
        candidates.append(pts)
    
    # Evaluate all candidates and keep the best few
    scored_candidates = []
    for pts in candidates:
        s = _score(pts)
        scored_candidates.append((s, pts))
    
    scored_candidates.sort(key=lambda x: -x[0])
    
    # Take top diverse candidates for local search
    top_candidates = []
    top_scores = set()
    for s, pts in scored_candidates:
        key = tuple(sorted(pts))
        if key not in top_scores:
            top_scores.add(key)
            top_candidates.append((s, pts))
        if len(top_candidates) >= 8:
            break
    
    # Phase 2: Hill-climb each top candidate
    best_overall_points = None
    best_overall_score = 0
    
    for s, pts in top_candidates:
        if time.time() > deadline - 10:
            break
        hp, hs = _hill_climb(pts, deadline)
        if hs > best_overall_score:
            best_overall_score = hs
            best_overall_points = hp
    
    # Phase 3: Simulated annealing from the best configuration
    if best_overall_points and time.time() < deadline - 15:
        sa_pts, sa_score = _simulated_annealing(best_overall_points, deadline, max_iter=3000)
        if sa_score > best_overall_score:
            best_overall_score = sa_score
            best_overall_points = sa_pts
    
    # Phase 4: Final hill-climb polish
    if best_overall_points and time.time() < deadline - 5:
        hp, hs = _hill_climb(best_overall_points, deadline, max_rounds=100)
        if hs > best_overall_score:
            best_overall_score = hs
            best_overall_points = hp
    
    if best_overall_points is None:
        # Fallback: return the best candidate from Phase 1
        best_overall_points = top_candidates[0][1] if top_candidates else [(x, y) for x in range(6) for y in range(5)]
    
    return best_overall_points

# EVOLVE-BLOCK-END
