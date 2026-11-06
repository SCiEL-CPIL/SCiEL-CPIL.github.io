"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import itertools
import time
import random

# EVOLVE-BLOCK-START
def _collinear_triples(points):
    """Find all triples of points that are collinear."""
    n = len(points)
    triples = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                if (x2 - x1) * (y3 - y1) == (x3 - x1) * (y2 - y1):
                    triples.append((i, j, k))
    return triples

def _has_four_collinear(points):
    """Check if any four points are collinear."""
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                dx1 = x2 - x1
                dy1 = y2 - y1
                dx2 = x3 - x1
                dy2 = y3 - y1
                if dx1 * dy2 != dy1 * dx2:
                    continue
                # Check if any other point is on this line
                for m in range(n):
                    if m in (i, j, k):
                        continue
                    x4, y4 = points[m]
                    if (x2 - x1) * (y4 - y1) == (x3 - x1) * (y2 - y1):
                        return True
    return False

def _score(points):
    """Compute the largest subset with no three collinear using branch and bound."""
    n = len(points)
    triples = _collinear_triples(points)
    
    if not triples:
        return n
    
    # Build adjacency: for each point, which triples contain it
    point_triples = [[] for _ in range(n)]
    for t in triples:
        for idx in t:
            point_triples[idx].append(t)
    
    # Minimum vertex cover on hypergraph: remove min points to hit all triples
    # Equivalent: max subset with no triple fully contained = n - min vertex cover
    
    # Branch and bound for minimum vertex cover
    best_cover = [n]  # worst case: remove all
    
    def _bvb(triples_remaining, removed, depth):
        if not triples_remaining:
            best_cover[0] = min(best_cover[0], len(removed))
            return
        if len(removed) >= best_cover[0]:
            return
        # Pick the triple with the most "forced" structure
        # Heuristic: pick a triple, try removing each of its 3 points
        t = triples_remaining[0]
        rest = triples_remaining[1:]
        
        for pt in t:
            if pt in removed:
                continue
            new_removed = removed | {pt}
            # Filter out triples that contain pt
            new_triples = [tr for tr in rest if pt not in tr]
            _bvb(new_triples, new_removed, depth + 1)
    
    _bvb(triples, frozenset(), 0)
    
    return n - best_cover[0]

def _try_config(points, time_limit=140):
    """Local search to minimize score."""
    start_time = time.time()
    current = [list(p) for p in points]
    current_score = _score(current)
    best = [list(p) for p in current]
    best_score = current_score
    
    # Candidate pool of points
    candidates = [(x, y) for x in range(-50, 51) for y in range(-50, 51)]
    candidates = [c for c in candidates if c not in [tuple(p) for p in current]]
    
    random.seed(42)
    
    while time.time() - start_time < time_limit - 10:
        # Try swapping a point
        idx = random.randint(0, 19)
        new_point = random.choice(candidates)
        
        old_point = tuple(current[idx])
        current[idx] = list(new_point)
        
        # Check validity: distinct and no 4 collinear
        if len(set(tuple(p) for p in current)) != 20:
            current[idx] = list(old_point)
            continue
        if _has_four_collinear(current):
            current[idx] = list(old_point)
            continue
        
        s = _score(current)
        if s < current_score:
            current_score = s
            if s < best_score:
                best_score = s
                best = [list(p) for p in current]
            # Update candidate pool
            candidates = [c for c in candidates if c != tuple(new_point)]
            candidates.append(old_point)
        else:
            current[idx] = list(old_point)
    
    return best, best_score

def construct():
    start_time = time.time()
    
    # Several initial configurations
    configs = []
    
    # Config 1: structured grid-like
    # Try points that create many 3-collinear lines
    # e.g., a 5x4 grid with some perturbations
    grid_pts = []
    for x in range(-4, 5):
        for y in range(-3, 4):
            if len(grid_pts) < 20:
                grid_pts.append([x * 10, y * 10])
    configs.append(grid_pts)
    
    # Config 2: two parallel lines with 10 points each won't work (4+ collinear)
    # Instead: spread points to create 3-collinear lines
    # Use a "star" pattern
    star_pts = []
    # 5 lines through origin, 4 points each? No, that gives 4 collinear.
    # 5 lines, each with 3 points, sharing the center
    # Center + 5*3 = 16, need 4 more
    # Let's try: center at (0,0), 5 directions, 3 points per direction at distances 1,2,3
    for angle in range(5):
        import math
        a = 2 * math.pi * angle / 5
        for d in [10, 20, 30]:
            x = int(round(d * math.cos(a)))
            y = int(round(d * math.sin(a)))
            if [x, y] not in star_pts:
                star_pts.append([x, y])
    # Add a few more
    for pt in [[0, 0], [5, 5], [-5, 5], [5, -5]]:
        if pt not in star_pts and len(star_pts) < 20:
            star_pts.append(list(pt))
    while len(star_pts) < 20:
        star_pts.append([random.randint(-10, 10), random.randint(-10, 10)])
    configs.append(star_pts)
    
    # Config 3: random
    random.seed(123)
    rand_pts = []
    while len(rand_pts) < 20:
        p = [random.randint(-50, 50), random.randint(-50, 50)]
        if p not in rand_pts:
            rand_pts.append(p)
    configs.append(rand_pts)
    
    best_overall = None
    best_score_overall = 999
    
    for cfg in configs:
        if time.time() - start_time > 135:
            break
        # Validate
        if len(set(tuple(p) for p in cfg)) != 20:
            continue
        if _has_four_collinear(cfg):
            continue
        try:
            result, score = _try_config(cfg, time_limit=135 - (time.time() - start_time))
            if score < best_score_overall:
                best_score_overall = score
                best_overall = result
        except Exception:
            continue
    
    if best_overall is None:
        # Fallback: parabola
        best_overall = [[x, x * x] for x in range(-10, 10)]
    
    return best_overall
# EVOLVE-BLOCK-END
