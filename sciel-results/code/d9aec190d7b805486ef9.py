"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START
def _doubled_area(p1, p2, p3):
    """Compute absolute doubled area (determinant) for three points."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

def _score(points):
    """Score = max multiplicity of a single positive doubled area among all triples."""
    from collections import Counter
    areas = []
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = _doubled_area(points[i], points[j], points[k])
                if a > 0:
                    areas.append(a)
    if not areas:
        return 0
    return max(Counter(areas).values())

def _evaluate(points):
    """Fast evaluation using vectorized numpy for 30 points."""
    n = len(points)
    pts = np.array(points, dtype=np.int64)
    # Compute all triple determinants
    best_count = 0
    from collections import Counter
    area_counts = Counter()
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = abs(int(pts[j,0]-pts[i,0])*(int(pts[k,1]-pts[i,1]) - int(pts[k,0]-pts[i,0])*0) 
                        - 0)  # placeholder, use direct
                # Direct computation
                x1,y1 = pts[i]
                x2,y2 = pts[j]
                x3,y3 = pts[k]
                det = abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))
                if det > 0:
                    area_counts[det] += 1
    if area_counts:
        best_count = max(area_counts.values())
    return best_count

def _evaluate_fast(points):
    """Even faster: precompute pair vectors."""
    n = len(points)
    best_count = 0
    from collections import Counter
    area_counts = Counter()
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i+1, n):
            x2, y2 = points[j]
            dx1, dy1 = x2-x1, y2-y1
            for k in range(j+1, n):
                x3, y3 = points[k]
                det = abs(dx1*(y3-y1) - (x3-x1)*dy1)
                if det > 0:
                    area_counts[det] += 1
    if area_counts:
        best_count = max(area_counts.values())
    return best_count

def _local_search(points, time_limit=55.0):
    """Local search with biased neighborhood moves to improve the configuration."""
    start_time = time.time()
    best_score = _evaluate_fast(points)
    best_points = list(points)
    current = list(points)
    current_score = best_score
    
    # All valid positions
    all_positions = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    current_set = set(current)
    
    # Biased neighborhood: focus on positions near existing points
    def get_neighborhood(pt, radius=5):
        x, y = pt
        neighbors = []
        for dx in range(-radius, radius+1):
            for dy in range(-radius, radius+1):
                np_ = (x+dx, y+dy)
                if -30 <= np_[0] <= 30 and -30 <= np_[1] <= 30 and np_ not in current_set:
                    neighbors.append(np_)
        return neighbors
    
    iterations = 0
    no_improve = 0
    max_no_improve = 200
    
    while time.time() - start_time < time_limit:
        iterations += 1
        improved = False
        
        # Try moving one point
        for idx in range(len(current)):
            if time.time() - start_time > time_limit - 5:
                break
            orig = current[idx]
            current_set.discard(orig)
            
            # Get neighborhood biased toward existing points
            neigh = get_neighborhood(orig, radius=4)
            random.shuffle(neigh)
            
            best_local_score = current_score
            best_local_pos = orig
            
            # Evaluate a subset of neighbors
            for new_pos in neigh[:30]:
                candidate = list(current)
                candidate[idx] = new_pos
                s = _evaluate_fast(candidate)
                if s > best_local_score:
                    best_local_score = s
                    best_local_pos = new_pos
                elif s == best_local_score and random.random() < 0.1:
                    best_local_pos = new_pos  # diversity
            
            if best_local_pos != orig:
                current[idx] = best_local_pos
                current_set.discard(orig)
                current_set.add(best_local_pos)
                current_score = best_local_score
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_points = list(current)
                no_improve = 0
            else:
                current_set.add(orig)
                no_improve += 1
            
            if no_improve >= max_no_improve:
                break
        
        if no_improve >= max_no_improve:
            # Reset and try different strategy
            no_improve = 0
            # Occasionally restart from best
            if random.random() < 0.3:
                current = list(best_points)
                current_score = best_score
                current_set = set(current)
        
        if improved and no_improve == 0:
            continue
    
    return best_points, best_score

def _spectral_init(target_area, n_points=30):
    """
    Initialize using spectral method: build matrix M where M[i,j] = number of 
    third lattice points k forming triangle of area target_area with i and j.
    Find largest eigenvector to identify densest subgraph.
    """
    # Grid positions
    positions = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    pos_to_idx = {p: i for i, p in enumerate(positions)}
    n_pos = len(positions)
    
    # Precompute for each pair (i,j), count of k such that area(i,j,k) == target_area
    # This is expensive, so we'll approximate with a smaller grid or use random sampling
    # For efficiency, let's use a reduced approach
    
    # Actually, for 61x61 = 3721 positions, computing M is O(n_pos^3) which is too slow.
    # Instead, let's use a heuristic: for each position, count how many other positions 
    # form the target area with it, then use that as a weight.
    
    # Simpler: just use random initialization with bias toward structured patterns
    return None

def _structured_init():
    """Create structured initial configurations likely to have many equal-area triangles."""
    configs = []
    
    # Grid-like configurations
    for w in range(3, 8):
        for h in range(3, 8):
            if w * h >= 30:
                pts = [(x, y) for x in range(w) for y in range(h)]
                if len(pts) > 30:
                    pts = pts[:30]
                if len(pts) == 30:
                    configs.append(pts)
    
    # Hexagonal-like patterns
    for offset in range(-5, 6):
        pts = []
        for x in range(-5, 6):
            for y in range(-5, 6):
                if (x + y) % 2 == 0:
                    pts.append((x + offset, y))
        if len(pts) >= 30:
            random.shuffle(pts)
            configs.append(pts[:30])
    
    # Concentric rectangles
    for scale in range(1, 6):
        pts = []
        for x in range(-scale, scale+1):
            for y in range(-scale, scale+1):
                if abs(x) == scale or abs(y) == scale:
                    pts.append((x, y))
        if len(pts) >= 30:
            random.shuffle(pts)
            configs.append(pts[:30])
    
    # Random
    all_pos = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    for _ in range(20):
        pts = random.sample(all_pos, 30)
        configs.append(pts)
    
    return configs

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin before 60s
    
    # Generate initial configurations
    configs = _structured_init()
    
    # Evaluate all initial configs
    best_points = None
    best_score = 0
    
    for cfg in configs:
        if time.time() > deadline - 10:
            break
        s = _evaluate_fast(cfg)
        if s > best_score:
            best_score = s
            best_points = list(cfg)
    
    # If no good config found, use seed
    if best_points is None:
        best_points = [(x, y) for x in range(6) for y in range(5)]
        best_score = _evaluate_fast(best_points)
    
    # Local search from best config
    if time.time() < deadline - 5:
        improved_points, improved_score = _local_search(best_points, time_limit=deadline - time.time() - 2)
        if improved_score > best_score:
            best_score = improved_score
            best_points = improved_points
    
    # Try a few more random restarts if time allows
    if time.time() < deadline - 10:
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            all_pos = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
            random_pts = random.sample(all_pos, 30)
            s = _evaluate_fast(random_pts)
            if s > best_score:
                best_score = s
                best_points = list(random_pts)
            # Quick local search
            if time.time() < deadline - 5:
                rp, rs = _local_search(random_pts, time_limit=min(5.0, deadline - time.time() - 2))
                if rs > best_score:
                    best_score = rs
                    best_points = rp
    
    return best_points
# EVOLVE-BLOCK-END
