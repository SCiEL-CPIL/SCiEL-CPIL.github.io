"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import Counter

def _compute_score(points):
    """Compute the largest multiplicity of a single positive doubled area."""
    n = len(points)
    areas = []
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            for k in range(j + 1, n):
                x3, y3 = points[k]
                det = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
                if det > 0:
                    areas.append(det)
    if not areas:
        return 0
    counts = Counter(areas)
    return max(counts.values())

def _inbounds(points, limit=30):
    for x, y in points:
        if x < -limit or x > limit or y < -limit or y > limit:
            return False
    return True

def _build_grid(w, h, x0=0, y0=0):
    return [(x0 + x, y0 + y) for x in range(w) for y in range(h)]

def _build_parallelogram(w, h, sx, sy, tx, ty, x0=0, y0=0):
    """Build a parallelogram grid: (x0 + i*sx + j*tx, y0 + i*sy + j*ty)"""
    pts = []
    for i in range(w):
        for j in range(h):
            pts.append((x0 + i * sx + j * tx, y0 + i * sy + j * ty))
    return pts

def _local_improve(points, max_iters=5000, timeout=45):
    """Local improvement via point moves."""
    start_time = time.time()
    best_points = list(points)
    best_score = _compute_score(best_points)
    
    # Precompute all points for quick distinctness check
    for iteration in range(max_iters):
        if time.time() - start_time > timeout:
            break
        
        # Try moving a random point to a random location
        idx = random.randint(0, len(best_points) - 1)
        old_point = best_points[idx]
        
        # Try a few candidate moves
        for _ in range(5):
            new_x = random.randint(-30, 30)
            new_y = random.randint(-30, 30)
            new_point = (new_x, new_y)
            
            # Check if point already exists
            if new_point in best_points:
                continue
            
            best_points[idx] = new_point
            new_score = _compute_score(best_points)
            
            if new_score > best_score:
                best_score = new_score
            else:
                best_points[idx] = old_point
                break  # Move didn't help, try different point
    
    return best_points, best_score

def _generate_candidates():
    """Generate diverse candidate configurations."""
    candidates = []
    
    # Rectangular grids
    grid_configs = [(6, 5), (5, 6), (10, 3), (3, 10), (15, 2), (2, 15)]
    for w, h in grid_configs:
        for x0 in range(-30, 31 - w + 1):
            for y0 in range(-30, 31 - h + 1):
                pts = _build_grid(w, h, x0, y0)
                if _inbounds(pts):
                    candidates.append(pts)
    
    # Parallelogram patterns
    for w, h in [(6, 5), (5, 6)]:
        for sx in [1, 2]:
            for sy in [0, 1]:
                for tx in [-1, 0, 1]:
                    for ty in [0, 1, 2]:
                        if sx * ty - sy * tx == 0:  # Degenerate
                            continue
                        for x0 in range(-15, 16):
                            for y0 in range(-15, 16):
                                pts = _build_parallelogram(w, h, sx, sy, tx, ty, x0, y0)
                                if _inbounds(pts):
                                    candidates.append(pts)
    
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85 second deadline with 5s safety margin
    
    best_points = None
    best_score = 0
    
    # Generate and evaluate candidates
    candidates = _generate_candidates()
    
    for pts in candidates:
        if time.time() > deadline - 20:  # Reserve time for local improvement
            break
        
        score = _compute_score(pts)
        if score > best_score:
            best_score = score
            best_points = list(pts)
    
    # Local improvement
    if best_points is not None and time.time() < deadline - 5:
        improved, s2 = _local_improve(best_points, max_iters=3000, timeout=deadline - time.time() - 5)
        if s2 > best_score:
            best_points, best_score = improved, s2
    
    # Fallback
    if best_points is None:
        best_points = [(x, y) for x in range(6) for y in range(5)]
    
    return best_points
# EVOLVE-BLOCK-END
