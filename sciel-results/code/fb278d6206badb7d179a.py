"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from collections import Counter

def _doubled_areas(points):
    """Compute all positive doubled areas for all triples."""
    n = len(points)
    areas = []
    for i in range(n):
        ax, ay = points[i]
        for j in range(i+1, n):
            bx, by = points[j]
            dx1 = bx - ax
            dy1 = by - ay
            for k in range(j+1, n):
                cx, cy = points[k]
                dx2 = cx - ax
                dy2 = cy - ay
                det = abs(dx1 * dy2 - dy1 * dx2)
                if det > 0:
                    areas.append(det)
    return areas

def _max_multiplicity(points):
    """Return the maximum count of any single positive doubled area."""
    areas = _doubled_areas(points)
    if not areas:
        return 0
    return Counter(areas).most_common(1)[0][1]

def _valid(points):
    """Check 30 distinct integer points in [-30,30]^2."""
    if len(points) != 30:
        return False
    if len(set(points)) != 30:
        return False
    for x, y in points:
        if not (isinstance(x, int) and isinstance(y, int)):
            return False
        if x < -30 or x > 30 or y < -30 or y > 30:
            return False
    return True

def _score(points):
    if not _valid(points):
        return 0
    return _max_multiplicity(points)

def _structured_candidates():
    """Generate various structured grid-like candidates."""
    candidates = []
    
    # Standard grids: m x n with m*n = 30
    for m, n in [(6,5), (5,6), (10,3), (3,10), (15,2), (2,15), (30,1), (1,30)]:
        pts = []
        for i in range(m):
            for j in range(n):
                x = i - m // 2
                y = j - n // 2
                pts.append((x, y))
        if _valid(pts):
            candidates.append(pts)
    
    # Grids with different spacings
    for m, n in [(6,5), (5,6), (10,3)]:
        for sx in [1, 2, 3]:
            for sy in [1, 2, 3]:
                pts = []
                for i in range(m):
                    for j in range(n):
                        x = i * sx - (m * sx) // 2
                        y = j * sy - (n * sy) // 2
                        pts.append((x, y))
                if _valid(pts):
                    candidates.append(pts)
    
    # Interleaved: points on lines
    for offset in range(0, 5):
        pts = []
        for line in range(3):
            for k in range(10):
                x = k * 6 - 30 + offset
                y = line * 10 - 10
                pts.append((x, y))
        if _valid(pts):
            candidates.append(pts)
    
    # Staggered rows
    for dx in [1, 2, 3]:
        for dy in [1, 2, 3]:
            pts = []
            for row in range(5):
                for col in range(6):
                    x = col * dx + (row % 2) * (dx // 2 + 1) - 15
                    y = row * dy - 10
                    pts.append((x, y))
            if _valid(pts):
                candidates.append(pts)
    
    # Arithmetic progression in both directions
    for step_x in [1, 2, 3, 4]:
        for step_y in [1, 2, 3, 4]:
            for start_x in range(-20, 20, 2):
                for start_y in range(-20, 20, 2):
                    pts = []
                    for i in range(30):
                        row = i // 6
                        col = i % 6
                        x = start_x + col * step_x
                        y = start_y + row * step_y
                        if -30 <= x <= 30 and -30 <= y <= 30:
                            pts.append((x, y))
                    if len(pts) == 30 and _valid(pts):
                        candidates.append(pts)
    
    return candidates

def _local_search(points, max_iterations=2000, time_limit=60.0):
    """Hill-climbing local search: move one point at a time."""
    start_time = time.time()
    best_score = _score(points)
    best_points = list(points)
    
    if best_score == 0:
        return best_points
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit - 5:
            break
        
        improved = False
        for idx in range(30):
            if time.time() - start_time > time_limit - 5:
                break
            
            px, py = best_points[idx]
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = px + dx, py + dy
                    if nx < -30 or nx > 30 or ny < -30 or ny > 30:
                        continue
                    
                    new_point = (nx, ny)
                    if new_point in best_points:
                        continue
                    
                    new_points = list(best_points)
                    new_points[idx] = new_point
                    
                    s = _score(new_points)
                    if s > best_score:
                        best_score = s
                        best_points = new_points
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            # Try larger jumps
            for idx in range(30):
                if time.time() - start_time > time_limit - 5:
                    break
                px, py = best_points[idx]
                for dx in range(-8, 9):
                    for dy in range(-8, 9):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = px + dx, py + dy
                        if nx < -30 or nx > 30 or ny < -30 or ny > 30:
                            continue
                        new_point = (nx, ny)
                        if new_point in best_points:
                            continue
                        new_points = list(best_points)
                        new_points[idx] = new_point
                        s = _score(new_points)
                        if s > best_score:
                            best_score = s
                            best_points = new_points
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
    
    return best_points

def construct():
    """Construct 30 points maximizing the multiplicity of a single doubled area."""
    start_time = time.time()
    deadline = start_time + 55  # Leave 5s margin for the 60s limit
    
    candidates = _structured_candidates()
    
    # Score all candidates
    best_points = None
    best_score = 0
    for cand in candidates:
        s = _score(cand)
        if s > best_score:
            best_score = s
            best_points = list(cand)
    
    if best_points is None:
        # Fallback to seed
        return [(x, y) for x in range(6) for y in range(5)]
    
    # Run local search on the best candidate
    if time.time() < deadline - 10:
        best_points = _local_search(best_points, max_iterations=2000, time_limit=deadline - time.time())
    
    # If we have time, try local search on other top candidates
    scored_candidates = []
    for cand in candidates:
        s = _score(cand)
        scored_candidates.append((s, cand))
    scored_candidates.sort(key=lambda x: -x[0])
    
    for s, cand in scored_candidates[:5]:
        if time.time() > deadline - 10:
            break
        if s >= best_score:
            continue
        result = _local_search(cand, max_iterations=500, time_limit=min(deadline - time.time() - 5, 10))
        rs = _score(result)
        if rs > best_score:
            best_score = rs
            best_points = result
    
    return best_points
# EVOLVE-BLOCK-END
