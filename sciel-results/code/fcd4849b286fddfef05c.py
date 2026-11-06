"""
Find 20 integer points with no 4 collinear, minimizing the largest no-3-collinear subset.
Uses systematic construction + exact branch-and-bound scoring.
"""
import time
import random
from itertools import combinations

_START_TIME = time.time()
_TIME_LIMIT = 145  # safety margin before 150s limit


def _elapsed():
    return time.time() - _START_TIME


def _timeout():
    return _elapsed() > _TIME_LIMIT


def _has_4_collinear(points):
    """Check if any 4 points are collinear."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            count = 2
            for k in range(n):
                if k == i or k == j:
                    continue
                x3, y3 = points[k]
                if (x3 - x1) * dy - (y3 - y1) * dx == 0:
                    count += 1
            if count >= 4:
                return True
    return False


def _valid(points):
    """Check validity: 20 distinct integer points, |x|,|y| <= 100, no 4 collinear."""
    if len(points) != 20:
        return False
    if len(set(tuple(p) for p in points)) != 20:
        return False
    for x, y in points:
        if not isinstance(x, int) or not isinstance(y, int):
            return False
        if abs(x) > 100 or abs(y) > 100:
            return False
    if _has_4_collinear(points):
        return False
    return True


def _compute_lines(points):
    """
    Compute all lines containing >= 3 points.
    Returns list of frozensets of point indices on each line.
    """
    n = len(points)
    lines = []
    seen_lines = set()
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            
            # Find all points on this line
            on_line = {i, j}
            for k in range(n):
                if k == i or k == j:
                    continue
                x3, y3 = points[k]
                if (x3 - x1) * dy - (y3 - y1) * dx == 0:
                    on_line.add(k)
            
            if len(on_line) >= 3:
                # Normalize line representation for dedup
                # Use the line equation or sorted indices
                key = frozenset(on_line)
                if key not in seen_lines:
                    seen_lines.add(key)
                    lines.append(key)
    
    return lines


def _largest_no_3_collinear(points):
    """
    Exact branch and bound: find largest subset with no 3 collinear.
    A subset is valid if no line contains 3+ points from the subset.
    """
    n = len(points)
    lines = _compute_lines(points)
    
    if not lines:
        return n
    
    # For each point, compute which lines it's on
    point_lines = [[] for _ in range(n)]
    for line in lines:
        for p in line:
            point_lines[p].append(line)
    
    best = [0]
    nodes_visited = [0]
    
    def search(subset_mask, remaining_mask):
        if _timeout():
            return
        
        nodes_visited[0] += 1
        
        current_size = bin(subset_mask).count('1')
        if current_size > best[0]:
            best[0] = current_size
        
        if remaining_mask == 0:
            return
        
        # Upper bound: current + remaining
        rem_count = bin(remaining_mask).count('1')
        if current_size + rem_count <= best[0]:
            return
        
        # Stronger upper bound: for each line, at most 2 points can be selected
        # Compute a better bound using the lines
        # For each point in remaining, count how many lines it shares with subset
        # A point can be added if for every line containing it, the subset has < 2 points on that line
        
        # Greedy upper bound: try to add as many as possible
        # For each candidate, check if it can be added
        # Upper bound = current_size + number of addable points (optimistic)
        
        candidates = []
        mask = remaining_mask
        while mask:
            bit = mask & (-mask)
            idx = bit.bit_length() - 1
            candidates.append(idx)
            mask ^= bit
        
        # Check which candidates are actually addable
        addable = 0
        for idx in candidates:
            # Check if idx can be added to subset
            can_add = True
            for line in point_lines[idx]:
                # Count how many points of subset are on this line
                count = 0
                for p in line:
                    if p != idx and (subset_mask >> p) & 1:
                        count += 1
                if count >= 2:
                    can_add = False
                    break
            if can_add:
                addable += 1
        
        if current_size + addable <= best[0]:
            return
        
        # Try adding each candidate
        # Sort by number of conflicts (fewer first for better pruning)
        def conflict_score(idx):
            score = 0
            for line in point_lines[idx]:
                for p in line:
                    if p != idx and (remaining_mask >> p) & 1:
                        score += 1
            return score
        
        candidates.sort(key=conflict_score)
        
        for idx in candidates:
            if _timeout():
                return
            
            # Check if idx can be added
            can_add = True
            for line in point_lines[idx]:
                count = 0
                for p in line:
                    if p != idx and (subset_mask >> p) & 1:
                        count += 1
                if count >= 2:
                    can_add = False
                    break
            
            if not can_add:
                continue
            
            new_subset = subset_mask | (1 << idx)
            new_remaining = remaining_mask & ~(1 << idx)
            search(new_subset, new_remaining)
    
    search(0, (1 << n) - 1)
    return best[0]


def _generate_line_based_config():
    """
    Generate a configuration where points are placed on lines of exactly 3 points.
    This maximizes collinear triples, making no-3-collinear subsets smaller.
    
    Strategy: Use a set of lines, each with exactly 3 points, covering all 20 points.
    Points can be on multiple lines (at intersections).
    """
    # Use a structured approach:
    # Place points on a small grid and select lines through them
    
    # Approach: Use 7+ lines of 3 points each
    # Let's try: create points at intersections of lines
    
    # Method: Pick 7 lines in the plane, each passing through 3 of our points.
    # Points are at intersections of pairs of lines.
    
    # Simpler: manually construct
    # Use horizontal, vertical, and diagonal lines
    
    # Let's try a 5x4 grid subset with deliberate collinearities
    # Actually, let's use a parametric approach
    
    # Generate points and then verify
    # Start with a base set and add collinear triples
    
    # Configuration: use lines y=0, y=1, y=2, y=3, y=4, and diagonals
    # Place 3 points on each of several lines
    
    # Let me try: 
    # Line y=0: (-6,0), (0,0), (6,0)
    # Line y=2: (-4,2), (0,2), (4,2)  
    # Line y=4: (-2,4), (0,4), (2,4)
    # Line x=0: (0,0), (0,2), (0,4) - but (0,0),(0,2),(0,4) already have 3, need exactly 3
    # Wait, x=0 has (0,0),(0,2),(0,4) = 3 points, good
    # Line x=2: (2,4) and need 2 more... (2,0)? (2,2)?
    
    # Let me be more systematic. Use a known structure.
    
    # Try: 20 points arranged so that there are many collinear triples
    # Use points from a small set of lines
    
    # Configuration attempt:
    pts = [
        [-8, -4], [-4, -4], [0, -4],  # y=-4
        [-6, -2], [-2, -2], [2, -2],   # y=-2
        [-8, 0], [-4, 0], [0, 0],      # y=0
        [-6, 2], [-2, 2], [2, 2],      # y=2
        [-4, 4], [0, 4], [4, 4],       # y=4
        [-8, 6], [-4, 6],              # y=6 (only 2, need to fix)
    ]
    # That's 17 points. Need 3 more.
    # Add points that create more collinear triples
    
    # Let me try a different approach: use a known optimal-ish structure
    
    # Use points on a 4x5 grid pattern with specific selections
    # to maximize collinear triples
    
    # Alternative: use the "no 4 collinear" constraint to our advantage
    # Place points so that many triples are collinear
    
    # Let me try a more systematic generation
    candidates = []
    
    # Use lines with slope 0, 1, -1, and vertical
    # Generate many candidate configurations and pick the best
    
    for trial in range(200):
        random.seed(trial)
        
        # Generate 20 points
        # Start with some structure
        pts = []
        
        # Place points on several horizontal lines (3 each)
        num_h_lines = 4
        for i in range(num_h_lines):
            y = -10 + i * 7
            for j in range(3):
                x = -15 + j * 10 + random.randint(-2, 2)
                pts.append([x, y])
        
        # That's 12 points, need 8 more
        # Add points on diagonal lines
        for i in range(8):
            # Try to place on diagonals
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            pts.append([x, y])
        
        # Check validity
        if len(set(tuple(p) for p in pts)) == 20 and _valid(pts):
            candidates.append(pts)
    
    if not candidates:
        # Fallback: generate random valid config
        for _ in range(50000):
            if _timeout():
                break
            pts = [[random.randint(-50, 50), random.randint(-50, 50)] for _ in range(20)]
            if len(set(tuple(p) for p in pts)) == 20 and _valid(pts):
                candidates.append(pts)
    
    if not candidates:
        # Last resort: parabola
        return [[x, x * x] for x in range(-10, 10)]
    
    # Score each candidate and return the best
    best_pts = None
    best_score = 999
    for pts in candidates[:50]:  # Limit to avoid timeout
        if _timeout():
            break
        score = _largest_no_3_collinear(pts)
        if score < best_score:
            best_score = score
            best_pts = pts
    
    if best_pts is None:
        best_pts = candidates[0]
    
    return best_pts


def _local_search(points):
    """
    Local search: try to improve by moving individual points.
    Focus on creating more collinear triples.
    """
    current = [list(p) for p in points]
    current_score = _largest_no_3_collinear(current)
    best = [list(p) for p in current]
    best_score = current_score
    
    stagnation = 0
    
    while not _timeout():
        improved = False
        stagnation += 1
        
        # Try moving each point to create more collinearities
        for idx in range(20):
            if _timeout():
                return best, best_score
            
            old_pt = current[idx][:]
            
            # Try to place point on a line with 2 existing points
            # Find pairs of other points and place on their line
            for a in range(20):
                if a == idx:
                    continue
                for b in range(a + 1, 20):
                    if b == idx:
                        continue
                    
                    x1, y1 = current[a]
                    x2, y2 = current[b]
                    dx = x2 - x1
                    dy = y2 - y1
                    
                    # Try points on line through a and b
                    for t in range(-5, 6):
                        if t == 0:
                            continue
                        nx = x1 + t * dx
                        ny = y1 + t * dy
                        
                        if abs(nx) > 100 or abs(ny) > 100:
                            continue
                        
                        # Check distinctness
                        if any(p == [nx, ny] for i, p in enumerate(current) if i != idx):
                            continue
                        
                        current[idx] = [nx, ny]
                        
                        if not _valid(current):
                            current[idx] = old_pt
                            continue
                        
                        new_score = _largest_no_3_collinear(current)
                        
                        if new_score < best_score:
                            best_score = new_score
                            best = [list(p) for p in current]
                            improved = True
                        else:
                            current[idx] = old_pt
            
            if _timeout():
                return best, best_score
        
        # Also try random perturbations
        for _ in range(200):
            if _timeout():
                return best, best_score
            
            idx = random.randint(0, 19)
            old_pt = current[idx][:]
            new_pt = [random.randint(-50, 50), random.randint(-50, 50)]
            
            if any(p == new_pt for i, p in enumerate(current) if i != idx):
                continue
            
            current[idx] = new_pt
            
            if not _valid(current):
                current[idx] = old_pt
                continue
            
            new_score = _largest_no_3_collinear(current)
            
            if new_score < best_score:
                best_score = new_score
                best = [list(p) for p in current]
                improved = True
            else:
                current[idx] = old_pt
        
        if not improved and stagnation > 10:
            # Big perturbation: move multiple points
            for _ in range(50):
                if _timeout():
                    return best, best_score
                
                # Move 3 random points
                indices = random.sample(range(20), 3)
                old_pts = [current[i][:] for i in indices]
                
                for i in indices:
                    current[i] = [random.randint(-50, 50), random.randint(-50, 50)]
                
                # Check distinctness
                if len(set(tuple(p) for p in current)) != 20:
                    for i, old in zip(indices, old_pts):
                        current[i] = old
                    continue
                
                if not _valid(current):
                    for i, old in zip(indices, old_pts):
                        current[i] = old
                    continue
                
                new_score = _largest_no_3_collinear(current)
                
                if new_score < best_score:
                    best_score = new_score
                    best = [list(p) for p in current]
                    improved = True
                else:
                    for i, old in zip(indices, old_pts):
                        current[i] = old
        
        if not improved and stagnation > 30:
            break
    
    return best, best_score


def construct():
    """
    Return exactly 20 distinct integer points [x, y] with |x|, |y| <= 100 and no four collinear.
    Minimize the largest subset with no three collinear.
    """
    random.seed(42)
    
    # Generate initial configuration
    points = _generate_line_based_config()
    
    if not _valid(points):
        # Fallback
        for _ in range(100000):
            if _timeout():
                break
            pts = [[random.randint(-50, 50), random.randint(-50, 50)] for _ in range(20)]
            if len(set(tuple(p) for p in pts)) == 20 and _valid(pts):
                points = pts
                break
        
        if not _valid(points):
            points = [[x, x * x] for x in range(-10, 10)]
    
    # Local search to improve
    best_points, best_score = _local_search(points)
    
    # Final validation
    if not _valid(best_points):
        best_points = points
    
    return best_points

