"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _line_key(p1, p2):
    """Return canonical line key for two points."""
    x1, y1 = p1
    x2, y2 = p2
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    if a == 0 and b == 0:
        return None
    g = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
    if g == 0:
        g = 1
    a, b, c = a // g, b // g, c // g
    if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
        a, b, c = -a, -b, -c
    return (a, b, c)

def _count_ordinary_fast(points):
    """Count ordinary lines using a dict of line->count."""
    n = len(points)
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is not None:
                line_counts[key] = line_counts.get(key, 0) + 1
    return sum(1 for v in line_counts.values() if v == 1)

def _is_valid(points):
    """Check: 12 distinct points, not all collinear."""
    if len(points) != 12:
        return False
    if len(set(tuple(p) for p in points)) != 12:
        return False
    (x1, y1), (x2, y2), (x3, y3) = points[0], points[1], points[2]
    if (x2 - x1) * (y3 - y1) != (y2 - y1) * (x3 - x1):
        return True
    for i in range(3, 12):
        (x4, y4) = points[i]
        if (x2 - x1) * (y4 - y1) != (y2 - y1) * (x4 - x1):
            return True
    return False

def _try_configurations():
    """Try several known structured configurations."""
    configs = []
    
    # 4+4+4 on three lines: y=0, y=1, y=x
    c1 = [[0,0],[1,0],[2,0],[3,0],
          [0,1],[1,1],[2,1],[3,1],
          [4,4],[5,5],[6,6],[7,7]]
    configs.append(c1)
    
    # 4+4+4 on three lines: y=0, y=1, y=2x
    c2 = [[0,0],[1,0],[2,0],[3,0],
          [0,1],[1,1],[2,1],[3,1],
          [0,0],[1,2],[2,4],[3,6]]  # wait, (0,0) duplicate
    c2 = [[0,0],[1,0],[2,0],[3,0],
          [0,1],[1,1],[2,1],[3,1],
          [1,2],[2,4],[3,6],[4,8]]
    configs.append(c2)
    
    # 3+3+3+3 on four lines
    c3 = [[0,0],[1,0],[2,0],
          [0,1],[1,1],[2,1],
          [0,2],[1,2],[2,2],
          [3,0],[4,1],[5,2]]
    configs.append(c3)
    
    # 6+6 on two parallel lines
    c4 = [[0,0],[1,0],[2,0],[3,0],[4,0],[5,0],
          [0,1],[1,1],[2,1],[3,1],[4,1],[5,1]]
    configs.append(c4)
    
    # 6+6 on two non-parallel lines
    c5 = [[0,0],[1,0],[2,0],[3,0],[4,0],[5,0],
          [1,1],[2,2],[3,3],[4,4],[5,5],[6,6]]
    configs.append(c5)
    
    # 4+4+4 with different slopes
    c6 = [[0,0],[1,0],[2,0],[3,0],
          [0,1],[1,1],[2,1],[3,1],
          [0,2],[1,4],[2,6],[3,8]]
    configs.append(c6)
    
    # Try to get close to Hesse-like configuration
    # 12 points: 3 groups of 4 on lines with specific intersection properties
    c7 = [[0,0],[2,0],[4,0],[6,0],
          [1,0],[3,0],[5,0],[7,0],  # wait, all on y=0
          ]
    # Let me redo: 4 on y=0, 4 on x=0, 4 on x+y=4
    c7 = [[1,0],[2,0],[3,0],[4,0],
          [0,1],[0,2],[0,3],[0,4],
          [0,4],[1,3],[2,2],[3,1]]  # (0,4) duplicate
    c7 = [[1,0],[2,0],[3,0],[4,0],
          [0,1],[0,2],[0,3],[0,4],
          [1,3],[2,2],[3,1],[4,0]]  # (4,0) duplicate
    c7 = [[1,0],[2,0],[3,0],[4,0],
          [0,1],[0,2],[0,3],[0,4],
          [2,2],[3,1],[4,0],[5,-1]]  # (4,0) still duplicate
    # Let me just use distinct points
    c7 = [[1,0],[2,0],[3,0],[4,0],
          [0,1],[0,2],[0,3],[0,4],
          [1,3],[2,2],[3,1],[5,0]]
    configs.append(c7)
    
    # 4+4+4 on lines y=0, y=x, y=-x+8
    c8 = [[0,0],[1,0],[2,0],[3,0],
          [4,4],[5,5],[6,6],[7,7],
          [8,0],[7,1],[6,2],[5,3]]
    configs.append(c8)
    
    # 4+4+4 on lines y=0, y=1, y=-x+4
    c9 = [[0,0],[1,0],[2,0],[3,0],
          [0,1],[1,1],[2,1],[3,1],
          [4,0],[3,1],[2,2],[1,3]]  # (3,1) duplicate
    c9 = [[0,0],[1,0],[2,0],[3,0],
          [0,1],[1,1],[2,1],[3,1],
          [4,0],[5,-1],[6,-2],[7,-3]]
    configs.append(c9)
    
    best_points = None
    best_score = float('inf')
    
    for cfg in configs:
        if not _is_valid(cfg):
            continue
        s = _count_ordinary_fast(cfg)
        if s < best_score:
            best_score = s
            best_points = [list(p) for p in cfg]
    
    return best_points, best_score

def _local_search(points, deadline):
    """Try single-point replacement moves to reduce ordinary line count."""
    current = [list(p) for p in points]
    current_score = _count_ordinary_fast(current)
    best = [list(p) for p in current]
    best_score = current_score
    
    # Generate candidate pool
    coords = set()
    for p in current:
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                coords.add((p[0] + dx, p[1] + dy))
    
    # Add structured candidates
    for x in range(-5, 15):
        for y in range(-5, 15):
            coords.add((x, y))
    
    candidate_list = list(coords)
    random.shuffle(candidate_list)
    
    no_improve = 0
    max_no_improve = 50
    
    while time.time() < deadline and no_improve < max_no_improve:
        improved = False
        random.shuffle(candidate_list)
        
        for i in range(12):
            if time.time() >= deadline:
                break
            old_pt = current[i]
            for cand in candidate_list:
                if tuple(cand) == tuple(old_pt):
                    continue
                if any(tuple(cand) == tuple(current[j]) for j in range(12) if j != i):
                    continue
                
                new_pts = [list(p) for p in current]
                new_pts[i] = list(cand)
                
                if not _is_valid(new_pts):
                    continue
                
                score = _count_ordinary_fast(new_pts)
                if score < current_score:
                    current = new_pts
                    current_score = score
                    improved = True
                    no_improve = 0
                    if current_score < best_score:
                        best_score = current_score
                        best = [list(p) for p in current]
                    break
            else:
                no_improve += 1
                continue
            break
        else:
            no_improve += 1
            continue
        
        if not improved:
            no_improve += 1
    
    return best, best_score

def construct():
    """Return 12 integer points minimizing ordinary lines."""
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Start with best known configuration
    best_points, best_score = _try_configurations()
    
    # Local search from best configuration
    if time.time() < deadline - 10:
        ls_points, ls_score = _local_search(best_points, deadline)
        if ls_score < best_score:
            best_score = ls_score
            best_points = ls_points
    
    # If still above 6, try more random restarts
    if best_score > 6 and time.time() < deadline - 15:
        random.seed(42)
        for restart in range(50):
            if time.time() >= deadline - 10:
                break
            
            # Generate random structured configuration
            pts = []
            # Choose 3 lines with 4 points each
            # Line 1: y = 0
            x1s = random.sample(range(-10, 10), 4)
            for x in x1s:
                pts.append([x, 0])
            
            # Line 2: y = x + c
            c2 = random.randint(-5, 5)
            x2s = random.sample(range(-10, 10), 4)
            for x in x2s:
                pt = [x, x + c2]
                if all(tuple(pt) != tuple(p) for p in pts):
                    pts.append(pt)
                else:
                    # Try another x
                    for x2 in range(-15, 15):
                        pt = [x2, x2 + c2]
                        if all(tuple(pt) != tuple(p) for p in pts):
                            pts.append(pt)
                            break
            
            # Line 3: y = mx + b
            m3 = random.randint(-3, 3)
            if m3 == 0:
                m3 = 1
            b3 = random.randint(-5, 5)
            x3s = random.sample(range(-10, 10), 4)
            for x in x3s:
                pt = [x, m3 * x + b3]
                if all(tuple(pt) != tuple(p) for p in pts):
                    pts.append(pt)
                else:
                    for x3 in range(-15, 15):
                        pt = [x3, m3 * x3 + b3]
                        if all(tuple(pt) != tuple(p) for p in pts):
                            pts.append(pt)
                            break
            
            if len(pts) != 12 or not _is_valid(pts):
                continue
            
            s = _count_ordinary_fast(pts)
            if s < best_score:
                best_score = s
                best_points = [list(p) for p in pts]
            
            # Local search
            if time.time() < deadline - 10:
                ls_points, ls_score = _local_search(pts, deadline)
                if ls_score < best_score:
                    best_score = ls_score
                    best_points = ls_points
    
    return [[int(p[0]), int(p[1])] for p in best_points]

# EVOLVE-BLOCK-END
