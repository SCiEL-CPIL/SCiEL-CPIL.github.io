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

def _count_ordinary(points):
    """Count ordinary lines."""
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
    # Check not all collinear
    (x1, y1), (x2, y2), (x3, y3) = points[0], points[1], points[2]
    if (x2 - x1) * (y3 - y1) != (y2 - y1) * (x3 - x1):
        return True
    for i in range(3, 12):
        (x4, y4) = points[i]
        if (x2 - x1) * (y4 - y1) != (y2 - y1) * (x4 - x1):
            return True
    return False

def _build_hesse_12():
    """
    Construct 12 points based on Hesse configuration extension.
    
    Hesse configuration: 9 points, 12 lines (4 lines with 3 points each in 3 parallel classes).
    We extend to 12 points by adding 3 more points strategically.
    
    Standard Hesse in affine plane:
    Points at (i, j) for i,j in {0,1,2} with i+j mod 3 = 0,1,2
    Actually, let's use a cleaner construction:
    
    3 parallel lines (horizontal): y=0, y=1, y=2
    3 parallel lines (vertical): x=0, x=1, x=2  
    3 parallel lines (diagonal): x+y=0, x+y=1, x+y=2 (mod 3 structure)
    
    The 9 points of Hesse are the 9 lattice points in [0,2]x[0,2].
    Each of the 12 lines (3 horizontal, 3 vertical, 3 with slope 1, 3 with slope -1)
    contains exactly 3 points.
    
    For 12 points, we add 3 more points. The key insight: add points that lie on 
    existing lines to avoid creating new ordinary lines.
    
    Let's try: take the 3x3 grid and add 3 points at infinity-like positions
    but in integer coordinates. Or better: use a 4x3 grid with careful placement.
    
    Alternative: Use the fact that for n=12, the minimum is 6 ordinary lines.
    This suggests 6 lines with 2 points, and the rest with 3+ points.
    
    Let's construct: 4 lines with 3 points each (12 point-line incidences from these),
    plus 6 lines with 2 points each. Total incidences: 4*3 + 6*2 = 24.
    Each point has degree: 24/12 = 2 on average, but some will have higher.
    
    Actually, let me try a specific known configuration:
    Take points on a conic and its dual structure.
    
    Simpler approach: Start with 3x4 grid and verify, then optimize.
    """
    # 3x4 grid: 12 points
    pts = [[i, j] for i in range(4) for j in range(3)]
    return pts

def _try_structured_configs():
    """Try several structured configurations and return the best."""
    candidates = []
    
    # 1. 3x4 grid
    candidates.append([[i, j] for i in range(4) for j in range(3)])
    
    # 2. 4x3 grid (transposed)
    candidates.append([[i, j] for i in range(3) for j in range(4)])
    
    # 3. Hesse-like: 3x3 grid plus 3 extra points on existing lines
    h9 = [[i, j] for i in range(3) for j in range(3)]
    # Add points that extend existing lines
    # Horizontal lines y=0,1,2 each have 3 points. Add (3,0), (3,1), (3,2)?
    # That makes vertical line x=3 with 3 points, and extends horizontals to 4 points.
    candidates.append(h9 + [[3, 0], [3, 1], [3, 2]])
    
    # 4. Another extension: add points at (0,3), (1,3), (2,3)
    candidates.append(h9 + [[0, 3], [1, 3], [2, 3]])
    
    # 5. Diagonal extension: add (3,0), (2,3), (0,3) to make triangle
    candidates.append(h9 + [[3, 0], [2, 3], [0, 3]])
    
    # 6. Try 4 points on each of 3 parallel lines
    # y=0: (0,0),(1,0),(2,0),(3,0)
    # y=1: (0,1),(1,1),(2,1),(3,1)  
    # y=2: (0,2),(1,2),(2,2),(3,2)
    # That's just 3x4 grid again.
    
    # 7. Try a configuration with 3 lines of 4 points each
    # Line 1 (y=0): (0,0),(1,0),(2,0),(3,0)
    # Line 2 (y=1): (0,1),(1,1),(2,1),(3,1)
    # Line 3 (y=2): (0,2),(1,2),(2,2),(3,2)
    # Same as 3x4.
    
    # 8. Try mixing: 2 lines of 4, 2 lines of 2, rest scattered
    # Let's try: 
    # y=0: (0,0),(2,0),(4,0),(6,0) - 4 points
    # y=1: (1,1),(3,1),(5,1),(7,1) - 4 points
    # x=0: (0,0),(0,1),(0,2) - but (0,0) already used, (0,1) not in set
    # This is getting complex. Let me just evaluate the candidates above.
    
    # 9. Try: 3x3 grid plus (3,3), (4,0), (0,4)
    candidates.append(h9 + [[3, 3], [4, 0], [0, 4]])
    
    # 10. Try: 3x3 grid plus (3,0), (0,3), (3,3)
    candidates.append(h9 + [[3, 0], [0, 3], [3, 3]])
    
    best_pts = None
    best_score = float('inf')
    
    for pts in candidates:
        if not _is_valid(pts):
            continue
        score = _count_ordinary(pts)
        if score < best_score:
            best_score = score
            best_pts = [list(p) for p in pts]
    
    return best_pts, best_score

def _local_search_fast(points, deadline):
    """Fast local search with incremental line counting."""
    current = [tuple(p) for p in points]
    n = len(current)
    
    # Build initial line counts
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(current[i], current[j])
            if key is not None:
                line_counts[key] = line_counts.get(key, 0) + 1
    ordinary_count = sum(1 for v in line_counts.values() if v == 1)
    
    best_score = ordinary_count
    best_points = [list(p) for p in current]
    
    # Generate candidate positions
    coords = set()
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = current[i]
            x2, y2 = current[j]
            dx = x2 - x1
            dy = y2 - y1
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx //= g
            dy //= g
            for t in range(-3, 4):
                coords.add((x1 + t * dx, y1 + t * dy))
    
    for p in current:
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                coords.add((p[0] + dx, p[1] + dy))
    
    candidate_list = list(coords)
    random.shuffle(candidate_list)
    
    no_improve = 0
    while time.time() < deadline and no_improve < 100:
        improved = False
        random.shuffle(candidate_list)
        
        for i in range(n):
            if time.time() >= deadline:
                break
            old_pt = current[i]
            for cand in candidate_list:
                if cand == old_pt:
                    continue
                if any(cand == current[j] for j in range(n) if j != i):
                    continue
                
                # Compute delta in ordinary count
                delta = 0
                
                # Remove lines through old_pt
                for j in range(n):
                    if j == i:
                        continue
                    key = _line_key(old_pt, current[j])
                    if key is not None:
                        c = line_counts[key]
                        if c == 2:
                            delta -= 1
                        elif c == 3:
                            delta += 1
                        line_counts[key] = c - 1
                        if line_counts[key] == 0:
                            del line_counts[key]
                
                # Add lines through new_pt
                for j in range(n):
                    if j == i:
                        continue
                    key = _line_key(cand, current[j])
                    if key is not None:
                        c = line_counts.get(key, 0)
                        if c == 1:
                            delta += 1
                        elif c == 2:
                            delta -= 1
                        line_counts[key] = c + 1
                
                new_ordinary = ordinary_count + delta
                
                if new_ordinary < ordinary_count:
                    current[i] = cand
                    ordinary_count = new_ordinary
                    improved = True
                    no_improve = 0
                    if new_ordinary < best_score:
                        best_score = new_ordinary
                        best_points = [list(p) for p in current]
                    break
                else:
                    # Undo: remove new lines, add old lines back
                    for j in range(n):
                        if j == i:
                            continue
                        key = _line_key(cand, current[j])
                        if key is not None:
                            c = line_counts[key]
                            line_counts[key] = c - 1
                            if line_counts[key] == 0:
                                del line_counts[key]
                    
                    for j in range(n):
                        if j == i:
                            continue
                        key = _line_key(old_pt, current[j])
                        if key is not None:
                            line_counts[key] = line_counts.get(key, 0) + 1
                    
                    # Continue trying other candidates
            else:
                no_improve += 1
                continue
            break
        else:
            no_improve += 1
            continue
        
        if not improved:
            # Random kick
            idx = random.randint(0, n - 1)
            cand = random.choice(candidate_list)
            if cand != current[idx] and not any(cand == current[j] for j in range(n) if j != idx):
                old_pt = current[idx]
                # Remove old lines
                for j in range(n):
                    if j == idx:
                        continue
                    key = _line_key(old_pt, current[j])
                    if key is not None:
                        c = line_counts[key]
                        if c == 2:
                            ordinary_count -= 1
                        elif c == 3:
                            ordinary_count += 1
                        line_counts[key] = c - 1
                        if line_counts[key] == 0:
                            del line_counts[key]
                
                # Add new lines
                current[idx] = cand
                for j in range(n):
                    if j == idx:
                        continue
                    key = _line_key(cand, current[j])
                    if key is not None:
                        c = line_counts.get(key, 0)
                        if c == 1:
                            ordinary_count += 1
                        elif c == 2:
                            ordinary_count -= 1
                        line_counts[key] = c + 1
                
                no_improve = 0
    
    return best_points, best_score

def construct():
    deadline = time.time() + 135  # Safety margin
    
    # Step 1: Try structured configurations
    best_points, best_score = _try_structured_configs()
    
    # Step 2: Local search from best structured config
    if time.time() < deadline - 15:
        ls_points, ls_score = _local_search_fast(best_points, deadline - 15)
        if ls_score < best_score:
            best_score = ls_score
            best_points = ls_points
    
    # Step 3: Multiple restarts with perturbation
    random.seed(42)
    for _ in range(10):
        if time.time() > deadline - 20:
            break
        
        # Perturb best solution
        perturbed = [list(p) for p in best_points]
        for _ in range(3):
            i = random.randint(0, 11)
            dx = random.randint(-5, 5)
            dy = random.randint(-5, 5)
            new_p = [perturbed[i][0] + dx, perturbed[i][1] + dy]
            if not any(new_p == p for p in perturbed if p is not perturbed[i]):
                perturbed[i] = new_p
        
        if not _is_valid(perturbed):
            continue
        
        s = _count_ordinary(perturbed)
        if s < best_score:
            best_score = s
            best_points = perturbed
        
        if time.time() < deadline - 15:
            ls_points, ls_score = _local_search_fast(perturbed, deadline - 15)
            if ls_score < best_score:
                best_score = ls_score
                best_points = ls_points
    
    # Final validation
    if not _is_valid(best_points):
        # Fallback to 3x4 grid
        best_points = [[i, j] for i in range(4) for j in range(3)]
    
    return [[int(p[0]), int(p[1])] for p in best_points]

# EVOLVE-BLOCK-END
