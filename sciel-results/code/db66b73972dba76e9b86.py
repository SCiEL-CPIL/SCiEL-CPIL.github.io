import time
import random
import math
from collections import defaultdict
from math import gcd

def _count_ordinary_lines(points):
    """Count lines containing exactly 2 of the points using pair-count inversion."""
    n = len(points)
    if n < 3:
        return 0
    
    line_pair_count = {}
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            
            A = dy
            B = -dx
            C = dx * y1 - dy * x1
            
            g = gcd(abs(A), abs(B))
            if g == 0:
                continue
            A //= g
            B //= g
            C //= g
            
            if A < 0 or (A == 0 and B < 0):
                A = -A
                B = -B
                C = -C
            
            key = (A, B, C)
            line_pair_count[key] = line_pair_count.get(key, 0) + 1
    
    return sum(1 for c in line_pair_count.values() if c == 1)


def _are_all_collinear(points):
    if len(points) < 3:
        return True
    x0, y0 = points[0]
    x1, y1 = points[1]
    for i in range(2, len(points)):
        x2, y2 = points[i]
        if (x2 - x0) * (y1 - y0) != (x1 - x0) * (y2 - y0):
            return False
    return True


def _get_line_membership(points):
    """Return dict mapping line_key -> set of point indices on that line."""
    n = len(points)
    lines = defaultdict(set)
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            
            A = dy
            B = -dx
            C = dx * y1 - dy * x1
            
            g = gcd(abs(A), abs(B))
            if g == 0:
                continue
            A //= g
            B //= g
            C //= g
            
            if A < 0 or (A == 0 and B < 0):
                A = -A
                B = -B
                C = -C
            
            key = (A, B, C)
            lines[key].add(i)
            lines[key].add(j)
    
    return lines


def _local_search(points, max_iters=3000, deadline=None):
    """Local search with targeted moves to reduce ordinary lines."""
    n = len(points)
    best = [list(p) for p in points]
    best_score = _count_ordinary_lines(best)
    
    if deadline is not None and time.time() > deadline:
        return best, best_score
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Get current line membership to identify ordinary lines
        lines = _get_line_membership(best)
        ordinary_lines = [key for key, pts in lines.items() if len(pts) == 2]
        
        if not ordinary_lines:
            break
        
        # Strategy 1: Try to move a point on an ordinary line to create a new collinearity
        if random.random() < 0.6 and ordinary_lines:
            line_key = random.choice(ordinary_lines)
            p1_idx, p2_idx = list(lines[line_key])
            # Try moving one of the two points
            move_idx = random.choice([p1_idx, p2_idx])
            old_x, old_y = best[move_idx]
            
            # Try to find a position where the moved point creates a new collinearity
            # with two other points not on the original line
            other_points = [i for i in range(n) if i != move_idx and i != p1_idx and i != p2_idx]
            
            # Try random positions near existing points
            improved = False
            for _ in range(20):
                # Pick two other points and try to place move_idx on their line
                if len(other_points) >= 2:
                    i1, i2 = random.sample(other_points, 2)
                    x1, y1 = best[i1]
                    x2, y2 = best[i2]
                    dx = x2 - x1
                    dy = y2 - y1
                    
                    # Check if the line through i1,i2 already contains the original position
                    # If not, try moving to a point on this line
                    # Parametrize: (x1 + t*dx, y1 + t*dy) for integer t
                    # We need to find t such that the new point is distinct and in bounds
                    
                    # Try several t values
                    for t in range(-5, 6):
                        new_x = x1 + t * dx
                        new_y = y1 + t * dy
                        
                        if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                            continue
                        
                        # Check distinctness
                        new_points = [list(p) for p in best]
                        new_points[move_idx] = [new_x, new_y]
                        
                        seen = set()
                        valid = True
                        for p in new_points:
                            tp = (p[0], p[1])
                            if tp in seen:
                                valid = False
                                break
                            seen.add(tp)
                        if not valid:
                            continue
                        
                        if _are_all_collinear(new_points):
                            continue
                        
                        score = _count_ordinary_lines(new_points)
                        if score < best_score:
                            best = new_points
                            best_score = score
                            improved = True
                            break
                    if improved:
                        break
            
            if improved:
                continue
        
        # Strategy 2: Random perturbation
        idx = random.randint(0, n - 1)
        old_x, old_y = best[idx]
        step = random.choice([1, 2, 3, 5, 10])
        new_x = old_x + random.randint(-step, step)
        new_y = old_y + random.randint(-step, step)
        
        if abs(new_x) > 10**9 or abs(new_y) > 10**9:
            continue
        
        new_points = [list(p) for p in best]
        new_points[idx] = [new_x, new_y]
        
        seen = set()
        valid = True
        for p in new_points:
            tp = (p[0], p[1])
            if tp in seen:
                valid = False
                break
            seen.add(tp)
        if not valid:
            continue
        
        if _are_all_collinear(new_points):
            continue
        
        score = _count_ordinary_lines(new_points)
        if score < best_score:
            best = new_points
            best_score = score
    
    return best, best_score


def _generate_structured_candidates():
    """Generate diverse structured candidates with high collinearity."""
    candidates = []
    
    # 1. 3x4 grid
    candidates.append([[x, y] for x in range(4) for y in range(3)])
    
    # 2. 4x3 grid
    candidates.append([[x, y] for x in range(3) for y in range(4)])
    
    # 3. 2x6 grid
    candidates.append([[x, y] for x in range(6) for y in range(2)])
    
    # 4. 6x2 grid
    candidates.append([[x, y] for x in range(2) for y in range(6)])
    
    # 5. Near-pencil: 11 on a line + 1 off
    pts = [[i, 0] for i in range(11)] + [[0, 1]]
    candidates.append(pts)
    
    # 6. Two parallel lines of 6
    pts = [[i, 0] for i in range(6)] + [[i, 2] for i in range(6)]
    candidates.append(pts)
    
    # 7. Three parallel lines of 4
    pts = [[i, 0] for i in range(4)] + [[i, 2] for i in range(4)] + [[i, 4] for i in range(4)]
    candidates.append(pts)
    
    # 8. 5+4+3 parallel lines
    pts = [[i, 0] for i in range(5)] + [[i, 2] for i in range(4)] + [[i, 4] for i in range(3)]
    candidates.append(pts)
    
    # 9. Diagonal-shifted grid
    pts = [[i, j + i] for i in range(4) for j in range(3)]
    candidates.append(pts)
    
    # 10. Parabolic
    pts = [[i, i * i] for i in range(12)]
    candidates.append(pts)
    
    # 11. Star: 4 lines of 3 through origin
    pts = []
    for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
        for t in [-1, 1, 2]:
            p = [t * dx, t * dy]
            if p not in pts:
                pts.append(p)
    if len(pts) == 12 and not _are_all_collinear(pts):
        candidates.append(pts)
    
    # 12. Cross: horizontal + vertical + diagonal
    pts = [[i, 0] for i in range(-3, 3)] + [[0, i] for i in range(-3, 3) if i != 0] + [[3, 3]]
    if len(pts) == 12:
        candidates.append(pts)
    
    # 13. Projective grid: 3 lines through P1, 3 through P2, plus P1,P2,P3
    # P1=(0,0), P2=(60,0), P3=(120,0)
    P1 = [0, 0]
    P2 = [60, 0]
    P3 = [120, 0]
    pts = [P1, P2, P3]
    slopes1 = [1, 2, 3]
    slopes2 = [-1, -2, -3]
    for m1 in slopes1:
        for m2 in slopes2:
            dx = m1 - m2
            if dx == 0:
                continue
            num = 60 * m2
            den = m2 - m1
            if num % den == 0:
                x = num // den
                y = m1 * x
                pts.append([x, y])
    if len(pts) == 12 and not _are_all_collinear(pts):
        candidates.append(pts)
    
    # 14. Random small grid points
    random.seed(42)
    for trial in range(30):
        pts = []
        for _ in range(12):
            x = random.randint(-4, 4)
            y = random.randint(-4, 4)
            if [x, y] not in pts:
                pts.append([x, y])
        while len(pts) < 12:
            x = random.randint(-8, 8)
            y = random.randint(-8, 8)
            if [x, y] not in pts:
                pts.append([x, y])
        if len(pts) == 12 and not _are_all_collinear(pts):
            candidates.append(pts)
    
    # 15. Random with collinear bias
    random.seed(123)
    for trial in range(20):
        dx = random.randint(-3, 3)
        dy = random.randint(-3, 3)
        if dx == 0 and dy == 0:
            dx, dy = 1, 0
        bx, by = random.randint(-3, 3), random.randint(-3, 3)
        k_on_line = random.randint(5, 7)
        pts = []
        for t in range(k_on_line):
            p = [bx + t * dx, by + t * dy]
            if p not in pts:
                pts.append(p)
        while len(pts) < 12:
            x = random.randint(-8, 8)
            y = random.randint(-8, 8)
            if [x, y] not in pts:
                pts.append([x, y])
        if len(pts) == 12 and not _are_all_collinear(pts):
            candidates.append(pts)
    
    return candidates


def _try_construction():
    """Try various constructions and return the best configuration."""
    candidates = _generate_structured_candidates()
    
    best_points = None
    best_score = float('inf')
    
    deadline = time.time() + 140  # Leave margin for the 150s limit
    
    # Score all candidates first
    scored = []
    for cand in candidates:
        s = _count_ordinary_lines(cand)
        scored.append((s, cand))
    scored.sort(key=lambda x: x[0])
    
    # Allocate time: top candidates get more iterations
    n_cands = len(scored)
    for rank, (init_score, cand) in enumerate(scored):
        if time.time() > deadline:
            break
        
        if rank < 3:
            max_iters = 3000
        elif rank < 8:
            max_iters = 1500
        else:
            max_iters = 500
        
        optimized, score = _local_search(cand, max_iters=max_iters, deadline=deadline)
        
        if score < best_score:
            best_score = score
            best_points = optimized
        
        if best_score <= 6:
            break
    
    if best_points is None:
        best_points = [[x, y] for x in range(4) for y in range(3)]
        best_score = _count_ordinary_lines(best_points)
    
    return best_points, best_score


def construct():
    """Construct 12 integer points minimizing ordinary lines."""
    points, score = _try_construction()
    
    # Ensure validity
    assert len(points) == 12
    assert len(set(tuple(p) for p in points)) == 12
    for p in points:
        assert abs(p[0]) <= 10**9
        assert abs(p[1]) <= 10**9
    assert not _are_all_collinear(points)
    
    return points
