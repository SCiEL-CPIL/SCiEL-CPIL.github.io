import time
import random
import math
from collections import defaultdict

def _normalize_line(p1, p2):
    """Return a canonical key for the line through p1 and p2."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    
    if dx == 0:
        return (0, 1, x1)
    if dy == 0:
        return (1, 0, y1)
    
    # Normalize direction vector
    g = math.gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    
    # Ensure canonical direction (positive x, or if x==0 then positive y)
    if dx < 0:
        dx = -dx
        dy = -dy
    elif dx == 0 and dy < 0:
        dy = -dy
    
    # Line: a*x + b*y = c where (a,b) is normal vector perpendicular to (dx,dy)
    # Normal vector: (-dy, dx) or (dy, -dx)
    a = -dy
    b = dx
    c = a * x1 + b * y1
    
    # Normalize (a, b, c) so that gcd(a,b,c) divides all and sign is canonical
    g2 = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
    if g2 > 0:
        a //= g2
        b //= g2
        c //= g2
    
    # Canonical sign: first non-zero of (a, b) should be positive
    if a < 0 or (a == 0 and b < 0):
        a = -a
        b = -b
        c = -c
    
    return (a, b, c)

def _build_lines(all_points):
    """Precompute all lines and which points lie on them."""
    point_to_idx = {p: i for i, p in enumerate(all_points)}
    lines = defaultdict(set)
    
    for i in range(len(all_points)):
        for j in range(i + 1, len(all_points)):
            key = _normalize_line(all_points[i], all_points[j])
            lines[key].add(all_points[i])
            lines[key].add(all_points[j])
    
    # Build list of (line_key, frozenset_of_points) for lines with >= 2 points
    line_list = []
    for key, pts in lines.items():
        if len(pts) >= 2:
            line_list.append((key, frozenset(pts)))
    
    return line_list

def _score(S, line_list):
    """Count lines with exactly 4 points from S."""
    count = 0
    for _, pts in line_list:
        inter = len(pts & S)
        if inter == 4:
            count += 1
    return count

def _is_valid(S, line_list):
    """Check no 5 collinear."""
    for _, pts in line_list:
        if len(pts & S) >= 5:
            return False
    return True

def _delta_score(current_S, p_out, p_in, line_list):
    """
    Compute score change when removing p_in from S and adding p_out.
    Returns (delta, new_valid) where delta is change in 4-line count,
    and new_valid is whether the new set is valid.
    """
    new_S = set(current_S)
    new_S.remove(p_in)
    new_S.add(p_out)
    
    # Check validity
    for _, pts in line_list:
        if len(pts & new_S) >= 5:
            return 0, False
    
    # Compute delta in score
    delta = 0
    for _, pts in line_list:
        old_inter = len(pts & current_S)
        new_inter = len(pts & new_S)
        
        # Only lines affected by p_in or p_out
        if p_in in pts or p_out in pts:
            if old_inter == 4 and new_inter != 4:
                delta -= 1
            elif old_inter != 4 and new_inter == 4:
                delta += 1
    
    return delta, True

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Use 6x6 grid for more candidate points
    N = 6
    all_points = [(x, y) for x in range(N) for y in range(N)]
    
    # Precompute lines
    line_list = _build_lines(all_points)
    
    # Filter to only lines that could contribute (have >= 4 points in full grid)
    # Actually keep all, but score only counts those with exactly 4 in S
    
    best_score = -1
    best_S = None
    
    # Try multiple random restarts
    num_restarts = 50
    max_iterations = 5000
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Generate initial configuration
        # Mix of random and structured starts
        if restart % 3 == 0:
            # Random start
            current_S = set(random.sample(all_points, 16))
        elif restart % 3 == 1:
            # Structured: take rows 0,1,2 fully (18 pts) then remove 2 random
            base = [(x, y) for x in range(N) for y in range(3)]
            current_S = set(random.sample(base, 16))
        else:
            # Structured: checkerboard-like pattern
            candidates = [(x, y) for x in range(N) for y in range(N) if (x + y) % 2 == 0]
            if len(candidates) >= 16:
                current_S = set(random.sample(candidates, 16))
            else:
                current_S = set(random.sample(all_points, 16))
        
        # Ensure validity
        if not _is_valid(current_S, line_list):
            # Try to repair
            valid = False
            for _ in range(100):
                # Remove a point that causes 5-collinearity, add another
                bad_points = set()
                for _, pts in line_list:
                    if len(pts & current_S) >= 5:
                        bad_points.update(pts & current_S)
                
                if not bad_points:
                    valid = True
                    break
                
                # Remove one bad point
                p_remove = random.choice(list(bad_points))
                p_remove_idx = all_points.index(p_remove)
                current_S.remove(p_remove)
                
                # Add a random point not in S
                available = [p for p in all_points if p not in current_S]
                if available:
                    p_add = random.choice(available)
                    current_S.add(p_add)
            
            if not valid:
                continue
        
        current_score = _score(current_S, line_list)
        if current_score > best_score:
            best_score = current_score
            best_S = set(current_S)
        
        # Local search with delta evaluation
        iterations = 0
        while iterations < max_iterations and time.time() < deadline:
            iterations += 1
            
            # Pick random swap
            p_in = random.choice(list(current_S))
            available = [p for p in all_points if p not in current_S]
            if not available:
                break
            p_out = random.choice(available)
            
            delta, new_valid = _delta_score(current_S, p_in, p_out, line_list)
            
            if new_valid:
                # Accept if better, or with small probability if equal
                if delta > 0:
                    current_S.remove(p_in)
                    current_S.add(p_out)
                    current_score += delta
                    if current_score > best_score:
                        best_score = current_score
                        best_S = set(current_S)
                elif delta == 0 and random.random() < 0.1:
                    current_S.remove(p_in)
                    current_S.add(p_out)
            else:
                # Invalid, skip
                pass
        
        # Occasional restart from best
        if restart > 0 and restart % 5 == 0 and best_S:
            current_S = set(best_S)
            current_score = _score(current_S, line_list)

    if best_S is None:
        # Fallback: return a valid random configuration
        current_S = set(random.sample(all_points, 16))
        while not _is_valid(current_S, line_list):
            current_S = set(random.sample(all_points, 16))
        best_S = current_S
    
    return [[x, y] for x, y in best_S]
