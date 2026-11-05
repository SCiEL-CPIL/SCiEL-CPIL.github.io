import random
import time
from math import gcd

def _get_line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    A = y1 - y2
    B = x2 - x1
    C = -(A * x1 + B * y1)
    
    g = gcd(abs(A), abs(B))
    if g == 0:
        g = abs(C)
    if g != 0:
        A //= g
        B //= g
        C //= g
        
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A = -A
        B = -B
        C = -C
        
    return (A, B, C)

def _evaluate(points):
    """Count lines with exactly 4 points, check for 5+. Returns (valid, score)."""
    n = len(points)
    line_points = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _get_line_key(points[i], points[j])
            if key not in line_points:
                line_points[key] = set()
            line_points[key].add(i)
            line_points[key].add(j)
    
    valid = True
    score = 0
    for key, idx_set in line_points.items():
        cnt = len(idx_set)
        if cnt >= 5:
            valid = False
            break
        elif cnt == 4:
            score += 1
    return valid, score

def _incremental_eval(points, idx, old_pt, new_pt):
    """
    Evaluate after moving point idx from old_pt to new_pt.
    Returns (valid, score) for the new configuration.
    """
    n = len(points)
    # Build line index for all points except idx, then add new_pt
    # Actually, let's just rebuild - it's fast enough for 16 points
    points[idx] = new_pt
    valid, score = _evaluate(points)
    points[idx] = old_pt
    return valid, score

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_points = None
    best_score = -1
    
    # Multiple diverse initial configurations
    def make_grid_4x4():
        return [[x, y] for x in range(4) for y in range(4)]
    
    def make_perturbed_grid(seed_offset):
        random.seed(seed_offset)
        pts = [[x, y] for x in range(4) for y in range(4)]
        for i in range(16):
            if random.random() < 0.3:
                pts[i][0] += random.randint(-2, 2)
                pts[i][1] += random.randint(-2, 2)
        # Ensure distinct
        seen = set()
        for p in pts:
            while (p[0], p[1]) in seen:
                p[1] += 1
            seen.add((p[0], p[1]))
        return pts
    
    def make_random_cluster():
        pts = []
        seen = set()
        while len(pts) < 16:
            x = random.randint(-5, 5)
            y = random.randint(-5, 5)
            if (x, y) not in seen:
                pts.append([x, y])
                seen.add((x, y))
        return pts
    
    def make_two_rows():
        # Two rows of 8 with some structure
        pts = []
        for i in range(8):
            pts.append([i, 0])
            pts.append([i, 3])
        return pts
    
    def make_l_shape():
        pts = []
        for i in range(4):
            for j in range(4):
                pts.append([i, j])
        # Perturb some
        for i in [0, 5, 10, 15]:
            pts[i][1] += 1
        return pts
    
    initial_configs = [
        make_grid_4x4,
        lambda: make_perturbed_grid(1),
        lambda: make_perturbed_grid(2),
        lambda: make_perturbed_grid(3),
        make_random_cluster,
        make_two_rows,
        make_l_shape,
        lambda: make_perturbed_grid(4),
        lambda: make_perturbed_grid(5),
        lambda: make_perturbed_grid(6),
    ]
    
    for trial in range(len(initial_configs)):
        if time.time() > deadline - 20:
            break
        
        points = initial_configs[trial]()
        
        # Validate initial config
        seen = set()
        valid_init = True
        for p in points:
            if abs(p[0]) > 1000 or abs(p[1]) > 1000:
                valid_init = False
                break
            if (p[0], p[1]) in seen:
                valid_init = False
                break
            seen.add((p[0], p[1]))
        
        if not valid_init:
            continue
        
        is_valid, current_score = _evaluate(points)
        if is_valid:
            if current_score > best_score:
                best_score = current_score
                best_points = [p[:] for p in points]
        
        # Simulated annealing with better parameters
        T = 2.0
        cooling = 0.9995
        min_T = 0.005
        no_improve_count = 0
        local_best = current_score if is_valid else -1
        local_best_pts = [p[:] for p in points] if is_valid else None
        
        step = 0
        while time.time() < deadline - 10 and T > min_T:
            step += 1
            
            idx = random.randint(0, 15)
            old_x, old_y = points[idx]
            
            # Generate move with varying magnitudes
            r = random.random()
            if r < 0.4:
                dx = random.randint(-1, 1)
                dy = random.randint(-1, 1)
            elif r < 0.7:
                dx = random.randint(-3, 3)
                dy = random.randint(-3, 3)
            elif r < 0.9:
                dx = random.randint(-6, 6)
                dy = random.randint(-6, 6)
            else:
                dx = random.randint(-15, 15)
                dy = random.randint(-15, 15)
                
            new_x = old_x + dx
            new_y = old_y + dy
            
            if abs(new_x) > 1000 or abs(new_y) > 1000:
                continue
                
            new_point = [new_x, new_y]
            
            # Check distinctness
            is_distinct = True
            for i, p in enumerate(points):
                if i != idx:
                    if p[0] == new_x and p[1] == new_y:
                        is_distinct = False
                        break
            if not is_distinct:
                continue
                
            # Evaluate
            points[idx] = new_point
            is_valid, new_score = _evaluate(points)
            
            # Acceptance
            if is_valid:
                if new_score > current_score:
                    accept = True
                elif new_score == current_score:
                    accept = random.random() < 0.3
                else:
                    delta = new_score - current_score
                    accept_prob = min(1.0, max(0.0, __import__('math').exp(delta / T)))
                    accept = random.random() < accept_prob
            else:
                accept = False
                
            if accept:
                current_score = new_score
                if current_score > local_best:
                    local_best = current_score
                    local_best_pts = [p[:] for p in points]
                    no_improve_count = 0
                    if current_score > best_score:
                        best_score = current_score
                        best_points = [p[:] for p in points]
                else:
                    no_improve_count += 1
            else:
                points[idx] = [old_x, old_y]
                no_improve_count += 1
                
            T *= cooling
            
            # Restart if stuck
            if no_improve_count > 500:
                no_improve_count = 0
                if local_best_pts is not None:
                    points = [p[:] for p in local_best_pts]
                    current_score = local_best
                T = max(T, 0.5)
        
        # End of trial
        if best_score >= 15:
            break
    
    if best_points is None:
        best_points = [[x, y] for x in range(4) for y in range(4)]
        
    return best_points
