import time
import random
import math

def _line_key(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if dx == 0 and dy == 0:
        return None
    g = math.gcd(abs(dx), abs(dy))
    dx_n, dy_n = dx // g, dy // g
    if dx_n < 0 or (dx_n == 0 and dy_n < 0):
        dx_n, dy_n = -dx_n, -dy_n
    a, b = dy_n, -dx_n
    c = a * p1[0] + b * p1[1]
    g2 = math.gcd(abs(a), abs(b))
    if g2 > 1:
        a, b, c = a // g2, b // g2, c // g2
    if a < 0 or (a == 0 and b < 0):
        a, b, c = -a, -b, -c
    return (a, b, c)

def _score_lines(points):
    n = len(points)
    line_pts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None: continue
            if key not in line_pts:
                line_pts[key] = set()
            line_pts[key].add(i)
            line_pts[key].add(j)
    
    score = 0
    valid = True
    for pts in line_pts.values():
        cnt = len(pts)
        if cnt == 4:
            score += 1
        elif cnt >= 5:
            valid = False
    return score, valid

def _get_candidates_on_line(line_key, existing_indices, points):
    a, b, c = line_key
    if not existing_indices:
        return []
    ref_idx = list(existing_indices)[0]
    x_ref, y_ref = points[ref_idx]
    
    dx_step = b
    dy_step = -a
    
    k_min = 10**9
    k_max = -10**9
    
    if dx_step > 0:
        k_min = max(k_min, math.ceil((-1000 - x_ref) / dx_step))
        k_max = min(k_max, math.floor((1000 - x_ref) / dx_step))
    elif dx_step < 0:
        k_min = max(k_min, math.ceil((1000 - x_ref) / dx_step))
        k_max = min(k_max, math.floor((-1000 - x_ref) / dx_step))
    else:
        if not (-1000 <= x_ref <= 1000):
            return []
        
    if dy_step > 0:
        k_min = max(k_min, math.ceil((-1000 - y_ref) / dy_step))
        k_max = min(k_max, math.floor((1000 - y_ref) / dy_step))
    elif dy_step < 0:
        k_min = max(k_min, math.ceil((1000 - y_ref) / dy_step))
        k_max = min(k_max, math.floor((-1000 - y_ref) / dy_step))
    else:
        if not (-1000 <= y_ref <= 1000):
            return []
            
    if k_min > k_max:
        return []
        
    existing_set = set(tuple(points[i]) for i in range(len(points)))
    candidates = []
    for k in range(k_min, k_max + 1):
        x = x_ref + k * dx_step
        y = y_ref + k * dy_step
        pt = (x, y)
        if pt not in existing_set:
            candidates.append(pt)
    return candidates

def _local_search(points, deadline, rng):
    current = [p[:] for p in points]
    current_score, h5 = _score_lines(current)
    if not h5:
        best = [p[:] for p in current]
        best_score = current_score
    else:
        best = [p[:] for p in current]
        best_score = -1
        
    T = 2.0
    T_min = 0.1
    cooling = 0.999
    
    no_improve = 0
    max_no_improve = 500
    
    while time.time() < deadline - 2 and no_improve < max_no_improve:
        move_type = rng.random()
        
        if move_type < 0.5:
            # Smart Move: Complete a 3-line
            n = len(current)
            line_pts = {}
            for i in range(n):
                for j in range(i + 1, n):
                    key = _line_key(current[i], current[j])
                    if key is None: continue
                    if key not in line_pts:
                        line_pts[key] = set()
                    line_pts[key].add(i)
                    line_pts[key].add(j)
            
            three_lines = [k for k, v in line_pts.items() if len(v) == 3]
            
            if three_lines:
                target_line = rng.choice(three_lines)
                pts_on_line = line_pts[target_line]
                candidates = _get_candidates_on_line(target_line, pts_on_line, current)
                
                if candidates:
                    new_pt = rng.choice(candidates)
                    replace_candidates = [i for i in range(n) if i not in pts_on_line]
                    if replace_candidates:
                        idx = rng.choice(replace_candidates)
                        old_pt = current[idx][:]
                        current[idx] = list(new_pt)
                        
                        new_score, new_h5 = _score_lines(current)
                        
                        if new_h5:
                            current[idx] = old_pt
                            no_improve += 1
                        else:
                            delta = new_score - current_score
                            if delta > 0 or rng.random() < math.exp(delta / T):
                                current_score = new_score
                                if current_score > best_score:
                                    best_score = current_score
                                    best = [p[:] for p in current]
                                    no_improve = 0
                                else:
                                    no_improve += 1
                            else:
                                current[idx] = old_pt
                                no_improve += 1
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
            else:
                no_improve += 1
                
        else:
            # Random Perturb
            idx = rng.randint(0, 15)
            old_pt = current[idx][:]
            step = rng.randint(1, 20)
            new_x = old_pt[0] + rng.randint(-step, step)
            new_y = old_pt[1] + rng.randint(-step, step)
            
            if abs(new_x) > 1000 or abs(new_y) > 1000:
                continue
            if any([new_x, new_y] == p for p in current):
                continue
                
            current[idx] = [new_x, new_y]
            new_score, new_h5 = _score_lines(current)
            
            if new_h5:
                current[idx] = old_pt
                no_improve += 1
            else:
                delta = new_score - current_score
                if delta > 0 or rng.random() < math.exp(delta / T):
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best = [p[:] for p in current]
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    current[idx] = old_pt
                    no_improve += 1
        
        T = max(T_min, T * cooling)
        
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    rng = random.Random(42)
    
    best_points = None
    best_score = -1
    
    # Restart with different seeds
    for seed in [42, 123, 456, 789, 101, 202, 303, 404]:
        if time.time() > deadline - 15:
            break
        rng = random.Random(seed)
        
        # Initial random points
        points = []
        for _ in range(16):
            x = rng.randint(-1000, 1000)
            y = rng.randint(-1000, 1000)
            points.append([x, y])
            
        s, h5 = _score_lines(points)
        if not h5:
            opt_pts, opt_score = _local_search(points, deadline, rng)
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_pts
        else:
            # Try to repair or just skip
            pass
            
    if best_points is None:
        # Fallback
        return [[0,0], [1,0], [2,0], [3,0], [0,1], [1,1], [2,1], [3,1], 
                [0,2], [1,2], [2,2], [3,2], [0,3], [1,3], [2,3], [3,3]]
        
    return best_points
