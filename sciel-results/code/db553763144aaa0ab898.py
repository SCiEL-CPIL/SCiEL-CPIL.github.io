import math
import time
import random
from collections import defaultdict

def _line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    g = math.gcd(abs(dx), abs(dy))
    if g == 0:
        return (0, 0, 0)
    dx //= g
    dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    A = dy
    B = -dx
    C = A * x1 + B * y1
    g2 = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
    if g2 != 0:
        A //= g2
        B //= g2
        C //= g2
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A = -A
        B = -B
        C = -C
    return (A, B, C)

def _score(pts):
    lines = defaultdict(set)
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(pts[i], pts[j])
            lines[key].add(i)
            lines[key].add(j)
    
    score = 0
    for key, indices in lines.items():
        cnt = len(indices)
        if cnt >= 5:
            return -1, False
        elif cnt == 4:
            score += 1
    return score, True

def _generate_initial(seed=0):
    rng = random.Random(seed)
    # Strategy 1: Perturbed 4x4 Grid
    # Strategy 2: Random dense cluster
    # Strategy 3: Points on a curve (e.g., y = x^2 % M)
    
    mode = seed % 3
    
    if mode == 0:
        # Perturbed 4x4
        pts = []
        for x in range(4):
            for y in range(4):
                px = x * 10 + rng.randint(-2, 2)
                py = y * 10 + rng.randint(-2, 2)
                pts.append([px, py])
    elif mode == 1:
        # Random points in a box
        pts = []
        seen = set()
        while len(pts) < 16:
            x = rng.randint(-50, 50)
            y = rng.randint(-50, 50)
            if (x, y) not in seen:
                seen.add((x, y))
                pts.append([x, y])
    else:
        # Points on a parabola, perturbed
        pts = []
        seen = set()
        for x in range(-4, 8):
            y = x * x
            # Perturb to ensure distinctness and integer bounds
            px = x * 5 + rng.randint(-1, 1)
            py = y * 2 + rng.randint(-1, 1)
            if (px, py) not in seen and abs(px) <= 1000 and abs(py) <= 1000:
                seen.add((px, py))
                pts.append([px, py])
        # Fill remaining
        while len(pts) < 16:
            x = rng.randint(-50, 50)
            y = rng.randint(-50, 50)
            if (x, y) not in seen:
                seen.add((x, y))
                pts.append([x, y])
                
    return pts[:16]

def _local_search_sa(cur_pts, cur_score, deadline, max_iters=5000):
    # Simulated Annealing-like search
    # We strictly reject invalid moves (5+ collinear)
    # We accept improving moves
    # We accept worsening moves with probability exp(-delta/T)
    
    pts = [list(p) for p in cur_pts]
    score = cur_score
    
    T = 10.0
    cooling = 0.999
    no_improve_count = 0
    best_score = score
    best_pts = [list(p) for p in pts]
    
    rng = random.Random(int(time.time() * 1000) % 10000)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Compute lines for current state
        lines = defaultdict(set)
        for i in range(16):
            for j in range(i + 1, 16):
                key = _line_key(pts[i], pts[j])
                lines[key].add(i)
                lines[key].add(j)
        
        three_lines = {k: v for k, v in lines.items() if len(v) == 3}
        
        # Choose a move type: 80% Shift, 20% Swap
        move_type = rng.random()
        
        new_pts = [list(p) for p in pts]
        new_score = score
        valid = True
        
        if move_type < 0.8:
            # Shift a point
            p_idx = rng.randint(0, 15)
            current_pos = tuple(pts[p_idx])
            
            # Generate candidates
            candidates = set()
            
            # 1. Intersections of 3-lines not containing p_idx
            tl_keys = [k for k, v in three_lines.items() if p_idx not in v]
            # Limit to avoid explosion if many 3-lines
            if len(tl_keys) > 20:
                tl_keys = rng.sample(tl_keys, 20)
            
            for i in range(len(tl_keys)):
                for j in range(i + 1, len(tl_keys)):
                    A1, B1, C1 = tl_keys[i]
                    A2, B2, C2 = tl_keys[j]
                    det = A1 * B2 - A2 * B1
                    if det == 0:
                        continue
                    x = (C1 * B2 - C2 * B1) / det
                    y = (A1 * C2 - A2 * C1) / det
                    if abs(x - round(x)) < 1e-9 and abs(y - round(y)) < 1e-9:
                        xi = int(round(x))
                        yi = int(round(y))
                        if -1000 <= xi <= 1000 and -1000 <= yi <= 1000:
                            candidates.add((xi, yi))
            
            # 2. Random nearby points
            # Radius increases with no_improve_count
            radius = 5 + no_improve_count // 10
            for _ in range(10):
                xi = current_pos[0] + rng.randint(-radius, radius)
                yi = current_pos[1] + rng.randint(-radius, radius)
                if -1000 <= xi <= 1000 and -1000 <= yi <= 1000:
                    candidates.add((xi, yi))
            
            if not candidates:
                continue
                
            cand = rng.choice(list(candidates))
            if cand == current_pos:
                continue
            
            new_pts[p_idx] = list(cand)
            
            # Check distinctness
            seen = set()
            for p in new_pts:
                tp = tuple(p)
                if tp in seen:
                    valid = False
                    break
                seen.add(tp)
            
            if valid:
                new_score, valid = _score(new_pts)
                if not valid:
                    new_score = -1
        else:
            # Swap two points
            i, j = rng.sample(range(16), 2)
            new_pts[i], new_pts[j] = pts[j], pts[i]
            
            # Check distinctness (swapping distinct points keeps them distinct)
            new_score, valid = _score(new_pts)
            if not valid:
                new_score = -1
        
        if not valid:
            T *= cooling
            continue
            
        # Evaluate acceptance
        delta = new_score - score
        if delta >= 0:
            # Always accept improvement or equal
            pts = new_pts
            score = new_score
            no_improve_count = 0
            if score > best_score:
                best_score = score
                best_pts = [list(p) for p in pts]
        else:
            # Accept with probability exp(delta/T)
            if T > 0.01:
                prob = math.exp(delta / T)
                if rng.random() < prob:
                    pts = new_pts
                    score = new_score
            no_improve_count += 1
            
        T *= cooling
        
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_pts = None
    best_score = -1
    
    # Try multiple initial seeds and strategies
    for seed in range(20):
        if time.time() > deadline:
            break
            
        cur_pts = _generate_initial(seed)
        cur_score, cur_valid = _score(cur_pts)
        
        if not cur_valid:
            continue
            
        if cur_score > best_score:
            best_score = cur_score
            best_pts = [list(p) for p in cur_pts]
        
        # Run local search
        # Allocate time based on remaining time
        remaining = deadline - time.time()
        if remaining < 5:
            break
            
        # Run SA for a portion of the time
        # We can just let the SA loop run until deadline, but we want to try multiple starts.
        # So we limit the SA iterations or time per start.
        sa_deadline = time.time() + min(remaining * 0.8, 10.0) # Max 10s per start
        
        cur_pts, cur_score = _local_search_sa(cur_pts, cur_score, sa_deadline)
        
        if cur_score > best_score:
            best_score = cur_score
            best_pts = [list(p) for p in cur_pts]
            
    # Fallback to simple grid if nothing better found
    if best_pts is None or best_score < 10:
        pts = []
        for x in range(4):
            for y in range(4):
                pts.append([x, y])
        best_pts = pts
        best_score, _ = _score(best_pts)
        
    return best_pts
