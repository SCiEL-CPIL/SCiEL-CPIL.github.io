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

def _build_lines(pts):
    lines = defaultdict(set)
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(pts[i], pts[j])
            lines[key].add(i)
            lines[key].add(j)
    return lines

def _score_from_lines(lines):
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
    mode = seed % 4
    
    if mode == 0:
        # Perturbed 4x4 grid
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
            x = rng.randint(-30, 30)
            y = rng.randint(-30, 30)
            if (x, y) not in seen:
                seen.add((x, y))
                pts.append([x, y])
    elif mode == 2:
        # Points on a parabola, perturbed
        pts = []
        seen = set()
        for x in range(-4, 8):
            y = x * x
            px = x * 5 + rng.randint(-1, 1)
            py = y * 2 + rng.randint(-1, 1)
            if (px, py) not in seen and abs(px) <= 1000 and abs(py) <= 1000:
                seen.add((px, py))
                pts.append([px, py])
        while len(pts) < 16:
            x = rng.randint(-30, 30)
            y = rng.randint(-30, 30)
            if (x, y) not in seen:
                seen.add((x, y))
                pts.append([x, y])
    else:
        # Structured: two overlapping 4x4 grids with offset
        pts = []
        seen = set()
        # First 8 points in a compact region
        for x in range(4):
            for y in range(2):
                px = x * 7 + rng.randint(-1, 1)
                py = y * 7 + rng.randint(-1, 1)
                if (px, py) not in seen:
                    seen.add((px, py))
                    pts.append([px, py])
        # Second 8 points offset
        for x in range(4):
            for y in range(2, 4):
                px = x * 7 + 3 + rng.randint(-1, 1)
                py = y * 7 + rng.randint(-1, 1)
                if (px, py) not in seen:
                    seen.add((px, py))
                    pts.append([px, py])
        while len(pts) < 16:
            x = rng.randint(-30, 30)
            y = rng.randint(-30, 30)
            if (x, y) not in seen:
                seen.add((x, y))
                pts.append([x, y])
    
    return pts[:16]

def _incremental_score(pts, p_idx, new_pos, old_lines):
    """Score the configuration if pts[p_idx] is moved to new_pos.
    Uses old_lines to avoid recomputing everything."""
    n = 16
    # Lines involving p_idx change; others stay the same
    # Build new lines: copy old, then fix lines involving p_idx
    
    # First, identify lines that will change
    # A line changes if it contains p_idx
    changed_line_keys = set()
    for key, indices in old_lines.items():
        if p_idx in indices:
            changed_line_keys.add(key)
    
    # New lines dict
    new_lines = defaultdict(set)
    for key, indices in old_lines.items():
        if key not in changed_line_keys:
            new_lines[key] = set(indices)
    
    # Add new lines from p_idx to all other points
    for j in range(n):
        if j == p_idx:
            continue
        key = _line_key(new_pos, pts[j])
        new_lines[key].add(p_idx)
        new_lines[key].add(j)
    
    # Check validity and score
    score = 0
    for key, indices in new_lines.items():
        cnt = len(indices)
        if cnt >= 5:
            return -1, False
        elif cnt == 4:
            score += 1
    return score, True

def _local_search_sa(cur_pts, cur_score, deadline, rng, max_iters=50000):
    pts = [list(p) for p in cur_pts]
    score = cur_score
    
    T = 15.0
    cooling = 0.9995
    no_improve_count = 0
    best_score = score
    best_pts = [list(p) for p in pts]
    
    # Precompute lines
    lines = _build_lines(pts)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Get 3-lines not containing each point (for candidate generation)
        three_lines = {k: v for k, v in lines.items() if len(v) == 3}
        
        move_type = rng.random()
        
        new_pts = [list(p) for p in pts]
        new_score = score
        valid = True
        
        if move_type < 0.85:
            # Shift a point
            p_idx = rng.randint(0, 15)
            current_pos = tuple(pts[p_idx])
            
            candidates = set()
            
            # 1. Intersections of pairs of 3-lines not containing p_idx
            tl_keys = [k for k, v in three_lines.items() if p_idx not in v]
            if len(tl_keys) > 30:
                tl_keys = rng.sample(tl_keys, 30)
            
            for i in range(len(tl_keys)):
                for j in range(i + 1, len(tl_keys)):
                    A1, B1, C1 = tl_keys[i]
                    A2, B2, C2 = tl_keys[j]
                    det = A1 * B2 - A2 * B1
                    if det == 0:
                        continue
                    x_num = C1 * B2 - C2 * B1
                    y_num = A1 * C2 - A2 * C1
                    if x_num % det == 0 and y_num % det == 0:
                        xi = x_num // det
                        yi = y_num // det
                        if -1000 <= xi <= 1000 and -1000 <= yi <= 1000:
                            candidates.add((xi, yi))
            
            # 2. Points on 3-lines extended (key idea from inspiration 2)
            for k, v in three_lines.items():
                if p_idx in v:
                    continue
                A, B, C = k
                idxs = list(v)
                p1 = pts[idxs[0]]
                p2 = pts[idxs[1]]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                g = math.gcd(abs(dx), abs(dy))
                if g == 0:
                    continue
                dx //= g
                dy //= g
                # Try steps from p1
                for t in range(-6, 7):
                    if t == 0:
                        continue
                    xi = p1[0] + t * dx
                    yi = p1[1] + t * dy
                    if -1000 <= xi <= 1000 and -1000 <= yi <= 1000:
                        candidates.add((xi, yi))
            
            # 3. Random nearby points
            radius = 8 + no_improve_count // 15
            for _ in range(15):
                xi = current_pos[0] + rng.randint(-radius, radius)
                yi = current_pos[1] + rng.randint(-radius, radius)
                if -1000 <= xi <= 1000 and -1000 <= yi <= 1000:
                    candidates.add((xi, yi))
            
            if not candidates:
                T *= cooling
                continue
                
            cand = rng.choice(list(candidates))
            if cand == current_pos:
                T *= cooling
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
                # Use incremental scoring
                new_score, valid = _incremental_score(pts, p_idx, cand, lines)
                if not valid:
                    new_score = -1
        else:
            # Swap two points
            i, j = rng.sample(range(16), 2)
            new_pts[i], new_pts[j] = pts[j], pts[i]
            new_score, valid = _score(new_pts)
            if not valid:
                new_score = -1
        
        if not valid:
            T *= cooling
            continue
            
        delta = new_score - score
        if delta >= 0:
            pts = new_pts
            score = new_score
            no_improve_count = 0
            lines = _build_lines(pts)
            if score > best_score:
                best_score = score
                best_pts = [list(p) for p in pts]
        else:
            if T > 0.01:
                prob = math.exp(delta / T)
                if rng.random() < prob:
                    pts = new_pts
                    score = new_score
                    lines = _build_lines(pts)
            no_improve_count += 1
            
        T *= cooling
        
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_pts = None
    best_score = -1
    
    rng = random.Random(42)
    
    # Try multiple initial seeds and strategies
    for seed in range(25):
        if time.time() > deadline:
            break
            
        cur_pts = _generate_initial(seed)
        cur_score, cur_valid = _score(cur_pts)
        
        if not cur_valid:
            continue
            
        if cur_score > best_score:
            best_score = cur_score
            best_pts = [list(p) for p in cur_pts]
        
        remaining = deadline - time.time()
        if remaining < 5:
            break
        
        # Run SA for a portion of the time
        sa_deadline = time.time() + min(remaining * 0.85, 12.0)
        
        cur_pts, cur_score = _local_search_sa(cur_pts, cur_score, sa_deadline, rng)
        
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
