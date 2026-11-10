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

def _delta_score(pts, p_idx, new_pos, lines_dict):
    """
    Compute delta in score when moving point p_idx from pts[p_idx] to new_pos.
    Returns (delta, valid).
    """
    n = len(pts)
    old_pos = pts[p_idx]
    
    # Lines affected: those containing old_pos or new_pos with other points
    affected_lines = set()
    
    # Old lines containing p_idx
    for key, idxs in lines_dict.items():
        if p_idx in idxs:
            affected_lines.add(key)
    
    # New lines from new_pos to other points
    for j in range(n):
        if j == p_idx:
            continue
        key = _line_key(new_pos, pts[j])
        affected_lines.add(key)
    
    delta = 0
    valid = True
    
    for key in affected_lines:
        # Old count on this line
        old_cnt = 0
        if key in lines_dict:
            old_cnt = len(lines_dict[key])
        
        # New count on this line
        new_cnt = 0
        # Count points on this line excluding p_idx
        for j in range(n):
            if j == p_idx:
                continue
            if key in lines_dict and j in lines_dict[key]:
                new_cnt += 1
            else:
                # Check if pts[j] is on this line
                A, B, C = key
                if A * pts[j][0] + B * pts[j][1] + C == 0:
                    new_cnt += 1
        # Add p_idx if new_pos is on this line
        A, B, C = key
        if A * new_pos[0] + B * new_pos[1] + C == 0:
            new_cnt += 1
        
        old_contrib = 1 if old_cnt == 4 else 0
        new_contrib = 1 if new_cnt == 4 else 0
        
        if new_cnt >= 5:
            valid = False
        
        delta += new_contrib - old_contrib
    
    return delta, valid

def _greedy_construct(seed):
    """
    Build a configuration by greedily selecting 4-point lines.
    Each line is defined by 4 points. We try to maximize shared points.
    """
    rng = random.Random(seed)
    
    # Start with a base structure: think of it as choosing lines one by one
    # Each new line should share 2 points with existing lines when possible
    
    pts = []
    used_points = set()
    lines_chosen = []
    
    # First line: 4 random points
    for _ in range(4):
        while True:
            x = rng.randint(-20, 20)
            y = rng.randint(-20, 20)
            if (x, y) not in used_points:
                used_points.add((x, y))
                pts.append([x, y])
                break
    
    lines_chosen.append(set(range(4)))
    
    # Add more lines, each sharing 2 points with existing configuration
    target_lines = 10  # Aim for ~10 four-point lines
    
    for line_num in range(target_lines):
        if time.time() > 140:
            break
        
        # Find pairs of existing points to serve as "anchors" for new line
        best_line = None
        best_shared = -1
        
        # Try all pairs of existing points
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                # Extend line through pts[i] and pts[j] to find 2 more integer points
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                g = math.gcd(abs(dx), abs(dy))
                if g == 0:
                    continue
                dx //= g
                dy //= g
                
                # Find other integer points on this line
                candidates = []
                for t in range(-10, 11):
                    xi = x1 + t * dx
                    yi = y1 + t * dy
                    if -1000 <= xi <= 1000 and -1000 <= yi <= 1000:
                        if (xi, yi) not in used_points:
                            candidates.append((xi, yi))
                
                if len(candidates) >= 2:
                    # Pick 2 candidates
                    c1, c2 = rng.sample(candidates, 2)
                    new_pts = [i, j]
                    # Check how many of these new points create additional 4-lines
                    shared = 2  # i and j are shared
                    
                    # Heuristic: prefer lines that pass through more existing structure
                    best_line = (i, j, c1, c2)
                    best_shared = shared
        
        if best_line is None:
            # Just add 4 random points
            for _ in range(4):
                while True:
                    x = rng.randint(-30, 30)
                    y = rng.randint(-30, 30)
                    if (x, y) not in used_points:
                        used_points.add((x, y))
                        pts.append([x, y])
                        break
            continue
        
        i, j, c1, c2 = best_line
        pts.append(list(c1))
        pts.append(list(c2))
        used_points.add(c1)
        used_points.add(c2)
        
        if len(pts) >= 16:
            break
    
    # If we have more than 16, trim; if less, fill randomly
    while len(pts) > 16:
        pts.pop()
    
    while len(pts) < 16:
        while True:
            x = rng.randint(-50, 50)
            y = rng.randint(-50, 50)
            if (x, y) not in used_points:
                used_points.add((x, y))
                pts.append([x, y])
                break
    
    return pts[:16]

def _local_search_fast(cur_pts, cur_score, deadline, max_rounds=50):
    """
    Local search using delta scoring for speed.
    """
    pts = [list(p) for p in cur_pts]
    score = cur_score
    best_score = score
    best_pts = [list(p) for p in pts]
    
    rng = random.Random(int(time.time() * 1000) % 10000)
    
    for rnd in range(max_rounds):
        if time.time() > deadline:
            break
        
        lines_dict = _build_lines(pts)
        
        # Find 3-point lines for candidate generation
        three_lines = {k: v for k, v in lines_dict.items() if len(v) == 3}
        four_lines = {k: v for k, v in lines_dict.items() if len(v) == 4}
        
        improved_this_round = False
        
        for p_idx in range(16):
            if time.time() > deadline:
                return best_pts, best_score
            
            current_pos = tuple(pts[p_idx])
            candidates = set()
            
            # 1. Intersections of pairs of 3-lines not containing p_idx
            tl_keys = [k for k, v in three_lines.items() if p_idx not in v]
            if len(tl_keys) > 15:
                tl_keys = rng.sample(tl_keys, 15)
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
            
            # 2. Intersections of 3-lines and 4-lines
            all_special = list(three_lines.keys()) + list(four_lines.keys())
            if len(all_special) > 20:
                all_special = rng.sample(all_special, 20)
            for i in range(len(all_special)):
                for j in range(i + 1, len(all_special)):
                    A1, B1, C1 = all_special[i]
                    A2, B2, C2 = all_special[j]
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
            
            # 3. Points extending 3-lines
            for k, v in three_lines.items():
                if p_idx in v:
                    continue
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
                for t in range(-4, 5):
                    if t == 0:
                        continue
                    xi = p1[0] + t * dx
                    yi = p1[1] + t * dy
                    if -1000 <= xi <= 1000 and -1000 <= yi <= 1000:
                        candidates.add((xi, yi))
            
            # 4. Random nearby
            for _ in range(8):
                xi = current_pos[0] + rng.randint(-12, 12)
                yi = current_pos[1] + rng.randint(-12, 12)
                if -1000 <= xi <= 1000 and -1000 <= yi <= 1000:
                    candidates.add((xi, yi))
            
            # 5. Global random
            for _ in range(5):
                xi = rng.randint(-1000, 1000)
                yi = rng.randint(-1000, 1000)
                candidates.add((xi, yi))
            
            if not candidates:
                continue
            
            # Remove current position
            candidates.discard(current_pos)
            
            if not candidates:
                continue
            
            # Evaluate candidates with delta scoring
            best_cand = None
            best_delta = 0
            current_set = set(map(tuple, pts))
            
            for cand in list(candidates)[:30]:  # Limit evaluations
                if cand in current_set:
                    continue
                
                delta, valid = _delta_score(pts, p_idx, cand, lines_dict)
                if not valid:
                    continue
                if delta > best_delta:
                    best_delta = delta
                    best_cand = cand
            
            if best_cand is not None and best_delta > 0:
                pts[p_idx] = list(best_cand)
                score += best_delta
                improved_this_round = True
                if score > best_score:
                    best_score = score
                    best_pts = [list(p) for p in pts]
        
        if not improved_this_round:
            break
    
    return best_pts, best_score

def _generate_structured(seed):
    """
    Generate structured initial configurations based on known patterns.
    """
    rng = random.Random(seed)
    
    mode = seed % 5
    
    if mode == 0:
        # 4x4 grid with small perturbations
        pts = []
        for x in range(4):
            for y in range(4):
                px = x * 8 + rng.randint(-1, 1)
                py = y * 8 + rng.randint(-1, 1)
                pts.append([px, py])
        return pts
    
    elif mode == 1:
        # Two interleaved 3x3 grids (offset)
        pts = []
        for x in range(3):
            for y in range(3):
                pts.append([x * 10, y * 10])
                pts.append([x * 10 + 5, y * 10 + 5])
        return pts[:16]
    
    elif mode == 2:
        # Points on a conic-like structure
        pts = []
        for i in range(16):
            t = i - 7.5
            x = int(round(t * 3 + rng.randint(-1, 1)))
            y = int(round(t * t * 0.5 + rng.randint(-1, 1)))
            pts.append([x, y])
        # Ensure distinct
        seen = set()
        unique_pts = []
        for p in pts:
            tp = tuple(p)
            if tp not in seen:
                seen.add(tp)
                unique_pts.append(list(p))
        while len(unique_pts) < 16:
            x = rng.randint(-20, 20)
            y = rng.randint(-20, 20)
            if (x, y) not in seen:
                seen.add((x, y))
                unique_pts.append([x, y])
        return unique_pts[:16]
    
    elif mode == 3:
        # Affine image of a known configuration
        base = [[0,0],[1,0],[2,0],[3,0],
                [0,1],[1,1],[2,1],[3,1],
                [0,2],[1,2],[2,2],[3,2],
                [0,3],[1,3],[2,3],[3,3]]
        a = rng.randint(2, 6)
        b = rng.randint(0, 3)
        c = rng.randint(0, 3)
        d = rng.randint(2, 6)
        ox = rng.randint(-10, 10)
        oy = rng.randint(-10, 10)
        pts = []
        for x, y in base:
            nx = a * x + b * y + ox
            ny = c * x + d * y + oy
            pts.append([nx, ny])
        return pts
    
    else:
        # Random with structure
        pts = []
        seen = set()
        # Place 8 points in a rough grid
        for i in range(4):
            for j in range(2):
                x = i * 12 + rng.randint(-2, 2)
                y = j * 12 + rng.randint(-2, 2)
                if (x, y) not in seen:
                    seen.add((x, y))
                    pts.append([x, y])
        # Add 8 more randomly
        while len(pts) < 16:
            x = rng.randint(-40, 40)
            y = rng.randint(-40, 40)
            if (x, y) not in seen:
                seen.add((x, y))
                pts.append([x, y])
        return pts[:16]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_pts = None
    best_score = -1
    
    # Phase 1: Try multiple diverse initial configurations
    for seed in range(30):
        if time.time() > deadline:
            break
        
        # Alternate between structured and greedy
        if seed % 2 == 0:
            cur_pts = _generate_structured(seed)
        else:
            cur_pts = _greedy_construct(seed)
        
        cur_score, cur_valid = _score(cur_pts)
        
        if not cur_valid:
            continue
        
        if cur_score > best_score:
            best_score = cur_score
            best_pts = [list(p) for p in cur_pts]
        
        # Phase 2: Local search from this start
        remaining = deadline - time.time()
        if remaining < 5:
            break
        
        sa_deadline = time.time() + min(remaining * 0.7, 8.0)
        cur_pts, cur_score = _local_search_fast(cur_pts, cur_score, sa_deadline)
        
        if cur_score > best_score:
            best_score = cur_score
            best_pts = [list(p) for p in cur_pts]
    
    # Fallback: simple grid if nothing better
    if best_pts is None or best_score < 8:
        pts = []
        for x in range(4):
            for y in range(4):
                pts.append([x, y])
        best_pts = pts
        best_score, _ = _score(best_pts)
    
    return best_pts
