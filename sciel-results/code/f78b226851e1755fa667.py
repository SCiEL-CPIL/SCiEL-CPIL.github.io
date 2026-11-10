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

def _delta_score(pts, new_pos, p_idx, lines_dict):
    """
    Compute the change in score when moving point p_idx to new_pos.
    Returns (delta_score, new_valid) where new_valid is False if any 5+ collinear.
    """
    n = len(pts)
    
    # Find all lines that currently contain p_idx
    old_lines = set()
    for key, idxs in lines_dict.items():
        if p_idx in idxs:
            old_lines.add(key)
    
    # Find all lines that the new point would form with other points
    new_lines = defaultdict(set)
    for j in range(n):
        if j == p_idx:
            continue
        key = _line_key(new_pos, pts[j])
        new_lines[key].add(j)
    
    affected = old_lines | set(new_lines.keys())
    
    delta = 0
    valid = True
    
    for key in affected:
        # Old full count of points on this line
        old_full = 0
        if key in lines_dict:
            old_full = len(lines_dict[key])
        
        # New full count of points on this line (including p_idx if it's on it)
        new_full = 0
        if key in new_lines:
            new_full = len(new_lines[key]) + 1
        
        # Score contribution change
        old_contrib = 1 if old_full == 4 else 0
        new_contrib = 1 if new_full == 4 else 0
        
        if new_full >= 5:
            valid = False
            
        delta += new_contrib - old_contrib
        
    return delta, valid

def _local_search(cur_pts, cur_score, deadline, max_rounds=100):
    improved = True
    rounds = 0
    
    while improved and rounds < max_rounds and time.time() < deadline:
        improved = False
        rounds += 1
        
        lines_dict = _build_lines(cur_pts)
        
        # Identify lines by count
        three_lines = {k: v for k, v in lines_dict.items() if len(v) == 3}
        four_lines = {k: v for k, v in lines_dict.items() if len(v) == 4}
        
        for p_idx in range(16):
            if time.time() > deadline:
                return cur_pts, cur_score, False
            
            current_pos = tuple(cur_pts[p_idx])
            candidates = set()
            
            # 1. Intersections of pairs of 3-lines not containing p_idx
            tl_keys = [k for k, v in three_lines.items() if p_idx not in v]
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
            
            # 2. Points on 3-lines extended
            for k, v in three_lines.items():
                if p_idx in v:
                    continue
                idxs = list(v)
                p1 = cur_pts[idxs[0]]
                p2 = cur_pts[idxs[1]]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                g = math.gcd(abs(dx), abs(dy))
                if g == 0:
                    continue
                dx //= g
                dy //= g
                for t in range(-3, 4):
                    if t == 0:
                        continue
                    xi = p1[0] + t * dx
                    yi = p1[1] + t * dy
                    if -1000 <= xi <= 1000 and -1000 <= yi <= 1000:
                        candidates.add((xi, yi))
            
            # 3. Intersections of 3-lines and 4-lines
            all_lines = list(three_lines.keys()) + list(four_lines.keys())
            for i in range(len(all_lines)):
                for j in range(i + 1, len(all_lines)):
                    A1, B1, C1 = all_lines[i]
                    A2, B2, C2 = all_lines[j]
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
            
            # 4. Random nearby
            for _ in range(8):
                xi = current_pos[0] + random.randint(-5, 5)
                yi = current_pos[1] + random.randint(-5, 5)
                if -1000 <= xi <= 1000 and -1000 <= yi <= 1000:
                    candidates.add((xi, yi))
            
            best_new_pos = None
            best_new_score = cur_score
            current_set = set(map(tuple, cur_pts))
            
            for cand in candidates:
                if cand == current_pos or cand in current_set:
                    continue
                
                delta, new_valid = _delta_score(cur_pts, cand, p_idx, lines_dict)
                new_score = cur_score + delta
                
                if new_valid and new_score > best_new_score:
                    best_new_score = new_score
                    best_new_pos = cand
            
            if best_new_pos is not None:
                cur_pts[p_idx] = list(best_new_pos)
                cur_score = best_new_score
                improved = True
    
    return cur_pts, cur_score, improved

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    best_pts = None
    best_score = -1
    
    # Start with the known 4x4 grid (score 10)
    grid_pts = [[x, y] for x in range(4) for y in range(4)]
    grid_score, grid_valid = _score(grid_pts)
    
    if grid_valid:
        best_score = grid_score
        best_pts = [list(p) for p in grid_pts]
    
    # Try local search from the grid
    if time.time() < deadline:
        cur_pts = [list(p) for p in grid_pts]
        cur_score = grid_score
        cur_pts, cur_score, _ = _local_search(cur_pts, cur_score, deadline)
        if cur_score > best_score:
            best_score = cur_score
            best_pts = [list(p) for p in cur_pts]
    
    # Try a few random restarts if time permits
    for seed in range(5):
        if time.time() > deadline:
            break
        
        rng = random.Random(seed)
        pts = []
        for x in range(4):
            for y in range(4):
                px = x + rng.choice([0, 0, 0, 0, 1, -1])
                py = y + rng.choice([0, 0, 0, 0, 1, -1])
                pts.append([px, py])
        
        seen = set()
        unique_pts = []
        for p in pts:
            tp = tuple(p)
            if tp not in seen:
                seen.add(tp)
                unique_pts.append(p)
        while len(unique_pts) < 16:
            x = rng.randint(-5, 5)
            y = rng.randint(-5, 5)
            tp = (x, y)
            if tp not in seen:
                seen.add(tp)
                unique_pts.append([x, y])
        
        cur_pts = unique_pts[:16]
        cur_score, cur_valid = _score(cur_pts)
        
        if cur_valid and cur_score > best_score:
            best_score = cur_score
            best_pts = [list(p) for p in cur_pts]
        
        if cur_valid and time.time() < deadline:
            cur_pts, cur_score, _ = _local_search(cur_pts, cur_score, deadline)
            if cur_score > best_score:
                best_score = cur_score
                best_pts = [list(p) for p in cur_pts]
    
    if best_pts is None:
        return grid_pts
        
    return best_pts
