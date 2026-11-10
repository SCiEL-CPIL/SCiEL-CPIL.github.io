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
    # Start with 4x4 grid but perturb slightly to break symmetries
    pts = []
    for x in range(4):
        for y in range(4):
            # Small random perturbation
            px = x + rng.choice([0, 0, 0, 1, -1])
            py = y + rng.choice([0, 0, 0, 1, -1])
            pts.append([px, py])
    # Ensure distinctness
    seen = set()
    unique_pts = []
    for p in pts:
        tp = tuple(p)
        if tp not in seen:
            seen.add(tp)
            unique_pts.append(p)
    # If we lost points, add new ones
    while len(unique_pts) < 16:
        x = rng.randint(-5, 5)
        y = rng.randint(-5, 5)
        tp = (x, y)
        if tp not in seen:
            seen.add(tp)
            unique_pts.append([x, y])
    return unique_pts[:16]

def _local_search(cur_pts, cur_score, deadline, max_iters=100):
    improved = True
    iters = 0
    while improved and iters < max_iters and time.time() < deadline:
        improved = False
        iters += 1
        
        lines = defaultdict(set)
        for i in range(16):
            for j in range(i + 1, 16):
                key = _line_key(cur_pts[i], cur_pts[j])
                lines[key].add(i)
                lines[key].add(j)
        
        three_lines = {k: v for k, v in lines.items() if len(v) == 3}
        
        for p_idx in range(16):
            if time.time() > deadline:
                return cur_pts, cur_score, False
            
            current_pos = tuple(cur_pts[p_idx])
            candidates = set()
            
            # Intersections of pairs of 3-lines not containing p_idx
            tl_keys = [k for k, v in three_lines.items() if p_idx not in v]
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
            
            # Points on 3-lines extended
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
            
            # Random nearby
            for _ in range(8):
                xi = current_pos[0] + random.randint(-4, 4)
                yi = current_pos[1] + random.randint(-4, 4)
                if -1000 <= xi <= 1000 and -1000 <= yi <= 1000:
                    candidates.add((xi, yi))
            
            best_new_pos = None
            best_new_score = cur_score
            current_set = set(map(tuple, cur_pts))
            
            for cand in candidates:
                if cand == current_pos or cand in current_set:
                    continue
                
                old_pos = cur_pts[p_idx]
                cur_pts[p_idx] = list(cand)
                new_score, new_valid = _score(cur_pts)
                cur_pts[p_idx] = old_pos
                
                if new_valid and new_score > best_new_score:
                    best_new_score = new_score
                    best_new_pos = cand
            
            if best_new_pos is not None:
                cur_pts[p_idx] = list(best_new_pos)
                cur_score = best_new_score
                improved = True
    
    return cur_pts, cur_score, improved

def _two_opt_swap(cur_pts, cur_score, deadline):
    improved = True
    while improved and time.time() < deadline:
        improved = False
        lines = defaultdict(set)
        for i in range(16):
            for j in range(i + 1, 16):
                key = _line_key(cur_pts[i], cur_pts[j])
                lines[key].add(i)
                lines[key].add(j)
        
        for i in range(16):
            if time.time() > deadline:
                return cur_pts, cur_score, False
            for j in range(i + 1, 16):
                old_i = list(cur_pts[i])
                old_j = list(cur_pts[j])
                cur_pts[i] = old_j
                cur_pts[j] = old_i
                new_score, new_valid = _score(cur_pts)
                cur_pts[i] = old_i
                cur_pts[j] = old_j
                if new_valid and new_score > cur_score:
                    cur_pts[i] = old_j
                    cur_pts[j] = old_i
                    cur_score = new_score
                    improved = True
    return cur_pts, cur_score, improved

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    best_pts = None
    best_score = -1
    
    # Try multiple initial seeds
    for seed in range(10):
        if time.time() > deadline:
            break
        
        cur_pts = _generate_initial(seed)
        cur_score, cur_valid = _score(cur_pts)
        
        if cur_valid and cur_score > best_score:
            best_score = cur_score
            best_pts = [list(p) for p in cur_pts]
        
        # Local search from this starting point
        if cur_valid:
            cur_pts, cur_score, _ = _local_search(cur_pts, cur_score, deadline)
            if cur_score > best_score:
                best_score = cur_score
                best_pts = [list(p) for p in cur_pts]
            
            # Try 2-opt swap
            cur_pts, cur_score, _ = _two_opt_swap(cur_pts, cur_score, deadline)
            if cur_score > best_score:
                best_score = cur_score
                best_pts = [list(p) for p in cur_pts]
    
    # If still no good result, use simple grid
    if best_pts is None or best_score < 10:
        pts = []
        for x in range(4):
            for y in range(4):
                pts.append([x, y])
        best_pts = pts
        best_score, _ = _score(best_pts)
    
    return best_pts
