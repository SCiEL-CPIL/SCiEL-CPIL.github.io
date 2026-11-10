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
    """Build line dictionary: key -> set of point indices"""
    lines = defaultdict(set)
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(pts[i], pts[j])
            lines[key].add(i)
            lines[key].add(j)
    return lines

def _delta_score(pts, old_pos, new_pos, p_idx, lines_dict):
    """
    Compute the change in score when moving point p_idx from old_pos to new_pos.
    Returns (delta_score, new_valid) where new_valid is False if any 5+ collinear.
    lines_dict is the current line mapping.
    """
    n = len(pts)
    # Find all lines that contain p_idx (old lines)
    old_lines = set()
    for key, idxs in lines_dict.items():
        if p_idx in idxs:
            old_lines.add(key)
    
    # Find all lines that the new point would be on with other points
    new_lines = defaultdict(set)
    for j in range(n):
        if j == p_idx:
            continue
        key = _line_key(new_pos, pts[j])
        new_lines[key].add(j)
        # Also add p_idx implicitly
    
    # The affected lines are: old_lines (lines that lose p_idx) and new_lines (lines that gain p_idx)
    # For each affected line, compute old count and new count
    
    # We need to check: for each line in old_lines ∪ new_lines, what's the new count?
    affected = old_lines | set(new_lines.keys())
    
    delta = 0
    valid = True
    
    for key in affected:
        # Old count: how many points (excluding p_idx) are on this line
        old_count = 0
        if key in lines_dict:
            old_count = len(lines_dict[key] - {p_idx})
        # New count: how many points (excluding p_idx) are on this line with new position
        new_count = 0
        if key in new_lines:
            new_count = len(new_lines[key])
        
        # But wait: some lines in old_lines might not be in new_lines (they lose p_idx, no new point there)
        # And some lines in new_lines might not be in old_lines (they gain p_idx)
        # For lines in both: old_count = len(lines_dict[key]) - 1, new_count = len(new_lines[key])
        # For lines only in old_lines: old_count = len(lines_dict[key]) - 1, new_count = 0
        # For lines only in new_lines: old_count = 0, new_count = len(new_lines[key])
        
        # Actually, let me reconsider. The line count for a line NOT involving p_idx doesn't change.
        # For lines involving p_idx:
        #   Old: the line had len(lines_dict[key]) points including p_idx
        #   New: the line has (number of other points on that line through new_pos) points
        
        # For a line key in old_lines:
        #   If key is also in new_lines: old_count_full = len(lines_dict[key]), new_count_full = len(new_lines[key]) + 1
        #   If key is NOT in new_lines: old_count_full = len(lines_dict[key]), new_count_full = 0 (p_idx left, no one there)
        
        # For a line key in new_lines but not old_lines:
        #   old_count_full = 0 (p_idx wasn't on it), new_count_full = len(new_lines[key]) + 1
        
        # Let me just compute old and new full counts for affected lines.
        
        old_full = 0
        if key in lines_dict:
            old_full = len(lines_dict[key])
        
        new_full = 0
        if key in new_lines:
            new_full = len(new_lines[key]) + 1  # +1 for p_idx at new position
        
        # But we need to be careful: a line might be in old_lines but the other points might have moved?
        # No, we're only moving p_idx, so other points stay.
        
        # Score contribution:
        old_contrib = 1 if old_full == 4 else 0
        new_contrib = 1 if new_full == 4 else 0
        
        if new_full >= 5:
            valid = False
        
        delta += new_contrib - old_contrib
    
    # Also need to check: are there any OTHER lines (not in affected) that now have 5+ points?
    # No, because only lines involving p_idx change. Lines not involving p_idx are unchanged.
    # But wait: could a line not in old_lines and not in new_lines suddenly have 5 points?
    # No, because p_idx wasn't on it before and isn't on it after.
    
    return delta, valid

def _generate_initial(seed=0):
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
    return unique_pts[:16]

def _generate_structured(seed=0):
    """Generate a structured configuration with known 4-lines."""
    rng = random.Random(seed)
    # Try a few different structured patterns
    patterns = [
        # 4x4 grid
        lambda: [[x, y] for x in range(4) for y in range(4)],
        # 4x4 grid shifted
        lambda: [[x, y] for x in range(4) for y in range(4)],
    ]
    # Just use the grid for now, with small random perturbations
    pts = [[x, y] for x in range(4) for y in range(4)]
    # Perturb a few points
    for _ in range(rng.randint(1, 3)):
        idx = rng.randint(0, 15)
        pts[idx][0] += rng.choice([-1, 1])
        pts[idx][1] += rng.choice([-1, 1])
    # Ensure distinctness
    seen = set()
    unique_pts = []
    for p in pts:
        tp = tuple(p)
        if tp not in seen:
            seen.add(tp)
            unique_pts.append(p)
    while len(unique_pts) < 16:
        x = rng.randint(-3, 6)
        y = rng.randint(-3, 6)
        tp = (x, y)
        if tp not in seen:
            seen.add(tp)
            unique_pts.append([x, y])
    return unique_pts[:16]

def _local_search(cur_pts, cur_score, deadline, max_iters=50):
    improved = True
    iters = 0
    while improved and iters < max_iters and time.time() < deadline:
        improved = False
        iters += 1
        
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
            
            # 2. Points on 3-lines extended (to make 4-lines)
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
                for t in range(-4, 5):
                    if t == 0:
                        continue
                    xi = p1[0] + t * dx
                    yi = p1[1] + t * dy
                    if -1000 <= xi <= 1000 and -1000 <= yi <= 1000:
                        candidates.add((xi, yi))
            
            # 3. Intersections of a 3-line and a 4-line (or two 4-lines) not containing p_idx
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
            for _ in range(6):
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
                
                # Use delta scoring
                delta, new_valid = _delta_score(cur_pts, current_pos, cand, p_idx, lines_dict)
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
    deadline = start_time + 140.0  # 10s safety margin
    
    best_pts = None
    best_score = -1
    
    # Try multiple initial seeds
    for seed in range(15):
        if time.time() > deadline:
            break
        
        if seed % 2 == 0:
            cur_pts = _generate_initial(seed)
        else:
            cur_pts = _generate_structured(seed)
        
        cur_score, cur_valid = _score(cur_pts)
        
        if cur_valid and cur_score > best_score:
            best_score = cur_score
            best_pts = [list(p) for p in cur_pts]
        
        # Local search from this starting point
        if cur_valid and time.time() < deadline:
            cur_pts, cur_score, _ = _local_search(cur_pts, cur_score, deadline)
            if cur_score > best_score:
                best_score = cur_score
                best_pts = [list(p) for p in cur_pts]
    
    # If still no good result, use simple grid
    if best_pts is None or best_score < 10:
        pts = [[x, y] for x in range(4) for y in range(4)]
        best_pts = pts
        best_score, _ = _score(best_pts)
    
    return best_pts
