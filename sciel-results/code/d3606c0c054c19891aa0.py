import math
import time
import random
from collections import defaultdict

def _line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    if dx == 0:
        return (0, 0, x1)
    if dy == 0:
        return (0, 1, y1)
    a = dy
    b = -dx
    c = a * x1 + b * y1
    g = math.gcd(abs(a), abs(b))
    if g > 1:
        a //= g
        b //= g
        c //= g
    if a < 0 or (a == 0 and b < 0):
        a = -a
        b = -b
        c = -c
    return (a, b, c)

def _build_line_map(points):
    n = len(points)
    line_map = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is not None:
                line_map[key].add(i)
                line_map[key].add(j)
    return line_map

def _score_from_line_map(line_map):
    count_4 = 0
    for key, idx_set in line_map.items():
        sz = len(idx_set)
        if sz >= 5:
            return -1
        if sz == 4:
            count_4 += 1
    return count_4

def _score(pts):
    line_map = _build_line_map(pts)
    return _score_from_line_map(line_map)

def _incremental_score(pts, idx, line_map, old_score):
    """Move point idx to pts[idx]. Returns (new_score, new_line_map) or (None, None) if invalid."""
    new_pos = tuple(pts[idx])
    # Check distinctness
    for j in range(16):
        if j != idx and tuple(pts[j]) == new_pos:
            return None, None

    # Remove idx from all lines it was on
    affected = set()
    for key in list(line_map.keys()):
        if idx in line_map[key]:
            affected.add(key)
            line_map[key].discard(idx)
            if not line_map[key]:
                del line_map[key]

    # Add idx to new lines
    for j in range(16):
        if j == idx:
            continue
        key = _line_key(pts[idx], pts[j])
        if key is not None:
            line_map[key].add(idx)
            line_map[key].add(j)
            affected.add(key)

    # Recompute score from scratch over affected lines only (but we need global count)
    # Since we only changed lines involving idx, we can recompute the 4-count
    # by tracking how many lines had size 4 before and after among affected.
    # Simpler: just recount from the full map (still O(lines) which is fine)
    new_score = _score_from_line_map(line_map)
    return new_score, line_map

def _generate_grid_4x4():
    return [[x, y] for x in range(4) for y in range(4)]

def _generate_perturbed_grid(rng, perturb_prob=0.3, max_perturb=3):
    pts = []
    for x in range(4):
        for y in range(4):
            if rng.random() < perturb_prob:
                nx = x + rng.randint(-max_perturb, max_perturb)
                ny = y + rng.randint(-max_perturb, max_perturb)
            else:
                nx, ny = x, y
            pts.append([nx, ny])
    seen = set()
    valid_pts = []
    for p in pts:
        if (p[0], p[1]) not in seen:
            seen.add((p[0], p[1]))
            valid_pts.append(p)
    while len(valid_pts) < 16:
        nx = rng.randint(-5, 8)
        ny = rng.randint(-5, 8)
        if (nx, ny) not in seen:
            seen.add((nx, ny))
            valid_pts.append([nx, ny])
    return valid_pts[:16]

def _generate_random_box(rng, half=30):
    pts = []
    seen = set()
    while len(pts) < 16:
        x = rng.randint(-half, half)
        y = rng.randint(-half, half)
        if (x, y) not in seen:
            seen.add((x, y))
            pts.append([x, y])
    return pts

def _local_search_sa(cur_pts, cur_score, deadline, seed=0):
    pts = [list(p) for p in cur_pts]
    score = cur_score
    line_map = _build_line_map(pts)

    T = 8.0
    cooling = 0.9995
    best_score = score
    best_pts = [list(p) for p in pts]
    no_improve = 0

    rng = random.Random(seed)
    start_t = time.time()

    while time.time() < deadline:
        # Adaptive move radius
        r = rng.random()
        if r < 0.5:
            radius = 3
        elif r < 0.85:
            radius = 8
        else:
            radius = 20

        idx = rng.randint(0, 15)
        dx = rng.randint(-radius, radius)
        dy = rng.randint(-radius, radius)
        if dx == 0 and dy == 0:
            continue

        new_x = pts[idx][0] + dx
        new_y = pts[idx][1] + dy
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            continue

        new_pts = [list(p) for p in pts]
        new_pts[idx] = [new_x, new_y]

        # Fast incremental scoring
        new_score, new_line_map = _incremental_score(new_pts, idx, line_map, score)
        if new_score is None:
            T *= cooling
            continue

        delta = new_score - score
        if delta >= 0:
            pts = new_pts
            score = new_score
            line_map = new_line_map
            no_improve = 0
            if score > best_score:
                best_score = score
                best_pts = [list(p) for p in pts]
        else:
            if T > 0.01:
                prob = math.exp(delta / T)
                if rng.random() < prob:
                    pts = new_pts
                    score = new_score
                    line_map = new_line_map
            no_improve += 1

        T *= cooling

    return best_pts, best_score

def _local_polish(pts, deadline):
    """Exhaustive small-move polish: try every shift of every point."""
    current = [list(p) for p in pts]
    cur_score = _score(current)
    if cur_score < 0:
        return pts, 0

    best_pts = [list(p) for p in current]
    best_score = cur_score
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(16):
            if time.time() > deadline:
                break
            cx, cy = current[idx]
            for dx in range(-4, 5):
                for dy in range(-4, 5):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = cx + dx, cy + dy
                    if abs(nx) > 1000 or abs(ny) > 1000:
                        continue
                    candidate = [list(p) for p in current]
                    candidate[idx] = [nx, ny]
                    s = _score(candidate)
                    if s < 0:
                        continue
                    if s > best_score:
                        best_score = s
                        best_pts = [list(p) for p in candidate]
                        current = [list(p) for p in candidate]
                        improved = True
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    best_pts = None
    best_score = -1

    # 1. Start with the 4x4 grid (known to have many 4-point lines)
    grid = _generate_grid_4x4()
    gs = _score(grid)
    if gs >= 0:
        best_score = gs
        best_pts = [list(p) for p in grid]

    # 2. Run SA from the grid
    if time.time() < deadline - 10:
        pts, sc = _local_search_sa(grid, gs, time.time() + min(40, deadline - time.time() - 10), seed=42)
        if sc > best_score:
            best_score = sc
            best_pts = [list(p) for p in pts]

    # 3. Try multiple perturbed grid starts
    rng = random.Random(12345)
    for trial in range(10):
        if time.time() > deadline - 10:
            break
        perturb_prob = 0.15 + 0.04 * trial
        max_perturb = 2 + trial // 4
        perturbed = _generate_perturbed_grid(rng, perturb_prob, max_perturb)
        ps = _score(perturbed)
        if ps < 0:
            continue
        if ps > best_score:
            best_score = ps
            best_pts = [list(p) for p in perturbed]

        remaining = deadline - time.time() - 10
        if remaining < 5:
            break
        sa_time = min(remaining * 0.7, 15.0)
        pts, sc = _local_search_sa(perturbed, ps, time.time() + sa_time, seed=100 + trial)
        if sc > best_score:
            best_score = sc
            best_pts = [list(p) for p in pts]

    # 4. Try a couple of random box starts
    for trial in range(3):
        if time.time() > deadline - 10:
            break
        rp = _generate_random_box(rng, half=25 + trial * 10)
        rs = _score(rp)
        if rs < 0:
            continue
        if rs > best_score:
            best_score = rs
            best_pts = [list(p) for p in rp]
        remaining = deadline - time.time() - 10
        if remaining < 5:
            break
        sa_time = min(remaining * 0.5, 12.0)
        pts, sc = _local_search_sa(rp, rs, time.time() + sa_time, seed=200 + trial)
        if sc > best_score:
            best_score = sc
            best_pts = [list(p) for p in pts]

    # 5. Final deterministic local polish
    if best_pts is not None and time.time() < deadline - 5:
        best_pts, best_score = _local_polish(best_pts, deadline - 2.0)

    # Fallback
    if best_pts is None or best_score < 0:
        best_pts = _generate_grid_4x4()
        best_score = _score(best_pts)

    return best_pts
