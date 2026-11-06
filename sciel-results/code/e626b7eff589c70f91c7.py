import time, math, random
from collections import defaultdict

def _line_score(pts):
    n = len(pts)
    line_pts = defaultdict(set)
    for i in range(n):
        x1, y1 = pts[i]
        for j in range(i+1, n):
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            g = math.gcd(abs(dx), abs(dy))
            if g:
                dx //= g; dy //= g
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx; dy = -dy
            c = dy * x1 - dx * y1
            key = (dx, dy, c)
            line_pts[key].add(i)
            line_pts[key].add(j)
    score = 0
    for key, s in line_pts.items():
        if len(s) == 4:
            score += 1
    return score

def _count_4_lines_fast(pts):
    """Faster scoring: use a dict mapping line key to count of points on it."""
    n = len(pts)
    line_count = defaultdict(int)
    line_members = defaultdict(set)
    for i in range(n):
        x1, y1 = pts[i]
        for j in range(i+1, n):
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            g = math.gcd(abs(dx), abs(dy))
            if g:
                dx //= g; dy //= g
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx; dy = -dy
            c = dy * x1 - dx * y1
            key = (dx, dy, c)
            line_members[key].add(i)
            line_members[key].add(j)
    score = 0
    for key, s in line_members.items():
        if len(s) == 4:
            score += 1
    return score

def construct():
    start = time.time()
    deadline = start + 115.0  # safety margin under 150s

    def in_bounds(p):
        return abs(p[0]) <= 1000 and abs(p[1]) <= 1000

    def valid(pts):
        if len(set(tuple(p) for p in pts)) != 16:
            return False
        return all(in_bounds(p) for p in pts)

    # Try multiple initial configurations
    def make_grid(offset_x, offset_y, step_x, step_y):
        pts = []
        for i in range(4):
            for j in range(4):
                x = offset_x + i * step_x
                y = offset_y + j * step_y
                pts.append([x, y])
        return pts

    # Initial candidates
    candidates = []
    # 4x4 grid variants
    for ox in range(-2, 3):
        for oy in range(-2, 3):
            for sx in [1, 2, 3]:
                for sy in [1, 2, 3]:
                    pts = make_grid(ox, oy, sx, sy)
                    if valid(pts):
                        candidates.append(pts)
    
    # Random initial configs
    random.seed(42)
    for _ in range(50):
        pts = []
        for _ in range(16):
            x = random.randint(-50, 50)
            y = random.randint(-50, 50)
            pts.append([x, y])
        if valid(pts):
            candidates.append(pts)

    best = None
    best_score = -1
    for cand in candidates:
        sc = _line_score(cand)
        if sc > best_score:
            best_score = sc
            best = [list(p) for p in cand]

    # Simulated annealing + local search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Local search phase
        deltas = []
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                if dx != 0 or dy != 0:
                    deltas.append((dx, dy))
        
        for i in range(16):
            if time.time() >= deadline:
                break
            x, y = best[i]
            for dx, dy in deltas:
                nx, ny = x + dx, y + dy
                if not in_bounds((nx, ny)):
                    continue
                cand = [list(p) for p in best]
                cand[i] = [nx, ny]
                if not valid(cand):
                    continue
                sc = _line_score(cand)
                if sc > best_score:
                    best = cand
                    best_score = sc
                    x, y = nx, ny
                    improved = True
                    if time.time() >= deadline:
                        break
            if time.time() >= deadline:
                break
        
        # Random kick with simulated annealing
        if time.time() < deadline:
            # Check if we've been stuck
            if not improved:
                for _ in range(500):
                    if time.time() >= deadline:
                        break
                    i = random.randrange(16)
                    x, y = best[i]
                    nx = x + random.randint(-10, 10)
                    ny = y + random.randint(-10, 10)
                    if not in_bounds((nx, ny)):
                        continue
                    cand = [list(p) for p in best]
                    cand[i] = [nx, ny]
                    if not valid(cand):
                        continue
                    sc = _line_score(cand)
                    if sc > best_score:
                        best = cand
                        best_score = sc
                        improved = True

    # Ensure distinct and return
    seen = set()
    out = []
    for p in best:
        t = tuple(p)
        if t not in seen:
            seen.add(t)
            out.append([p[0], p[1]])
    return out[:16]
