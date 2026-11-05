import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        return ('v', y1)
    if dy == 0:
        return ('h', x1)
    g = _gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    if dx < 0:
        dx, dy = -dx, -dy
    c = dy * x1 - dx * y1
    return ('s', dx, dy, c)

def _find_lines(points):
    n = len(points)
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
    return lines

def _is_valid(points):
    n = len(points)
    if n != 20:
        return False
    seen = set()
    for p in points:
        if p in seen:
            return False
        seen.add(p)
        if abs(p[0]) > 100 or abs(p[1]) > 100:
            return False
    lines = _find_lines(points)
    for key, pts in lines.items():
        if len(pts) >= 4:
            return False
    return True

def _compute_score(points, deadline):
    n = len(points)
    lines = _find_lines(points)
    triples = []
    for key, pts in lines.items():
        if len(pts) == 3:
            triples.append(frozenset(pts))
    
    if not triples:
        return n
    
    # Precompute for each point, list of triples it's in
    point_to_triples = [[] for _ in range(n)]
    for ti, t in enumerate(triples):
        for p in t:
            point_to_triples[p].append(ti)
    
    # Order points by number of triples (most constrained first)
    order = sorted(range(n), key=lambda p: -len(point_to_triples[p]))
    
    best = [0]
    
    def branch(chosen, idx):
        if time.time() > deadline:
            return
        cnt = len(chosen)
        if idx == n:
            if cnt > best[0]:
                best[0] = cnt
            return
        # Upper bound: cnt + remaining points
        if cnt + (n - idx) <= best[0]:
            return
        
        p = order[idx]
        
        # Try including p
        can_include = True
        for ti in point_to_triples[p]:
            t = triples[ti]
            others = [q for q in t if q != p]
            if len(others) == 2 and others[0] in chosen and others[1] in chosen:
                can_include = False
                break
        if can_include:
            chosen.add(p)
            branch(chosen, idx + 1)
            chosen.discard(p)
        
        # Try excluding p
        branch(chosen, idx + 1)
    
    branch(set(), 0)
    return best[0]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_score = 20
    best_points = [[x, x * x] for x in range(-10, 10)]
    
    random.seed(42)
    attempts = 0
    
    while time.time() < deadline:
        attempts += 1
        
        if attempts <= 50:
            # Structured grid-like with perturbations to create triples
            base_x = random.randint(-40, 40)
            base_y = random.randint(-40, 40)
            sx = random.randint(4, 10)
            sy = random.randint(4, 10)
            points = []
            for i in range(5):
                for j in range(4):
                    x = base_x + i * sx + random.randint(-1, 1)
                    y = base_y + j * sy + random.randint(-1, 1)
                    if -100 <= x <= 100 and -100 <= y <= 100:
                        points.append([x, y])
            while len(points) < 20:
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                points.append([x, y])
            points = points[:20]
        elif attempts <= 100:
            # Deliberate triple construction: pick lines and place 3 points on each
            points = []
            used = set()
            # Place 6-7 triples (18-21 points, overlap to get 20)
            for _ in range(8):
                if len(points) >= 20:
                    break
                # Pick a random line
                x1 = random.randint(-100, 100)
                y1 = random.randint(-100, 100)
                dx = random.randint(-20, 20)
                dy = random.randint(-20, 20)
                if dx == 0 and dy == 0:
                    continue
                # Place 3 points on this line
                placed = 0
                for k in range(5):
                    if len(points) >= 20:
                        break
                    x = x1 + k * dx
                    y = y1 + k * dy
                    if -100 <= x <= 100 and -100 <= y <= 100 and (x, y) not in used:
                        points.append([x, y])
                        used.add((x, y))
                        placed += 1
            while len(points) < 20:
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                if (x, y) not in used:
                    points.append([x, y])
                    used.add((x, y))
            points = points[:20]
        else:
            # Fully random
            points = []
            used = set()
            while len(points) < 20:
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                if (x, y) not in used:
                    points.append([x, y])
                    used.add((x, y))
        
        if len(set(tuple(p) for p in points)) != 20:
            continue
        if not _is_valid(points):
            continue
        
        score = _compute_score(points, deadline)
        if score < best_score:
            best_score = score
            best_points = [list(p) for p in points]
            if best_score <= 8:
                break
    
    return best_points
# EVOLVE-BLOCK-END
