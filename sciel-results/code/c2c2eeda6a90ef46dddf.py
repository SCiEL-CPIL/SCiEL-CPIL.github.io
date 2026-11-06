import time
import random
import itertools
from collections import defaultdict
from math import gcd

def construct():
    t0 = time.time()
    deadline = t0 + 142.0   # safety margin vs 150s kill

    def elapsed():
        return time.time() - t0

    def are_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])

    def line_key(a, b):
        # canonical integer line through a,b: (A,B,C) with gcd-normalized, sign-fixed
        A = b[1] - a[1]
        B = a[0] - b[0]
        C = b[0] * a[1] - a[0] * b[1]
        g = gcd(abs(A), gcd(abs(B), abs(C)))
        if g:
            A //= g; B //= g; C //= g
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A, B, C = -A, -B, -C
        return (A, B, C)

    def has_4_collinear(pts):
        n = len(pts)
        for i in range(n):
            counts = defaultdict(int)
            xi, yi = pts[i]
            for j in range(n):
                if i == j:
                    continue
                xj, yj = pts[j]
                dx = xj - xi; dy = yj - yi
                g = gcd(abs(dx), abs(dy))
                if g == 0:
                    continue
                counts[(dx // g, dy // g)] += 1
            for c in counts.values():
                if c >= 3:
                    return True
        return False

    # Fast exact branch & bound: largest subset with no 3 collinear.
    # Uses line triple counter for O(lines) validity check and cheap upper bound.
    def max_no_3_collinear(points, node_cap=4000000):
        n = len(points)
        # line membership: line -> list of point indices
        line_map = defaultdict(list)
        for i in range(n):
            for j in range(i + 1, n):
                k = line_key(points[i], points[j])
                line_map[k].append(i)
                line_map[k].append(j)
        # de-dup
        for k in line_map:
            line_map[k] = sorted(set(line_map[k]))
        # lines that can form a triple (>=3 points)
        active_lines = [lst for lst in line_map.values() if len(lst) >= 3]
        line_sets = [set(lst) for lst in active_lines]
        line_counts = [0] * len(line_sets)
        # For each point, list of (line_idx, mask_of_other_members)
        pt_line_info = [[] for _ in range(n)]
        for li, lst in enumerate(active_lines):
            m = 0
            for p in lst:
                m |= (1 << p)
            for p in lst:
                others = m & ~(1 << p)
                pt_line_info[p].append((li, others))
        best = 0
        nodes = 0
        # ordering: points with most line-membership first (more constrained)
        order = sorted(range(n), key=lambda p: -len(pt_line_info[p]))
        # map new index -> point
        new_pts = [points[p] for p in order]
        info = [pt_line_info[p] for p in order]
        best = 0
        nodes = 0
        max_nodes = node_cap
        def backtrack(idx, mask):
            nonlocal best, nodes
            nodes += 1
            if nodes > max_nodes:
                return
            cur = bin(mask).count('1')
            # upper bound: cur + remaining
            if cur + (n - idx) <= best:
                return
            if idx == n:
                if cur > best:
                    best = cur
                return
            # try include
            p = idx
            ok = True
            for (li, others) in info[p]:
                if line_counts[li] + (bin(mask & others).count('1')) >= 2:
                    # adding p would make 3 on this line
                    ok = False
                    break
            if ok:
                for (li, _) in info[p]:
                    line_counts[li] += 1
                backtrack(idx + 1, mask | (1 << p))
                for (li, _) in info[p]:
                    line_counts[li] -= 1
            # try exclude
            backtrack(idx + 1, mask)
        backtrack(0, 0)
        return best

    # ---- Structured deterministic construction: pack points onto shared hub lines ----
    def build_hub_config(seed=0):
        rng = random.Random(seed)
        # Choose a few hub lines through a common-ish region to maximize overlap.
        # We'll use lines through origin with distinct slopes, plus a couple offset lines.
        # Place points at integer intersections/parametric positions, ensuring no 4 collinear
        # by capping 3 per line and using distinct points.
        pts = []
        used = set()
        def add(p):
            if p in used:
                return False
            if abs(p[0]) > 100 or abs(p[1]) > 100:
                return False
            used.add(p)
            pts.append(p)
            return True

        # Hub lines: through (0,0) with slopes 0,1,-1,2,-2 (dy/dx)
        # Parametrize: for slope m = dy/dx, points (t, m*t) integer.
        # We'll pick 3 points per line from a parameter set, avoiding reuse.
        param_sets = {
            0:   [(-4, 0), (-2, 0), (2, 0), (4, 0)],
            1:   [(-4, -4), (-2, -2), (2, 2), (4, 4)],
            -1:  [(-4, 4), (-2, 2), (2, -2), (4, -4)],
            2:   [(-3, -6), (-1, -2), (1, 2), (3, 6)],
            -2:  [(-3, 6), (-1, 2), (1, -2), (3, -6)],
        }
        # Add 3 per line = 15 points
        for slope, plist in param_sets.items():
            chosen = 0
            for p in plist:
                if chosen >= 3:
                    break
                if add(p):
                    chosen += 1
        # Need 5 more; use a second hub at (5,5) with slopes 0,1,-1
        hub2 = (5, 5)
        extra_sets = {
            0:   [(5, 1), (5, 3), (5, 9), (5, 11)],
            1:   [(1, 1), (3, 3), (9, 9), (11, 11)],
            -1:  [(1, 9), (3, 7), (9, -1), (11, -3)],
        }
        for slope, plist in extra_sets.items():
            chosen = 0
            for p in plist:
                if chosen >= 2:
                    break
                if add(p):
                    chosen += 1
            if len(pts) >= 20:
                break
        # Fill remaining with near-hub random distinct points (avoid 4-collinear by checking)
        while len(pts) < 20:
            x = rng.randint(-30, 30)
            y = rng.randint(-30, 30)
            p = (x, y)
            if p in used:
                continue
            cand = pts + [p]
            if has_4_collinear(cand):
                continue
            used.add(p)
            pts.append(p)
        return [list(p) for p in pts]

    best_config = None
    best_score = 10**9

    # Deterministic hub configs with a few seeds
    for seed in [0, 1, 2, 3, 4, 5]:
        if elapsed() > deadline - 30:
            break
        cfg = build_hub_config(seed)
        if len(set(map(tuple, cfg))) != 20:
            continue
        if has_4_collinear(cfg):
            continue
        if any(abs(p[0]) > 100 or abs(p[1]) > 100 for p in cfg):
            continue
        s = max_no_3_collinear(cfg)
        if s < best_score:
            best_score = s
            best_config = [list(p) for p in cfg]

    # Local improvement: move one point to a position that increases triple density
    # (reduces the no-3-collinear subset), while preserving validity.
    if best_config is not None and elapsed() < deadline - 15:
        cur = [list(p) for p in best_config]
        cur_score = best_score
        improved = True
        while improved and elapsed() < deadline - 8:
            improved = False
            rng = random.Random(int(elapsed() * 1000) & 0xffff)
            for _ in range(60):
                if elapsed() > deadline - 8:
                    break
                idx = rng.randint(0, 19)
                ox, oy = cur[idx]
                # candidate new position: try to land on a line already used by 2 others
                # (creates a new triple) while not making 4 collinear.
                best_move = None
                best_delta = 0
                for trial in range(80):
                    if trial < 40:
                        # near current
                        nx = ox + rng.randint(-6, 6)
                        ny = oy + rng.randint(-6, 6)
                    else:
                        nx = rng.randint(-60, 60)
                        ny = rng.randint(-60, 60)
                    if nx == ox and ny == oy:
                        continue
                    if abs(nx) > 100 or abs(ny) > 100:
                        continue
                    cand = cur[:]
                    cand[idx] = [nx, ny]
                    if len(set(map(tuple, cand))) != 20:
                        continue
                    if has_4_collinear(cand):
                        continue
                    # quick triple count to prefer denser configs
                    triples = 0
                    for i in range(20):
                        for j in range(i + 1, 20):
                            if i == idx or j == idx:
                                continue
                            if are_collinear(cand[idx], cand[i], cand[j]):
                                triples += 1
                    # score proxy: more triples -> likely smaller no-3 subset
                    if best_move is None or triples > best_move[0]:
                        best_move = (triples, cand)
                if best_move is None:
                    continue
                _, cand = best_move
                # exact check
                ns = max_no_3_collinear(cand)
                if ns < cur_score:
                    cur = cand
                    cur_score = ns
                    best_score = ns
                    best_config = [list(p) for p in cur]
                    improved = True
                    break
                elif ns == cur_score and rng.random() < 0.3:
                    # accept neutral move to escape local structure
                    cur = cand
                    improved = True
                    break

    if best_config is None:
        # fallback: simple parabola (valid, no 4 collinear)
        return [[x, x * x] for x in range(-10, 10)]

    return best_config
