import time
import itertools
import random

_DEADLINE = time.time() + 145  # safety margin before 150s kill


def _check_no_four(points):
    # True if no four points are collinear
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            cnt = 2
            for k in range(n):
                if k == i or k == j:
                    continue
                x3, y3 = points[k]
                if dx * (y3 - y1) == dy * (x3 - x1):
                    cnt += 1
                    if cnt >= 4:
                        return False
    return True


def _max_3free(points):
    # exact branch and bound: largest subset with no three collinear
    n = len(points)
    # adjacency: edge (i,j) if a third point is collinear with i,j
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            for k in range(n):
                if k == i or k == j:
                    continue
                x3, y3 = points[k]
                if dx * (y3 - y1) == dy * (x3 - x1):
                    adj[i].add(j)
                    adj[j].add(i)
                    break

    best = 0
    # order points by degree (higher first) for stronger pruning
    order = sorted(range(n), key=lambda v: -len(adj[v]))
    pos = [0] * n
    for idx, v in enumerate(order):
        pos[v] = idx

    # state: remaining vertices (as list), current selected set
    # Use recursion with candidate list
    def dfs(cands, selected):
        nonlocal best
        if not cands:
            if len(selected) > best:
                best = len(selected)
            return
        # upper bound: len(selected) + len(cands)
        if len(selected) + len(cands) <= best:
            return
        # pick first candidate
        v = cands[0]
        rest = cands[1:]
        # branch 1: include v
        # v must not conflict with selected
        ok = True
        for s in selected:
            if s in adj[v]:
                ok = False
                break
        if ok:
            # filter rest: remove neighbors of v
            new_rest = [c for c in rest if c not in adj[v]]
            dfs(new_rest, selected + [v])
        # branch 2: exclude v
        dfs(rest, selected)

    dfs(order, [])
    return best


def _gen_grid(a, b):
    # 5 rows x 4 cols
    pts = []
    for r in range(5):
        for c in range(4):
            x = c + a * r
            y = r + b * r
            pts.append([x, y])
    return pts


def construct():
    global _DEADLINE
    best_pts = None
    best_score = 10**9

    # Deterministic candidate set
    candidates = []
    for a in range(-6, 7):
        for b in range(-6, 7):
            if a == 0 and b == 0:
                continue
            candidates.append((a, b))

    # Shuffle with fixed seed for reproducibility but explore broadly
    rnd = random.Random(12345)
    rnd.shuffle(candidates)

    # Also add some random offsets
    extra = []
    for _ in range(60):
        a = rnd.randint(-10, 10)
        b = rnd.randint(-10, 10)
        if a == 0 and b == 0:
            continue
        extra.append((a, b))
    candidates.extend(extra)

    for a, b in candidates:
        if time.time() > _DEADLINE - 5:
            break
        pts = _gen_grid(a, b)
        # distinct check
        if len(set(tuple(p) for p in pts)) != 20:
            continue
        # bounds check
        if any(abs(x) > 100 or abs(y) > 100 for x, y in pts):
            continue
        # no-four check
        if not _check_no_four(pts):
            continue
        s = _max_3free(pts)
        if s < best_score:
            best_score = s
            best_pts = pts
            if best_score <= 7:
                # good enough, stop
                break

    if best_pts is None:
        # fallback: parabola (score 20, valid)
        return [[x, x * x] for x in range(-10, 10)]
    return best_pts
