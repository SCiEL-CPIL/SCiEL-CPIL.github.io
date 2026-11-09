import time
import random

DEADLINE = 140.0

def _collinear(p, q, r):
    return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]) == 0

def _has_four_collinear(points):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            count = 2
            for k in range(n):
                if k != i and k != j:
                    if _collinear(points[i], points[j], points[k]):
                        count += 1
                        if count >= 4:
                            return True
    return False

def _build_triples(points):
    n = len(points)
    triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _collinear(points[i], points[j], points[k]):
                    triples.append((i, j, k))
    return triples

def _max_no_three_collinear(points, node_limit=200000):
    n = len(points)
    triples = _build_triples(points)
    if not triples:
        return n
    v_triples = [[] for _ in range(n)]
    for t in triples:
        for v in t:
            v_triples[v].append(t)
    best = [0]
    nodes = [0]
    def branch(selected, remaining, cur):
        nodes[0] += 1
        if nodes[0] > node_limit:
            return
        if cur + len(remaining) <= best[0]:
            return
        if not remaining:
            if cur > best[0]:
                best[0] = cur
            return
        v = remaining[0]
        rest = remaining[1:]
        conflict = False
        for t in v_triples[v]:
            others = [x for x in t if x != v]
            if all(o in selected for o in others):
                conflict = True
                break
        if not conflict:
            new_sel = selected | {v}
            new_rem = []
            for u in rest:
                ok = True
                for t in v_triples[u]:
                    others = [x for x in t if x != u]
                    if all(o in new_sel for o in others):
                        ok = False
                        break
                if ok:
                    new_rem.append(u)
            branch(new_sel, new_rem, cur + 1)
        branch(selected, rest, cur)
    branch(frozenset(), list(range(n)), 0)
    return best[0]

def construct():
    start = time.time()
    best_score = 20
    best_points = [[x, x * x] for x in range(-10, 10)]
    def elapsed():
        return time.time() - start
    def valid(pts):
        if len(set(tuple(p) for p in pts)) != 20:
            return False
        for p in pts:
            if abs(p[0]) > 100 or abs(p[1]) > 100:
                return False
        return not _has_four_collinear(pts)
    for _ in range(1000):
        if elapsed() > DEADLINE - 20:
            break
        pts = []
        for _ in range(3000):
            if len(pts) >= 20:
                break
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if [x, y] not in pts:
                pts.append([x, y])
        if len(pts) < 20 or not valid(pts):
            continue
        score = _max_no_three_collinear(pts)
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in pts]
        if elapsed() > DEADLINE - 10:
            break
        for _ in range(200):
            if elapsed() > DEADLINE - 10:
                break
            i = random.randint(0, 19)
            old = pts[i]
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if [x, y] in pts:
                continue
            pts[i] = [x, y]
            if valid(pts):
                ns = _max_no_three_collinear(pts)
                if ns <= score:
                    score = ns
                    if score < best_score:
                        best_score = score
                        best_points = [p[:] for p in pts]
                else:
                    pts[i] = old
    return best_points
