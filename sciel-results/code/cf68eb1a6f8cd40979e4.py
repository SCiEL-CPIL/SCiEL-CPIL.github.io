import time
import math
import random
import sys

def construct():
    random.seed(42)
    start = time.time()
    DEADLINE = 140.0

    def elapsed():
        return time.time() - start

    def line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1
        g = math.gcd(A, math.gcd(B, C))
        if g == 0:
            return None
        A //= g; B //= g; C //= g
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A = -A; B = -B; C = -C
        return (A, B, C)

    def has_4_collinear(pts):
        n = len(pts)
        if n < 4:
            return False
        seen = set()
        for i in range(n):
            for j in range(i + 1, n):
                key = line_key(pts[i], pts[j])
                if key is None or key in seen:
                    continue
                seen.add(key)
                A, B, C = key
                cnt = 0
                for k in range(n):
                    x3, y3 = pts[k]
                    if A * x3 + B * y3 + C == 0:
                        cnt += 1
                if cnt >= 4:
                    return True
        return False

    def build_triple_structure(pts):
        """Precompute which triples are collinear. Returns list of frozensets of {i,j,k}."""
        n = len(pts)
        triples = []
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                for k in range(j + 1, n):
                    x3, y3 = pts[k]
                    if dx * (y3 - y1) == dy * (x3 - x1):
                        triples.append((i, j, k))
        return triples

    def count_collinear_triples(pts):
        """Count all collinear triples in pts."""
        n = len(pts)
        cnt = 0
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                for k in range(j + 1, n):
                    x3, y3 = pts[k]
                    if dx * (y3 - y1) == dy * (x3 - x1):
                        cnt += 1
        return cnt

    def exact_score(pts):
        n = len(pts)
        if n < 3:
            return n
        triples = build_triple_structure(pts)
        best = [0]

        order = sorted(range(n), key=lambda i: sum(1 for t in triples if i in t), reverse=True)
        triples_by_point = [[] for _ in range(n)]
        for t in triples:
            for idx in t:
                others = [x for x in t if x != idx]
                triples_by_point[idx].append(tuple(others))

        sys.setrecursionlimit(10000)

        def bbb(step, cur_mask, cur_size):
            if elapsed() > DEADLINE - 5:
                return
            if cur_size > best[0]:
                best[0] = cur_size
            if step == n:
                return
            if cur_size + (n - step) <= best[0]:
                return
            idx = order[step]
            # Option exclude
            bbb(step + 1, cur_mask, cur_size)
            # Option include
            ok = True
            for a, b in triples_by_point[idx]:
                if (cur_mask & (1 << a)) and (cur_mask & (1 << b)):
                    ok = False
                    break
            if ok:
                bbb(step + 1, cur_mask | (1 << idx), cur_size + 1)

        bbb(0, 0, 0)
        return best[0]

    def build_dense_config(seed=0):
        """Build 20 points with many 3-collinear triples, no 4 collinear."""
        rng = random.Random(seed)
        pts = []
        used = set()

        directions = [
            (1, 0), (0, 1), (1, 1), (1, -1), (2, 1), (1, 2), (2, -1), (1, -2),
            (3, 1), (1, 3), (3, -1), (-3, 1), (3, 2), (2, 3), (3, -2), (2, -3),
            (4, 1), (1, 4), (4, -1), (-4, 1), (4, 2), (2, 4), (4, -2), (2, -4),
            (5, 1), (1, 5), (5, -1), (-5, 1), (5, 2), (2, 5), (5, -2), (2, -5),
        ]
        rng.shuffle(directions)

        attempts = 0
        while len(pts) < 20 and attempts < 3000:
            attempts += 1
            d = rng.choice(directions)
            for _ in range(50):
                sx = rng.randint(-95, 95)
                sy = rng.randint(-95, 95)
                cand = [(sx, sy), (sx + d[0], sy + d[1]), (sx + 2*d[0], sy + 2*d[1])]
                if any(not (-100 <= x <= 100 and -100 <= y <= 100) for x, y in cand):
                    continue
                if len(set(cand)) != 3:
                    continue
                new_pts = [p for p in cand if p not in used]
                if not new_pts:
                    continue
                trial = pts + new_pts
                if has_4_collinear(trial):
                    continue
                pts = trial
                for p in new_pts:
                    used.add(p)
                if len(pts) >= 20:
                    break

        if len(pts) < 20:
            while len(pts) < 20:
                p = (rng.randint(-100, 100), rng.randint(-100, 100))
                if p in used:
                    continue
                trial = pts + [p]
                if has_4_collinear(trial):
                    continue
                pts.append(p)
                used.add(p)
        return pts[:20]

    # Build initial candidates
    candidates = []
    for seed in range(10):
        if elapsed() > DEADLINE - 60:
            break
        cfg = build_dense_config(seed)
        if len(cfg) == 20 and not has_4_collinear(cfg):
            candidates.append(cfg)

    if not candidates:
        pts = []
        used = set()
        rng = random.Random(7)
        while len(pts) < 20:
            p = (rng.randint(-100, 100), rng.randint(-100, 100))
            if p in used:
                continue
            if has_4_collinear(pts + [p]):
                continue
            pts.append(p)
            used.add(p)
        candidates = [pts]

    # Pick best initial by triple count (proxy for lower score)
    best_pts = None
    best_triples = -1
    for cfg in candidates:
        tc = count_collinear_triples(cfg)
        if tc > best_triples:
            best_triples = tc
            best_pts = cfg

    cur_pts = [tuple(p) for p in best_pts]
    cur_triples = best_triples
    best_score = exact_score(cur_pts)
    best_pts = cur_pts[:]

    # Simulated annealing with incremental scoring
    rng = random.Random(123)
    T = 5.0
    T_min = 0.01
    cooling = 0.995
    steps = 0
    max_steps = 50000

    while elapsed() < DEADLINE - 15 and steps < max_steps:
        steps += 1
        idx = rng.randint(0, 19)
        old = cur_pts[idx]
        if rng.random() < 0.6:
            np = (old[0] + rng.randint(-6, 6), old[1] + rng.randint(-6, 6))
        else:
            np = (rng.randint(-100, 100), rng.randint(-100, 100))
        if not (-100 <= np[0] <= 100 and -100 <= np[1] <= 100):
            continue
        if np in cur_pts:
            continue

        trial = cur_pts[:]
        trial[idx] = np
        if has_4_collinear(trial):
            continue

        # Incremental: count change in triples
        # Triples involving idx: for each pair (a,b) from remaining 19, check if collinear
        new_triples = 0
        for i in range(20):
            if i == idx:
                continue
            for j in range(i + 1, 20):
                if j == idx:
                    continue
                x1, y1 = trial[i]
                x2, y2 = trial[j]
                x3, y3 = trial[idx]
                dx = x2 - x1
                dy = y2 - y1
                if dx * (y3 - y1) == dy * (x3 - x1):
                    new_triples += 1

        # Old triples involving idx
        old_triples = 0
        for i in range(20):
            if i == idx:
                continue
            for j in range(i + 1, 20):
                if j == idx:
                    continue
                x1, y1 = cur_pts[i]
                x2, y2 = cur_pts[j]
                x3, y3 = old
                dx = x2 - x1
                dy = y2 - y1
                if dx * (y3 - y1) == dy * (x3 - y1):
                    old_triples += 1

        delta = new_triples - old_triples
        # We want to MAXIMIZE triples (minimize score)
        if delta > 0 or rng.random() < math.exp(delta / T):
            cur_pts = trial
            cur_triples += delta
            # Periodically check exact score
            if steps % 50 == 0:
                s = exact_score(cur_pts)
                if s < best_score:
                    best_score = s
                    best_pts = cur_pts[:]
        T *= cooling
        if T < T_min:
            T = T_min

    # Final exact score check
    final_score = exact_score(cur_pts)
    if final_score < best_score:
        best_score = final_score
        best_pts = cur_pts[:]

    return [[x, y] for x, y in best_pts]

if __name__ == "__main__":
    pts = construct()
    print(pts)
