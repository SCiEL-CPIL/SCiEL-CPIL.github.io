import time
import math
import random
import sys

def construct():
    random.seed(42)
    start = time.time()
    DEADLINE = 130.0  # safety margin before 150s kill

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

    def precompute_triples(pts):
        n = len(pts)
        triples_with_idx = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                for k in range(j + 1, n):
                    x3, y3 = pts[k]
                    if dx * (y3 - y1) == dy * (x3 - x1):
                        triples_with_idx[i].append((j, k))
                        triples_with_idx[j].append((i, k))
                        triples_with_idx[k].append((i, j))
        return triples_with_idx

    def exact_score(pts):
        n = len(pts)
        if n < 3:
            return n
        triples_with_idx = precompute_triples(pts)
        best = [0]

        # Order: points in more triples first
        order = sorted(range(n), key=lambda i: len(triples_with_idx[i]), reverse=True)
        
        import sys
        sys.setrecursionlimit(10000)

        def bbb(step, cur_mask, cur_size):
            if elapsed() > DEADLINE - 5:
                return
            if cur_size > best[0]:
                best[0] = cur_size
            if step == n:
                return
            # prune
            if cur_size + (n - step) <= best[0]:
                return
            idx = order[step]
            # Option exclude
            bbb(step + 1, cur_mask, cur_size)
            # Option include: check conflict
            ok = True
            for a, b in triples_with_idx[idx]:
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

        # Strategy: pick several 3-point lines with distinct "directions"
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
            # pick a starting point
            for _ in range(50):
                sx = rng.randint(-95, 95)
                sy = rng.randint(-95, 95)
                # 3 points
                cand = [(sx, sy), (sx + d[0], sy + d[1]), (sx + 2*d[0], sy + 2*d[1])]
                # all in bounds and distinct
                if any(not (-100 <= x <= 100 and -100 <= y <= 100) for x, y in cand):
                    continue
                if len(set(cand)) != 3:
                    continue
                # Try to add the ones not already present
                new_pts = []
                for p in cand:
                    if p not in used:
                        new_pts.append(p)
                if not new_pts:
                    continue
                # Check adding new_pts doesn't create 4-collinear with existing
                trial = pts + new_pts
                if has_4_collinear(trial):
                    continue
                pts = trial
                for p in new_pts:
                    used.add(p)
                if len(pts) >= 20:
                    break

        if len(pts) < 20:
            # fill randomly
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

    # Build several candidates
    candidates = []
    for seed in range(20):
        if elapsed() > DEADLINE - 50:
            break
        cfg = build_dense_config(seed)
        if len(cfg) == 20 and not has_4_collinear(cfg):
            candidates.append(cfg)

    if not candidates:
        # fallback
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

    # Evaluate and pick best
    best_pts = None
    best_val = float('inf')
    for cfg in candidates:
        if elapsed() > DEADLINE - 10:
            break
        s = exact_score(cfg)
        if s < best_val:
            best_val = s
            best_pts = cfg

    if best_pts is None:
        best_pts = candidates[0]
        best_val = exact_score(best_pts)

    # Local search to improve
    cur_pts = [tuple(p) for p in best_pts]
    cur_val = best_val
    iters = 0
    while elapsed() < DEADLINE - 10 and iters < 600:
        iters += 1
        idx = random.randint(0, 19)
        old = cur_pts[idx]
        if random.random() < 0.5:
            np = (old[0] + random.randint(-8, 8), old[1] + random.randint(-8, 8))
        else:
            np = (random.randint(-100, 100), random.randint(-100, 100))
        if not (-100 <= np[0] <= 100 and -100 <= np[1] <= 100):
            continue
        if np in cur_pts:
            continue
        trial = cur_pts[:]
        trial[idx] = np
        if has_4_collinear(trial):
            continue
        s = exact_score(trial)
        if s < cur_val:
            cur_val = s
            cur_pts = trial
            best_val = s
            best_pts = cur_pts
        elif s == cur_val and random.random() < 0.2 and elapsed() < DEADLINE - 25:
            cur_pts = trial
            best_pts = cur_pts

    return [[x, y] for x, y in best_pts]

if __name__ == "__main__":
    pts = construct()
    print(pts)
