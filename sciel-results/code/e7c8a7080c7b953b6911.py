import random
import time
from itertools import combinations

def construct():
    t0 = time.time()
    DEADLINE = 135  # safety margin before 150s kill

    def now():
        return time.time() - t0

    def collinear(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) == (c[0] - a[0]) * (b[1] - a[1])

    def has_4_collinear(pts):
        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                cnt = 0
                for k in range(n):
                    if k != i and k != j and collinear(pts[i], pts[j], pts[k]):
                        cnt += 1
                if cnt >= 2:
                    return True
        return False

    def get_triples(pts):
        n = len(pts)
        triples = []
        for i, j, k in combinations(range(n), 3):
            if collinear(pts[i], pts[j], pts[k]):
                triples.append((i, j, k))
        return triples

    def exact_score(pts):
        """Exact max subset with no three collinear via branch & bound."""
        n = len(pts)
        triples = get_triples(pts)
        if not triples:
            return n

        # Build: for each point, which pairs of other points form a triple with it
        # conflicts[i] = list of (j, k) such that {i,j,k} is a triple
        conflicts = [[] for _ in range(n)]
        for (a, b, c) in triples:
            conflicts[a].append((b, c))
            conflicts[b].append((a, c))
            conflicts[c].append((a, b))

        # Degree: number of triples each point is in
        deg = [0] * n
        for (a, b, c) in triples:
            deg[a] += 1
            deg[b] += 1
            deg[c] += 1

        # Order by degree descending
        order = sorted(range(n), key=lambda x: -deg[x])
        pos = [0] * n
        for new_i, old_i in enumerate(order):
            pos[old_i] = new_i

        # Rebuild conflicts in new order
        new_conflicts = [[] for _ in range(n)]
        for (a, b, c) in triples:
            na, nb, nc = pos[a], pos[b], pos[c]
            new_conflicts[na].append((nb, nc))
            new_conflicts[nb].append((na, nc))
            new_conflicts[nc].append((na, nb))

        best = [0]

        def bb(idx, selected, cnt):
            if cnt + (n - idx) <= best[0]:
                return
            if idx == n:
                if cnt > best[0]:
                    best[0] = cnt
                return
            # Try include
            ok = True
            for (j, k) in new_conflicts[idx]:
                if (selected & (1 << j)) and (selected & (1 << k)):
                    ok = False
                    break
            if ok:
                bb(idx + 1, selected | (1 << idx), cnt + 1)
            # Try exclude
            bb(idx + 1, selected, cnt)

        bb(0, 0, 0)
        return best[0]

    def valid(pts):
        if len(pts) != 20:
            return False
        if len(set(tuple(p) for p in pts)) < 20:
            return False
        for p in pts:
            if abs(p[0]) > 100 or abs(p[1]) > 100:
                return False
        if has_4_collinear(pts):
            return False
        return True

    def gen_clustered(seed_val):
        """Generate 20 points: 6 lines × 3 points + 2 strays, with jitter."""
        rng = random.Random(seed_val)
        pts = []
        used = set()

        # 6 lines with distinct slopes/intercepts
        line_params = []
        slopes = [0, 1, -1, 2, -2, 1/2, -1/2, 3, -3]
        used_slopes = set()
        for _ in range(6):
            m = rng.choice([s for s in slopes if s not in used_slopes])
            used_slopes.add(m)
            b = rng.randint(-30, 30)
            line_params.append((m, b))

        # 3 points per line
        for (m, b) in line_params:
            xs = sorted(rng.sample(range(-80, 81), 3))
            for x in xs:
                y = m * x + b
                # Jitter to avoid exact collinearity issues and 4-collinear
                jx = rng.randint(-2, 2)
                jy = rng.randint(-2, 2)
                px, py = x + jx, int(y) + jy
                px = max(-100, min(100, px))
                py = max(-100, min(100, py))
                if (px, py) not in used:
                    pts.append([px, py])
                    used.add((px, py))

        # 2 strays
        for _ in range(2):
            for _ in range(200):
                px = rng.randint(-100, 100)
                py = rng.randint(-100, 100)
                if (px, py) not in used:
                    pts.append([px, py])
                    used.add((px, py))
                    break

        # Fix any 4-collinear
        for _ in range(300):
            if not has_4_collinear(pts):
                break
            n = 20
            moved = False
            for i in range(n):
                for j in range(i + 1, n):
                    extra = [k for k in range(n) if k != i and k != j and collinear(pts[i], pts[j], pts[k])]
                    if len(extra) >= 2:
                        victim = rng.choice(extra)
                        ox, oy = pts[victim]
                        for _ in range(50):
                            nx = ox + rng.randint(-8, 8)
                            ny = oy + rng.randint(-8, 8)
                            nx = max(-100, min(100, nx))
                            ny = max(-100, min(100, ny))
                            if (nx, ny) not in [tuple(q) for q in pts if q is not pts[victim]]:
                                pts[victim] = [nx, ny]
                                if not has_4_collinear(pts):
                                    moved = True
                                    break
                        if not moved:
                            pts[victim] = [ox, oy]
                        break
                if moved:
                    break

        return pts if len(pts) == 20 else None

    def gen_parabola(seed_val):
        """Points near a parabola: few collinear triples, but we want many."""
        rng = random.Random(seed_val)
        pts = []
        used = set()
        for i in range(20):
            x = i - 10
            y = (x * x) // 2 + rng.randint(-2, 2)
            x += rng.randint(-1, 1)
            x = max(-100, min(100, x))
            y = max(-100, min(100, y))
            if (x, y) not in used:
                pts.append([x, y])
                used.add((x, y))
        while len(pts) < 20:
            x = rng.randint(-100, 100)
            y = rng.randint(-100, 100)
            if (x, y) not in used:
                pts.append([x, y])
                used.add((x, y))
        # Fix 4-collinear
        for _ in range(200):
            if not has_4_collinear(pts):
                break
            n = 20
            moved = False
            for i in range(n):
                for j in range(i + 1, n):
                    extra = [k for k in range(n) if k != i and k != j and collinear(pts[i], pts[j], pts[k])]
                    if len(extra) >= 2:
                        victim = rng.choice(extra)
                        ox, oy = pts[victim]
                        for _ in range(30):
                            nx = ox + rng.randint(-6, 6)
                            ny = oy + rng.randint(-6, 6)
                            nx = max(-100, min(100, nx))
                            ny = max(-100, min(100, ny))
                            if (nx, ny) not in [tuple(q) for q in pts if q is not pts[victim]]:
                                pts[victim] = [nx, ny]
                                if not has_4_collinear(pts):
                                    moved = True
                                    break
                        if not moved:
                            pts[victim] = [ox, oy]
                        break
                if moved:
                    break
        return pts if len(pts) == 20 else None

    def local_search(pts, budget):
        """Perturb points to reduce score."""
        pts = [p[:] for p in pts]
        cur = exact_score(pts)
        best = cur
        best_pts = [p[:] for p in pts]
        rng = random.Random(int(now() * 10000))

        while now() < budget:
            improved = False
            for i in range(20):
                if now() > budget:
                    break
                ox, oy = pts[i]
                for _ in range(12):
                    if now() > budget:
                        break
                    if rng.random() < 0.6:
                        nx = ox + rng.randint(-5, 5)
                        ny = oy + rng.randint(-5, 5)
                    else:
                        nx = rng.randint(-100, 100)
                        ny = rng.randint(-100, 100)
                    nx = max(-100, min(100, nx))
                    ny = max(-100, min(100, ny))
                    if (nx, ny) in [tuple(q) for q in pts if q is not pts[i]]:
                        continue
                    pts[i] = [nx, ny]
                    if has_4_collinear(pts):
                        pts[i] = [ox, oy]
                        continue
                    new_s = exact_score(pts)
                    if new_s < cur:
                        cur = new_s
                        if cur < best:
                            best = cur
                            best_pts = [q[:] for q in pts]
                        improved = True
                        break
                    else:
                        pts[i] = [ox, oy]
                if improved:
                    break
            if not improved:
                break
        return best_pts, best

    # Main
    best_pts = None
    best_score = 21

    # Phase 1: Try many clustered configurations
    for seed in range(60):
        if now() > DEADLINE - 20:
            break
        pts = gen_clustered(seed)
        if pts is None or not valid(pts):
            continue
        s = exact_score(pts)
        if s < best_score:
            best_score = s
            best_pts = [p[:] for p in pts]

    # Phase 2: Try parabola configs
    for seed in range(20):
        if now() > DEADLINE - 15:
            break
        pts = gen_parabola(seed)
        if pts is None or not valid(pts):
            continue
        s = exact_score(pts)
        if s < best_score:
            best_score = s
            best_pts = [p[:] for p in pts]

    # Phase 3: Local search on best
    if best_pts is not None and now() < DEADLINE - 5:
        lp, ls = local_search(best_pts, DEADLINE - 3)
        if ls < best_score:
            best_score = ls
            best_pts = lp

    # Fallback
    if best_pts is None:
        pts = []
        used = set()
        rng = random.Random(42)
        for _ in range(20):
            while True:
                x = rng.randint(-100, 100)
                y = rng.randint(-100, 100)
                if (x, y) not in used:
                    used.add((x, y))
                    pts.append([x, y])
                    break
        for _ in range(200):
            if not has_4_collinear(pts):
                break
            n = 20
            for i in range(n):
                for j in range(i + 1, n):
                    extra = [k for k in range(n) if k != i and k != j and collinear(pts[i], pts[j], pts[k])]
                    if len(extra) >= 2:
                        v = rng.choice(extra)
                        ox, oy = pts[v]
                        for _ in range(30):
                            nx = ox + rng.randint(-10, 10)
                            ny = oy + rng.randint(-10, 10)
                            nx = max(-100, min(100, nx))
                            ny = max(-100, min(100, ny))
                            if (nx, ny) not in [tuple(q) for q in pts if q is not pts[v]]:
                                pts[v] = [nx, ny]
                                if not has_4_collinear(pts):
                                    break
                        break
                break
        best_pts = pts
        best_score = exact_score(pts)

    return best_pts
