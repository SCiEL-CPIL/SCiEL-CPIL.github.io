import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DEADLINE = 78.0

    def elapsed():
        return time.time() - t0

    # Precompute triple indices
    triples = []
    for i in range(30):
        for j in range(i+1, 30):
            for k in range(j+1, 30):
                triples.append((i, j, k))

    def score_of(points):
        cnt = defaultdict(int)
        for i, j, k in triples:
            xi, yi = points[i]
            xj, yj = points[j]
            xk, yk = points[k]
            d = abs(xi*(yj-yk) + xj*(yk-yi) + xk*(yi-yj))
            if d > 0:
                cnt[d] += 1
        if not cnt:
            return 0
        return max(cnt.values())

    # Start with 6x5 grid
    points = [(x, y) for x in range(6) for y in range(5)]
    best_score = score_of(points)
    best_points = points[:]

    # Try other grids
    for m, n in [(5,6), (10,3), (15,2), (30,1), (6,5)]:
        if m*n == 30:
            pts = [(x, y) for x in range(m) for y in range(n)]
            s = score_of(pts)
            if s > best_score:
                best_score = s
                best_points = pts[:]

    # Local search
    points = best_points[:]
    no_improve = 0
    max_no_improve = 500

    while elapsed() < DEADLINE:
        idx = random.randint(0, 29)
        old_x, old_y = points[idx]
        new_x = random.randint(-30, 30)
        new_y = random.randint(-30, 30)
        if (new_x, new_y) == (old_x, old_y):
            continue
        if any(points[i] == (new_x, new_y) for i in range(30) if i != idx):
            continue
        points[idx] = (new_x, new_y)
        s = score_of(points)
        if s > best_score:
            best_score = s
            best_points = points[:]
            no_improve = 0
        else:
            points[idx] = (old_x, old_y)
            no_improve += 1
            if no_improve > max_no_improve:
                random.seed(int(elapsed()*1000) % 100000)
                no_improve = 0
                max_no_improve = min(max_no_improve + 300, 3000)

    return best_points
# EVOLVE-BLOCK-END
