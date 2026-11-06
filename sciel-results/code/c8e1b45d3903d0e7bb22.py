import time
import random

def _count_ordinary(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    ordinary = 0
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            dx = xj - xi
            dy = yj - yi
            count = 2
            for k in range(n):
                if k == i or k == j:
                    continue
                xk, yk = points[k]
                if dx * (yk - yi) == dy * (xk - xi):
                    count += 1
            if count == 2:
                ordinary += 1
    return ordinary

def construct():
    start_time = time.time()
    deadline = start_time + 120  # safety margin before 150s kill

    random.seed(42)

    # Build diverse initial configurations
    inits = []
    # 3x4 grid (4 cols, 3 rows)
    inits.append([[x, y] for x in range(4) for y in range(3)])
    # 4x3 grid (3 cols, 4 rows)
    inits.append([[x, y] for x in range(3) for y in range(4)])
    # Scaled grid
    inits.append([[x * 2, y * 3] for x in range(4) for y in range(3)])
    # Random small configs
    for _ in range(8):
        pts = []
        while len(pts) < 12:
            p = [random.randint(-15, 15), random.randint(-15, 15)]
            if p not in pts:
                pts.append(p)
        inits.append(pts)
    # Hexagonal-ish
    hex_pts = [[0,0],[1,0],[2,0],[0,1],[1,1],[2,1],[0,2],[1,2],[2,2],[3,0],[3,1],[3,2]]
    inits.append(hex_pts)

    best = float('inf')
    best_points = None

    for init in inits:
        if time.time() > deadline:
            break
        pts = [p[:] for p in init]
        cur = _count_ordinary(pts)
        if cur < best:
            best = cur
            best_points = [p[:] for p in pts]
        if best == 6:
            return [[p[0], p[1]] for p in best_points]

        # Local search: single-point moves
        no_improve = 0
        for it in range(20000):
            if time.time() > deadline:
                break
            i = random.randint(0, 11)
            # Adaptive step size
            step = random.randint(1, 10) if no_improve > 200 else random.randint(1, 5)
            dx = random.randint(-step, step)
            dy = random.randint(-step, step)
            if dx == 0 and dy == 0:
                continue
            new_pts = [p[:] for p in pts]
            new_pts[i] = [new_pts[i][0] + dx, new_pts[i][1] + dy]
            # Check distinctness
            if len(set(map(tuple, new_pts))) != 12:
                continue
            c = _count_ordinary(new_pts)
            if c <= cur:
                pts = new_pts
                cur = c
                no_improve = 0
                if cur < best:
                    best = cur
                    best_points = [p[:] for p in pts]
                    if best == 6:
                        return [[p[0], p[1]] for p in best_points]
            else:
                no_improve += 1

        # Two-point swap moves
        for it in range(5000):
            if time.time() > deadline:
                break
            i, j = random.sample(range(12), 2)
            dx = random.randint(-10, 10)
            dy = random.randint(-10, 10)
            if dx == 0 and dy == 0:
                continue
            new_pts = [p[:] for p in pts]
            new_pts[i] = [new_pts[i][0] + dx, new_pts[i][1] + dy]
            new_pts[j] = [new_pts[j][0] - dx, new_pts[j][1] - dy]
            if len(set(map(tuple, new_pts))) != 12:
                continue
            c = _count_ordinary(new_pts)
            if c <= cur:
                pts = new_pts
                cur = c
                if cur < best:
                    best = cur
                    best_points = [p[:] for p in pts]
                    if best == 6:
                        return [[p[0], p[1]] for p in best_points]

    # Final safety: ensure not all collinear and within bounds
    if best_points is None:
        best_points = [[x, y] for x in range(4) for y in range(3)]

    # Verify not all collinear
    if len(best_points) >= 3:
        (x0, y0), (x1, y1), (x2, y2) = best_points[0], best_points[1], best_points[2]
        if (x1 - x0) * (y2 - y0) == (y1 - y0) * (x2 - x0):
            # Perturb one point
            best_points[2] = [best_points[2][0] + 1, best_points[2][1]]

    return [[p[0], p[1]] for p in best_points]
