import time

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DEADLINE = 75.0  # safety margin before 90s kill

    def elapsed():
        return time.time() - t0

    def doubled_areas(points):
        """Return dict: doubled_area -> count (positive only)."""
        n = len(points)
        from collections import defaultdict
        cnt = defaultdict(int)
        for i in range(n):
            xi, yi = points[i]
            for j in range(i + 1, n):
                xj, yj = points[j]
                for k in range(j + 1, n):
                    xk, yk = points[k]
                    # doubled area = |xi*(yj-yk) + xj*(yk-yi) + xk*(yi-yj)|
                    d = abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
                    if d > 0:
                        cnt[d] += 1
        return cnt

    def score_of(points):
        if len(points) != 30 or len(set(points)) != 30:
            return -1
        for (x, y) in points:
            if not (-30 <= x <= 30 and -30 <= y <= 30):
                return -1
        cnt = doubled_areas(points)
        if not cnt:
            return 0
        return max(cnt.values())

    best_points = None
    best_score = -1

    # Family A: m consecutive points on y=0, n points on y=1 (same height), distinct x.
    # Doubled area for (2 line pts dist d, 1 off pt) = d. Multiplicity of area=1 = (m-1)*n.
    for m in range(2, 31):
        n = 30 - m
        if n < 1:
            continue
        if elapsed() > DEADLINE:
            break
        # line points x = 0 .. m-1
        # off points: choose n distinct integer x in [-30,30], y=1
        # To keep distinct and in range, use x = 0 .. n-1 (n<=30 ok)
        line = [(i, 0) for i in range(m)]
        off = [(i, 1) for i in range(n)]
        pts = line + off
        s = score_of(pts)
        if s > best_score:
            best_score = s
            best_points = pts[:]

    # Family B: m on y=0, n on y=h with h>1, off x spread to also maybe boost? area=d*h, same multiplicity pattern.
    for h in [1, 2, 3, 5, 7]:
        for m in range(2, 31):
            n = 30 - m
            if n < 1:
                continue
            if elapsed() > DEADLINE:
                break
            line = [(i, 0) for i in range(m)]
            off = [(i, h) for i in range(n)]
            pts = line + off
            s = score_of(pts)
            if s > best_score:
                best_score = s
                best_points = pts[:]
        if elapsed() > DEADLINE:
            break

    # Family C: 3 parallel lines to possibly increase? 2 from one line + 1 from another, area = d * (height diff).
    # Try splitting 30 into 3 lines of 10 each at y=0,1,2. Area for 2 on line a, 1 on line b = |d|*|ya-yb|.
    # Multiplicity of area=1: need |d|=1 and |ya-yb|=1. Pairs: lines (0,1) and (1,2).
    # For lines 0&1: 2 from y0 (dist1: 9 pairs) * 1 from y1 (10) = 90; plus 2 from y1 (9)*1 from y0 (10)=90 → 180
    # For lines 1&2: 2 from y1(9)*1 from y2(10)=90; 2 from y2(9)*1 from y1(10)=90 → 180
    # Total area=1 multiplicity = 360? Let's verify by computation.
    for split in [(10, 10, 10), (12, 9, 9), (11, 10, 9), (14, 8, 8), (16, 7, 7), (15, 8, 7)]:
        if sum(split) != 30:
            continue
        if elapsed() > DEADLINE:
            break
        for ys in [(0, 1, 2), (0, 1, 3), (0, 2, 4)]:
            pts = []
            for count, y in zip(split, ys):
                pts += [(i, y) for i in range(count)]
            if len(pts) != 30 or len(set(pts)) != 30:
                continue
            s = score_of(pts)
            if s > best_score:
                best_score = s
                best_points = pts[:]
        if elapsed() > DEADLINE:
            break

    # Family E: Rectangular Grids K x L where K*L=30
    for K in range(2, 31):
        if 30 % K != 0:
            continue
        L = 30 // K
        if elapsed() > DEADLINE:
            break
        # Grid points (x, y) for x in [0, L-1], y in [0, K-1]
        # Check bounds
        if L - 1 > 30 or K - 1 > 30:
            continue
        pts = [(x, y) for x in range(L) for y in range(K)]
        if len(pts) != 30 or len(set(pts)) != 30:
            continue
        s = score_of(pts)
        if s > best_score:
            best_score = s
            best_points = pts[:]

    if best_points is None:
        # fallback seed
        return [(x, y) for x in range(6) for y in range(5)]

    return best_points
# EVOLVE-BLOCK-END
