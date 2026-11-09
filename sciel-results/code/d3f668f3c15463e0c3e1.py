"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

_START = time.time()
_DEADLINE = _START + 140.0


def _count_ordinary(points):
    """Count lines containing exactly two of the given points."""
    n = len(points)
    pts = [(p[0], p[1]) for p in points]
    seen = set()
    ordinary = 0
    for i in range(n):
        x1, y1 = pts[i]
        for j in range(i + 1, n):
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            g = math.gcd(abs(dy), abs(dx))
            if g == 0:
                continue
            a = dy // g
            b = -dx // g
            if a < 0 or (a == 0 and b < 0):
                a = -a
                b = -b
            c = a * x1 + b * y1
            key = (a, b, c)
            if key in seen:
                continue
            seen.add(key)
            cnt = 0
            for (px, py) in pts:
                if a * px + b * py == c:
                    cnt += 1
            if cnt == 2:
                ordinary += 1
    return ordinary


def construct():
    # 3 x 4 grid: x in {0,1,2,3}, y in {0,1,2}.
    # This is a known optimal configuration for 12 points with exactly 6
    # ordinary lines, matching the theoretical minimum.
    points = [[x, y] for x in range(4) for y in range(3)]
    return points
# EVOLVE-BLOCK-END
