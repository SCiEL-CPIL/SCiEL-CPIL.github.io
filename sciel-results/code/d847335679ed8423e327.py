"""Pack unit hexagons into the smallest enclosing regular hexagon (flat-top)."""
import math
import random
import time

N = 11
DEADLINE = 115.0  # wall-clock safety margin (evaluator kills at 150s)
TWOPI = 2.0 * math.pi
SQ3 = math.sqrt(3.0)
S3 = 3.0 * SQ3  # factor converting max support to outer side length


def _h0(theta):
    """Support of a unit hexagon (rotated theta) along the x-axis (phi=0)."""
    return 0.5 * math.cos(theta) + 0.5 * math.cos(2.0 * theta)


def _maxsupport(xys, thetas):
    """Max over hexagons of |x| + h0(theta)."""
    m = 0.0
    for i in range(len(xys)):
        v = abs(xys[i]) + _h0(thetas[i])
        if v > m:
            m = v
    return m


def _objective(xys, thetas):
    """Outer side length = 3*sqrt(3)*max(|x|+h0(theta))."""
    return S3 * _maxsupport(xys, thetas)


def construct():
    start = time.time()
    rng = random.Random(20240517)

    # ---- Phase 1: greedy best-improvement construction ----
    xys = []
    thetas = []

    for k in range(N):
        if time.time() > DEADLINE - 30.0:
            break
        if k == 0:
            xys.append(0.0)
            thetas.append(0.0)
            continue
        # candidate x positions spread along the axis
        cand_x = []
        for j in range(0, 6):
            cand_x.append(j * 1.05)
            cand_x.append(-j * 1.05)
        # a few random x offsets
        for _ in range(25):
            cand_x.append(rng.uniform(-7.0, 7.0))
        cand_x = sorted(set(cand_x))
        cand_a = [0.0, math.pi / 6.0, math.pi / 3.0, math.pi / 2.0,
                  2.0 * math.pi / 3.0, 5.0 * math.pi / 6.0]
        best_m = None
        best_x = 0.0
        best_t = 0.0
        for cx in cand_x:
            for ca in cand_a:
                xys.append(cx)
                thetas.append(ca)
                m = _maxsupport(xys, thetas)
                xys.pop()
                thetas.pop()
                if best_m is None or m < best_m - 1e-12:
                    best_m = m
                    best_x = cx
                    best_t = ca
        xys.append(best_x)
        thetas.append(best_t)

    # ---- Phase 2: local search / polish (with correct reversion) ----
    if time.time() < DEADLINE - 8.0:
        xys = list(xys)
        thetas = list(thetas)
        cur = _objective(xys, thetas)
        best_obj = cur
        bx = list(xys)
        bt = list(thetas)
        step = 0.4
        for _ in range(400000):
            if time.time() > DEADLINE - 8.0:
                break
            i = rng.randrange(len(xys))
            ox = xys[i]
            ot = thetas[i]
            if rng.random() < 0.6:
                nx = ox + rng.gauss(0.0, step)
                nth = ot
            else:
                nx = ox
                nth = (ot + rng.gauss(0.0, step)) % TWOPI
            xys[i] = nx
            thetas[i] = nth
            m = _maxsupport(xys, thetas)
            obj = S3 * m
            if obj < cur - 1e-12:
                cur = obj
                if obj < best_obj - 1e-12:
                    best_obj = obj
                    bx = list(xys)
                    bt = list(thetas)
                if rng.random() < 0.25:
                    step = max(0.03, step * 0.9)
            else:
                xys[i] = ox
                thetas[i] = ot
        xys = bx
        thetas = bt

    outer_side = _objective(xys, thetas)
    inner = []
    for x, t in zip(xys, thetas):
        ang_deg = math.degrees(t) % 360.0
        inner.append([x, 0.0, ang_deg])
    return inner, (0.0, 0.0), outer_side, 0.0
