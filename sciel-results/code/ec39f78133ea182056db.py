import time
from itertools import permutations

_DEADLINE = 145.0  # safety margin vs 150s kill

def _rotations(w, h, d):
    return sorted(set(permutations((w, h, d))))

def load(boxes, container):
    t0 = time.time()
    W, H, D = container
    W, H, D = float(W), float(H), float(D)
    container_vol = W * H * D
    n = len(boxes)

    # Precompute rotation options per box
    rots = [_rotations(b[0], b[1], b[2]) for b in boxes]

    # Ordering strategies -> list of box indices
    def make_order(key):
        return sorted(range(n), key=lambda i: key(boxes[i]), reverse=True)

    keys = [
        lambda b: b[0] * b[1] * b[2],
        lambda b: max(b),
        lambda b: 2 * (b[0] * b[1] + b[1] * b[2] + b[0] * b[2]),
        lambda b: min(b),
        lambda b: max(b) / (min(b) + 1e-9),
    ]
    strategies = []
    seen = set()
    for k in keys:
        order = make_order(k)
        tup = tuple(order)
        if tup not in seen:
            seen.add(tup)
            strategies.append(order)

    def run(order, deadline):
        # spatial hash grid
        cell = max(1.0, min(W, H, D) / 40.0)
        grid = {}
        placed = []  # (x, y, z, w, h, d)
        placed_idx = []

        def overlaps(x, y, z, w, h, d):
            x0, y0, z0 = x, y, z
            x1, y1, z1 = x + w, y + h, z + d
            cx0, cy0, cz0 = int(x0 // cell), int(y0 // cell), int(z0 // cell)
            cx1, cy1, cz1 = int((x1 - 1e-9) // cell), int((y1 - 1e-9) // cell), int((z1 - 1e-9) // cell)
            for gx in range(cx0, cx1 + 1):
                for gy in range(cy0, cy1 + 1):
                    for gz in range(cz0, cz1 + 1):
                        lst = grid.get((gx, gy, gz))
                        if not lst:
                            continue
                        for (px, py, pz, pw, ph, pd) in lst:
                            if (x0 < px + pw - 1e-9 and px < x1 - 1e-9 and
                                y0 < py + ph - 1e-9 and py < y1 - 1e-9 and
                                z0 < pz + pd - 1e-9 and pz < z1 - 1e-9):
                                return True
            return False

        def insert(x, y, z, w, h, d):
            x0, y0, z0 = x, y, z
            x1, y1, z1 = x + w, y + h, z + d
            cx0, cy0, cz0 = int(x0 // cell), int(y0 // cell), int(z0 // cell)
            cx1, cy1, cz1 = int((x1 - 1e-9) // cell), int((y1 - 1e-9) // cell), int((z1 - 1e-9) // cell)
            for gx in range(cx0, cx1 + 1):
                for gy in range(cy0, cy1 + 1):
                    for gz in range(cz0, cz1 + 1):
                        grid.setdefault((gx, gy, gz), []).append((x, y, z, w, h, d))

        # candidate x,y positions: 0, box dims, container dims
        xs = sorted(set([0.0] + [r[0] for r in rots[0]] + [W]))
        ys = sorted(set([0.0] + [r[1] for r in rots[0]] + [H]))
        # but rots differ per box; use global candidate sets
        allx = {0.0, W}
        ally = {0.0, H}
        allz = {0.0, D}
        for rlist in rots:
            for r in rlist:
                allx.add(r[0]); ally.add(r[1]); allz.add(r[2])
                allx.add(W - r[0]); ally.add(H - r[1]); allz.add(D - r[2])
        xs = sorted(allx)
        ys = sorted(ally)
        zs = sorted(allz)

        # score: prefer low z (bottom), then low y, then low x
        def score_pos(x, y, z):
            return (z, y, x)

        result = [-1] * n
        for bi in order:
            if time.time() > deadline:
                break
            bw, bh, bd = boxes[bi]
            best = None
            for (rw, rh, rd) in rots[bi]:
                if rw > W + 1e-9 or rh > H + 1e-9 or rd > D + 1e-9:
                    continue
                for z in zs:
                    if z + rd > D + 1e-9:
                        break
                    for y in ys:
                        if y + rh > H + 1e-9:
                            break
                        for x in xs:
                            if x + rw > W + 1e-9:
                                break
                            if not overlaps(x, y, z, rw, rh, rd):
                                s = score_pos(x, y, z)
                                if best is None or s < best[0]:
                                    best = (s, x, y, z, rw, rh, rd)
            if best is not None:
                s, x, y, z, rw, rh, rd = best
                insert(x, y, z, rw, rh, rd)
                placed_idx.append(bi)
                placed.append((x, y, z, rw, rh, rd))
                result[bi] = (0, x, y, z, rw, rh, rd)
        return result, placed

    # Allocate time across strategies
    n_strat = len(strategies)
    per = (deadline - time.time()) / max(1, n_strat)
    best_result = None
    best_vol = -1.0
    for si, order in enumerate(strategies):
        strat_deadline = deadline - 0.5  # keep safety margin
        res, placed = run(order, strat_deadline)
        vol = sum(w * h * d for (_, x, y, z, w, h, d) in res if _ == 0)
        if vol > best_vol:
            best_vol = vol
            best_result = res
        if time.time() > deadline - 1.0:
            break

    if best_result is None:
        # fallback: no placement
        best_result = [(-1, 0, 0, 0, b[0], b[1], b[2]) for b in boxes]
    return best_result
