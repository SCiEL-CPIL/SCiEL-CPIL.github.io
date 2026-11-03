"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random
import numpy as np

N = 12
TIME_LIMIT = 148.0

def _hex_verts(cx, cy, s=1.0):
    """Vertices of a regular hexagon centered at (cx,cy), side s, flat-top."""
    a = math.sqrt(3) / 2.0
    return [(cx + s, cy), (cx + s/2, cy + a), (cx - s/2, cy + a),
            (cx - s, cy), (cx - s/2, cy - a), (cx + s/2, cy - a)]

def _separates(v1, v2):
    """SAT test: do two convex hexagons (lists of 6 (x,y)) NOT overlap?"""
    for poly in (v1, v2):
        for i in range(6):
            x1, y1 = poly[i]
            x2, y2 = poly[(i+1) % 6]
            nx = -(y2 - y1)
            ny = (x2 - x1)
            pmin = min(nx*x + ny*y for x, y in v1)
            pmax = max(nx*x + ny*y for x, y in v1)
            qmin = min(nx*x + ny*y for x, y in v2)
            qmax = max(nx*x + ny*y for x, y in v2)
            if pmax <= qmin + 1e-9 or qmax <= pmin + 1e-9:
                return True
    return False

def _valid(cx, cy):
    """Check all 12 hexagons are mutually disjoint."""
    verts = [_hex_verts(cx[i], cy[i]) for i in range(N)]
    for i in range(N):
        for j in range(i+1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx*dx + dy*dy > (2*math.sqrt(3))**2:
                continue
            if not _separates(verts[i], verts[j]):
                return False
    return True

def _outer_side(cx, cy, ocx, ocy, oangle_deg):
    """Min side of outer hexagon (center, rotation) enclosing all inner hexes."""
    apothem = math.sqrt(3) / 2.0
    a = math.radians(oangle_deg)
    normals = [(math.cos(a + math.radians(90 + 60*k)),
                math.sin(a + math.radians(90 + 60*k))) for k in range(6)]
    max_need = 0.0
    for i in range(N):
        for vx, vy in _hex_verts(cx[i], cy[i]):
            for nx, ny in normals:
                val = (nx*(vx - ocx) + ny*(vy - ocy)) / apothem
                if val > max_need:
                    max_need = val
    return max_need

def _optimize_outer(cx, cy):
    """Grid search outer hex center & rotation."""
    all_v = []
    for i in range(N):
        all_v.extend(_hex_verts(cx[i], cy[i]))
    cex = sum(v[0] for v in all_v) / len(all_v)
    cey = sum(v[1] for v in all_v) / len(all_v)
    best = float('inf')
    best_c, best_a = (0.0, 0.0), 0.0
    for da in range(6):
        ang = 60.0 * da
        for dcx in np.linspace(cex - 1.0, cex + 1.0, 7):
            for dcy in np.linspace(cey - 1.0, cey + 1.0, 7):
                s = _outer_side(cx, cy, dcx, dcy, ang)
                if s < best:
                    best = s
                    best_c = (float(dcx), float(dcy))
                    best_a = ang
    # Fine-tune angles
    for ang in np.linspace(0, 60, 13):
        s = _outer_side(cx, cy, best_c[0], best_c[1], float(ang))
        if s < best:
            best = s
            best_a = float(ang)
    return best, best_c, best_a

def _sa_refine(cx, cy, t0, max_iter=3000):
    """Simulated annealing on positions, fixed 0° rotation."""
    cx = list(cx)
    cy = list(cy)
    verts = [_hex_verts(cx[i], cy[i]) for i in range(N)]
    # Precompute current overlap count
    def count_overlaps():
        cnt = 0
        for i in range(N):
            for j in range(i+1, N):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                if dx*dx + dy*dy > (2*math.sqrt(3))**2:
                    continue
                if not _separates(verts[i], verts[j]):
                    cnt += 1
        return cnt
    cur_ov = count_overlaps()
    best_ov = cur_ov
    best_cx, best_cy = list(cx), list(cy)
    T = 0.5
    cooling = 0.9997
    for it in range(max_iter):
        if time.time() - t0 > TIME_LIMIT - 5:
            break
        idx = random.randrange(N)
        dx = random.gauss(0, 0.3)
        dy = random.gauss(0, 0.3)
        old_x, old_y = cx[idx], cy[idx]
        cx[idx] += dx
        cy[idx] += dy
        new_verts = _hex_verts(cx[idx], cy[idx])
        # Only check pairs involving idx
        new_ov = 0
        for j in range(N):
            if j == idx:
                continue
            ddx = cx[idx] - cx[j]
            ddy = cy[idx] - cy[j]
            if ddx*ddx + ddy*ddy > (2*math.sqrt(3))**2:
                continue
            if not _separates(new_verts, verts[j]):
                new_ov += 1
        delta = new_ov - cur_ov
        if delta <= 0 or random.random() < math.exp(-delta / T):
            cur_ov = new_ov
            verts[idx] = new_verts
            if cur_ov < best_ov:
                best_ov = cur_ov
                best_cx, best_cy = list(cx), list(cy)
        else:
            cx[idx] = old_x
            cy[idx] = old_y
        T *= cooling
    return best_cx, best_cy, best_ov

def _nm_refine(cx, cy, t0):
    """Nelder-Mead on objective = outer_side + penalty*overlaps."""
    from scipy.optimize import minimize
    x0 = np.array(cx + cy)
    def obj(x):
        cx_l = list(x[:N])
        cy_l = list(x[N:])
        if not _valid(cx_l, cy_l):
            # Count overlaps as penalty
            verts = [_hex_verts(cx_l[i], cy_l[i]) for i in range(N)]
            cnt = 0
            for i in range(N):
                for j in range(i+1, N):
                    dx = cx_l[i] - cx_l[j]
                    dy = cy_l[i] - cy_l[j]
                    if dx*dx + dy*dy > (2*math.sqrt(3))**2:
                        continue
                    if not _separates(verts[i], verts[j]):
                        cnt += 1
            return 100.0 + cnt
        s, _, _ = _optimize_outer(cx_l, cy_l)
        return s
    res = minimize(obj, x0, method='Nelder-Mead',
                   options={'maxiter': 200, 'xatol': 0.01, 'fatol': 0.01})
    if res.fun < 100:  # valid
        return list(res.x[:N]), list(res.x[N:])
    return list(cx), list(cy)

def construct():
    t0 = time.time()
    random.seed(123)
    np.random.seed(123)

    def flower():
        cx, cy = [0.0], [0.0]
        r = 2 * math.sqrt(3)
        for i in range(6):
            a = math.radians(60*i)
            cx.append(r*math.cos(a))
            cy.append(r*math.sin(a))
        r2 = r * 1.85
        for i in range(5):
            a = math.radians(60*i + 30)
            cx.append(r2*math.cos(a))
            cy.append(r2*math.sin(a))
        return cx, cy

    def ring11():
        cx, cy = [0.0], [0.0]
        r = 2 * math.sqrt(3)
        for i in range(11):
            a = math.radians(360*i/11 + 15)
            cx.append(r*math.cos(a))
            cy.append(r*math.sin(a))
        return cx, cy

    def lattice():
        cx, cy = [], []
        rh = math.sqrt(3)
        for row in range(4):
            y = (row - 1.5) * rh
            for col in range(3):
                x = (col - 1) * 2.0 + (0.5 if row % 2 else 0.0)
                cx.append(x)
                cy.append(y)
        # Remove 2 random to get 12, add 2 elsewhere
        # Actually just take first 12
        return cx[:12], cy[:12]

    def random_init():
        cx, cy = [], []
        for _ in range(N):
            cx.append(random.uniform(-3, 3))
            cy.append(random.uniform(-3, 3))
        return cx, cy

    inits = [flower(), ring11(), lattice(), random_init()]

    best_side = float('inf')
    best_result = None

    for init_idx, (cx, cy) in enumerate(inits):
        if time.time() - t0 > TIME_LIMIT - 10:
            break
        # SA refine
        cx_s, cy_s, ov = _sa_refine(cx, cy, t0, max_iter=2500)
        if time.time() - t0 > TIME_LIMIT - 8:
            break
        # NM refine
        cx_n, cy_n = _nm_refine(cx_s, cy_s, t0)
        if time.time() - t0 > TIME_LIMIT - 5:
            break
        # Final outer optimization
        s, oc, oa = _optimize_outer(cx_n, cy_n)
        if s < best_side:
            best_side = s
            inner = [[cx_n[i], cy_n[i], 0.0] for i in range(N)]
            best_result = (inner, oc, s, oa)

    if best_result is None:
        # Fallback: seed solution
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    return best_result

# EVOLVE-BLOCK-END
