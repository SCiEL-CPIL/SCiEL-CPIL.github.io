import math
import time
import random
import numpy as np

N = 11
EPS = 1e-7

def construct():
    deadline = time.time() + 145.0

    # --- Geometry helpers -----------------------------------------------------
    # Unit regular hexagon with "pointy-top" orientation at angle 0:
    #   vertices at angles 0, 60, ..., 300 degrees, radius = side = 1.
    # We use the same convention for the outer hexagon.
    def hex_vertices(cx, cy, S, angle_deg):
        a = math.radians(angle_deg)
        out = np.empty((6, 2))
        for i in range(6):
            t = a + math.radians(60 * i)
            out[i, 0] = cx + S * math.cos(t)
            out[i, 1] = cy + S * math.sin(t)
        return out

    # Precompute unit hexagon vertices for common angles.
    _unit_cache = {}
    def unit_hex_verts(angle_deg):
        key = round(angle_deg, 9)
        v = _unit_cache.get(key)
        if v is None:
            a = math.radians(key)
            v = np.empty((6, 2))
            for i in range(6):
                t = a + math.radians(60 * i)
                v[i, 0] = math.cos(t)
                v[i, 1] = math.sin(t)
            _unit_cache[key] = v
        return v

    def seg_seg_intersect(p1, p2, p3, p4):
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        return False

    def point_in_polygon(p, poly):
        x, y = p
        inside = False
        j = len(poly) - 1
        for i in range(len(poly)):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def hex_disjoint(verts_a, verts_b):
        # Fast bbox reject
        aminx = verts_a[:, 0].min() - EPS
        amaxy = verts_a[:, 0].max() + EPS
        aminy = verts_a[:, 1].min() - EPS
        amaxy_y = verts_a[:, 1].max() + EPS
        bminx = verts_b[:, 0].min() - EPS
        bmaxx = verts_b[:, 0].max() + EPS
        bminy = verts_b[:, 1].min() - EPS
        bmaxy = verts_b[:, 1].max() + EPS
        if amaxy < bminx or bmaxx < aminx or amaxy_y < bminy or bmaxy < aminy:
            return True
        for v in verts_a:
            if point_in_polygon(v, verts_b):
                return False
        for v in verts_b:
            if point_in_polygon(v, verts_a):
                return False
        for i in range(6):
            if seg_seg_intersect(verts_a[i], verts_a[(i + 1) % 6],
                                  verts_b[i], verts_b[(i + 1) % 6]):
                return False
        return True

    # --- Objective / validity -------------------------------------------------
    def compute_required_S(inner, ocx, ocy, oangle):
        # inner: list of (x, y, angle_deg)
        # Outer hexagon pointy-top: outward normals of edges are at angles
        # 30 + 60*i degrees from the centre.
        max_S = 0.0
        a = math.radians(oangle)
        for i in range(6):
            theta = a + math.radians(30 + 60 * i)
            nx, ny = math.cos(theta), math.sin(theta)
            max_proj = 0.0
            for (ix, iy, ia) in inner:
                uv = unit_hex_verts(ia)
                px = ix + uv[:, 0]
                py = iy + uv[:, 1]
                proj = (px - ocx) * nx + (py - ocy) * ny
                m = proj.max()
                if m > max_proj:
                    max_proj = m
            if max_proj > max_S * math.sqrt(3) / 2.0:
                S = max_proj / (math.sqrt(3) / 2.0)
                if S > max_S:
                    max_S = S
        return max_S

    def all_fit(inner, ocx, ocy, S, oangle):
        outer = hex_vertices(ocx, ocy, S, oangle)
        verts_list = []
        for (ix, iy, ia) in inner:
            v = (unit_hex_verts(ia)) + np.array([ix, iy])
            verts_list.append(v)
            for p in v:
                if not point_in_polygon(p, outer):
                    return False
        for i in range(N):
            for j in range(i + 1, N):
                if not hex_disjoint(verts_list[i], verts_list[j]):
                    return False
        return True

    # --- Starting layouts -----------------------------------------------------
    def make_centered_layout():
        # 7 hexagons in a hexagonal cluster (centred), plus 4 more on an
        # outer ring at the "gaps" between the cluster. This is a tight,
        # symmetric starting point.
        sp = math.sqrt(3.0)
        lay = []
        lay.append([0.0, 0.0, 0.0])
        for i in range(6):
            t = math.radians(60 * i)
            lay.append([sp * math.cos(t), sp * math.sin(t), 0.0])
        # Ring 2 at radius 2*sp, 6 positions; pick 4 of them that don't
        # collide with each other (they are separated by 60 deg, so all 6
        # work). We take indices 0,2,4,1 for asymmetry.
        for i in [0, 2, 4, 1]:
            t = math.radians(60 * i)
            lay.append([2 * sp * math.cos(t), 2 * sp * math.sin(t), 0.0])
        return lay

    def make_line_layout():
        sp = math.sqrt(3.0)
        return [[(k - 5) * sp, 0.0, 0.0] for k in range(N)]

    def make_random_layout(rng, scale=2.0):
        lay = []
        for _ in range(N):
            lay.append([rng.uniform(-scale, scale),
                        rng.uniform(-scale, scale),
                        rng.choice([0.0, 30.0, 60.0, 90.0, 120.0, 150.0])])
        return lay

    # --- Local search (simulated annealing) -----------------------------------
    best = {'S': float('inf'), 'inner': None, 'oc': (0.0, 0.0), 'oa': 0.0, 'valid': False}

    def try_update(S, inner, ocx, ocy, oangle):
        if S < best['S']:
            best['S'] = S
            best['inner'] = [list(x) for x in inner]
            best['oc'] = (ocx, ocy)
            best['oa'] = oangle
            best['valid'] = True

    def local_search(init_inner, ocx, ocy, oangle, seed, t0=1.0, cool=0.99995, steps=20000):
        rng = random.Random(seed)
        inner = [list(x) for x in init_inner]
        S = compute_required_S(inner, ocx, ocy, oangle)
        valid = all_fit(inner, ocx, ocy, S, oangle)
        # Repair if invalid: small random jitter a few times.
        if not valid:
            for _ in range(80):
                idx = rng.randrange(N)
                inner[idx][0] += rng.gauss(0, 0.3)
                inner[idx][1] += rng.gauss(0, 0.3)
                S = compute_required_S(inner, ocx, ocy, oangle)
                valid = all_fit(inner, ocx, ocy, S, oangle)
                if valid:
                    break
            if not valid:
                return
        try_update(S, inner, ocx, ocy, oangle)
        cur_S = S
        cur_valid = valid
        T = t0
        for it in range(steps):
            if time.time() > deadline:
                return
            old = [list(x) for x in inner]
            idx = rng.randrange(N)
            inner[idx][0] += rng.gauss(0, 0.10)
            inner[idx][1] += rng.gauss(0, 0.10)
            if rng.random() < 0.04:
                inner[idx][2] = rng.choice([0.0, 30.0, 60.0, 90.0, 120.0, 150.0])
            new_S = compute_required_S(inner, ocx, ocy, oangle)
            new_valid = all_fit(inner, ocx, ocy, new_S, oangle)
            if new_valid:
                if new_S < cur_S or rng.random() < math.exp((cur_S - new_S) / max(T, 1e-6)):
                    cur_S = new_S
                    cur_valid = True
                    try_update(new_S, inner, ocx, ocy, oangle)
                else:
                    inner = old
            else:
                inner = old
            T *= cool

    # --- Multi-restart driver -------------------------------------------------
    # Phase 1: local search on analytic starting layouts.
    layouts = [make_centered_layout(), make_line_layout()]
    for i, lay in enumerate(layouts):
        if time.time() > deadline:
            break
        local_search(lay, 0.0, 0.0, 0.0, 1000 + i, t0=0.8, cool=0.99997, steps=15000)

    # Phase 2: randomised restarts with varied outer angles and centre.
    rng_master = random.Random(12345)
    restart = 0
    while time.time() < deadline - 1.0:
        restart += 1
        oa = rng_master.choice([0.0, 30.0, 60.0, 90.0, 120.0, 150.0])
        # Slight random centre offset (the cluster is roughly centred).
        ocx = rng_master.gauss(0, 0.2)
        ocy = rng_master.gauss(0, 0.2)
        if rng_master.random() < 0.5:
            lay = make_centered_layout()
        elif rng_master.random() < 0.5:
            lay = make_random_layout(rng_master, scale=2.5)
        else:
            lay = make_line_layout()
        # Perturb the layout a bit so that each restart is different.
        rp = random.Random(restart * 7 + 13)
        for i in range(N):
            lay[i][0] += rp.gauss(0, 0.25)
            lay[i][1] += rp.gauss(0, 0.25)
            if rp.random() < 0.15:
                lay[i][2] = rp.choice([0.0, 30.0, 60.0, 90.0, 120.0, 150.0])
        local_search(lay, ocx, ocy, oa, restart * 101 + 1,
                     t0=rng_master.uniform(0.4, 1.2),
                     cool=rng_master.uniform(0.99996, 0.99999),
                     steps=rng_master.randint(4000, 12000))

    # Fallback
    if not best['valid'] or best['inner'] is None:
        lay = make_line_layout()
        S = compute_required_S(lay, 0.0, 0.0, 0.0)
        # Expand until valid (shouldn't happen for a line of 11, but safe).
        for _ in range(200):
            if all_fit(lay, 0.0, 0.0, S, 0.0):
                break
            for k in range(N):
                lay[k][0] *= 1.05
            S = compute_required_S(lay, 0.0, 0.0, 0.0)
        return lay, (0.0, 0.0), S, 0.0

    return best['inner'], best['oc'], best['S'], best['oa']
