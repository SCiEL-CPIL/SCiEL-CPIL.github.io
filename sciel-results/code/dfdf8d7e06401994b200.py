import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools

def hexagon_packing_11():
    start_time = time.time()
    deadline = 600.0
    
    SQRT3 = math.sqrt(3.0)
    
    def hex_verts(cx, cy, angle_deg):
        a = np.deg2rad(angle_deg)
        vs = []
        for i in range(6):
            t = a + np.deg2rad(60 * i)
            vs.append([cx + math.cos(t), cy + math.sin(t)])
        return np.array(vs)
    
    def check_valid(inner, outer, side):
        """Check validity per task contract."""
        if not (np.all(np.isfinite(inner)) and np.all(np.isfinite(outer))):
            return False
        if side < SQRT3 * 11**0.5 - 1e-9:
            return False
        # side >= sqrt(11)
        if side < math.sqrt(11.0) - 1e-9:
            return False
        
        # outer center
        ocx, ocy = outer[0], outer[1]
        
        # get outer vertices
        oa = np.deg2rad(outer[2])
        ov = []
        for i in range(6):
            t = oa + np.deg2rad(60 * i)
            ov.append([ocx + side * math.cos(t), ocy + side * math.sin(t)])
        ov = np.array(ov)
        
        # inner hex vertices
        iverts_list = []
        for i in range(11):
            iverts_list.append(hex_verts(inner[i,0], inner[i,1], inner[i,2]))
        
        # check separation: for every pair, on at least one of 12 edge normals,
        # projected intervals are apart by > 1e-6
        for i in range(11):
            v1 = iverts_list[i]
            for j in range(i+1, 11):
                v2 = iverts_list[j]
                ok = False
                # 12 normals (6 from each)
                for vv in (v1, v2):
                    for k in range(6):
                        p = vv[k]
                        q = vv[(k+1) % 6]
                        e = q - p
                        n = np.array([e[1], -e[0]])
                        nn = np.linalg.norm(n)
                        if nn < 1e-12:
                            continue
                        n = n / nn
                        proj1 = v1 @ n
                        proj2 = v2 @ n
                        min1, max1 = np.min(proj1), np.max(proj1)
                        min2, max2 = np.min(proj2), np.max(proj2)
                        # separated if max1 + 1e-6 < min2 or max2 + 1e-6 < min1
                        if max1 + 1e-6 < min2 or max2 + 1e-6 < min1:
                            ok = True
                            break
                    if ok:
                        break
                if not ok:
                    return False
        
        # check containment: every inner vertex p satisfies
        # cross(p2 - p1, p - p1) >= -1e-6 for each outer edge
        for i in range(6):
            p1 = ov[i]
            p2 = ov[(i+1) % 6]
            edge = p2 - p1
            for i_hex in range(11):
                for p in iverts_list[i_hex]:
                    cross = edge[0] * (p[1] - p1[1]) - edge[1] * (p[0] - p1[0])
                    if cross < -1e-6:
                        return False
                    if cross > 0:
                        # could be outside depending on winding; we need all inside
                        # check with consistent winding
                        pass
        # For containment, we need all inner vertices inside or on boundary.
        # The cross product sign depends on winding. Let's check using a point
        # definitely inside (outer center) to determine sign convention.
        cx, cy = ocx, ocy
        for i in range(6):
            p1 = ov[i]
            p2 = ov[(i+1) % 6]
            edge = p2 - p1
            # cross with center
            ccross = edge[0] * (cy - p1[1]) - edge[1] * (cx - p1[0])
            sign = 1.0 if ccross >= 0 else -1.0
            for i_hex in range(11):
                for p in iverts_list[i_hex]:
                    cross = edge[0] * (p[1] - p1[1]) - edge[1] * (p[0] - p1[0])
                    if sign * cross < -1e-6:
                        return False
        
        # shifted inner hexagons keep edge lengths within 1e-8 of 1
        for i in range(11):
            v = iverts_list[i]
            for k in range(6):
                e = v[(k+1)%6] - v[k]
                if abs(np.linalg.norm(e) - 1.0) > 1e-8:
                    return False
        return True
    
    def compute_side(inner, outer_angle_deg):
        """Compute minimal side for outer hex centered at origin with given angle."""
        # outer normals at angles outer_angle + 60*i
        oa = np.deg2rad(outer_angle_deg)
        # For regular hexagon, the 6 edge normals are at oa+30+60*i (pointing to edges)
        # Actually vertices at oa+60*i. Edge between vertex i and i+1 has normal at oa+30+60*i.
        # Apothem = S * sqrt(3)/2.
        # The support in edge-normal direction must be <= apothem.
        # So S = 2/sqrt(3) * max over edge-normal directions of max projection.
        oa = np.deg2rad(outer_angle_deg)
        max_h = 0.0
        for i in range(6):
            t = oa + np.deg2rad(30 + 60 * i)
            n = np.array([math.cos(t), math.sin(t)])
            for i_hex in range(11):
                v = hex_verts(inner[i_hex,0], inner[i_hex,1], inner[i_hex,2])
                proj = np.max(v @ n)
                if proj > max_h:
                    max_h = proj
        return (2.0 / SQRT3) * max_h
    
    def objective(params):
        if time.time() > deadline - 5:
            return 1e10
        # params: 11*3 + 1 = 34
        data = params[:33].reshape(11, 3)
        theta = params[33]
        # side
        side = compute_side(data, theta)
        # separation penalty
        pen = 0.0
        verts = [hex_verts(data[i,0], data[i,1], data[i,2]) for i in range(11)]
        for i in range(11):
            v1 = verts[i]
            for j in range(i+1, 11):
                v2 = verts[j]
                # minimal gap check: if any normal shows separation > 1e-6, fine
                # else add penetration penalty
                min_pen = 0.0
                separated = False
                for vv in (v1, v2):
                    for k in range(6):
                        p = vv[k]
                        q = vv[(k+1) % 6]
                        e = q - p
                        n = np.array([e[1], -e[0]])
                        nn = np.linalg.norm(n)
                        if nn < 1e-12:
                            continue
                        n = n / nn
                        proj1 = v1 @ n
                        proj2 = v2 @ n
                        min1, max1 = np.min(proj1), np.max(proj1)
                        min2, max2 = np.min(proj2), np.max(proj2)
                        if max1 + 1e-6 < min2 or max2 + 1e-6 < min1:
                            separated = True
                            break
                        # overlap
                        ov = np.minimum(max1, max2) - np.maximum(min1, min2)
                        if ov > min_pen:
                            min_pen = ov
                    if separated:
                        break
                if not separated:
                    # penetration: penalize
                    pen += min_pen * min_pen
        # containment penalty
        oa = np.deg2rad(theta)
        ov = []
        for i in range(6):
            t = oa + np.deg2rad(60 * i)
            ov.append([side * math.cos(t), side * math.sin(t)])
        ov = np.array(ov)
        cx, cy = 0.0, 0.0
        contain_pen = 0.0
        for i in range(6):
            p1 = ov[i]
            p2 = ov[(i+1) % 6]
            edge = p2 - p1
            ccross = edge[0] * (cy - p1[1]) - edge[1] * (cx - p1[0])
            sign = 1.0 if ccross >= 0 else -1.0
            for i_hex in range(11):
                for p in verts[i_hex]:
                    cross = edge[0] * (p[1] - p1[1]) - edge[1] * (p[0] - p1[0])
                    if sign * cross < 0:
                        contain_pen += (-sign * cross) ** 2
        return side + 100.0 * pen + 100.0 * contain_pen
    
    # Initial guess: flower (7) + 4 more, with S ~ 4
    d = SQRT3
    init = np.zeros(34)
    # center hex
    init[0:3] = [0, 0, 0]
    # ring 1: 6 hexes
    for i in range(6):
        a = np.deg2rad(30 + 60 * i)
        idx = 1 + i
        init[idx*3 + 0] = d * math.cos(a)
        init[idx*3 + 1] = d * math.sin(a)
        init[idx*3 + 2] = 0.0
    # ring 2: 4 hexes at gaps 0, 90, 180, 270? use 0, 120, 240 and one more
    # place at angles 0, 90, 180, 270 with radius ~3
    for k, ang in enumerate([0, 90, 180, 270]):
        idx = 7 + k
        a = np.deg2rad(ang)
        r = 3.0
        init[idx*3 + 0] = r * math.cos(a)
        init[idx*3 + 1] = r * math.sin(a)
        init[idx*3 + 2] = ang
    init[33] = 0.0
    
    best = None
    best_val = float('inf')
    
    def try_opt(x0, maxiter=1500):
        nonlocal best, best_val
        if time.time() > deadline - 30:
            return
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                           options={'maxiter': maxiter, 'xatol': 1e-7, 'fatol': 1e-7, 'disp': False})
            if time.time() > deadline - 5:
                return
            data = res.x[:33].reshape(11, 3)
            theta = res.x[33]
            side = compute_side(data, theta)
            # check validity
            outer = np.array([0.0, 0.0, theta])
            if check_valid(data, outer, side):
                if side < best_val:
                    best_val = side
                    best = (data.copy(), outer.copy(), side)
        except Exception:
            pass
    
    # Try the main init
    try_opt(init, 2000)
    
    # Try several different init variants
    np.random.seed(1)
    variants = []
    # vary outer angle
    for ang in [0, 30, -30]:
        x = init.copy()
        x[33] = ang
        variants.append(x)
    # vary ring 2 angles/positions
    for k in range(6):
        x = init.copy()
        base_angs = [0, 90, 180, 270]
        # perturb ring2
        for i in range(4):
            idx = 7 + i
            a = np.deg2rad(base_angs[i] + np.random.uniform(-20, 20))
            r = 3.0 + np.random.uniform(-0.3, 0.3)
            x[idx*3 + 0] = r * math.cos(a)
            x[idx*3 + 1] = r * math.sin(a)
            x[idx*3 + 2] = base_angs[i] + np.random.uniform(-15, 15)
        variants.append(x)
    # random full init around flower
    for k in range(8):
        x = np.zeros(34)
        x[0:3] = [0, 0, 0]
        for i in range(6):
            a = np.deg2rad(30 + 60 * i)
            idx = 1 + i
            x[idx*3 + 0] = d * math.cos(a) + np.random.uniform(-0.1, 0.1)
            x[idx*3 + 1] = d * math.sin(a) + np.random.uniform(-0.1, 0.1)
            x[idx*3 + 2] = 0.0
        for i in range(4):
            idx = 7 + i
            a = np.random.uniform(0, 2*np.pi)
            r = np.random.uniform(2.5, 3.5)
            x[idx*3 + 0] = r * math.cos(a)
            x[idx*3 + 1] = r * math.sin(a)
            x[idx*3 + 2] = np.random.uniform(0, 60)
        x[33] = np.random.uniform(-30, 30)
        variants.append(x)
    
    for x in variants:
        if time.time() > deadline - 60:
            break
        try_opt(x, 1500)
    
    # local polish: if we have a best, do a fine refinement
    if best is not None and time.time() < deadline - 40:
        data, outer, side = best
        x0 = np.concatenate([data.flatten(), [outer[2]]])
        for _ in range(3):
            if time.time() > deadline - 20:
                break
            try:
                res = minimize(objective, x0, method='Nelder-Mead',
                               options={'maxiter': 800, 'xatol': 1e-9, 'fatol': 1e-9, 'disp': False})
                d2 = res.x[:33].reshape(11, 3)
                t2 = res.x[33]
                s2 = compute_side(d2, t2)
                o2 = np.array([0.0, 0.0, t2])
                if check_valid(d2, o2, s2) and s2 < best_val:
                    best_val = s2
                    best = (d2.copy(), o2.copy(), s2)
                    x0 = res.x
            except Exception:
                pass
    
    if best is not None:
        return best
    
    # Fallback: use the S=4 configuration
    inner = np.zeros((11, 3))
    inner[0] = [0, 0, 0]
    d = SQRT3
    for i in range(6):
        a = np.deg2rad(30 + 60 * i)
        inner[1+i] = [d * math.cos(a), d * math.sin(a), 0.0]
    for k, ang in enumerate([0, 90, 180, 270]):
        a = np.deg2rad(ang)
        r = 3.0
        inner[7+k] = [r * math.cos(a), r * math.sin(a), float(ang)]
    outer = np.array([0.0, 0.0, 0.0])
    side = compute_side(inner, 0.0)
    # ensure validity; if not, increase side
    if not check_valid(inner, outer, side):
        side = 4.0
    return inner, outer, side

if __name__ == "__main__":
    r = hexagon_packing_11()
    print(r)
