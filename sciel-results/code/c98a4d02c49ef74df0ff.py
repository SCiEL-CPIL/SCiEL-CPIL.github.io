import math
import time
import random
import numpy as np
from scipy.optimize import differential_evolution

N = 11

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145.0
    best_S = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    best_valid = False

    # Precompute unit hex vertices (flat-top: vertices at 0, 60, 120, 180, 240, 300)
    base_verts = [(math.cos(math.radians(60*i)), math.sin(math.radians(60*i))) for i in range(6)]

    def transform_verts(cx, cy, S, angle_deg):
        a = math.radians(angle_deg)
        ca, sa = math.cos(a), math.sin(a)
        return [(cx + S*(vx*ca - vy*sa), cy + S*(vx*sa + vy*ca)) for vx, vy in base_verts]

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    def segments_intersect(p1, p2, p3, p4):
        d1 = cross(p3, p4, p1); d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3); d4 = cross(p1, p2, p4)
        if ((d1>0 and d2<0) or (d1<0 and d2>0)) and ((d3>0 and d4<0) or (d3<0 and d4>0)):
            return True
        return False

    def point_in_polygon(p, poly):
        x, y = p
        inside = False
        j = len(poly) - 1
        for i in range(len(poly)):
            xi, yi = poly[i]; xj, yj = poly[j]
            if ((yi > y) != (yj > y)) and (x < (xj-xi)*(y-yi)/(yj-yi) + xi):
                inside = not inside
            j = i
        return inside

    def hexagons_disjoint(va, vb):
        for v in va:
            if point_in_polygon(v, vb): return False
        for v in vb:
            if point_in_polygon(v, va): return False
        for i in range(6):
            if segments_intersect(va[i], va[(i+1)%6], vb[i], vb[(i+1)%6]):
                return False
        return True

    def compute_outer_hex(inner, ocx, ocy, oangle):
        inner_verts = [transform_verts(ix, iy, 1.0, ia) for ix, iy, ia in inner]
        a = math.radians(oangle)
        apothem_factor = math.sqrt(3) / 2
        max_S = 0.0
        for i in range(6):
            theta = a + math.radians(30 + 60*i)
            nx, ny = math.cos(theta), math.sin(theta)
            max_proj = 0.0
            for iv in inner_verts:
                for v in iv:
                    proj = (v[0]-ocx)*nx + (v[1]-ocy)*ny
                    if proj > max_proj: max_proj = proj
            if max_proj > 0:
                S_needed = max_proj / apothem_factor
                if S_needed > max_S: max_S = S_needed
        return max_S

    def all_fit(inner, ocx, ocy, S, oangle):
        outer_verts = transform_verts(ocx, ocy, S, oangle)
        inner_verts = [transform_verts(ix, iy, 1.0, ia) for ix, iy, ia in inner]
        for iv in inner_verts:
            for v in iv:
                if not point_in_polygon(v, outer_verts):
                    return False
        for i in range(len(inner_verts)):
            for j in range(i+1, len(inner_verts)):
                if not hexagons_disjoint(inner_verts[i], inner_verts[j]):
                    return False
        return True

    def update_best(inner, ocx, ocy, S, oangle):
        nonlocal best_S, best_inner, best_center, best_angle, best_valid
        if S < best_S and all_fit(inner, ocx, ocy, S, oangle):
            best_S = S
            best_inner = [list(x) for x in inner]
            best_center = (ocx, ocy)
            best_angle = oangle
            best_valid = True
            return True
        return False

    # ---- Structural change: Differential Evolution over the full parameter space ----
    # Parameters: for each hex: x, y, angle (3*N = 33 params)
    # Objective: outer hexagon side length, penalized heavily if invalid
    # This is a fundamentally different search method than local search / lattice search.

    def objective(params):
        if time.time() > deadline:
            return 1e10
        inner = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i+1]
            a = params[3*i+2]
            inner.append([x, y, a])
        
        # Center of mass
        cx = sum(p[0] for p in inner) / N
        cy = sum(p[1] for p in inner) / N
        
        # Subtract centroid to center configuration at origin
        inner_c = [[p[0]-cx, p[1]-cy, p[2]] for p in inner]
        
        S = compute_outer_hex(inner_c, 0.0, 0.0, 0.0)
        
        # Check validity
        if not all_fit(inner_c, 0.0, 0.0, S, 0.0):
            # Penalty for invalid: distance to valid
            return S + 100.0
        
        return S

    # Bounds for DE
    # Each hex can be within roughly [-6, 6] in x and y, angle in [0, 60]
    bounds = [(-6, 6), (-6, 6), (0, 60)] * N

    # Run differential evolution with limited budget
    de_result = None
    try:
        de_result = differential_evolution(
            objective,
            bounds,
            maxiter=200,
            popsize=20,
            tol=1e-5,
            seed=42,
            init='latinhypercube',
            polish=False,
            constraints=[]
        )
    except Exception:
        pass

    if de_result and de_result.fun < best_S:
        params = de_result.x
        inner = []
        for i in range(N):
            inner.append([params[3*i], params[3*i+1], params[3*i+2]])
        cx = sum(p[0] for p in inner) / N
        cy = sum(p[1] for p in inner) / N
        inner_c = [[p[0]-cx, p[1]-cy, p[2]] for p in inner]
        S = compute_outer_hex(inner_c, 0.0, 0.0, 0.0)
        if all_fit(inner_c, 0.0, 0.0, S, 0.0):
            best_S = S
            best_inner = [list(x) for x in inner_c]
            best_center = (0.0, 0.0)
            best_angle = 0.0
            best_valid = True

    # ---- Refinement via local search starting from DE result ----
    if best_valid and best_inner is not None:
        def local_refine(inner, seed):
            nonlocal best_S, best_inner, best_center, best_angle, best_valid
            rng = random.Random(seed)
            cur = [list(x) for x in inner]
            cx = sum(p[0] for p in cur) / N
            cy = sum(p[1] for p in cur) / N
            cur = [[p[0]-cx, p[1]-cy, p[2]] for p in cur]
            cur_S = compute_outer_hex(cur, 0.0, 0.0, 0.0)
            
            for it in range(3000):
                if time.time() > deadline:
                    break
                old = [list(x) for x in cur]
                idx = rng.randrange(N)
                cur[idx][0] += rng.gauss(0, 0.08)
                cur[idx][1] += rng.gauss(0, 0.08)
                if rng.random() < 0.05:
                    cur[idx][2] = rng.choice([0, 15, 30, 45, 60])
                
                new_S = compute_outer_hex(cur, 0.0, 0.0, 0.0)
                if new_S < cur_S and all_fit(cur, 0.0, 0.0, new_S, 0.0):
                    cur_S = new_S
                    if new_S < best_S:
                        best_S = new_S
                        best_inner = [list(x) for x in cur]
                        best_center = (0.0, 0.0)
                        best_angle = 0.0
                        best_valid = True
                else:
                    cur = old

        local_refine(best_inner, 12345)

    # ---- Fallback: lattice blob if DE failed ----
    if not best_valid or best_inner is None:
        sp = math.sqrt(3) * 1.001
        a1 = (sp, 0)
        a2 = (sp/2, sp*math.sqrt(3)/2)
        points = []
        for i in range(-8, 9):
            for j in range(-8, 9):
                x = i*a1[0] + j*a2[0]
                y = i*a1[1] + j*a2[1]
                points.append((x, y))
        points.sort(key=lambda p: p[0]**2 + p[1]**2)
        selected = points[:N]
        cx = sum(p[0] for p in selected)/N
        cy = sum(p[1] for p in selected)/N
        inner = [[p[0]-cx, p[1]-cy, 0.0] for p in selected]
        S = compute_outer_hex(inner, 0.0, 0.0, 0.0)
        if all_fit(inner, 0.0, 0.0, S, 0.0):
            return inner, (0.0, 0.0), S, 0.0
        # Scale up until valid
        for _ in range(200):
            inner = [[p[0]*1.02, p[1]*1.02, p[2]] for p in inner]
            S = compute_outer_hex(inner, 0.0, 0.0, 0.0)
            if all_fit(inner, 0.0, 0.0, S, 0.0):
                return inner, (0.0, 0.0), S, 0.0

    if best_valid and best_inner is not None:
        return best_inner, best_center, best_S, best_angle

    # Absolute fallback
    sp = math.sqrt(3)
    inner = [[(k - N//2)*sp, 0.0, 0.0] for k in range(N)]
    S = compute_outer_hex(inner, 0.0, 0.0, 0.0)
    for _ in range(200):
        if all_fit(inner, 0.0, 0.0, S, 0.0):
            return inner, (0.0, 0.0), S, 0.0
        inner = [[p[0]*1.05, p[1]*1.05, p[2]] for p in inner]
        S = compute_outer_hex(inner, 0.0, 0.0, 0.0)
    return inner, (0.0, 0.0), S, 0.0
# EVOLVE-BLOCK-END
