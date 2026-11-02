import math
import numpy as np
import time
from scipy.optimize import linprog, minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 145.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    def hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        verts = []
        for k in range(6):
            theta = angle_rad + k * math.pi / 3.0
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            verts.append((x, y))
        return np.array(verts)

    def outer_normals(outer_angle_deg):
        outer_angle_rad = math.radians(outer_angle_deg)
        normals = []
        for k in range(6):
            theta = outer_angle_rad + math.pi/6.0 + k * math.pi / 3.0
            normals.append((math.cos(theta), math.sin(theta)))
        return np.array(normals)

    def solve_lp(all_verts, normals):
        """Solve LP to find min enclosing hexagon apothem D and center (cx,cy)."""
        A_ub = []
        b_ub = []
        for nx, ny in normals:
            for vx, vy in all_verts:
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-(vx * nx + vy * ny))
                A_ub.append([nx, ny, -1.0])
                b_ub.append(vx * nx + vy * ny)
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        bounds = [(None, None), (None, None), (0, None)]
        c_obj = np.array([0.0, 0.0, 1.0])
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                return res.x[2], (res.x[0], res.x[1])
            else:
                return float('inf'), None
        except Exception:
            return float('inf'), None

    def compute_S(pts, outer_angle_deg):
        """Compute outer hexagon side length for given inner hexagon positions."""
        all_verts = []
        for x, y, ang in pts:
            verts = hex_verts(x, y, 1.0, ang)
            all_verts.append(verts)
        all_verts = np.vstack(all_verts)
        normals = outer_normals(outer_angle_deg)
        D, center = solve_lp(all_verts, normals)
        if center is None:
            return float('inf'), None
        S = D / APOTHEM
        return S, center

    def objective(params):
        pts = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i+1]
            ang = params[3*i+2]
            pts.append((x, y, ang))
        outer_angle = params[3*N]
        S, _ = compute_S(pts, outer_angle)
        return S

    # Generate compact initial configurations
    def axial_to_xy(q, r):
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return x, y

    configs = []

    # Config 1: Central + 6 ring + 4 outer
    pts1 = [(0.0, 0.0, 0.0)]
    for k in range(6):
        theta = k * math.pi / 3.0
        x = 2.0 * math.cos(theta)
        y = 2.0 * math.sin(theta)
        pts1.append((x, y, 0.0))
    for k in range(4):
        theta = k * math.pi / 2.0
        x = 3.5 * math.cos(theta)
        y = 3.5 * math.sin(theta)
        pts1.append((x, y, 0.0))
    configs.append(pts1)

    # Config 2: Axial compact shape
    axial2 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,-2)]
    pts2 = []
    for q,r in axial2:
        x,y = axial_to_xy(q,r)
        pts2.append((x,y,0.0))
    configs.append(pts2)

    # Config 3: 3-4-3-1
    pts3 = []
    for i in range(3):
        pts3.append((i*2.0, 0.0, 0.0))
    for i in range(4):
        pts3.append((i*2.0+1.0, 1.5*math.sqrt(3), 0.0))
    for i in range(3):
        pts3.append((i*2.0, 3.0*math.sqrt(3), 0.0))
    pts3.append((1.0, 4.5*math.sqrt(3), 0.0))
    configs.append(pts3)

    # Config 4: With rotations
    pts4 = [(0.0, 0.0, 0.0)]
    for k in range(6):
        theta = k * math.pi / 3.0
        x = 1.8 * math.cos(theta)
        y = 1.8 * math.sin(theta)
        pts4.append((x, y, 30.0))
    for k in range(4):
        theta = k * math.pi / 2.0 + math.pi/4
        x = 3.2 * math.cos(theta)
        y = 3.2 * math.sin(theta)
        pts4.append((x, y, 0.0))
    configs.append(pts4)

    # Config 5: Axial shape 2
    axial5 = [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3)]
    pts5 = []
    for q,r in axial5:
        x,y = axial_to_xy(q,r)
        pts5.append((x,y,0.0))
    configs.append(pts5)

    # Config 6: Tight ring
    pts6 = [(0.0, 0.0, 0.0)]
    for k in range(6):
        theta = k * math.pi / 3.0
        x = 1.7 * math.cos(theta)
        y = 1.7 * math.sin(theta)
        pts6.append((x, y, 15.0))
    for k in range(4):
        theta = k * math.pi / 2.0
        x = 3.0 * math.cos(theta)
        y = 3.0 * math.sin(theta)
        pts6.append((x, y, 0.0))
    configs.append(pts6)

    # Config 7: Another axial
    axial7 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(1,2),(-1,2),(0,3)]
    pts7 = []
    for q,r in axial7:
        x,y = axial_to_xy(q,r)
        pts7.append((x,y,0.0))
    configs.append(pts7)

    # Config 8: Diamond
    pts8 = [(0.0, 0.0, 0.0)]
    for k in range(4):
        theta = k * math.pi / 2.0
        x = 2.0 * math.cos(theta)
        y = 2.0 * math.sin(theta)
        pts8.append((x, y, 0.0))
    for k in range(4):
        theta = k * math.pi / 2.0 + math.pi/4
        x = 2.5 * math.cos(theta)
        y = 2.5 * math.sin(theta)
        pts8.append((x, y, 0.0))
    pts8.append((0.0, 3.5, 0.0))
    configs.append(pts8)

    # Config 9: Row arrangement
    pts9 = []
    for i in range(5):
        pts9.append((i*2.0, 0.0, 0.0))
    for i in range(3):
        pts9.append((i*2.0+1.0, 1.5*math.sqrt(3), 0.0))
    for i in range(3):
        pts9.append((i*2.0+1.0, -1.5*math.sqrt(3), 0.0))
    configs.append(pts9)

    # Config 10: Hex lattice with gaps
    axial10 = [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(0,2),(1,2),(0,3),(1,3)]
    pts10 = []
    for q,r in axial10:
        x,y = axial_to_xy(q,r)
        pts10.append((x,y,0.0))
    configs.append(pts10)

    best_S = float('inf')
    best_params = None

    # Evaluate configs with multiple outer angles
    for config in configs:
        if time.time() - START_TIME > TIME_LIMIT:
            break
        for outer_ang in [0.0, 30.0, 15.0, 45.0, 60.0]:
            params = []
            for x, y, ang in config:
                params.extend([x, y, ang])
            params.append(outer_ang)
            S = objective(params)
            if S < best_S:
                best_S = S
                best_params = params[:]

    if best_params is None:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Local optimization
    if time.time() - START_TIME < TIME_LIMIT - 15:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-7})
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x[:]
        except Exception:
            pass

    # More perturbations
    if time.time() - START_TIME < TIME_LIMIT - 8:
        np.random.seed(123)
        for trial in range(8):
            if time.time() - START_TIME > TIME_LIMIT - 3:
                break
            perturbed = best_params + np.random.normal(0, 0.05, len(best_params))
            try:
                res = minimize(objective, perturbed, method='Nelder-Mead',
                              options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-7})
                if res.fun < best_S:
                    best_S = res.fun
                    best_params = res.x[:]
            except Exception:
                pass

    # Extract result
    pts = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        ang = best_params[3*i+2]
        pts.append((x, y, ang))
    outer_angle = best_params[3*N]

    S, center = compute_S(pts, outer_angle)
    if center is None:
        cx = float(np.mean([p[0] for p in pts]))
        cy = float(np.mean([p[1] for p in pts]))
        S = 10.0
    else:
        cx, cy = center

    inner = []
    for x, y, ang in pts:
        inner.append([float(x), float(y), float(ang)])

    return inner, [float(cx), float(cy)], float(S), float(outer_angle)
