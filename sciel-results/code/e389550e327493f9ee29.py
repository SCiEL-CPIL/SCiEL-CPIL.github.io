import math
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar, linprog

N = 11
SQRT3 = math.sqrt(3.0)
APOTHEM = SQRT3 / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def _time_ok():
    return time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN

def construct():
    if not _time_ok():
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Axial coordinates for candidate packings (hex grid, side-1 hexagons)
    # The key insight: for a hexagonal packing, the optimal outer hexagon
    # is aligned with the lattice (rotation 0 or multiples of 30°).
    # We need to find the minimal enclosing regular hexagon (with any rotation)
    # of the set of hexagon centers, then add the apothem.

    # Generate many candidate packings
    candidates = []

    # Standard 3-3-3-1-1 arrangement (center + 6 around + 4 more)
    # This is the classic 11-hexagon packing
    # Center at origin, ring of 6, then 4 more
    # Let's use axial coordinates (q, r)
    # Standard hex grid: neighbors of (0,0) are (1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)

    # Packing 1: compact cluster
    # 3 in row 0, 4 in row 1, 3 in row 2, 1 in row 3, 1 in row 4 (roughly)
    # Let me think more carefully.

    # For 11 hexagons, the best known packing has side length around 3.
    # Let's try various configurations.

    # Configuration A: 4-3-4 (rows of hexagons in a hexagonal pattern)
    # In axial coords, a row at constant q or r
    # Let's use the standard hexagonal arrangement

    # Actually, let me think about this differently.
    # 11 hexagons: 1 center + 6 surrounding + 4 more
    # The 4 more can be placed to form a more compact shape

    # Let me just enumerate several plausible packings

    # Packing: center + 6 neighbors + 4 second-ring
    # Second ring positions (axial): (2,0),(2,-1),(1,-2),(-1,-2),(-2,-1),(-2,0),(-1,1),(0,2),(1,2),(2,1)
    # That's 10 second-ring positions. We pick 4 of them.

    # For minimal bounding hexagon, we want a compact, roughly circular shape.
    # Best to pick 4 that are spread around.

    # Option 1: (2,0), (0,2), (-2,0), (0,-2) - but (0,-2) is not a valid 2nd ring
    # 2nd ring in axial: (2,0),(1,-1),(0,-2),(-1,-2),(-2,-1),(-2,0),(-1,1),(0,2),(1,2),(2,1)
    # Wait, let me recalculate. In axial coords (q,r):
    # 1st ring: (1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)
    # 2nd ring: (2,0),(2,-1),(1,-2),(0,-2),(-1,-2),(-2,-1),(-2,0),(-1,1),(0,2),(1,2),(2,1)
    # Hmm, that's 11. Let me be more careful.

    # In cube coords (x,y,z) with x+y+z=0:
    # 1st ring: (1,-1,0),(1,0,-1),(0,1,-1),(-1,1,0),(-1,0,1),(0,-1,1)
    # 2nd ring: (2,-2,0),(2,-1,-1),(1,0,-2),(0,1,-2),(-1,2,-1),(-2,2,0),(-2,1,1),(-1,0,2),(0,-1,2),(1,-2,1),(2,-1,1)
    # That's 11 in 2nd ring.

    # Convert to axial: q=x, r=z
    # 1st ring axial: (1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)
    # 2nd ring axial: (2,0),(2,-1),(1,-2),(0,-2),(-1,-2),(-2,-1),(-2,0),(-1,1),(0,2),(1,2),(2,1)

    # Wait, (-1,1) appears in both? No:
    # 1st ring: (1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)
    # 2nd ring: (2,0),(2,-1),(1,-2),(0,-2),(-1,-2),(-2,-1),(-2,0),(-1,1),(0,2),(1,2),(2,1)
    # (-1,1) is in 1st ring. Let me recheck.
    # Cube (−1,1,0) → axial (q,r) = (−1,0). No wait.
    # Cube to axial: q = x, r = z. So (x,y,z) → (x,z).
    # 1st ring cube: (1,-1,0)→(1,0), (1,0,-1)→(1,-1), (0,1,-1)→(0,-1), (-1,1,0)→(-1,0), (-1,0,1)→(-1,1), (0,-1,1)→(0,1)
    # 2nd ring cube: (2,-2,0)→(2,0), (2,-1,-1)→(2,-1), (1,0,-2)→(1,-2), (0,1,-2)→(0,-2), (-1,2,-1)→(-1,-2), (-2,2,0)→(-2,0), (-2,1,1)→(-2,1), (-1,0,2)→(-1,2), (0,-1,2)→(0,2), (1,-2,1)→(1,1), (2,-1,1)→(2,1)

    # OK so 2nd ring axial: (2,0),(2,-1),(1,-2),(0,-2),(-1,-2),(-2,0),(-2,1),(-1,2),(0,2),(1,1),(2,1)

    # For 11 hexagons: 1 center + 6 first ring + 4 second ring
    # Pick 4 from 11 second-ring positions to minimize bounding hexagon

    # By symmetry, good choices are those that maintain 3-fold or 6-fold symmetry
    # 3-fold: pick every 3rd: (2,0), (0,-2), (-2,1) - but that's only 3
    # For 4, break symmetry slightly

    # Let me just try several specific packings

    packings = []

    # P1: center + 6 neighbors + 4 specific 2nd ring
    # Choose 2nd ring positions that are roughly evenly spaced
    # Angles of 2nd ring (approx): 0°, 30°, 60°, 90°, 120°, 150°, 180°, 210°, 240°, 270°, 300°
    # (2,0)≈0°, (2,-1)≈-30°, (1,-2)≈-60°, (0,-2)≈-90°, (-1,-2)≈-120°, (-2,0)≈180°, (-2,1)≈150°, (-1,2)≈120°, (0,2)≈90°, (1,1)≈60°, (2,1)≈30°

    # Pick 4 roughly evenly: 0°, 90°, 180°, 270° → (2,0),(0,2),(-2,0),(1,-2)
    p1 = [(0,0)] + [(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)] + [(2,0),(0,2),(-2,0),(1,-2)]
    packings.append(p1)

    # P2: (2,0),(2,-1),(-2,1),(0,-2)
    p2 = [(0,0)] + [(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)] + [(2,0),(2,-1),(-2,1),(0,-2)]
    packings.append(p2)

    # P3: (2,0),(0,2),(-2,0),(-2,-1)
    p3 = [(0,0)] + [(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)] + [(2,0),(0,2),(-2,0),(-2,-1)]
    packings.append(p3)

    # P4: (2,0),(1,-2),(-2,0),(0,2)
    p4 = [(0,0)] + [(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)] + [(2,0),(1,-2),(-2,0),(0,2)]
    packings.append(p4)

    # P5: (2,0),(0,-2),(-2,1),(1,1)
    p5 = [(0,0)] + [(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)] + [(2,0),(0,-2),(-2,1),(1,1)]
    packings.append(p5)

    # P6: (2,-1),(1,-2),(-2,0),(0,2)
    p6 = [(0,0)] + [(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)] + [(2,-1),(1,-2),(-2,0),(0,2)]
    packings.append(p6)

    # P7: (2,0),(2,-1),(-2,1),(-1,-2)
    p7 = [(0,0)] + [(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)] + [(2,0),(2,-1),(-2,1),(-1,-2)]
    packings.append(p7)

    # P8: (2,0),(0,2),(-2,0),(-1,2)
    p8 = [(0,0)] + [(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)] + [(2,0),(0,2),(-2,0),(-1,2)]
    packings.append(p8)

    # P9: (2,0),(2,1),(-2,0),(0,-2)
    p9 = [(0,0)] + [(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)] + [(2,0),(2,1),(-2,0),(0,-2)]
    packings.append(p9)

    # P10: (2,0),(1,1),(-2,0),(0,-2)
    p10 = [(0,0)] + [(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)] + [(2,0),(1,1),(-2,0),(0,-2)]
    packings.append(p10)

    # Also try non-standard packings where not all 6 neighbors are present
    # P11: 4 in a row + 4 in next row + 3 in next
    p11 = [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)]
    packings.append(p11)

    # P12: 3-4-4
    p12 = [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2)]
    packings.append(p12)

    # P13: 4-4-3
    p13 = [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2)]
    packings.append(p13)

    # P14: 5-3-3
    p14 = [(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2)]
    packings.append(p14)

    # P15: 3-3-5
    p15 = [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(3,2),(4,2)]
    packings.append(p15)

    # P16: 2-4-2-3 (staggered)
    p16 = [(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)]
    packings.append(p16)

    # P17: diamond shape
    p17 = [(0,0),(1,0),(0,1),(-1,1),(2,0),(1,-1),(0,-1),(-1,0),(-2,1),(-1,2),(1,1)]
    packings.append(p17)

    # P18: another diamond
    p18 = [(0,0),(1,0),(0,1),(-1,1),(2,0),(1,-1),(0,-1),(-1,0),(-2,1),(-1,2),(0,2)]
    packings.append(p18)

    # P19: cross shape
    p19 = [(0,0),(1,0),(2,0),(-1,0),(-2,0),(0,1),(0,2),(0,-1),(0,-2),(1,1),(-1,1)]
    packings.append(p19)

    # P20: T shape
    p20 = [(0,0),(1,0),(2,0),(-1,0),(-2,0),(0,1),(1,1),(2,1),(-1,1),(-2,1),(0,2)]
    packings.append(p20)

    # P21: L shape
    p21 = [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(3,2),(3,1)]
    packings.append(p21)

    # P22: zigzag
    p22 = [(0,0),(1,0),(1,1),(2,1),(2,2),(3,2),(0,1),(0,2),(-1,2),(-1,1),(-2,1)]
    packings.append(p22)

    # P23: another compact
    p23 = [(0,0),(1,0),(0,1),(1,1),(2,1),(2,0),(-1,0),(-1,1),(0,2),(1,2),(2,2)]
    packings.append(p23)

    # P24: hexagonal with one gap filled differently
    p24 = [(0,0),(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1),(2,0),(2,-1),(-2,1),(0,2)]
    packings.append(p24)

    # P25
    p25 = [(0,0),(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1),(2,0),(1,-2),(-2,0),(0,2)]
    packings.append(p25)

    # P26
    p26 = [(0,0),(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1),(2,0),(0,-2),(-2,0),(1,1)]
    packings.append(p26)

    # P27
    p27 = [(0,0),(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1),(2,0),(2,-1),(-2,0),(0,2)]
    packings.append(p27)

    # P28
    p28 = [(0,0),(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1),(2,0),(1,-2),(-2,1),(0,2)]
    packings.append(p28)

    # P29: 4-3-4 with offset
    p29 = [(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2),(4,2)]
    packings.append(p29)

    # P30: 4-3-4 with different offset
    p30 = [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2),(4,2)]
    packings.append(p30)

    candidates = packings

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = SQRT3 * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def compute_min_side(pts, beta):
        """Compute minimal side length of enclosing regular hexagon with rotation beta."""
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))

        c_obj = np.array([0.0, 0.0, 1.0])
        A_ub = []
        b_ub = []

        for nx, ny in normals:
            for px, py in pts:
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-(px * nx + py * ny))
                A_ub.append([nx, ny, -1.0])
                b_ub.append(px * nx + py * ny)

        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        bounds = [(None, None), (None, None), (0, None)]

        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                D = res.x[2]
                cx = res.x[0]
                cy = res.x[1]
                S = (D + APOTHEM) / APOTHEM
                return S, (cx, cy), D
            else:
                return float('inf'), None, None
        except Exception:
            return float('inf'), None, None

    def smooth_side(pts, beta, cx, cy, temperature=0.1):
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))

        projs = np.array([
            ((px - cx) * nx + (py - cy) * ny)
            for nx, ny in normals
            for px, py in pts
        ])
        all_projs = np.concatenate([projs, -projs])

        max_val = np.max(all_projs)
        if max_val > 100:
            return float('inf')

        shifted = (all_projs - max_val) / temperature
        log_sum = np.log(np.sum(np.exp(shifted))) + max_val / temperature
        D = temperature * log_sum
        S = (D + APOTHEM) / APOTHEM
        return S

    def optimize_for_packing(pts):
        """Optimize rotation and center for a given set of points."""
        best_S = float('inf')
        best_beta = 0.0
        best_center = None

        if not _time_ok():
            return float('inf'), 0.0, None

        # Grid search over rotation
        for i in range(72):
            beta = i * (math.pi / 6) / 72
            S, center, D = compute_min_side(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center

        # Refine with bounded scalar optimization
        try:
            if _time_ok():
                res = minimize_scalar(lambda b: compute_min_side(pts, b)[0],
                                     bounds=(0, math.pi / 6), method='bounded',
                                     options={'maxiter': 300, 'xatol': 1e-12})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = compute_min_side(pts, best_beta)
        except Exception:
            pass

        # Smooth optimization for joint (cx, cy, beta)
        try:
            if _time_ok():
                temps = [0.3, 0.1, 0.05, 0.02]
                start_betas = [0.0, math.pi/12, math.pi/9, best_beta]

                for temp in temps:
                    if not _time_ok():
                        break
                    for sb in start_betas:
                        if not _time_ok():
                            break

                        def smooth_obj(params, t=temp):
                            cx, cy, beta = params
                            beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                            return smooth_side(pts, beta, cx, cy, temperature=t)

                        init = [best_center[0], best_center[1], sb]
                        res = minimize(smooth_obj, init, method='Nelder-Mead',
                                     options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12})

                        if res.fun < 1e10:
                            cx_r, cy_r, beta_r = res.x
                            beta_r = ((beta_r % (math.pi/6)) + math.pi/6) % (math.pi/6)
                            S_exact, c_exact, _ = compute_min_side(pts, beta_r)
                            if S_exact < best_S:
                                best_S = S_exact
                                best_beta = beta_r
                                best_center = c_exact
        except Exception:
            pass

        # Final polish with exact objective
        try:
            if _time_ok():
                def exact_obj(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    normals = []
                    for k in range(6):
                        ang = beta + math.pi / 6.0 + k * math.pi / 3.0
                        normals.append((math.cos(ang), math.sin(ang)))

                    max_proj = 0
                    for nx, ny in normals:
                        for px, py in pts:
                            proj = (px - cx) * nx + (py - cy) * ny
                            max_proj = max(max_proj, abs(proj))

                    S = (max_proj + APOTHEM) / APOTHEM
                    return S

                polish_starts = [
                    [best_center[0], best_center[1], best_beta],
                    [best_center[0] + 0.05, best_center[1], best_beta],
                    [best_center[0], best_center[1] + 0.05, best_beta],
                    [best_center[0] - 0.05, best_center[1], best_beta],
                    [best_center[0], best_center[1] - 0.05, best_beta],
                    [best_center[0], best_center[1], best_beta + 0.02],
                    [best_center[0], best_center[1], best_beta - 0.02],
                ]

                for init in polish_starts:
                    if not _time_ok():
                        break
                    res = minimize(exact_obj, init, method='Nelder-Mead',
                                 options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14})
                    if res.fun < best_S:
                        best_S = res.fun
                        best_center = (res.x[0], res.x[1])
                        best_beta = ((res.x[2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
        except Exception:
            pass

        return best_S, best_beta, best_center

    # Also try joint optimization of positions for the best packings
    def joint_optimize(axial_list, max_iter=200):
        """Optimize both positions and rotation jointly."""
        pts0 = axial_to_cartesian(axial_list)
        n = len(pts0)

        # Parameters: 2*n positions + 1 rotation = 2n+1
        # But we can fix one hexagon to remove translation invariance
        # Let's fix the first hexagon at origin
        # Params: 2*(n-1) positions + 1 rotation

        def objective(params):
            if not _time_ok():
                return 1e10
            beta = params[-1]
            beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)

            # Reconstruct points
            pts = np.zeros((n, 2))
            pts[0] = [0.0, 0.0]
            for i in range(1, n):
                pts[i] = [params[2*(i-1)], params[2*(i-1)+1]]

            normals = []
            for k in range(6):
                ang = beta + math.pi / 6.0 + k * math.pi / 3.0
                normals.append((math.cos(ang), math.sin(ang)))

            # Find center and D via linprog
            c_obj = np.array([0.0, 0.0, 1.0])
            A_ub = []
            b_ub = []
            for nx, ny in normals:
                for px, py in pts:
                    A_ub.append([-nx, -ny, -1.0])
                    b_ub.append(-(px * nx + py * ny))
                    A_ub.append([nx, ny, -1.0])
                    b_ub.append(px * nx + py * ny)

            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)
            bounds = [(None, None), (None, None), (0, None)]

            try:
                res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
                if res.status != 0:
                    return 1e10
                D = res.x[2]
                S = (D + APOTHEM) / APOTHEM

                # Check for overlaps
                for i in range(n):
                    for j in range(i+1, n):
                        dx = pts[i, 0] - pts[j, 0]
                        dy = pts[i, 1] - pts[j, 1]
                        dist = math.sqrt(dx*dx + dy*dy)
                        if dist < 2.0 - 1e-6:
                            return S + (2.0 - dist) * 1000
                return S
            except Exception:
                return 1e10

        # Initial params
        init_params = np.zeros(2*(n-1) + 1)
        for i in range(1, n):
            init_params[2*(i-1)] = pts0[i, 0]
            init_params[2*(i-1)+1] = pts0[i, 1]
        init_params[-1] = 0.0

        try:
            res = minimize(objective, init_params, method='Nelder-Mead',
                          options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.fun < 1e10:
                beta = ((res.x[-1] % (math.pi/6)) + math.pi/6) % (math.pi/6)
                pts = np.zeros((n, 2))
                pts[0] = [0.0, 0.0]
                for i in range(1, n):
                    pts[i] = [res.x[2*(i-1)], res.x[2*(i-1)+1]]
                S, center, D = compute_min_side(pts, beta)
                return S, beta, center, pts
            else:
                return float('inf'), 0.0, None, None
        except Exception:
            return float('inf'), 0.0, None, None

    # Evaluate all packings
    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None
    best_global_axial = None

    for axial in candidates:
        if not _time_ok():
            break
        pts = axial_to_cartesian(axial)
        S, beta, center = optimize_for_packing(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center
            best_global_axial = axial

    # Try joint optimization on the best few packings
    if _time_ok() and best_global_axial is not None:
        # Sort candidates by their rigid S and try joint opt on top 5
        ranked = []
        for axial in candidates:
            if not _time_ok():
                break
            pts = axial_to_cartesian(axial)
            S, beta, center = optimize_for_packing(pts)
            ranked.append((S, axial))
        ranked.sort(key=lambda x: x[0])

        for S_r, axial_r in ranked[:5]:
            if not _time_ok():
                break
            S_j, beta_j, center_j, pts_j = joint_optimize(axial_r, max_iter=300)
            if S_j < best_global_S:
                best_global_S = S_j
                best_global_pts = pts_j
                best_global_beta = beta_j
                best_global_center = center_j
                best_global_axial = axial_r

    if best_global_center is None or best_global_S == float('inf'):
        pts = axial_to_cartesian(candidates[0])
        best_global_center = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))
        best_global_beta = 0.0
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = pts[:, 0] * nx + pts[:, 1] * ny
            max_proj = max(max_proj, float(np.max(projs)), float(-np.min(projs)))
        best_global_S = (max_proj + APOTHEM) / APOTHEM
        best_global_pts = pts

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)

    # Scale points slightly to ensure no overlap (safety)
    SCALE = 1.0 + 1e-10
    pts_scaled = []
    for x, y in best_global_pts:
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        pts_scaled.append((sx, sy))

    pts_scaled = np.array(pts_scaled)

    # Compute final side length
    max_proj = 0
    for k in range(6):
        ang = best_global_beta + math.pi / 6.0 + k * math.pi / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        projs = (pts_scaled[:, 0] - cx) * nx + (pts_scaled[:, 1] - cy) * ny
        max_proj = max(max_proj, float(np.max(projs)), float(-np.min(projs)))

    D_scaled = max_proj
    S_scaled = (D_scaled + APOTHEM) / APOTHEM

    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(angle_deg)])

    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(angle_deg)

    return inner, center, side, rotation
