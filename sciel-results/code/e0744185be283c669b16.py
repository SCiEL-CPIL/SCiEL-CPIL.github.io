import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9
TIME_BUDGET_PER_CANDIDATE = 3.0
TIME_BUDGET_RELAX = 8.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    candidates = []

    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    candidates.append(axial_1)

    axial_2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_2)

    axial_3 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1)
    ]
    candidates.append(axial_3)

    axial_4 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    candidates.append(axial_4)

    axial_5 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_5)

    axial_6 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_6)

    axial_7 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_7)

    axial_8 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)
    ]
    candidates.append(axial_8)

    axial_9 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (-2, 2), (-1, 3), (1, -2)
    ]
    candidates.append(axial_9)

    axial_10 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_10)

    axial_11 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3)
    ]
    candidates.append(axial_11)

    axial_12 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3),
        (0, 4)
    ]
    candidates.append(axial_12)

    axial_13 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, 2)
    ]
    candidates.append(axial_13)

    axial_14 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, -1)
    ]
    candidates.append(axial_14)

    axial_15 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1), (0, -1)
    ]
    candidates.append(axial_15)

    axial_16 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 2)
    ]
    candidates.append(axial_16)

    axial_17 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (1, 2), (-1, 2), (0, 3)
    ]
    candidates.append(axial_17)

    axial_18 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_18)

    axial_19 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 3)
    ]
    candidates.append(axial_19)

    axial_20 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, 1)
    ]
    candidates.append(axial_20)

    axial_21 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_21)

    axial_22 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3)
    ]
    candidates.append(axial_22)

    axial_23 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_23)

    axial_24 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3),
        (0, 4)
    ]
    candidates.append(axial_24)

    axial_25 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_25)

    axial_26 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_26)

    axial_27 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, -2)
    ]
    candidates.append(axial_27)

    axial_28 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3), (2, 3),
        (1, 4), (2, 4)
    ]
    candidates.append(axial_28)

    axial_29 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    candidates.append(axial_29)

    axial_30 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (-1, 2)
    ]
    candidates.append(axial_30)

    axial_31 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_31)

    axial_32 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_32)

    axial_33 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (-2, 2), (0, 3), (1, -2)
    ]
    candidates.append(axial_33)

    axial_34 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2), (3, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_34)

    axial_35 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, 2)
    ]
    candidates.append(axial_35)

    axial_36 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_36)

    axial_37 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3), (3, 3),
        (0, 4)
    ]
    candidates.append(axial_37)

    axial_38 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 2), (-1, -2)
    ]
    candidates.append(axial_38)

    axial_39 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_39)

    axial_40 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3),
        (1, 4), (2, 4)
    ]
    candidates.append(axial_40)

    axial_41 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3),
        (0, 4)
    ]
    candidates.append(axial_41)

    axial_42 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3),
        (0, 4)
    ]
    candidates.append(axial_42)

    axial_43 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_43)

    axial_44 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3),
        (0, 4), (1, 4)
    ]
    candidates.append(axial_44)

    axial_45 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_45)

    axial_46 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3),
        (0, 4)
    ]
    candidates.append(axial_46)

    axial_47 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_47)

    axial_48 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, 2)
    ]
    candidates.append(axial_48)

    axial_49 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3),
        (1, 4)
    ]
    candidates.append(axial_49)

    axial_50 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_50)

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def calc_S_for_beta(pts, beta):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None
        
        normals = np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])
        
        n_pts = len(pts)
        c_obj = np.array([0.0, 0.0, 1.0])
        A_ub = []
        b_ub = []
        
        for nx, ny in normals:
            for i in range(n_pts):
                px, py = pts[i]
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

    def exact_S(pts, beta, cx, cy):
        normals = np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])
        pts_centered = pts - np.array([cx, cy])
        projs = np.abs(pts_centered @ normals.T)
        max_proj = float(np.max(projs))
        S = (max_proj + APOTHEM) / APOTHEM
        return S

    def smooth_calc_S(pts, beta, cx, cy, temperature=0.1):
        normals = np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])
        
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T
        all_projs = np.concatenate([projs.ravel(), (-projs).ravel()])
        
        max_val = np.max(all_projs)
        if max_val > 100:
            return float('inf')
        
        shifted = (all_projs - max_val) / temperature
        shifted = np.clip(shifted, -50, 50)
        log_sum = np.log(np.sum(np.exp(shifted))) + max_val / temperature
        D = temperature * log_sum
        S = (D + APOTHEM) / APOTHEM
        return S

    def hex_pair_min_dist(p1, p2, a1, a2):
        """Compute minimum distance between two unit hexagons with centers p1,p2 and angles a1,a2."""
        d = np.array(p2) - np.array(p1)
        dist = np.linalg.norm(d)
        if dist < 1e-15:
            return 0.0
        # For two regular hexagons of side 1, the minimum distance between their boundaries
        # when centers are separated by vector d.
        # The support function of a unit hexagon in direction u is APOTHEM * |cos(angle_diff)|
        # where angle_diff is the angle between u and the nearest normal direction.
        # Actually, for a regular hexagon with flat-top orientation, the support in direction theta
        # is APOTHEM / max_k |cos(theta - (pi/6 + k*pi/3))| ... this gets complex.
        # Simpler: the distance between two convex polygons. For hexagons, the minimum
        # distance is approximately: if dist > 2*APOTHEM*sqrt(3)/2 ... 
        # Let's use: min_dist = dist - 2 * support_hex(theta_d) where theta_d is direction of d
        # support of unit hexagon (flat top) in direction phi:
        # The hexagon vertices at angles 0, 60, 120, 180, 240, 300 degrees (pointy-top) or
        # for flat-top: normals at 30, 90, 150, 210, 270, 330 degrees
        # support(phi) = APOTHEM / max_k |cos(phi - (pi/6 + k*pi/3))|
        # But this is only valid when the hexagon is axis-aligned. With rotation, we adjust.
        # For hexagon with rotation a, the support in direction phi is:
        # support(phi - a) for a unit hexagon with normals at pi/6 + k*pi/3
        # support_base(phi) = APOTHEM / max_k |cos(phi - (pi/6 + k*pi/3))| for k=0..5
        # Actually simpler: for a regular hexagon, the distance from center to boundary in direction phi
        # is APOTHEM / |cos(phi mod pi/3 - pi/6)| ... let me just compute it directly.
        
        # The distance from center to edge in direction phi for a unit hexagon (flat-top, normals at 30+60k deg):
        # If we define the hexagon with vertices at (cos(t), sin(t)) * 1 for t in {0, 60, 120, 180, 240, 300} deg
        # then the apothem is sqrt(3)/2 and the support function in direction phi is:
        # r(phi) = APOTHEM / cos(phi - pi/6) for phi in [0, pi/3] (by symmetry, extend periodically)
        # More precisely: let phi_mod = phi % (pi/3), then r = APOTHEM / cos(phi_mod - pi/6)
        # Wait, that's not right either. Let me think again.
        
        # For a regular hexagon with circumradius 1 (side length 1), the apothem is sqrt(3)/2.
        # The distance from center to the boundary in direction phi:
        # The hexagon has 6 edges. Each edge is at distance APOTHEM from center.
        # The edge normals are at angles pi/6 + k*pi/3 for k=0..5 (for flat-top).
        # The support function h(phi) = max_k (APOTHEM * |cos(phi - (pi/6 + k*pi/3))|) ... no.
        # Actually h(phi) = max over vertices of dot(v, u) where u is unit vector in direction phi.
        # For a regular hexagon with circumradius 1, vertices at angles 0, pi/3, 2pi/3, pi, 4pi/3, 5pi/3.
        # h(phi) = max_k cos(phi - k*pi/3)
        # The apothem = cos(pi/6) = sqrt(3)/2.
        
        # For our purpose, we need the minimum distance between two hexagons.
        # A simpler approximation: the "effective radius" of a hexagon in direction phi is:
        # r_eff(phi) = APOTHEM / cos((phi mod pi/3) - pi/6)  ... this peaks at circumradius when phi aligns with vertex.
        # Actually: for flat-top hexagon with normals at pi/6 + k*pi/3:
        # In direction phi, find the nearest normal direction. Let delta = ((phi - pi/6) mod pi/3) - pi/6
        # Then r = APOTHEM / cos(delta)
        
        def hex_support(phi, rot):
            # Support of unit hexagon rotated by rot, in direction phi
            phi_rel = phi - rot
            # Reduce to [0, pi/3)
            phi_mod = phi_rel % (math.pi / 3)
            # delta from nearest normal (normals at 0, pi/3, ... after reduction)
            # Actually normals are at pi/6 + k*pi/3, so relative to that:
            # Let's use: the distance from center to boundary in direction phi for a unit hexagon
            # with vertices at circumradius 1.
            # h(phi) = max_k cos(phi - k*pi/3) for vertices at k*pi/3
            # = cos(phi mod pi/3 - pi/3/2) ... no
            # Let me just compute: for unit hexagon (circumradius 1), vertices at angles 0, 60, ..., 300 deg
            # h(phi) = max(cos(phi), cos(phi-pi/3), cos(phi-2pi/3), cos(phi-pi), cos(phi-4pi/3), cos(phi-5pi/3))
            # By symmetry, h(phi) = cos(phi_mod) where phi_mod = ((phi + pi/6) % (pi/3)) - pi/6
            # Hmm, let me just use the formula: h(phi) = APOTHEM / cos(((phi) % (pi/3)) - pi/6)
            # Check: phi=0 (toward vertex): (0 % pi/3) - pi/6 = -pi/6, cos(-pi/6)=sqrt(3)/2, APOTHEM/(sqrt(3)/2)=1. Correct (circumradius).
            # phi=pi/6 (toward edge normal): (pi/6 % pi/3) - pi/6 = 0, cos(0)=1, APOTHEM/1=APOTHEM. Correct.
            phi_m = phi_rel % (math.pi / 3)
            delta = phi_m - math.pi / 6
            return APOTHEM / math.cos(delta)
        
        # Direction from p1 to p2
        theta_d = math.atan2(d[1], d[0])
        
        # Support of hex1 in direction theta_d (outward from p1 toward p2)
        s1 = hex_support(theta_d, a1)
        # Support of hex2 in direction theta_d + pi (outward from p2 away from p1)
        s2 = hex_support(theta_d + math.pi, a2)
        
        # Minimum distance between boundaries
        min_dist = dist - s1 - s2
        return min_dist

    def relaxation_objective(params, pts_init, beta):
        """
        params: [cx, cy, beta, x0, y0, x1, y1, ..., x10, y10]
        Optimize container center, rotation, and free positions of 11 hexagons.
        Objective: S + penalty * sum(max(0, -min_dist_ij)^2)
        """
        n = len(pts_init)
        cx = params[0]
        cy = params[1]
        beta_r = params[2]
        beta_r = ((beta_r % (math.pi/6)) + math.pi/6) % (math.pi/6)
        
        # Extract positions
        positions = []
        for i in range(n):
            xi = params[3 + 2*i]
            yi = params[3 + 2*i + 1]
            positions.append((xi, yi))
        
        # Compute S for the container
        normals = np.array([
            (math.cos(beta_r + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta_r + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])
        
        max_proj = 0.0
        for i, (px, py) in enumerate(positions):
            dx = px - cx
            dy = py - cy
            for nx, ny in normals:
                proj = abs(dx * nx + dy * ny)
                if proj > max_proj:
                    max_proj = proj
        
        S = (max_proj + APOTHEM) / APOTHEM
        
        # Non-overlap penalty
        penalty = 0.0
        for i in range(n):
            for j in range(i+1, n):
                md = hex_pair_min_dist(positions[i], positions[j], 0.0, 0.0)
                if md < 0:
                    penalty += md * md
        
        return S + 100.0 * penalty

    def relax_positions(pts_init):
        """Use lattice relaxation to find a tighter configuration."""
        n = len(pts_init)
        best_S = float('inf')
        best_positions = None
        best_beta = 0.0
        best_center = None
        
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None, None
        
        # Start from the rigid lattice configuration
        start_params = [float(np.mean(pts_init[:, 0])), float(np.mean(pts_init[:, 1])), 0.0]
        for x, y in pts_init:
            start_params.append(float(x))
            start_params.append(float(y))
        
        # Try multiple beta starting points
        beta_starts = [0.0, math.pi/12, math.pi/6 * 0.25, math.pi/6 * 0.5, math.pi/6 * 0.75]
        
        for b0 in beta_starts:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            params0 = list(start_params)
            params0[2] = b0
            
            try:
                res = minimize(relaxation_objective, params0, args=(pts_init, b0),
                             method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12})
                
                if res.fun < 1e10:
                    # Extract results
                    cx_r = res.x[0]
                    cy_r = res.x[1]
                    beta_r = ((float(res.x[2]) % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    
                    positions = []
                    for i in range(n):
                        positions.append((float(res.x[3 + 2*i]), float(res.x[3 + 2*i + 1])))
                    
                    # Verify no overlap and compute exact S
                    valid = True
                    for i in range(n):
                        for j in range(i+1, n):
                            md = hex_pair_min_dist(positions[i], positions[j], 0.0, 0.0)
                            if md < -1e-6:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        # Compute exact S
                        normals = np.array([
                            (math.cos(beta_r + math.pi / 6.0 + k * math.pi / 3.0),
                             math.sin(beta_r + math.pi / 6.0 + k * math.pi / 3.0))
                            for k in range(6)
                        ])
                        max_proj = 0.0
                        for (px, py) in positions:
                            dx = px - cx_r
                            dy = py - cy_r
                            for nx, ny in normals:
                                proj = abs(dx * nx + dy * ny)
                                if proj > max_proj:
                                    max_proj = proj
                        S_exact = (max_proj + APOTHEM) / APOTHEM
                        
                        if S_exact < best_S:
                            best_S = S_exact
                            best_positions = positions
                            best_beta = beta_r
                            best_center = (cx_r, cy_r)
            except Exception:
                pass
        
        # Polish with Nelder-Mead on the relaxed positions
        if best_positions is not None and time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            def polish_obj(params):
                cx = params[0]
                cy = params[1]
                beta_r = ((float(params[2]) % (math.pi/6)) + math.pi/6) % (math.pi/6)
                positions = []
                for i in range(n):
                    positions.append((float(params[3 + 2*i]), float(params[3 + 2*i + 1])))
                
                normals = np.array([
                    (math.cos(beta_r + math.pi / 6.0 + k * math.pi / 3.0),
                     math.sin(beta_r + math.pi / 6.0 + k * math.pi / 3.0))
                    for k in range(6)
                ])
                max_proj = 0.0
                for (px, py) in positions:
                    dx = px - cx
                    dy = py - cy
                    for nx, ny in normals:
                        proj = abs(dx * nx + dy * ny)
                        if proj > max_proj:
                            max_proj = proj
                S = (max_proj + APOTHEM) / APOTHEM
                
                penalty = 0.0
                for i in range(n):
                    for j in range(i+1, n):
                        md = hex_pair_min_dist(positions[i], positions[j], 0.0, 0.0)
                        if md < 0:
                            penalty += md * md
                return S + 50.0 * penalty
            
            init = [best_center[0], best_center[1], best_beta]
            for (x, y) in best_positions:
                init.append(x)
                init.append(y)
            
            try:
                res = minimize(polish_obj, init, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True})
                if res.fun < best_S:
                    cx_r = float(res.x[0])
                    cy_r = float(res.x[1])
                    beta_r = ((float(res.x[2]) % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    positions = []
                    for i in range(n):
                        positions.append((float(res.x[3 + 2*i]), float(res.x[3 + 2*i + 1])))
                    
                    # Verify
                    valid = True
                    for i in range(n):
                        for j in range(i+1, n):
                            md = hex_pair_min_dist(positions[i], positions[j], 0.0, 0.0)
                            if md < -1e-6:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        normals = np.array([
                            (math.cos(beta_r + math.pi / 6.0 + k * math.pi / 3.0),
                             math.sin(beta_r + math.pi / 6.0 + k * math.pi / 3.0))
                            for k in range(6)
                        ])
                        max_proj = 0.0
                        for (px, py) in positions:
                            dx = px - cx_r
                            dy = py - cy_r
                            for nx, ny in normals:
                                proj = abs(dx * nx + dy * ny)
                                if proj > max_proj:
                                    max_proj = proj
                        S_exact = (max_proj + APOTHEM) / APOTHEM
                        if S_exact < best_S:
                            best_S = S_exact
                            best_positions = positions
                            best_beta = beta_r
                            best_center = (cx_r, cy_r)
            except Exception:
                pass
        
        return best_S, best_beta, best_center, best_positions

    def optimize_shape(pts):
        best_S = float('inf')
        best_beta = 0.0
        best_center = None
        best_free_positions = None
        
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None, None
        
        # Coarse grid search over beta
        n_grid = 60
        for i in range(n_grid + 1):
            beta = i * (math.pi / 6) / n_grid
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center
        
        # Refine beta with bounded scalar optimization
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                refine_starts = [best_beta, 0.0, math.pi/12, math.pi/6 * 0.25, math.pi/6 * 0.75, math.pi/6 * 0.1, math.pi/6 * 0.4, math.pi/6 * 0.6, math.pi/6 * 0.9]
                for rb in refine_starts:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    lo = max(0, rb - 0.25)
                    hi = min(math.pi/6, rb + 0.25)
                    res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0], bounds=(lo, hi), method='bounded',
                                         options={'maxiter': 500, 'xatol': 1e-14})
                    if res.fun < best_S:
                        best_S = res.fun
                        best_beta = res.x
                        _, best_center, _ = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass
        
        # Smooth surrogate optimization
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                temperatures = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
                start_configs = [
                    (best_center[0], best_center[1], best_beta),
                    (best_center[0], best_center[1], best_beta + 0.05),
                    (best_center[0], best_center[1], best_beta - 0.05),
                    (best_center[0], best_center[1], 0.0),
                    (best_center[0], best_center[1], math.pi/12),
                    (best_center[0], best_center[1], math.pi/6 * 0.3),
                    (best_center[0], best_center[1], math.pi/6 * 0.7),
                ]
                
                for temp in temperatures:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    for sc in start_configs:
                        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                            break
                        
                        def smooth_obj(params, t=temp):
                            cx, cy, beta = params
                            beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                            return smooth_calc_S(pts, beta, cx, cy, temperature=t)
                        
                        res = minimize(smooth_obj, list(sc), method='Nelder-Mead',
                                     options={'maxiter': 800, 'xatol': 1e-13, 'fatol': 1e-13,
                                              'adaptive': True})
                        
                        if res.fun < 1e10:
                            cx_r, cy_r, beta_r = res.x
                            beta_r = ((beta_r % (math.pi/6)) + math.pi/6) % (math.pi/6)
                            S_exact = exact_S(pts, beta_r, cx_r, cy_r)
                            if S_exact < best_S:
                                best_S = S_exact
                                best_beta = beta_r
                                best_center = (float(cx_r), float(cy_r))
        except Exception:
            pass
        
        # Final high-precision polish
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def objective(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    return exact_S(pts, beta, cx, cy)
                
                bcx, bcy = best_center
                bb = best_beta
                polish_starts = [[bcx, bcy, bb]]
                for dr in [0.05, -0.05, 0.02, -0.02, 0.01, -0.01]:
                    for ang in range(0, 360, 15):
                        dx = dr * math.cos(math.radians(ang))
                        dy = dr * math.sin(math.radians(ang))
                        polish_starts.append([bcx + dx, bcy + dy, bb])
                for db in [-0.03, -0.015, -0.008, -0.004, -0.002, 0.002, 0.004, 0.008, 0.015, 0.03]:
                    polish_starts.append([bcx, bcy, bb + db])
                for db in [-0.015, -0.008, 0.008, 0.015]:
                    for dr in [0.03, -0.03, 0.015, -0.015]:
                        polish_starts.append([bcx + dr, bcy, bb + db])
                        polish_starts.append([bcx, bcy + dr, bb + db])
                
                for init in polish_starts:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    res = minimize(objective, init, method='Nelder-Mead',
                                 options={'maxiter': 15000, 'xatol': 1e-16, 'fatol': 1e-16,
                                          'adaptive': True})
                    if res.fun < best_S:
                        best_S = res.fun
                        best_center = (float(res.x[0]), float(res.x[1]))
                        best_beta = ((float(res.x[2]) % (math.pi/6)) + math.pi/6) % (math.pi/6)
        except Exception:
            pass
            
        # Ultra-fine final polish
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def objective(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    return exact_S(pts, beta, cx, cy)
                
                bcx, bcy = best_center
                bb = best_beta
                ultra_starts = [
                    [bcx, bcy, bb],
                    [bcx + 0.008, bcy, bb],
                    [bcx - 0.008, bcy, bb],
                    [bcx, bcy + 0.008, bb],
                    [bcx, bcy - 0.008, bb],
                    [bcx + 0.004, bcy + 0.004, bb],
                    [bcx - 0.004, bcy - 0.004, bb],
                    [bcx, bcy, bb + 0.003],
                    [bcx, bcy, bb - 0.003],
                    [bcx, bcy, bb + 0.001],
                    [bcx, bcy, bb - 0.001],
                ]
                
                for init in ultra_starts:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    res = minimize(objective, init, method='Nelder-Mead',
                                 options={'maxiter': 30000, 'xatol': 1e-18, 'fatol': 1e-18,
                                          'adaptive': True})
                    if res.fun < best_S:
                        best_S = res.fun
                        best_center = (float(res.x[0]), float(res.x[1]))
                        best_beta = ((float(res.x[2]) % (math.pi/6)) + math.pi/6) % (math.pi/6)
        except Exception:
            pass
        
        # Lattice relaxation: allow free positions
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                S_rel, beta_rel, center_rel, pos_rel = relax_positions(pts)
                if S_rel < best_S:
                    best_S = S_rel
                    best_beta = beta_rel
                    best_center = center_rel
                    best_free_positions = pos_rel
        except Exception:
            pass
        
        return best_S, best_beta, best_center, best_free_positions

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None
    best_global_free_pos = None

    def hex_radius(axial_list):
        pts = axial_to_cartesian(axial_list)
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        max_proj = 0.0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = (pts[:, 0] - cx) * nx + (pts[:, 1] - cy) * ny
            max_proj = max(max_proj, float(np.max(projs)), float(-np.min(projs)))
        return max_proj
    
    sorted_indices = sorted(range(len(candidates)), key=lambda i: hex_radius(candidates[i]))
    
    for idx in sorted_indices:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        axial = candidates[idx]
        pts = axial_to_cartesian(axial)
        
        candidate_start = time.time()
        S, beta, center, free_pos = optimize_shape(pts)
        elapsed = time.time() - candidate_start
        
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center
            best_global_free_pos = free_pos
        
        if elapsed > TIME_BUDGET_PER_CANDIDATE:
            remaining = TIME_LIMIT - SAFETY_MARGIN - (time.time() - START_TIME)
            if remaining < 15.0:
                break

    if best_global_center is None or best_global_S == float('inf'):
        best_global_pts = axial_to_cartesian(candidates[0])
        best_global_center = (float(np.mean(best_global_pts[:, 0])), float(np.mean(best_global_pts[:, 1])))
        best_global_beta = 0.0
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = best_global_pts[:, 0] * nx + best_global_pts[:, 1] * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        best_global_S = (max_proj + APOTHEM) / APOTHEM

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)
    
    # Use free positions if available, otherwise use lattice positions
    if best_global_free_pos is not None:
        final_positions = best_global_free_pos
    else:
        # Use a very small safety margin to ensure no overlap
        SCALE = 1.0 + 1e-12
        final_positions = []
        for x, y in best_global_pts:
            sx = cx + (x - cx) * SCALE
            sy = cy + (y - cy) * SCALE
            final_positions.append((sx, sy))
    
    max_proj = 0.0
    for (px, py) in final_positions:
        dx = px - cx
        dy = py - cy
        for k in range(6):
            ang = best_global_beta + math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            proj = abs(dx * nx + dy * ny)
            if proj > max_proj:
                max_proj = proj
    
    D_scaled = max_proj
    S_scaled = (D_scaled + APOTHEM) / APOTHEM
    
    inner = []
    for (x, y) in final_positions:
        inner.append([float(x), float(y), float(angle_deg)])
        
    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
