import math
import numpy as np
import time
from scipy.optimize import minimize_scalar, minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Define candidate point sets (axial coordinates)
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
        (0, 1), (1, 1), (2, 1),
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

    # Additional promising configurations
    axial_29 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_29)

    axial_30 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, -1), (0, 2)
    ]
    candidates.append(axial_30)

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def h_unit_hex(angle):
        # Support function of a unit regular hexagon in direction 'angle'
        # Symmetry period is pi/3
        # In the sector [0, pi/6], h(a) = (sqrt(3)/2) / cos(a)
        a = angle % (math.pi / 3.0)
        if a > math.pi / 6.0:
            a = math.pi / 3.0 - a
        return (math.sqrt(3) / 2.0) / math.cos(a)

    def calc_S_for_theta_alpha(pts, theta, alpha):
        # pts are centered at origin
        max_val = 0.0
        for k in range(6):
            n_angle = theta + math.pi / 6.0 + k * math.pi / 3.0
            nx = math.cos(n_angle)
            ny = math.sin(n_angle)
            # h_P(n)
            projs = pts[:, 0] * nx + pts[:, 1] * ny
            h_p = np.max(projs)
            # h_H_unit(n - alpha)
            h_u = h_unit_hex(n_angle - alpha)
            val = h_p + h_u
            if val > max_val:
                max_val = val
        return (2.0 / math.sqrt(3.0)) * max_val

    def optimize_shape(pts):
        best_S = float('inf')
        best_theta = 0.0
        best_alpha = 0.0
        best_center = None

        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, 0.0, None

        # Center the points at origin for symmetry
        cx0 = np.mean(pts[:, 0])
        cy0 = np.mean(pts[:, 1])
        pts_centered = pts - np.array([cx0, cy0])

        # Coarse grid search over alpha
        # alpha is the rotation of inner hexagons relative to outer
        # Due to symmetry, we only need to check [0, pi/6]
        for i_alpha in range(13):
            alpha = i_alpha * (math.pi / 6.0) / 12.0
            
            # Minimize over theta for this alpha
            def obj_theta(theta):
                return calc_S_for_theta_alpha(pts_centered, theta, alpha)
            
            try:
                if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                    res = minimize_scalar(obj_theta, bounds=(0, math.pi / 6), method='bounded',
                                         options={'maxiter': 200, 'xatol': 1e-10})
                    if res.fun < best_S:
                        best_S = res.fun
                        best_theta = res.x
                        best_alpha = alpha
            except Exception:
                pass
        
        return best_S, best_theta, best_alpha, (cx0, cy0)

    best_global_S = float('inf')
    best_global_pts = None
    best_global_theta = 0.0
    best_global_alpha = 0.0
    best_global_center = None

    for axial in candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)
        S, theta, alpha, center = optimize_shape(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_theta = theta
            best_global_alpha = alpha
            best_global_center = center

    if best_global_center is None or best_global_S == float('inf'):
        best_global_pts = axial_to_cartesian(candidates[0])
        best_global_center = (float(np.mean(best_global_pts[:, 0])), float(np.mean(best_global_pts[:, 1])))
        best_global_theta = 0.0
        best_global_alpha = 0.0
        # Fallback calculation
        pts_c = best_global_pts - np.array(best_global_center)
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = pts_c[:, 0] * nx + pts_c[:, 1] * ny
            h_p = np.max(projs)
            h_u = h_unit_hex(ang)
            val = h_p + h_u
            if val > max_proj: max_proj = val
        best_global_S = (2.0 / math.sqrt(3.0)) * max_proj

    cx, cy = best_global_center
    outer_angle_deg = math.degrees(best_global_theta)
    inner_angle_deg = math.degrees(best_global_theta + best_global_alpha)

    # Use a very small safety margin to ensure no overlap
    SCALE = 1.0 + 1e-12
    pts_scaled = []
    for x, y in best_global_pts:
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        pts_scaled.append((sx, sy))

    pts_scaled = np.array(pts_scaled)

    # Recalculate S with scaled points and final angles
    max_val = 0.0
    for k in range(6):
        n_angle = best_global_theta + math.pi / 6.0 + k * math.pi / 3.0
        nx = math.cos(n_angle)
        ny = math.sin(n_angle)
        projs = (pts_scaled[:, 0] - cx) * nx + (pts_scaled[:, 1] - cy) * ny
        h_p = np.max(projs)
        h_u = h_unit_hex(n_angle - (best_global_theta + best_global_alpha - best_global_theta)) # alpha is relative
        # Wait, in calc_S_for_theta_alpha, h_u was h_unit_hex(n_angle - alpha).
        # Here n_angle is the outer normal. The inner hexagon is rotated by alpha+theta.
        # So the relative angle is n_angle - (alpha+theta) = (theta + pi/6 + k*pi/3) - alpha - theta = pi/6 + k*pi/3 - alpha.
        # This is constant for a given alpha!
        # So h_u is the same for all k? No, n_angle changes, but the relative angle to the inner hexagon normals changes.
        # Let's re-verify.
        # Outer normal direction: $\theta + \pi/6 + k\pi/3$.
        # Inner hexagon orientation: $\phi = \theta + \alpha$.
        # Relative angle: $\theta + \pi/6 + k\pi/3 - (\theta + \alpha) = \pi/6 + k\pi/3 - \alpha$.
        # Yes, the relative orientation of the outer normal to the inner hexagon is fixed for each k, independent of theta.
        # So h_u is constant for a given alpha.
        h_u = h_unit_hex(math.pi/6.0 + k*math.pi/3.0 - best_global_alpha)
        val = h_p + h_u
        if val > max_val:
            max_val = val
            
    S_scaled = (2.0 / math.sqrt(3.0)) * max_val

    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(inner_angle_deg)])

    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(outer_angle_deg)

    return inner, center, side, rotation
