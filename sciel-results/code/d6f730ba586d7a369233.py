import numpy as np
from scipy.optimize import minimize
import time

N = 12
SQRT3 = np.sqrt(3)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def hex_vertices(cx, cy, side, angle_rad):
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        x = cx + side * np.cos(angles)
        y = cy + side * np.sin(angles)
        return np.column_stack([x, y])

    def get_normals(verts):
        edges = np.roll(verts, -1, axis=0) - verts
        normals = np.column_stack([edges[:, 1], -edges[:, 0]])
        norms = np.linalg.norm(normals, axis=1)
        norms = np.where(norms < 1e-12, 1.0, norms)
        normals = normals / norms[:, np.newaxis]
        return normals

    def penetration_depth(v1, v2):
        normals1 = get_normals(v1)
        normals2 = get_normals(v2)
        normals = np.vstack([normals1, normals2])
        p1 = v1 @ normals.T
        p2 = v2 @ normals.T
        min1 = p1.min(axis=0)
        max1 = p1.max(axis=0)
        min2 = p2.min(axis=0)
        max2 = p2.max(axis=0)
        overlaps = np.minimum(max1, max2) - np.maximum(min1, min2)
        if np.min(overlaps) <= 1e-9:
            return 0.0
        return np.min(overlaps)

    def inside_violation(inner_verts, outer_verts):
        outer_normals = get_normals(outer_verts)
        p_inner = inner_verts @ outer_normals.T
        p_outer_max = (outer_verts @ outer_normals.T).max(axis=0)
        p_inner_max = p_inner.max(axis=0)
        viol = p_inner_max - p_outer_max
        return max(0.0, np.max(viol))

    def objective(params, alpha_rad, theta_rad):
        if time.time() > deadline:
            return 1e10

        positions = params[:2 * N].reshape(N, 2)
        s = params[2 * N]

        if s < 2.0:
            return 1e6

        outer_verts = hex_vertices(0, 0, s, theta_rad)

        total_penalty = 0.0
        inner_verts_list = []

        for i in range(N):
            x, y = positions[i]
            verts = hex_vertices(x, y, 1.0, alpha_rad)
            inner_verts_list.append(verts)
            viol = inside_violation(verts, outer_verts)
            total_penalty += viol * 1000.0

        for i in range(N):
            for j in range(i + 1, N):
                pd = penetration_depth(inner_verts_list[i], inner_verts_list[j])
                total_penalty += pd * 1000.0

        return s + total_penalty

    def make_honeycomb_init():
        """
        3x4 honeycomb grid of flat-top hexagons (rotated 30 deg).
        This configuration is valid (non-overlapping) and compact.
        """
        pos = []
        # Flat-top hexagons (angle 30 deg) have vertical edges on left/right.
        # Horizontal spacing between centers in same row: sqrt(3)
        # Vertical spacing between rows: 1.5
        # Horizontal shift for odd rows: sqrt(3)/2
        
        dx = SQRT3
        dy = 1.5
        shift = SQRT3 / 2
        
        for row in range(3):
            for col in range(4):
                x = col * dx
                if row % 2 == 1:
                    x += shift
                y = row * dy
                pos.append([x, y])
        
        # Center the configuration
        pos = np.array(pos)
        center = pos.mean(axis=0)
        pos = pos - center
        return pos

    inits = [make_honeycomb_init()]
    
    # We will fix the rotation of inner hexagons to 30 degrees (flat-top) 
    # and outer to 0 or 30. 
    # Actually, let's let the optimizer adjust positions, but fix rotations to 
    # simplify the landscape. The honeycomb works best with aligned orientations.
    
    best_s = np.inf
    best_params = None
    best_alpha = np.radians(30)
    best_theta = np.radians(30)

    # Try a few rotation combinations for the outer hexagon
    # Inner hexagons are fixed at 30 deg (flat-top) for the honeycomb init
    thetas = [np.radians(0), np.radians(30), np.radians(15)]
    
    for theta in thetas:
        if time.time() > deadline - 10:
            break
            
        init_pos = inits[0]
        # Estimate initial s: bounding box of the honeycomb
        # Width: 3 * sqrt(3) + sqrt(3)/2 + 2 * 1 (approx)
        # Height: 2 * 1.5 + 2 * 1 (approx)
        # A safe initial s is 4.5
        init_s = 4.5
        x0 = np.concatenate([init_pos.flatten(), [init_s]])

        try:
            # Nelder-Mead is robust for this non-smooth penalty
            res = minimize(
                objective,
                x0,
                args=(best_alpha, theta),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
            )

            pos = res.x[:2 * N].reshape(N, 2)
            s = res.x[2 * N]

            outer_verts = hex_vertices(0, 0, s, theta)
            valid = True
            inner_verts_list = []
            for i in range(N):
                verts = hex_vertices(pos[i, 0], pos[i, 1], 1.0, best_alpha)
                inner_verts_list.append(verts)
                if inside_violation(verts, outer_verts) > 1e-4:
                    valid = False
                    break

            if valid:
                for i in range(N):
                    for j in range(i + 1, N):
                        if penetration_depth(inner_verts_list[i], inner_verts_list[j]) > 1e-4:
                            valid = False
                            break
                    if not valid:
                        break

            if valid and s < best_s:
                best_s = s
                best_params = res.x.copy()
                best_theta = theta

        except Exception:
            pass

    if best_params is None:
        # Fallback: use the initial honeycomb with a conservative s
        best_s = 5.0
        best_params = np.concatenate([inits[0].flatten(), [best_s]])
        best_alpha = np.radians(30)
        best_theta = np.radians(30)

    final_pos = best_params[:2 * N].reshape(N, 2)
    final_s = best_params[2 * N]

    inner = [[final_pos[i, 0], final_pos[i, 1], np.degrees(best_alpha)] for i in range(N)]
    outer_center = (0.0, 0.0)
    outer_side = final_s
    outer_angle_deg = np.degrees(best_theta)

    return inner, outer_center, outer_side, outer_angle_deg
