import numpy as np

N = 12

def construct():
    # Define centers
    sqrt3 = np.sqrt(3)
    centers = [
        [0.0, 0.0],
        [sqrt3, 0.0], [-sqrt3, 0.0],
        [sqrt3/2, 1.5], [-sqrt3/2, 1.5],
        [sqrt3/2, -1.5], [-sqrt3/2, -1.5],
        [3*sqrt3/2, 1.5],
        [0.0, 3.0],
        [-3*sqrt3/2, 1.5],
        [-3*sqrt3/2, -1.5],
        [0.0, -3.0]
    ]
    
    # Initial angles for inner hexagons
    # Rotate each to point a vertex away from origin
    inner_angles = []
    for c in centers:
        if np.linalg.norm(c) < 1e-6:
            inner_angles.append(0.0)
        else:
            # Angle of center
            theta = np.arctan2(c[1], c[0])
            # We want a vertex to point in direction theta
            # Vertices are at angles alpha + k*60
            # So alpha + k*60 = theta
            # alpha = theta - k*60
            # Choose k such that alpha is in [0, 60)
            k = int(np.floor((theta) / (np.pi/3)))
            alpha = theta - k * (np.pi/3)
            # Convert to degrees
            inner_angles.append(np.degrees(alpha))

    # Optimize outer angle
    def calc_S(outer_angle_deg, inner_angles_deg):
        outer_angle_rad = np.radians(outer_angle_deg)
        max_s = 0.0
        for i, c in enumerate(centers):
            a_rad = np.radians(inner_angles_deg[i])
            # Vertices of this hexagon
            for k in range(6):
                v_angle = a_rad + k * np.pi/3
                v_rel = [np.cos(v_angle), np.sin(v_angle)]
                v_abs = [c[0] + v_rel[0], c[1] + v_rel[1]]
                # Rotate by -outer_angle
                x = v_abs[0] * np.cos(-outer_angle_rad) - v_abs[1] * np.sin(-outer_angle_rad)
                y = v_abs[0] * np.sin(-outer_angle_rad) + v_abs[1] * np.cos(-outer_angle_rad)
                # Hex norm
                s1 = abs(x)
                s2 = abs(x + y/sqrt3)
                s3 = abs(x - y/sqrt3)
                s_req = max(s1, s2, s3)
                if s_req > max_s:
                    max_s = s_req
        return max_s

    # Grid search for outer angle
    best_s = float('inf')
    best_outer = 0.0
    for oa in np.linspace(0, 60, 121):
        s = calc_S(oa, inner_angles)
        if s < best_s:
            best_s = s
            best_outer = oa

    # Refine inner angles?
    # For each inner hex, try a few angles
    # This is O(12 * 60 * 60) which is fine.
    # Let's do a simple local search for inner angles
    for _ in range(10): # iterations
        improved = False
        for i in range(N):
            best_a = inner_angles[i]
            best_s_i = best_s
            # Try 12 angles around current
            for da in np.linspace(-30, 30, 12):
                new_angles = inner_angles.copy()
                new_angles[i] = (best_a + da) % 60
                s = calc_S(best_outer, new_angles)
                if s < best_s_i:
                    best_s_i = s
                    best_a = new_angles[i]
            if best_a != inner_angles[i]:
                inner_angles[i] = best_a
                best_s = best_s_i
                improved = True
        if not improved:
            break

    # Prepare output
    inner = []
    for i in range(N):
        inner.append([centers[i][0], centers[i][1], inner_angles[i]])
    
    return inner, (0.0, 0.0), best_s, best_outer

# EVOLVE-BLOCK-END
