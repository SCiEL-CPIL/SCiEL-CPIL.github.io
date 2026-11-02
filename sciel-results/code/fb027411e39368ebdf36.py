import math
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    inradius = math.sqrt(3) / 2.0
    eps = 1e-9
    d = math.sqrt(3) + eps  # lattice spacing for same-oriented hexagons
    
    # Triangular lattice vectors
    v1x, v1y = d, 0.0
    v2x, v2y = d / 2.0, d * math.sqrt(3) / 2.0
    
    # 3-4-3 hexagonal cluster in axial coordinates
    positions_axial = [
        (-1, 0), (0, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1), (2, 1),
        (-1, 2), (0, 2), (1, 2),
    ]
    
    centers = []
    for q, r in positions_axial:
        x = q * v1x + r * v2x
        y = q * v1y + r * v2y
        centers.append((x, y))
    
    # Center the cluster
    cx = sum(c[0] for c in centers) / len(centers)
    cy = sum(c[1] for c in centers) / len(centers)
    centers = [(x - cx, y - cy) for x, y in centers]
    
    inner = [[x, y, 0.0] for x, y in centers]
    
    # Collect all inner hexagon vertices
    all_vertices = []
    for x, y, ang in inner:
        ang_rad = math.radians(ang)
        for k in range(6):
            theta_v = ang_rad + k * math.pi / 3
            vx = x + math.cos(theta_v)
            vy = y + math.sin(theta_v)
            all_vertices.append((vx, vy))
    
    def required_R(theta_outer):
        max_r = 0.0
        for vx, vy in all_vertices:
            r = math.sqrt(vx * vx + vy * vy)
            if r < 1e-12:
                continue
            alpha = math.atan2(vy, vx)
            min_dist = float('inf')
            for k in range(6):
                diff = (alpha - theta_outer - k * math.pi / 3) % (2 * math.pi)
                if diff > math.pi:
                    diff = 2 * math.pi - diff
                min_dist = min(min_dist, diff)
            cos_min = math.cos(min_dist)
            if cos_min < 1e-12:
                return float('inf')
            val = r / cos_min
            if val > max_r:
                max_r = val
        return max_r
    
    # Coarse scan
    best_R = float('inf')
    best_theta = 0.0
    n_steps = 180
    for i in range(n_steps + 1):
        theta = i * (math.pi / 3) / n_steps
        R = required_R(theta)
        if R < best_R:
            best_R = R
            best_theta = theta
    
    # Fine scan if time permits
    if time.time() < deadline - 5.0:
        step = (math.pi / 3) / (n_steps * 20)
        for i in range(-30, 31):
            theta = best_theta + i * step
            R = required_R(theta)
            if R < best_R:
                best_R = R
                best_theta = theta
    
    # Ultra-fine scan
    if time.time() < deadline - 2.0:
        step = (math.pi / 3) / (n_steps * 200)
        for i in range(-15, 16):
            theta = best_theta + i * step
            R = required_R(theta)
            if R < best_R:
                best_R = R
                best_theta = theta
    
    outer_center = (0.0, 0.0)
    outer_side = best_R
    outer_angle_deg = math.degrees(best_theta)
    
    return inner, outer_center, outer_side, outer_angle_deg
# EVOLVE-BLOCK-END
