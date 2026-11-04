"""
Packing 12 unit regular hexagons inside a regular hexagon, minimizing outer side length.
Strategy: Multi-start L-BFGS-B optimization with smooth SDF penalties and independent rotations.
"""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 12
_START_TIME = time.time()
DEADLINE = 145.0  # 5s safety margin

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return DEADLINE - _elapsed()

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        x = cx + side * math.cos(theta)
        y = cy + side * math.sin(theta)
        verts.append((x, y))
    return verts

def _project_on_axis(verts, axis):
    """Project vertices onto axis, return (min, max)."""
    dots = [v[0]*axis[0] + v[1]*axis[1] for v in verts]
    return min(dots), max(dots)

def _get_axes(verts):
    """Get edge normals for SAT."""
    axes = []
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-12:
            continue
        nx, ny = -dy / length, dx / length
        axes.append((nx, ny))
    return axes

def _sat_overlap(verts1, verts2, tol=1e-9):
    """
    Separating Axis Theorem: check if two convex polygons overlap.
    Returns True if they overlap (i.e., no separating axis found).
    """
    axes = _get_axes(verts1) + _get_axes(verts2)
    for axis in axes:
        min1, max1 = _project_on_axis(verts1, axis)
        min2, max2 = _project_on_axis(verts2, axis)
        if max1 < min2 - tol or max2 < min1 - tol:
            return False
    return True

def _outer_hex_side(inner_hexes, outer_center, outer_angle):
    """
    Compute the minimum outer side length such that all inner hexagons fit inside.
    inner_hexes: list of (cx, cy, angle_deg)
    outer_center: (cx, cy)
    outer_angle: rotation of outer hexagon in degrees
    """
    cx, cy = outer_center
    angle_rad = math.radians(outer_angle)
    
    # The outer hexagon has vertices at outer_angle + 60k.
    # The inward normals (pointing to center) are at outer_angle + 90 + 60k.
    # A point (px, py) is inside if dot((px-cx, py-cy), normal) <= apothem * side
    # So side >= dot(...) / apothem
    
    apothem = math.sqrt(3) / 2.0
    max_dist = 0.0
    
    for (ix, iy, iang) in inner_hexes:
        verts = _hex_vertices(ix, iy, 1.0, iang)
        for (px, py) in verts:
            # Check against all 6 inward normals of the outer hexagon
            for k in range(6):
                theta = angle_rad + math.pi/2.0 + k * math.pi / 3.0
                nx, ny = math.cos(theta), math.sin(theta)
                d = (px - cx) * nx + (py - cy) * ny
                if d > max_dist:
                    max_dist = d
    
    return max_dist / apothem

def _sdf_point_to_hex(px, py, cx, cy, angle_deg):
    """
    Approximate signed distance from point (px,py) to regular hexagon centered at (cx,cy) with side 1.
    Returns positive if outside, 0 on boundary, negative inside (approx).
    For smooth optimization, we return the maximum penetration depth.
    """
    angle_rad = math.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    # Rotate into hexagon's local frame
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    lx = dx * cos_a + dy * sin_a
    ly = -dx * sin_a + dy * cos_a
    
    apothem = math.sqrt(3) / 2.0
    
    # Hexagon with vertices at 0, 60, ... has edge normals at 30, 90, ...
    # In local coords, the constraints are:
    # |lx| <= 1 (actually apothem for flat-top? No, side=1 means apothem=sqrt(3)/2)
    # The 6 half-planes:
    # 1. lx <= 1 (vertex at 0) -> Normal (1, 0) is not an edge normal. Edge normal is at 30 deg.
    # Let's stick to the normal formulation.
    # Normals at 30, 90, 150, 210, 270, 330 degrees.
    max_val = 0.0
    for k in range(6):
        theta = math.pi/6.0 + k * math.pi / 3.0
        nx = math.cos(theta)
        ny = math.sin(theta)
        val = lx * nx + ly * ny - apothem
        if val > max_val:
            max_val = val
    
    return max_val

def _sdf_hex_hex(cx1, cy1, a1, cx2, cy2, a2):
    """
    Smooth overlap penalty between two hexagons.
    We sample the vertices of hex1 and check their SDF to hex2, and vice versa.
    This is a convex approximation of the overlap.
    """
    verts1 = _hex_vertices(cx1, cy1, 1.0, a1)
    total = 0.0
    for (vx, vy) in verts1:
        d = _sdf_point_to_hex(vx, vy, cx2, cy2, a2)
        if d > 0:
            total += d * d
            
    verts2 = _hex_vertices(cx2, cy2, 1.0, a2)
    for (vx, vy) in verts2:
        d = _sdf_point_to_hex(vx, vy, cx1, cy1, a1)
        if d > 0:
            total += d * d
            
    return total

def _objective(params, outer_center, outer_angle):
    """
    Objective function for optimization.
    params: [x0, y0, a0, x1, y1, a1, ...]
    Returns: outer_side + penalty_weight * sum(overlap_penalties)
    """
    inner = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i+1]
        a = params[3*i+2]
        inner.append((x, y, a))
    
    side = _outer_hex_side(inner, outer_center, outer_angle)
    
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d = _sdf_hex_hex(inner[i][0], inner[i][1], inner[i][2],
                            inner[j][0], inner[j][1], inner[j][2])
            penalty += d
    
    # Penalty weight needs to be high enough to prevent overlaps, but not so high it ruins the gradient
    return side + 50.0 * penalty

def _is_valid(inner, outer_center, outer_angle, tol=1e-9):
    """Check if configuration is valid: no overlaps, all inside outer hexagon."""
    # Check overlaps
    polys = [_hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2]) for i in range(N)]
    for i in range(N):
        for j in range(i+1, N):
            if _sat_overlap(polys[i], polys[j], tol):
                return False
    return True

def _generate_initial_configs():
    """Generate diverse initial configurations."""
    configs = []
    
    # 1. Ring
    ring = []
    for i in range(12):
        angle = i * 30.0
        r = 2.8
        x = r * math.cos(math.radians(angle))
        y = r * math.sin(math.radians(angle))
        ring.append([x, y, 0.0])
    configs.append(ring)
    
    # 2. Lattice (3x4)
    lattice = []
    for r in range(3):
        for c in range(4):
            x = c * 2.1 + (1.05 if r % 2 else 0)
            y = r * 1.82
            lattice.append([x - 3.0, y - 1.82, 0.0])
    configs.append(lattice)
    
    # 3. Flower (1 center + 6 + 5)
    flower = [[0.0, 0.0, 0.0]]
    for i in range(6):
        angle = i * 60.0
        r = math.sqrt(3)
        x = r * math.cos(math.radians(angle))
        y = r * math.sin(math.radians(angle))
        flower.append([x, y, 0.0])
    for i in range(5):
        angle = i * 72.0 + 36.0
        r = 2.2 * math.sqrt(3)
        x = r * math.cos(math.radians(angle))
        y = r * math.sin(math.radians(angle))
        flower.append([x, y, 0.0])
    configs.append(flower)
    
    # 4. Random
    np.random.seed(42)
    rand = []
    for i in range(12):
        x = np.random.uniform(-3, 3)
        y = np.random.uniform(-3, 3)
        a = np.random.uniform(0, 60)
        rand.append([x, y, a])
    configs.append(rand)
    
    # 5. Two rings
    two_rings = []
    for i in range(6):
        angle = i * 60.0
        r = 1.5
        two_rings.append([r * math.cos(math.radians(angle)), r * math.sin(math.radians(angle)), 0.0])
    for i in range(6):
        angle = i * 60.0 + 30.0
        r = 3.5
        two_rings.append([r * math.cos(math.radians(angle)), r * math.sin(math.radians(angle)), 0.0])
    configs.append(two_rings)
    
    return configs

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_side = float('inf')
    best_params = None
    best_outer_angle = 0.0
    
    outer_center = (0.0, 0.0)
    
    # Try a few outer angles
    outer_angles = [0.0, 30.0]
    
    configs = _generate_initial_configs()
    
    for oa in outer_angles:
        if _time_left() < 20.0:
            break
            
        for config in configs:
            if _time_left() < 10.0:
                break
                
            # Flatten config to params
            params = np.array([v for item in config for v in item])
            
            def obj(p):
                return _objective(p, outer_center, oa)
            
            try:
                # Use L-BFGS-B
                result = minimize(
                    obj, params, method='L-BFGS-B',
                    options={'maxiter': 200, 'maxfun': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
                )
            except Exception:
                continue
                
            if _time_left() < 5.0:
                continue
            
            # Extract inner hexagons
            inner = []
            for i in range(N):
                x = result.x[3*i]
                y = result.x[3*i+1]
                a = result.x[3*i+2]
                inner.append((x, y, a))
                
            # Check validity
            if _is_valid(inner, outer_center, oa):
                side = _outer_hex_side(inner, outer_center, oa)
                
                if side < best_side:
                    best_side = side
                    best_params = result.x.copy()
                    best_outer_angle = oa
                    
                # Local polish: try to reduce side length further by moving hexagons
                # This is a simple hill-climbing step
                if _time_left() > 15.0:
                    current_inner = inner
                    current_side = side
                    step = 0.05
                    for _ in range(50):
                        if _time_left() < 10.0:
                            break
                        improved = False
                        for i in range(N):
                            for dim in range(2): # x, y
                                for direction in [-1, 1]:
                                    trial = [list(h) for h in current_inner]
                                    trial[i][dim] += direction * step
                                    
                                    if _is_valid(trial, outer_center, oa):
                                        trial_side = _outer_hex_side(trial, outer_center, oa)
                                        if trial_side < current_side - 1e-8:
                                            current_inner = trial
                                            current_side = trial_side
                                            improved = True
                        if not improved:
                            step *= 0.8
                            if step < 0.001:
                                break
                            
                    if current_side < best_side:
                        best_side = current_side
                        # Update best_params from current_inner
                        best_params = np.array([v for h in current_inner for v in h])
                        best_outer_angle = oa

    # Final verification and return
    if best_params is not None:
        final_inner = []
        for i in range(N):
            x = best_params[3*i]
            y = best_params[3*i+1]
            a = best_params[3*i+2]
            final_inner.append([x, y, a])
            
        # One last check for best outer angle
        final_best_side = float('inf')
        final_best_oa = best_outer_angle
        for oa in [0.0, 30.0, 15.0, 45.0]:
            side = _outer_hex_side(final_inner, (0.0, 0.0), oa)
            if side < final_best_side:
                final_best_side = side
                final_best_oa = oa
        
        # Verify no overlaps
        if _is_valid(final_inner, (0.0, 0.0), final_best_oa):
            return final_inner, (0.0, 0.0), final_best_side, final_best_oa
    
    # Fallback: safe default (large hexagon)
    inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    return inner, (0.0, 0.0), 40.0, 0.0

if __name__ == "__main__":
    inner, center, side, angle = construct()
    print(f"Best side: {side}")
