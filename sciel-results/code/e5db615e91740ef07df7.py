"""Pack 12 unit regular hexagons inside a regular hexagon, minimizing outer side length."""
import numpy as np
import time
from scipy.optimize import minimize

N = 12
DEADLINE = time.time() + 110.0  # safety margin before 120s limit

def hexagon_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx, cy) with given side and rotation."""
    angle = np.radians(angle_deg)
    vertices = []
    for i in range(6):
        theta = angle + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        vertices.append((x, y))
    return np.array(vertices)

def point_in_polygon(point, polygon):
    """Ray casting to check if point is inside polygon."""
    x, y = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def hexagon_distance_to_hexagon(v1, v2):
    """Check if two hexagons (given as vertex arrays) are disjoint (distance > 0).
    Returns minimum distance between edges (approximate via vertex checks)."""
    # For now, use a simple check: if any vertex of one is inside the other, they overlap
    for p in v1:
        if point_in_polygon(p, v2):
            return -1.0  # overlap
    for p in v2:
        if point_in_polygon(p, v1):
            return -1.0  # overlap
    # Check edge-edge intersection or minimum distance
    # Simplified: check if any edge of one intersects any edge of the other
    def segments_intersect(p1, p2, p3, p4):
        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)
    
    for i in range(6):
        for j in range(6):
            p1 = v1[i]
            p2 = v1[(i+1) % 6]
            p3 = v2[j]
            p4 = v2[(j+1) % 6]
            if segments_intersect(p1, p2, p3, p4):
                return -1.0  # overlap
    # No overlap; estimate minimum distance as min vertex-vertex distance (crude)
    min_dist = float('inf')
    for p in v1:
        for q in v2:
            d = np.linalg.norm(p - q)
            min_dist = min(min_dist, d)
    return min_dist

def objective(params):
    """Objective: outer hexagon side length, penalized if inner hexagons overlap or go outside."""
    # params: for each inner hex: (x, y, angle_deg), then outer: (cx, cy, side, angle_deg)
    n_inner = N
    inner_params = params[:3 * n_inner]
    outer_cx = params[3 * n_inner]
    outer_cy = params[3 * n_inner + 1]
    outer_side = params[3 * n_inner + 2]
    outer_angle = params[3 * n_inner + 3]
    
    # Get outer hexagon vertices
    outer_verts = hexagon_vertices(outer_cx, outer_cy, outer_side, outer_angle)
    
    penalty = 0.0
    inner_verts_list = []
    
    for i in range(n_inner):
        ix = inner_params[3 * i]
        iy = inner_params[3 * i + 1]
        iangle = inner_params[3 * i + 2]
        iv = hexagon_vertices(ix, iy, 1.0, iangle)
        inner_verts_list.append(iv)
        
        # Check all vertices of inner hex are inside outer hex
        for p in iv:
            if not point_in_polygon(p, outer_verts):
                # Penalize by distance outside (approximate)
                # Find minimum distance to outer hex edges
                min_d = float('inf')
                for j in range(6):
                    e1 = outer_verts[j]
                    e2 = outer_verts[(j+1) % 6]
                    # Point-to-segment distance
                    d = np.linalg.norm(e2 - e1)
                    if d > 1e-10:
                        t = np.dot(p - e1, e2 - e1) / (d * d)
                        t = max(0.0, min(1.0, t))
                        proj = e1 + t * (e2 - e1)
                        dist = np.linalg.norm(p - proj)
                        min_d = min(min_d, dist)
                    else:
                        dist = np.linalg.norm(p - e1)
                        min_d = min(min_d, dist)
                penalty += max(0.0, min_d) ** 2 * 100
    
    # Check inner-inner overlaps
    for i in range(n_inner):
        for j in range(i + 1, n_inner):
            d = hexagon_distance_to_hexagon(inner_verts_list[i], inner_verts_list[j])
            if d < 0:  # overlap
                penalty += 10.0
    
    return outer_side + penalty

def construct():
    global DEADLINE
    DEADLINE = time.time() + 110.0
    
    # Initial guess: honeycomb-like arrangement
    # Place 12 hexagons in a compact cluster
    # Honeycomb: rows of 4, 3, 4 or similar
    # Let's try a 3x4 grid with hexagonal offsets
    
    # Initial positions
    init_params = []
    # Row 1: 4 hexagons
    for k in range(4):
        x = k * 1.8
        y = 0.0
        init_params.extend([x, y, 0.0])
    # Row 2: 4 hexagons, offset
    for k in range(4):
        x = k * 1.8 + 0.9
        y = 1.5
        init_params.extend([x, y, 0.0])
    # Row 3: 4 hexagons
    for k in range(4):
        x = k * 1.8
        y = 3.0
        init_params.extend([x, y, 0.0])
    
    # Outer hexagon initial guess
    # Center of cluster
    cx = 2.7
    cy = 1.5
    side = 8.0
    angle = 0.0
    init_params.extend([cx, cy, side, angle])
    
    init_params = np.array(init_params)
    
    # Optimize
    best_result = None
    best_obj = float('inf')
    
    # Try multiple random restarts
    np.random.seed(42)
    n_restarts = 5
    
    for r in range(n_restarts):
        if time.time() > DEADLINE:
            break
        
        if r == 0:
            x0 = init_params.copy()
        else:
            # Add random perturbation
            x0 = init_params.copy()
            x0[:3*N] += np.random.normal(0, 1.0, size=3*N)
            x0[3*N] += np.random.normal(0, 0.5)
            x0[3*N+1] += np.random.normal(0, 0.5)
            x0[3*N+2] += np.random.normal(0, 1.0)
            x0[3*N+3] += np.random.normal(0, 30)
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < best_obj:
                best_obj = res.fun
                best_result = res.x
        except Exception:
            pass
        
        if time.time() > DEADLINE:
            break
    
    if best_result is None:
        # Fallback: return initial
        best_result = init_params
    
    # Extract
    inner = []
    for i in range(N):
        x = best_result[3 * i]
        y = best_result[3 * i + 1]
        a = best_result[3 * i + 2]
        inner.append([x, y, a])
    
    outer_cx = best_result[3 * N]
    outer_cy = best_result[3 * N + 1]
    outer_side = best_result[3 * N + 2]
    outer_angle = best_result[3 * N + 3]
    
    # Center everything relative to outer hex center
    inner_shifted = [[x - outer_cx, y - outer_cy, a] for x, y, a in inner]
    
    return inner_shifted, (0.0, 0.0), outer_side, outer_angle

if __name__ == "__main__":
    inner, center, side, angle = construct()
    print(f"Side length: {side:.6f}")
    print(f"Angle: {angle:.4f}")
    print(f"Center: {center}")
    for x, y, a in inner:
        print(f"  ({x:.4f}, {y:.4f}, {a:.2f}°)")
