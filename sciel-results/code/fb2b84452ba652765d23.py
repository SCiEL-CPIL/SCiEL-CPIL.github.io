import numpy as np
from scipy.optimize import minimize

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3)
        outer_hex_data: np.ndarray of shape (3,)
        outer_hex_side_length: float
    """
    sqrt3 = np.sqrt(3)
    
    # Compact packing of 11 unit hexagons
    inner_hex_data = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [1.0, sqrt3, 0.0],
        [-1.0, sqrt3, 0.0],
        [-2.0, 0.0, 0.0],
        [-1.0, -sqrt3, 0.0],
        [1.0, -sqrt3, 0.0],
        [3.0, sqrt3, 0.0],
        [3.0, -sqrt3, 0.0],
        [-3.0, sqrt3, 0.0],
        [-3.0, -sqrt3, 0.0],
    ])
    
    def get_hexagon_vertices(cx, cy, s, a_deg):
        a = np.radians(a_deg)
        angles = a + np.radians(60 * np.arange(6))
        xs = cx + s * np.cos(angles)
        ys = cy + s * np.sin(angles)
        return np.column_stack([xs, ys])
    
    all_vertices = []
    for i in range(11):
        cx, cy, a_deg = inner_hex_data[i]
        verts = get_hexagon_vertices(cx, cy, 1.0, a_deg)
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)
    
    # Find min enclosing regular hexagon by checking containment
    def check_hexagon(hx, hy, s, theta_deg, points):
        theta = np.radians(theta_deg)
        angles = theta + np.radians(60 * np.arange(6))
        vx = hx + s * np.cos(angles)
        vy = hy + s * np.sin(angles)
        
        for i in range(6):
            x1, y1 = vx[i], vy[i]
            x2, y2 = vx[(i+1)%6], vy[(i+1)%6]
            ex, ey = x2 - x1, y2 - y1
            nx, ny = ey, -ex
            nlen = np.sqrt(nx**2 + ny**2)
            if nlen < 1e-12:
                continue
            nx, ny = nx/nlen, ny/nlen
            mx, my = (x1+x2)/2, (y1+y2)/2
            if nx*(hx-mx) + ny*(hy-my) < 0:
                nx, ny = -nx, -ny
            for j in range(len(points)):
                if (points[j,0]-x1)*nx + (points[j,1]-y1)*ny > 1e-9:
                    return False
        return True
    
    centroid = np.mean(all_vertices, axis=0)
    max_dist = np.max(np.linalg.norm(all_vertices - centroid, axis=1))
    s0 = max_dist * 2.0
    
    best_s = np.inf
    best_params = None
    
    for theta0 in np.arange(0, 60, 5):
        hx, hy = centroid
        # Binary search on side length for this orientation
        lo, hi = max_dist, max_dist * 3
        for _ in range(50):
            mid = (lo + hi) / 2
            if check_hexagon(hx, hy, mid, theta0, all_vertices):
                hi = mid
            else:
                lo = mid
        if hi < best_s:
            best_s = hi
            best_params = (hx, hy, theta0, hi)
    
    hx, hy, theta0, s = best_params
    s += 0.005  # small margin
    
    outer_hex_data = np.array([hx, hy, theta0])
    outer_hex_side_length = float(s)
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length
