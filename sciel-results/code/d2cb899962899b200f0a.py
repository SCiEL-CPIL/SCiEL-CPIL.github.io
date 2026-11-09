import numpy as np

N = 12

def _hex_to_cartesian(q, r):
    x = 1.5 * q
    y = np.sqrt(3) * (r + 0.5 * q)
    return x, y

def _get_hex_vertices(cx, cy, angle_rad, side=1.0):
    angles = np.deg2rad([0, 60, 120, 180, 240, 300]) + angle_rad
    xs = cx + side * np.cos(angles)
    ys = cy + side * np.sin(angles)
    return np.column_stack((xs, ys))

def _min_enclosing_hex_side(vertices, theta_rad):
    cos_t = np.cos(-theta_rad)
    sin_t = np.sin(-theta_rad)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    rotated_vertices = vertices @ R.T
    xs = rotated_vertices[:, 0]
    ys = rotated_vertices[:, 1]
    abs_x = np.abs(xs)
    abs_y = np.abs(ys)
    s1 = abs_x
    s2 = (2.0 / np.sqrt(3)) * abs_y
    s3 = abs_x + (1.0 / np.sqrt(3)) * abs_y
    s_per_vertex = np.maximum(s1, np.maximum(s2, s3))
    return np.max(s_per_vertex)

# EVOLVE-BLOCK-START
def construct():
    # Generate 12 centers in a hexagonal lattice
    centers = []
    # 7 central
    centers.append((0, 0))
    for q, r in [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]:
        centers.append((q, r))
    # 5 more
    for q, r in [(2,0), (2,-1), (1,-2), (0,-2), (-1,-2)]:
        centers.append((q, r))
        
    cart_centers = [_hex_to_cartesian(q, r) for q, r in centers]
    
    all_vertices = []
    for cx, cy in cart_centers:
        verts = _get_hex_vertices(cx, cy, 0.0)
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)
    
    best_S = float('inf')
    best_theta = 0.0
    thetas = np.linspace(0, np.pi/6, 61)
    
    for theta in thetas:
        S = _min_enclosing_hex_side(all_vertices, theta)
        if S < best_S:
            best_S = S
            best_theta = theta
            
    inner = [[cx, cy, 0.0] for cx, cy in cart_centers]
    outer_center = (0.0, 0.0)
    outer_side = best_S
    outer_angle = np.rad2deg(best_theta)
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
