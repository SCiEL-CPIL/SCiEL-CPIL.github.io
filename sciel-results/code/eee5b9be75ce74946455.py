import numpy as np
import time
import math
from scipy.optimize import linprog

N = 11
EPS = 1e-9

def hex_vertices(cx, cy, side, angle_deg=0.0):
    """Return vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    angle = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle + i * math.pi / 3
        x = cx + side * math.cos(theta)
        y = cy + side * math.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def min_enclosing_hex_side(points):
    """
    Compute the minimum side length of a regular hexagon that encloses the given points.
    points: Nx2 array of (x,y) coordinates.
    Returns (side, cx, cy, angle_deg)
    """
    # The problem is to minimize S subject to:
    # For all points p_i and all normals n_k(theta):
    # (p_i - c) dot n_k(theta) <= S * sqrt(3)/2
    # (p_i - c) dot -n_k(theta) <= S * sqrt(3)/2
    
    # This is non-linear due to theta.
    # We can iterate over theta.
    best_side = float('inf')
    best_c = (0,0)
    best_angle = 0.0
    
    # The optimal theta is likely aligned with the points or 0, 30, 60.
    # Let's sample theta from 0 to 60 degrees.
    for theta_deg in np.linspace(0, 60, 61):
        theta = math.radians(theta_deg)
        # Normals of the outer hexagon
        normals = []
        for k in range(6):
            phi = theta + k * math.pi / 3
            normals.append([math.cos(phi), math.sin(phi)])
        normals = np.array(normals)
        
        # Variables: cx, cy, S
        # x = [cx, cy, S]
        # Minimize S -> c = [0, 0, 1]
        c_obj = [0, 0, 1]
        
        A_ub = []
        b_ub = []
        
        # Constraints:
        # (p_i - c) dot n_k <= S * sqrt(3)/2
        # p_i dot n_k - c dot n_k <= S * sqrt(3)/2
        # -cx * n_kx - cy * n_ky - S * sqrt(3)/2 <= -p_i dot n_k  (Wait, signs)
        
        # Let's write:
        # cx * n_kx + cy * n_ky + S * (sqrt(3)/2) >= p_i dot n_k
        # cx * n_kx + cy * n_ky - S * (sqrt(3)/2) <= p_i dot n_k  (This is not right, c is variable)
        
        # Correct form:
        # (px - cx) * nx + (py - cy) * ny <= S * sqrt(3)/2
        # px*nx + py*ny - cx*nx - cy*ny <= S*sqrt(3)/2
        # -cx*nx - cy*ny - S*sqrt(3)/2 <= -(px*nx + py*ny)
        
        # And for the opposite side:
        # (px - cx) * nx + (py - cy) * ny >= -S * sqrt(3)/2
        # cx*nx + cy*ny - S*sqrt(3)/2 <= px*nx + py*ny
        
        for i in range(len(points)):
            px, py = points[i]
            for k in range(6):
                nx, ny = normals[k]
                proj = px * nx + py * ny
                sqrt3_2 = math.sqrt(3) / 2
                
                # Inequality 1: -nx*cx - ny*cy - sqrt3_2*S <= -proj
                A_ub.append([-nx, -ny, -sqrt3_2])
                b_ub.append(-proj)
                
                # Inequality 2: nx*cx + ny*cy - sqrt3_2*S <= proj
                A_ub.append([nx, ny, -sqrt3_2])
                b_ub.append(proj)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        # S must be positive
        bounds = [(None, None), (None, None), (0, None)]
        
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.success:
            S = res.x[2]
            if S < best_side:
                best_side = S
                best_c = (res.x[0], res.x[1])
                best_angle = theta_deg
                
    return best_side, best_c[0], best_c[1], best_angle

def generate_lattice_configs():
    """Generate candidate configurations based on hexagonal lattice subgraphs."""
    configs = []
    sqrt3 = math.sqrt(3)
    
    # Helper to convert axial (q, r) to Cartesian (x, y)
    # Pointy-top or Flat-top?
    # Standard flat-top:
    # x = 1.5 * q
    # y = sqrt(3) * (r + 0.5 * q)
    
    def axial_to_cart(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + 0.5 * q)
        return (x, y)
    
    # 1. H2 cluster (7 hexagons) + 4 extra
    # H2 axial coords:
    # (0,0)
    # (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    base = [(0,0), (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]
    
    # Candidates for the 4 extra hexagons
    # They should be adjacent to the base cluster to keep it compact.
    # Possible neighbors of the base cluster:
    # (2,0), (2,-1), (2,-2), (1,-2), (0,-2), (-1,-1), (-2,0), (-2,1), (-2,2), (-1,2), (0,2), (1,1)
    # Let's pick 4 from these.
    # To be compact, we should fill the "corners" of the hexagonal shape.
    # The corners of H2 are at (2,0) type positions?
    # Let's just try a few sets.
    
    extra_candidates = [
        (2, 0), (2, -1), (1, -2), (0, -2), (-1, -1), (-2, 0), (-2, 1), (-1, 2), (0, 2), (1, 1)
    ]
    
    # Try adding 4 specific ones
    sets_of_4 = [
        [(2, 0), (2, -1), (1, -2), (0, -2)], # One side
        [(2, 0), (-2, 0), (0, 2), (0, -2)], # Opposite corners
        [(2, 0), (2, -1), (-2, 0), (-2, 1)], # Two opposite sides
        [(1, 1), (2, 0), (2, -1), (1, -2)], # Corner
        [(0, 2), (1, 1), (2, 0), (2, -1)], # Another corner
        [(-1, 2), (0, 2), (1, 1), (2, 0)], # Another
        [(-2, 1), (-1, 2), (0, 2), (1, 1)], # Another
        [(-2, 0), (-2, 1), (-1, 2), (0, 2)], # Another
        [(-2, 0), (-1, -1), (0, -2), (1, -2)], # Bottom
        [(2, 0), (1, -2), (-1, -1), (-2, 0)], # Alternating
    ]
    
    for extras in sets_of_4:
        coords = base + extras
        # Check for duplicates
        if len(set(coords)) < 11:
            continue
        centers = [axial_to_cart(q, r) for q, r in coords]
        configs.append(centers)
        
    # 2. Linear strip (11 in a row)
    centers = []
    for i in range(11):
        centers.append((1.5 * i, 0))
    configs.append(centers)
    
    # 3. Two rows (6 and 5)
    # Row 1: 6 hexagons
    # Row 2: 5 hexagons, offset
    centers = []
    for i in range(6):
        centers.append((1.5 * i, 0))
    for i in range(5):
        centers.append((1.5 * (i + 0.5), sqrt3))
    configs.append(centers)
    
    # 4. Three rows (4, 4, 3)
    centers = []
    for i in range(4):
        centers.append((1.5 * i, 0))
    for i in range(4):
        centers.append((1.5 * (i + 0.5), sqrt3))
    for i in range(3):
        centers.append((1.5 * (i + 1.0), 2 * sqrt3))
    configs.append(centers)
    
    return configs

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_side = float('inf')
    best_inner = None
    best_center = (0, 0)
    best_angle = 0.0
    
    configs = generate_lattice_configs()
    
    for centers in configs:
        if time.time() > deadline:
            break
            
        # Check if centers are valid (no overlaps)
        # For lattice configs, they should be valid.
        # But let's verify with a simple check.
        valid = True
        for i in range(N):
            for j in range(i + 1, N):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < math.sqrt(3) - 1e-6: # Should be >= sqrt(3)
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
            
        points = np.array(centers)
        side, cx, cy, angle = min_enclosing_hex_side(points)
        
        if side < best_side:
            best_side = side
            best_inner = centers
            best_center = (cx, cy)
            best_angle = angle
            
    # If no valid config found (should not happen), fallback
    if best_inner is None:
        # Fallback: linear row
        centers = [(1.5 * i, 0) for i in range(N)]
        points = np.array(centers)
        best_side, cx, cy, best_angle = min_enclosing_hex_side(points)
        best_inner = centers
        best_center = (cx, cy)
        
    inner = [[float(cx), float(cy), 0.0] for cx, cy in best_inner]
    return inner, (float(best_center[0]), float(best_center[1])), float(best_side), float(best_angle)
