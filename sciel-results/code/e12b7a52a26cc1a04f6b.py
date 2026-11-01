"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    angle = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def point_in_hexagon(px, py, cx, cy, side, angle_deg=0):
    """Check if point (px,py) is inside hexagon with center (cx,cy), side length side, rotation angle_deg."""
    dx = px - cx
    dy = py - cy
    a = -np.radians(angle_deg)
    rx = dx * np.cos(a) - dy * np.sin(a)
    ry = dx * np.sin(a) + dy * np.cos(a)
    inradius = side * np.sqrt(3) / 2
    return (abs(rx) <= inradius and 
            abs(ry) <= side and
            abs(rx * np.sqrt(3)/2 - ry/2) <= inradius and
            abs(rx * np.sqrt(3)/2 + ry/2) <= inradius)

def min_enclosing_hexagon(points, max_iterations=50):
    """Find minimum side length of regular hexagon enclosing all points."""
    best_side = float('inf')
    best_params = (0, 0, 0, 0)
    
    for angle_deg in [0, 30, 60, 90, 120, 150]:
        a = np.radians(angle_deg)
        normals = []
        for i in range(6):
            theta = a + i * np.pi / 3
            normals.append([np.cos(theta), np.sin(theta)])
        normals = np.array(normals)
        
        c = np.mean(points, axis=0)
        S = 1.0
        
        for _ in range(max_iterations):
            max_proj = 0
            for p in points:
                for n in normals:
                    proj = abs(np.dot(p - c, n))
                    if proj > max_proj:
                        max_proj = proj
            S = max_proj / (np.sqrt(3) / 2)
            
            if S < best_side:
                best_side = S
                best_params = (c[0], c[1], S, angle_deg)
            
            max_val = 0
            best_p = None
            best_n = None
            best_sign = 1
            for p in points:
                for n in normals:
                    proj = np.dot(p - c, n)
                    if abs(proj) > max_val:
                        max_val = abs(proj)
                        best_p = p
                        best_n = n
                        best_sign = 1 if proj > 0 else -1
            
            step = 0.1 * best_sign * np.dot(best_p - c, best_n) / (np.dot(best_n, best_n) + 1e-10)
            c_new = c + step * best_n
            S_new = 0
            for p in points:
                for n in normals:
                    proj = abs(np.dot(p - c_new, n))
                    S_new = max(S_new, proj)
            S_new /= (np.sqrt(3) / 2)
            
            if S_new < S:
                c = c_new
            else:
                step = 0.01 * best_sign * np.dot(best_p - c, best_n) / (np.dot(best_n, best_n) + 1e-10)
                c_new = c + step * best_n
                S_new = 0
                for p in points:
                    for n in normals:
                        proj = abs(np.dot(p - c_new, n))
                        S_new = max(S_new, proj)
                S_new /= (np.sqrt(3) / 2)
                if S_new < S:
                    c = c_new
                else:
                    break
    
    return best_params[0], best_params[1], best_params[2], best_params[3]

def get_all_vertices(inner):
    all_verts = []
    for x, y, angle in inner:
        verts = hex_vertices(x, y, 1.0, angle)
        all_verts.extend(verts.tolist())
    return np.array(all_verts)

def check_valid(inner, center, side, angle_deg):
    all_verts = get_all_vertices(inner)
    for px, py in all_verts:
        if not point_in_hexagon(px, py, center[0], center[1], side, angle_deg):
            return False
    return True

def construct():
    sqrt3 = np.sqrt(3)
    candidates = []
    
    # Arrangement A: 3-4-3-2 rows
    arr_A = []
    row_y = [0, 1.5, 3.0, 4.5]
    row_counts = [3, 4, 3, 2]
    for r, (y, count) in enumerate(zip(row_y, row_counts)):
        offset = 0 if r % 2 == 0 else sqrt3/2
        start_x = -(count - 1) * sqrt3 / 2 + offset
        for i in range(count):
            x = start_x + i * sqrt3
            arr_A.append([x, y, 0.0])
    candidates.append(arr_A)
    
    # Arrangement B: 4-3-4-1 rows
    arr_B = []
    row_y = [0, 1.5, 3.0, 4.5]
    row_counts = [4, 3, 4, 1]
    for r, (y, count) in enumerate(zip(row_y, row_counts)):
        offset = 0 if r % 2 == 0 else sqrt3/2
        start_x = -(count - 1) * sqrt3 / 2 + offset
        for i in range(count):
            x = start_x + i * sqrt3
            arr_B.append([x, y, 0.0])
    candidates.append(arr_B)
    
    # Arrangement C: 2-4-4-2 rows
    arr_C = []
    row_y = [0, 1.5, 3.0, 4.5]
    row_counts = [2, 4, 4, 2]
    for r, (y, count) in enumerate(zip(row_y, row_counts)):
        offset = 0 if r % 2 == 0 else sqrt3/2
        start_x = -(count - 1) * sqrt3 / 2 + offset
        for i in range(count):
            x = start_x + i * sqrt3
            arr_C.append([x, y, 0.0])
    candidates.append(arr_C)
    
    # Arrangement D: 3-3-3-3 rows
    arr_D = []
    row_y = [0, 1.5, 3.0, 4.5]
    row_counts = [3, 3, 3, 3]
    for r, (y, count) in enumerate(zip(row_y, row_counts)):
        offset = 0 if r % 2 == 0 else sqrt3/2
        start_x = -(count - 1) * sqrt3 / 2 + offset
        for i in range(count):
            x = start_x + i * sqrt3
            arr_D.append([x, y, 0.0])
    candidates.append(arr_D)
    
    # Arrangement E: 5-3-4 rows
    arr_E = []
    row_y = [0, 1.5, 3.0]
    row_counts = [5, 3, 4]
    for r, (y, count) in enumerate(zip(row_y, row_counts)):
        offset = 0 if r % 2 == 0 else sqrt3/2
        start_x = -(count - 1) * sqrt3 / 2 + offset
        for i in range(count):
            x = start_x + i * sqrt3
            arr_E.append([x, y, 0.0])
    candidates.append(arr_E)
    
    # Arrangement F: 4-4-4 rows
    arr_F = []
    row_y = [0, 1.5, 3.0]
    row_counts = [4, 4, 4]
    for r, (y, count) in enumerate(zip(row_y, row_counts)):
        offset = 0 if r % 2 == 0 else sqrt3/2
        start_x = -(count - 1) * sqrt3 / 2 + offset
        for i in range(count):
            x = start_x + i * sqrt3
            arr_F.append([x, y, 0.0])
    candidates.append(arr_F)
    
    # Arrangement G: 6-6 rows
    arr_G = []
    row_y = [0, 1.5]
    row_counts = [6, 6]
    for r, (y, count) in enumerate(zip(row_y, row_counts)):
        offset = 0 if r % 2 == 0 else sqrt3/2
        start_x = -(count - 1) * sqrt3 / 2 + offset
        for i in range(count):
            x = start_x + i * sqrt3
            arr_G.append([x, y, 0.0])
    candidates.append(arr_G)
    
    best_result = None
    best_side = float('inf')
    
    for inner in candidates:
        all_verts = get_all_vertices(inner)
        cx, cy, side, angle = min_enclosing_hexagon(all_verts)
        
        if check_valid(inner, (cx, cy), side, angle):
            if side < best_side:
                best_side = side
                best_result = (inner, (cx, cy), side, angle)
    
    if best_result is not None:
        return best_result
    
    inner = candidates[0]
    all_verts = get_all_vertices(inner)
    cx = np.mean([v[0] for v in all_verts])
    cy = np.mean([v[1] for v in all_verts])
    max_dist = max(np.sqrt((v[0]-cx)**2 + (v[1]-cy)**2) for v in all_verts)
    side = max_dist * 2
    return inner, (cx, cy), side, 0.0
# EVOLVE-BLOCK-END
