import numpy as np
import time
from scipy.optimize import minimize

N = 12

def _hex_vertices(cx, cy, s, angle_deg):
    """Get vertices of a regular hexagon with side s, center (cx,cy), rotation angle_deg."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        x = cx + s * np.cos(theta)
        y = cy + s * np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def _min_enclosing_hexagon(inner_positions, inner_angles, inner_sides):
    """
    Find minimal regular hexagon (center, side, angle) containing all inner hexagons.
    """
    all_verts = []
    for (cx, cy, ang), s in zip(inner_positions, inner_sides):
        verts = _hex_vertices(cx, cy, s, ang)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    best_side = float('inf')
    best_center = None
    best_angle = 0.0
    
    # Sample rotations
    for phi_deg in np.linspace(0, 60, 600):
        phi = np.radians(phi_deg)
        normals = []
        for i in range(6):
            theta = phi + i * np.pi / 3 + np.pi / 6
            normals.append([np.cos(theta), np.sin(theta)])
        normals = np.array(normals)
        
        projs = all_verts @ normals.T
        max_projs = projs.max(axis=0)
        min_projs = projs.min(axis=0)
        widths = max_projs - min_projs
        
        required_apothem = np.max(widths) / 2
        s_candidate = required_apothem * 2 / np.sqrt(3)
        
        lo = max_projs - required_apothem
        hi = min_projs + required_apothem
        
        if np.any(lo > hi + 1e-12):
            continue
        
        mid = (lo + hi) / 2
        c, residuals, rank, sv = np.linalg.lstsq(normals, mid, rcond=None)
        
        c_projs = normals @ c
        if np.any(c_projs < lo - 1e-9) or np.any(c_projs > hi + 1e-9):
            continue
        
        if s_candidate < best_side:
            best_side = s_candidate
            best_center = c
            best_angle = phi_deg
    
    if best_center is None:
        centroid = all_verts.mean(axis=0)
        return centroid[0], centroid[1], 100.0, 0.0
    
    return best_center[0], best_center[1], best_side, best_angle

def _sat_overlap_depth(h1, h2):
    """Minimum penetration depth between two convex polygons using SAT."""
    min_depth = float('inf')
    axes = []
    for poly in [h1, h2]:
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i + 1) % n]
            edge = p2 - p1
            norm = np.linalg.norm(edge)
            if norm < 1e-10:
                continue
            normal = np.array([-edge[1], edge[0]]) / norm
            axes.append(normal)
    
    for axis in axes:
        proj1 = np.dot(h1, axis)
        proj2 = np.dot(h2, axis)
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 <= min2 or max2 <= min1:
            return 0.0
        
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_depth:
            min_depth = overlap
    
    return min_depth

def _total_penalty(params, outer_side, outer_center, outer_angle):
    """Total penalty: containment + pairwise disjointness."""
    inner_verts_list = []
    for i in range(N):
        cx, cy, ang = params[3*i], params[3*i+1], params[3*i+2]
        inner_verts_list.append(_hex_vertices(cx, cy, 1.0, ang))
    
    outer_verts = _hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    
    # Containment penalty: sum of squared distances of vertices outside
    cont_pen = 0.0
    for iv in inner_verts_list:
        for v in iv:
            # Check if v is inside outer_verts
            # Use cross product method
            inside = True
            for i in range(6):
                v1 = outer_verts[i]
                v2 = outer_verts[(i + 1) % 6]
                edge = v2 - v1
                rel = v - v1
                cross = edge[0] * rel[1] - edge[1] * rel[0]
                if cross < 0:
                    inside = False
                    # Approximate distance to edge
                    edge_len = np.linalg.norm(edge)
                    dist = np.abs(cross) / edge_len
                    cont_pen += dist * dist
                    break
    # Disjointness penalty
    disj_pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            depth = _sat_overlap_depth(inner_verts_list[i], inner_verts_list[j])
            if depth > 0:
                disj_pen += depth * depth * 100.0
    
    return cont_pen + disj_pen

def _generate_lattice_configurations():
    """Generate a few high-quality lattice configurations."""
    configs = []
    sqrt3 = np.sqrt(3)
    
    # Config 1: 1-6-5 Flower
    # Center (0,0)
    pts = [(0,0)]
    # Ring 1: 6 points
    for i in range(6):
        angle = i * np.pi / 3
        x = sqrt3 * np.cos(angle)
        y = sqrt3 * np.sin(angle)
        pts.append((x, y))
    # Ring 2: 5 points. Choose 5 from the 12 available.
    # The 12 are at angles 0, 30, 60... with radius 3 (for 0, 60, 120, 180, 240, 300) 
    # and radius 3 (for 30, 90...)? No.
    # Lattice vectors: v1=(sqrt3,0), v2=(sqrt3/2, 3/2).
    # Points in ring 2:
    # 2*v1 = (2*sqrt3, 0)
    # v1+v2 = (1.5*sqrt3, 1.5)
    # 2*v2 = (sqrt3, 3)
    # -2*v1 = (-2*sqrt3, 0)
    # -v1-v2 = (-1.5*sqrt3, -1.5)
    # -2*v2 = (-sqrt3, -3)
    # v1-v2 = (0.5*sqrt3, -1.5)
    # -v1+v2 = (-0.5*sqrt3, 1.5)
    # v2-v1 = (-0.5*sqrt3, 1.5) -- duplicate?
    # Let's just generate all lattice points in a box and filter by distance.
    lattice_pts = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = sqrt3 * (i + j/2)
            y = 1.5 * j
            lattice_pts.append((x, y))
    
    # Sort by distance to origin
    lattice_pts.sort(key=lambda p: p[0]**2 + p[1]**2)
    # Take first 12
    config1 = lattice_pts[:12]
    configs.append(config1)
    
    # Config 2: 2-4-4-2 Strip
    # Rows of 2, 4, 4, 2
    # y = 0, 1.5, 3, 4.5
    # x offsets
    pts2 = []
    # Row 0 (y=0): 2 pts at x = -0.5*sqrt3, 0.5*sqrt3
    pts2.append((-0.5*sqrt3, 0))
    pts2.append((0.5*sqrt3, 0))
    # Row 1 (y=1.5): 4 pts at x = -1.5*sqrt3, -0.5*sqrt3, 0.5*sqrt3, 1.5*sqrt3
    for k in [-1.5, -0.5, 0.5, 1.5]:
        pts2.append((k*sqrt3, 1.5))
    # Row 2 (y=3): 4 pts
    for k in [-1.5, -0.5, 0.5, 1.5]:
        pts2.append((k*sqrt3, 3))
    # Row 3 (y=4.5): 2 pts
    pts2.append((-0.5*sqrt3, 4.5))
    pts2.append((0.5*sqrt3, 4.5))
    configs.append(pts2)
    
    # Config 3: 3-3-3-3 Strip
    pts3 = []
    for r in range(4):
        y = r * 1.5
        for k in [-1.5, -0.5, 0.5]:
            # Offset alternating?
            offset = 0 if r % 2 == 0 else 0.5
            pts3.append(((k + offset)*sqrt3, y))
    configs.append(pts3)
    
    return configs

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_score = float('inf')
    best_result = None
    
    configs = _generate_lattice_configurations()
    
    for config in configs:
        if time.time() > deadline - 20.0:
            break
            
        # Convert to list of (x, y, angle)
        # Initial angles 0
        inner = [(x, y, 0.0) for (x, y) in config]
        inner_angles = [r[2] for r in inner]
        inner_sides = [1.0] * 12
        
        cx, cy, side, angle = _min_enclosing_hexagon(inner, inner_angles, inner_sides)
        side_safe = side * 1.0001 + 1e-6
        
        if side_safe < best_score:
            best_score = side_safe
            best_result = (inner, (cx, cy), side_safe, angle)
            
        # Local optimization
        if time.time() < deadline - 10.0:
            params0 = np.array([x for (x, y, a) in inner for x in [x, y, a]])
            
            # Try a few outer orientations
            for outer_angle in [angle, angle + 15.0, angle - 15.0]:
                if time.time() > deadline - 5.0:
                    break
                
                # Use a slightly smaller outer side to force optimization
                outer_side = side * 0.99
                outer_center = (cx, cy)
                
                def objective(p):
                    return _total_penalty(p, outer_side, outer_center, outer_angle)
                
                try:
                    result = minimize(
                        objective, params0,
                        method='L-BFGS-B',
                        options={'maxiter': 100, 'maxfun': 2000, 'ftol': 1e-12, 'gtol': 1e-8}
                    )
                    opt_params = result.x
                    opt_val = result.fun
                except Exception:
                    continue
                
                if opt_val < 1e-4:
                    inner_opt = [(opt_params[3*i], opt_params[3*i+1], opt_params[3*i+2]) for i in range(N)]
                    inner_angles_opt = [r[2] for r in inner_opt]
                    cx_opt, cy_opt, side_opt, angle_opt = _min_enclosing_hexagon(inner_opt, inner_angles_opt, [1.0]*12)
                    side_opt_safe = side_opt * 1.0001 + 1e-6
                    
                    if side_opt_safe < best_score:
                        best_score = side_opt_safe
                        best_result = (inner_opt, (cx_opt, cy_opt), side_opt_safe, angle_opt)
    
    if best_result is None:
        inner = [(4.0 * k, 0.0, 0.0) for k in range(-6, 6)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_result
