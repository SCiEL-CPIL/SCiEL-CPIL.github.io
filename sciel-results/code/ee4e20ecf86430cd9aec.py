import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    sqrt3 = math.sqrt(3.0)
    
    def get_hex_vertices(cx, cy, angle_deg):
        angles = [math.radians(angle_deg + 60*k) for k in range(6)]
        return np.array([(cx + math.cos(a), cy + math.sin(a)) for a in angles])
    
    def point_in_polygon(pt, poly):
        x, y = pt
        n = len(poly)
        inside = False
        p1x, p1y = poly[0]
        for i in range(1, n + 1):
            p2x, p2y = poly[i % n]
            if (y > p1y) != (y > p2y):
                if p2y != p1y:
                    xint = (p2x - p1x) * (y - p1y) / (p2y - p1y) + p1x
                    if x < xint:
                        inside = not inside
            p1x, p1y = p2x, p2y
        return inside
    
    def segments_intersect(p1, p2, p3, p4):
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) - (B[1]-A[1]) * (C[0]-A[0])
        c1 = ccw(p1, p3, p4)
        c2 = ccw(p2, p3, p4)
        c3 = ccw(p1, p2, p3)
        c4 = ccw(p1, p2, p4)
        return (c1 > 0) != (c2 > 0) and (c3 > 0) != (c4 > 0)
    
    def polygons_overlap(v1, v2):
        for v in v1:
            if point_in_polygon(v, v2):
                return True
        for v in v2:
            if point_in_polygon(v, v1):
                return True
        n1 = len(v1)
        n2 = len(v2)
        for i in range(n1):
            p1 = v1[i]
            p2 = v1[(i+1)%n1]
            for j in range(n2):
                p3 = v2[j]
                p4 = v2[(j+1)%n2]
                if segments_intersect(p1, p2, p3, p4):
                    return True
        return False
    
    def point_segment_dist(pt, p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        if dx == 0 and dy == 0:
            return math.hypot(pt[0]-p1[0], pt[1]-p1[1])
        t = ((pt[0]-p1[0])*dx + (pt[1]-p1[1])*dy) / (dx*dx + dy*dy)
        t = max(0.0, min(1.0, t))
        px = p1[0] + t*dx
        py = p1[1] + t*dy
        return math.hypot(pt[0]-px, pt[1]-py)
    
    def point_poly_dist(pt, poly):
        min_d = float('inf')
        n = len(poly)
        for i in range(n):
            d = point_segment_dist(pt, poly[i], poly[(i+1)%n])
            if d < min_d:
                min_d = d
        return min_d
    
    def compute_L(all_vertices, center=(0,0), angle_deg=0):
        # Compute minimum side length of regular hexagon with given center and angle
        # containing all vertices
        cx, cy = center
        base_angle = math.radians(angle_deg)
        normals = [base_angle + math.radians(30 + 60*k) for k in range(6)]
        max_proj = 0.0
        for v in all_vertices:
            dx = v[0] - cx
            dy = v[1] - cy
            for ang in normals:
                proj = dx * math.cos(ang) + dy * math.sin(ang)
                if proj > max_proj:
                    max_proj = proj
        return max_proj / (sqrt3 / 2.0)
    
    def smooth_overlap_penalty(v1, v2, alpha=10.0):
        """Compute a smooth penalty for overlap between two polygons."""
        pen = 0.0
        for v in v1:
            if point_in_polygon(v, v2):
                d = point_poly_dist(v, v2)
                pen += d * d
        for v in v2:
            if point_in_polygon(v, v1):
                d = point_poly_dist(v, v1)
                pen += d * d
        # Check edge intersections with soft penalty
        n1 = len(v1)
        n2 = len(v2)
        for i in range(n1):
            p1 = v1[i]
            p2 = v1[(i+1)%n1]
            for j in range(n2):
                p3 = v2[j]
                p4 = v2[(j+1)%n2]
                if segments_intersect(p1, p2, p3, p4):
                    pen += 5.0
        return pen
    
    def objective(params):
        if time.time() > deadline:
            return 1e10
        
        positions = []
        all_vertices = []
        verts_list = []
        
        for i in range(N):
            x = params[3*i]
            y = params[3*i+1]
            th = params[3*i+2]
            positions.append((x, y, th))
            verts = get_hex_vertices(x, y, th)
            verts_list.append(verts)
            all_vertices.extend(verts)
        
        L_req = compute_L(all_vertices)
        
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                v1 = verts_list[i]
                v2 = verts_list[j]
                if polygons_overlap(v1, v2):
                    penalty += smooth_overlap_penalty(v1, v2)
        
        return L_req + penalty
    
    # Generate initial configurations
    def make_grid_init():
        """Hex grid initialization - 12 hexagons in a compact arrangement"""
        centers = []
        # Use axial coordinates for hex grid
        # For a compact cluster of 12, try various patterns
        # Pattern: rows of 4, 4, 4 or 3, 4, 3, 2 etc.
        
        # Try a 2-4-4-2 pattern (rows offset)
        # Row 0: 2 hexes
        # Row 1: 4 hexes  
        # Row 2: 4 hexes
        # Row 3: 2 hexes
        # This gives 12 total
        
        dx = 1.5  # horizontal spacing between adjacent hex centers in same row
        dy = sqrt3  # vertical spacing between rows
        
        # Actually, for flat-top hexagons with side 1:
        # Adjacent hex centers in same row: distance = 2 * inradius = sqrt(3)
        # Wait, for flat-top hex with side 1, the width is 2, height is sqrt(3)
        # Center to center horizontally (same row) = 2 * inradius = sqrt(3)
        # Center to center vertically (adjacent row) = 1.5
        
        # Let me use pointy-top for easier grid math
        # For pointy-top hex with side 1:
        # Width = sqrt(3), Height = 2
        # Horizontal spacing = sqrt(3)
        # Vertical spacing = 1.5
        # Odd rows offset by sqrt(3)/2
        
        h_sp = sqrt3
        v_sp = 1.5
        offset = sqrt3 / 2
        
        # 12 hexagons in a hexagonal-ish cluster
        # Try: 3 rows of 4
        pattern = [
            (0, 0), (1, 0), (2, 0), (3, 0),
            (0, 1), (1, 1), (2, 1), (3, 1),
            (0, 2), (1, 2), (2, 2), (3, 2),
        ]
        
        params = []
        for col, row in pattern:
            x = col * h_sp + (row % 2) * offset
            y = row * v_sp
            # Center the cluster
            x -= 1.5 * h_sp + offset/2
            y -= 1.0 * v_sp
            params.extend([x, y, 0.0])
        
        return params
    
    def make_ring_init():
        """One center + ring of 6 + ring of 5 more"""
        params = []
        # Center
        params.extend([0.0, 0.0, 0.0])
        
        # Inner ring of 6 at distance sqrt(3)
        for k in range(6):
            ang = math.radians(60 * k)
            x = sqrt3 * math.cos(ang)
            y = sqrt3 * math.sin(ang)
            params.extend([x, y, 0.0])
        
        # Outer ring of 5 more
        # Place them at distance 2*sqrt(3) at angles between the inner ring
        for k in range(5):
            ang = math.radians(60 * k + 30)
            x = 2 * sqrt3 * math.cos(ang)
            y = 2 * sqrt3 * math.sin(ang)
            params.extend([x, y, 0.0])
        
        return params
    
    def make_random_init(seed=42):
        np.random.seed(seed)
        params = []
        for _ in range(N):
            x = np.random.uniform(-3, 3)
            y = np.random.uniform(-3, 3)
            th = np.random.uniform(0, 360)
            params.extend([x, y, th])
        return params
    
    # Try multiple initializations
    inits = [
        make_grid_init(),
        make_ring_init(),
        make_random_init(42),
        make_random_init(123),
    ]
    
    best_params = None
    best_score = float('inf')
    
    for init in inits:
        if time.time() > deadline - 20:
            break
        
        try:
            res = minimize(objective, init, method='Nelder-Mead',
                          options={'maxiter': 8000, 'xatol': 1e-6, 'fatol': 1e-6,
                                   'adaptive': True})
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        best_params = np.array(inits[0])
    
    # Final polish with tighter tolerances
    if time.time() < deadline - 10:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8,
                                   'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Extract final positions
    positions = []
    all_vertices = []
    verts_list = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        th = best_params[3*i+2]
        positions.append([x, y, th])
        verts = get_hex_vertices(x, y, th)
        verts_list.append(verts)
        all_vertices.extend(verts)
    
    L_final = compute_L(all_vertices)
    
    # Verify disjointness
    is_valid = True
    for i in range(N):
        for j in range(i+1, N):
            if polygons_overlap(verts_list[i], verts_list[j]):
                is_valid = False
                break
        if not is_valid:
            break
    
    if not is_valid:
        # Fallback to grid configuration
        params_fb = make_grid_init()
        positions = []
        all_vertices = []
        for i in range(N):
            x = params_fb[3*i]
            y = params_fb[3*i+1]
            th = params_fb[3*i+2]
            positions.append([x, y, th])
            verts = get_hex_vertices(x, y, th)
            all_vertices.extend(verts)
        L_final = compute_L(all_vertices)
    
    return positions, (0.0, 0.0), L_final, 0.0
