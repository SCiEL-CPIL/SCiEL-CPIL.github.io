import math
import numpy as np
import time
from scipy.optimize import linprog

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def get_hex_vertices(cx, cy, angle_deg):
        """Get the 6 vertices of a unit regular hexagon at (cx,cy) rotated by angle_deg."""
        a = math.radians(angle_deg)
        vertices = []
        for k in range(6):
            theta = a + k * math.pi / 3.0
            vx = cx + math.cos(theta)
            vy = cy + math.sin(theta)
            vertices.append((vx, vy))
        return np.array(vertices)

    def check_separation(p1, a1, p2, a2, tol=1e-7):
        """
        Check if two unit hexagons at positions p1,p2 with angles a1,a2 (degrees)
        are non-overlapping. Two convex polygons are separated iff there exists a
        separating axis. For two hexagons, check all 6 edge normals of each.
        """
        v1 = get_hex_vertices(p1[0], p1[1], a1)
        v2 = get_hex_vertices(p2[0], p2[1], a2)
        
        # Get all edge normals from both hexagons
        normals = []
        for k in range(6):
            # Edge from vertex k to vertex k+1
            edge1 = v1[(k+1) % 6] - v1[k]
            n1 = np.array([-edge1[1], edge1[0]])
            n1_norm = np.linalg.norm(n1)
            if n1_norm > 1e-12:
                normals.append(n1 / n1_norm)
            
            edge2 = v2[(k+1) % 6] - v2[k]
            n2 = np.array([-edge2[1], edge2[0]])
            n2_norm = np.linalg.norm(n2)
            if n2_norm > 1e-12:
                normals.append(n2 / n2_norm)
        
        for n in normals:
            proj1 = v1 @ n
            proj2 = v2 @ n
            if proj1.max() > proj2.min() + tol or proj2.max() > proj1.min() + tol:
                return False
        return True

    def solve_lp_for_fixed_angles(angles, outer_beta):
        """
        Given fixed inner hexagon angles (degrees) and outer hexagon rotation (radians),
        solve LP to minimize outer side length S.
        
        Variables: [x0,y0,...,x10,y10, cx, cy, S]
        Total: 2N + 3 = 25
        
        Objective: minimize S
        Constraints:
          1. For each inner hex i and each of its 6 edge normals n_i:
             For each of 6 outer normals m_k:
               - Inner vertices must be inside outer hex:
                 For each vertex v of inner hex i: m_k · (v - (cx,cy)) <= S * APOTHEM
                 And: -m_k · (v - (cx,cy)) <= S * APOTHEM
          2. Non-overlap between inner hexes i and j:
             For each separating axis n (from either hex's edges):
               proj_i_max - proj_j_min >= 0 (one direction)
               OR proj_j_max - proj_i_min >= 0 (other direction)
             Since we need linear constraints, we use the condition that there EXISTS
             a separating axis. But this is non-convex. Instead, we'll use the fact
             that for hexagons with the SAME orientation, separation is equivalent to
             checking the 6 axial directions.
             
        For simplicity in this first pass, assume all inner hexagons have the SAME
        orientation. Then non-overlap reduces to checking 6 directions.
        """
        n_var = 2 * N + 3  # x0..yN-1, cx, cy, S
        idx = lambda i, comp: 2 * i + comp  # 0=x, 1=y
        cx_idx = 2 * N
        cy_idx = 2 * N + 1
        S_idx = 2 * N + 2
        
        c_obj = np.zeros(n_var)
        c_obj[S_idx] = 1.0
        
        A_ub = []
        b_ub = []
        
        # Outer hex normals
        outer_normals = []
        for k in range(6):
            ang = outer_beta + math.pi / 6.0 + k * math.pi / 3.0
            outer_normals.append([math.cos(ang), math.sin(ang)])
        
        # Inner hex vertices (unit hex at origin, rotated)
        inner_angles_rad = [math.radians(a) for a in angles]
        
        # Constraint 1: Inner vertices inside outer hex
        for i in range(N):
            a_rad = inner_angles_rad[i]
            for k in range(6):
                theta = a_rad + k * math.pi / 3.0
                vx_off = math.cos(theta)
                vy_off = math.sin(theta)
                # Vertex of inner hex i: (x_i + vx_off, y_i + vy_off)
                for m in outer_normals:
                    mx, my = m
                    # m · (vertex - center) <= S * APOTHEM
                    # m · (x_i + vx_off - cx, y_i + vy_off - cy) <= S * APOTHEM
                    row = np.zeros(n_var)
                    row[idx(i, 0)] = mx
                    row[idx(i, 1)] = my
                    row[cx_idx] = -mx
                    row[cy_idx] = -my
                    row[S_idx] = -APOTHEM
                    A_ub.append(row)
                    b_ub.append(mx * vx_off + my * vy_off)
                    
                    # -m · (vertex - center) <= S * APOTHEM
                    row2 = np.zeros(n_var)
                    row2[idx(i, 0)] = -mx
                    row2[idx(i, 1)] = -my
                    row2[cx_idx] = mx
                    row2[cy_idx] = my
                    row2[S_idx] = -APOTHEM
                    A_ub.append(row2)
                    b_ub.append(-mx * vx_off - my * vy_off)
        
        # Constraint 2: Non-overlap between inner hexes
        # For same-oriented hexagons, use the 6 axial directions
        for i in range(N):
            for j in range(i + 1, N):
                # Check 6 directions
                for k in range(6):
                    ang = math.pi / 6.0 + k * math.pi / 3.0
                    nx = math.cos(ang)
                    ny = math.sin(ang)
                    
                    # Separation: |n · (pos_i - pos_j)| >= sqrt(3)
                    # This is: n·(pos_i - pos_j) >= sqrt(3) OR n·(pos_i - pos_j) <= -sqrt(3)
                    # We need to handle both cases. Use the fact that for same orientation,
                    # the projection difference along any edge normal must be at least 2*APOTHEM = sqrt(3)
                    
                    # Case 1: n·(pos_i - pos_j) >= sqrt(3)
                    # n·pos_i - n·pos_j >= sqrt(3)
                    # -n·pos_i + n·pos_j <= -sqrt(3)
                    row1 = np.zeros(n_var)
                    row1[idx(i, 0)] = -nx
                    row1[idx(i, 1)] = -ny
                    row1[idx(j, 0)] = nx
                    row1[idx(j, 1)] = ny
                    A_ub.append(row1)
                    b_ub.append(-math.sqrt(3))
                    
                    # Case 2: -n·(pos_i - pos_j) >= sqrt(3)
                    # n·pos_i - n·pos_j <= -sqrt(3)
                    row2 = np.zeros(n_var)
                    row2[idx(i, 0)] = nx
                    row2[idx(i, 1)] = ny
                    row2[idx(j, 0)] = -nx
                    row2[idx(j, 1)] = -ny
                    A_ub.append(row2)
                    b_ub.append(-math.sqrt(3))
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        bounds = [(None, None)] * n_var
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                positions = []
                for i in range(N):
                    positions.append((res.x[idx(i, 0)], res.x[idx(i, 1)]))
                cx = res.x[cx_idx]
                cy = res.x[cy_idx]
                S = res.x[S_idx]
                return S, positions, (cx, cy)
            else:
                return float('inf'), None, None
        except Exception:
            return float('inf'), None, None

    def solve_lp_general(angles, outer_beta):
        """
        General LP solver for arbitrary inner angles.
        Uses the separating axis theorem with all relevant normals.
        For non-convex separation, we use a heuristic: add constraints for
        all 12 normals (6 from each pair) and require separation in at least one.
        Since LP can't handle OR, we use the convex relaxation approach:
        require that the minimum projection gap >= 0 across all axes,
        which is sufficient but not necessary.
        """
        n_var = 2 * N + 3
        idx = lambda i, comp: 2 * i + comp
        cx_idx = 2 * N
        cy_idx = 2 * N + 1
        S_idx = 2 * N + 2
        
        c_obj = np.zeros(n_var)
        c_obj[S_idx] = 1.0
        
        A_ub = []
        b_ub = []
        
        outer_normals = []
        for k in range(6):
            ang = outer_beta + math.pi / 6.0 + k * math.pi / 3.0
            outer_normals.append([math.cos(ang), math.sin(ang)])
        
        inner_angles_rad = [math.radians(a) for a in angles]
        
        # Inner vertices
        inner_verts = []
        for i in range(N):
            a_rad = inner_angles_rad[i]
            verts = []
            for k in range(6):
                theta = a_rad + k * math.pi / 3.0
                verts.append((math.cos(theta), math.sin(theta)))
            inner_verts.append(verts)
        
        # Constraint 1: Inner vertices inside outer hex
        for i in range(N):
            for vx_off, vy_off in inner_verts[i]:
                for m in outer_normals:
                    mx, my = m
                    row = np.zeros(n_var)
                    row[idx(i, 0)] = mx
                    row[idx(i, 1)] = my
                    row[cx_idx] = -mx
                    row[cy_idx] = -my
                    row[S_idx] = -APOTHEM
                    A_ub.append(row)
                    b_ub.append(mx * vx_off + my * vy_off)
                    
                    row2 = np.zeros(n_var)
                    row2[idx(i, 0)] = -mx
                    row2[idx(i, 1)] = -my
                    row2[cx_idx] = mx
                    row2[cy_idx] = my
                    row2[S_idx] = -APOTHEM
                    A_ub.append(row2)
                    b_ub.append(-mx * vx_off - my * vy_off)
        
        # Constraint 2: Non-overlap using separating axes
        for i in range(N):
            for j in range(i + 1, N):
                # Get all edge normals from both hexagons
                normals = []
                for a_rad in [inner_angles_rad[i], inner_angles_rad[j]]:
                    for k in range(6):
                        theta1 = a_rad + k * math.pi / 3.0
                        theta2 = a_rad + (k + 1) * math.pi / 3.0
                        edge = (math.cos(theta2) - math.cos(theta1), 
                               math.sin(theta2) - math.sin(theta1))
                        n = (-edge[1], edge[0])
                        norm = math.sqrt(n[0]**2 + n[1]**2)
                        if norm > 1e-12:
                            normals.append((n[0]/norm, n[1]/norm))
                
                # For each normal, add separation constraint
                # We use: n · (pos_i - pos_j) >= min_proj OR n · (pos_i - pos_j) <= -min_proj
                # For linearization, we compute the required projection gap
                for nx, ny in normals:
                    # Compute min and max projection of each hex onto this normal
                    proj_i = [(vx * nx + vy * ny) for vx, vy in inner_verts[i]]
                    proj_j = [(vx * nx + vy * ny) for vx, vy in inner_verts[j]]
                    gap = (max(proj_j) - min(proj_i))  # minimum gap when centered at origin
                    
                    # Separation: n·(pos_i - pos_j) >= gap or n·(pos_i - pos_j) <= -gap
                    # We'll add both as separate constraints (this is a relaxation)
                    
                    # n·pos_i - n·pos_j >= gap  =>  -n·pos_i + n·pos_j <= -gap
                    row1 = np.zeros(n_var)
                    row1[idx(i, 0)] = -nx
                    row1[idx(i, 1)] = -ny
                    row1[idx(j, 0)] = nx
                    row1[idx(j, 1)] = ny
                    A_ub.append(row1)
                    b_ub.append(-gap)
                    
                    # -n·pos_i + n·pos_j >= gap  =>  n·pos_i - n·pos_j <= -gap
                    row2 = np.zeros(n_var)
                    row2[idx(i, 0)] = nx
                    row2[idx(i, 1)] = ny
                    row2[idx(j, 0)] = -nx
                    row2[idx(j, 1)] = -ny
                    A_ub.append(row2)
                    b_ub.append(-gap)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        bounds = [(None, None)] * n_var
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                positions = []
                for i in range(N):
                    positions.append((res.x[idx(i, 0)], res.x[idx(i, 1)]))
                cx = res.x[cx_idx]
                cy = res.x[cy_idx]
                S = res.x[S_idx]
                return S, positions, (cx, cy)
            else:
                return float('inf'), None, None
        except Exception:
            return float('inf'), None, None

    # Sequential optimization:
    # 1. Start with all inner hexagons at angle 0
    # 2. Solve LP for positions and outer hex
    # 3. For each inner hexagon, try optimizing its angle while fixing others
    # 4. Iterate
    
    best_S = float('inf')
    best_positions = None
    best_angles = None
    best_beta = 0.0
    best_center = None
    
    # Initial: all angles 0
    angles = [0.0] * N
    beta_options = [i * math.pi / 6.0 for i in range(6)]
    
    for beta in beta_options:
        S, positions, center = solve_lp_general(angles, beta)
        if S < best_S:
            best_S = S
            best_positions = positions
            best_angles = angles.copy()
            best_beta = beta
            best_center = center
    
    # Iterate: optimize each angle
    for iteration in range(5):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        improved = False
        for i in range(N):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            # Try different angles for hexagon i
            for angle_try in [a * 30.0 for a in range(12)]:
                angles_try = best_angles.copy()
                angles_try[i] = angle_try
                
                S_try, pos_try, center_try = solve_lp_general(angles_try, best_beta)
                if S_try < best_S - 1e-8:
                    best_S = S_try
                    best_positions = pos_try
                    best_angles = angles_try
                    best_center = center_try
                    improved = True
        
        # Also try optimizing outer beta
        for beta_try in [i * math.pi / 6.0 for i in range(6)]:
            S_try, pos_try, center_try = solve_lp_general(best_angles, beta_try)
            if S_try < best_S - 1e-8:
                best_S = S_try
                best_positions = pos_try
                best_beta = beta_try
                best_center = center_try
                improved = True
        
        if not improved:
            break
    
    # Fallback: use hexagonal spiral
    if best_positions is None or best_S == float('inf'):
        axial_spiral = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (-2, 0), (0, 2), (1, -2)
        ]
        best_positions = [axial_to_cartesian([pt])[0] for pt in axial_spiral]
        best_angles = [0.0] * N
        best_beta = 0.0
        cx = sum(p[0] for p in best_positions) / N
        cy = sum(p[1] for p in best_positions) / N
        best_center = (cx, cy)
        
        # Compute S
        max_proj = 0
        for k in range(6):
            ang = best_beta + math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            for px, py in best_positions:
                proj = (px - cx) * nx + (py - cy) * ny
                max_proj = max(max_proj, abs(proj))
        best_S = (max_proj + APOTHEM) / APOTHEM
    
    cx, cy = best_center
    angle_deg = math.degrees(best_beta)
    
    # Scale slightly to ensure validity
    SCALE = 1.0 + 1e-8
    pts_scaled = []
    for i in range(N):
        px, py = best_positions[i]
        sx = cx + (px - cx) * SCALE
        sy = cy + (py - cy) * SCALE
        pts_scaled.append((sx, sy))
    
    # Recompute S
    max_proj = 0
    for k in range(6):
        ang = best_beta + math.pi / 6.0 + k * math.pi / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        for px, py in pts_scaled:
            proj = (px - cx) * nx + (py - cy) * ny
            max_proj = max(max_proj, abs(proj))
    
    S_final = (max_proj + APOTHEM) / APOTHEM
    
    inner = []
    for i in range(N):
        inner.append([float(pts_scaled[i][0]), float(pts_scaled[i][1]), float(best_angles[i])])
    
    center = [float(cx), float(cy)]
    side = float(S_final)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
