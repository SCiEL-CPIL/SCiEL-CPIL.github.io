import math
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar, linprog

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    def hex_vertices(cx, cy, s, angle_deg):
        """Return 6 vertices of a regular hexagon centered at (cx,cy) with side s and rotation angle_deg."""
        angle = math.radians(angle_deg)
        # Flat-top hexagon: vertices at angles 0, 60, 120, ... from center
        # For a regular hexagon with side s, the circumradius = s
        # Vertices at angles: angle + k*60 degrees, k=0..5
        # Actually for flat-top, the vertices are at 0, 60, 120, 180, 240, 300 degrees
        # The apothem (distance from center to side) = s * sqrt(3)/2
        verts = []
        for k in range(6):
            a = angle + k * math.pi / 3.0
            verts.append((cx + s * math.cos(a), cy + s * math.sin(a)))
        return np.array(verts)

    def point_in_hexagon(px, py, cx, cy, s, angle_deg):
        """Check if point (px,py) is inside or on boundary of hexagon."""
        angle = math.radians(angle_deg)
        # The hexagon is the intersection of 6 half-planes
        # For each side, the normal points inward
        # Side normals are at angles: angle + 30 + k*60 degrees (perpendicular to vertices)
        # Actually, for a hexagon with vertices at angle + k*60, the sides are between consecutive vertices
        # The inward normal to the side between vertex k and k+1 is at angle + (k+0.5)*60 + 90... 
        # Let me think more carefully.
        # For a regular hexagon centered at origin with flat top (vertex at angle 0):
        # Vertices: (s,0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s,0), (-s/2,-s*sqrt(3)/2), (s/2,-s*sqrt(3)/2)
        # The 6 sides have outward normals at angles 30, 90, 150, 210, 270, 330 degrees
        # i.e., angle + 30 + k*60 for k=0..5 where angle is the vertex angle offset
        # So inward normal is at angle + 30 + k*60 + 180 = angle + 210 + k*60
        # Distance from center to each side (apothem) = s * sqrt(3)/2
        # Point (px,py) relative to center: (px-cx, py-cy)
        # For each of 6 sides: dot((px-cx, py-cy), inward_normal_k) <= apothem
        # Actually: dot((px-cx, py-cy), outward_normal_k) <= apothem
        dx = px - cx
        dy = py - cy
        apothem = s * APOTHEM
        for k in range(6):
            # Outward normal angle for side k
            n_angle = angle + math.pi / 6.0 + k * math.pi / 3.0
            nx = math.cos(n_angle)
            ny = math.sin(n_angle)
            if dx * nx + dy * ny > apothem + 1e-9:
                return False
        return True

    def min_separation_hexagons(cx1, cy1, a1, cx2, cy2, a2, s1=1.0, s2=1.0):
        """Compute minimum distance between boundaries of two hexagons.
        Returns the minimum distance (negative if overlapping).
        Uses support function approach: for each of 12 edge directions, compute gap.
        """
        # Two convex polygons: minimum distance between them
        # We can use the fact that min distance between two convex sets = 
        # max over directions of (support1(dir) + support2(-dir) - distance between centers projected)
        # Actually, simpler: compute all 6*6 = 36 vertex-edge distances and 6*6 edge-vertex distances
        # But that's expensive. Let's use a sampling approach with refinement.
        
        # Better approach: The minimum distance between two convex polygons can be found
        # by checking each edge of one against each vertex of the other and vice versa.
        # For hexagons, that's 6*6 + 6*6 = 72 checks.
        
        verts1 = hex_vertices(cx1, cy1, s1, a1)
        verts2 = hex_vertices(cx2, cy2, s2, a2)
        
        min_dist = float('inf')
        
        # For each edge of hex1, find min distance to each vertex of hex2
        for i in range(6):
            p1 = verts1[i]
            p2 = verts1[(i+1) % 6]
            # Edge vector
            ex = p2[0] - p1[0]
            ey = p2[1] - p1[1]
            elen2 = ex*ex + ey*ey
            if elen2 < 1e-20:
                continue
            for v in verts2:
                # Project v onto edge
                t = ((v[0]-p1[0])*ex + (v[1]-p1[1])*ey) / elen2
                t = max(0.0, min(1.0, t))
                qx = p1[0] + t*ex
                qy = p1[1] + t*ey
                d = math.sqrt((v[0]-qx)**2 + (v[1]-qy)**2)
                if d < min_dist:
                    min_dist = d
        
        # For each edge of hex2, find min distance to each vertex of hex1
        for i in range(6):
            p1 = verts2[i]
            p2 = verts2[(i+1) % 6]
            ex = p2[0] - p1[0]
            ey = p2[1] - p1[1]
            elen2 = ex*ex + ey*ey
            if elen2 < 1e-20:
                continue
            for v in verts1:
                t = ((v[0]-p1[0])*ex + (v[1]-p1[1])*ey) / elen2
                t = max(0.0, min(1.0, t))
                qx = p1[0] + t*ex
                qy = p1[1] + t*ey
                d = math.sqrt((v[0]-qx)**2 + (v[1]-qy)**2)
                if d < min_dist:
                    min_dist = d
        
        return min_dist

    def is_feasible(positions, angles, outer_center, outer_side, outer_angle):
        """Check if all hexagons are disjoint and inside the outer hexagon."""
        ocx, ocy = outer_center
        os = outer_side
        oa = outer_angle
        
        # Check each hexagon is inside outer
        for i in range(N):
            cx, cy = positions[i]
            a = angles[i]
            verts = hex_vertices(cx, cy, 1.0, a)
            for vx, vy in verts:
                if not point_in_hexagon(vx, vy, ocx, ocy, os, oa):
                    return False
        
        # Check pairwise separation
        for i in range(N):
            for j in range(i+1, N):
                sep = min_separation_hexagons(
                    positions[i][0], positions[i][1], angles[i],
                    positions[j][0], positions[j][1], angles[j]
                )
                if sep < -1e-7:
                    return False
        return True

    def compute_outer_size(positions, angles, outer_angle):
        """Given positions and angles of inner hexagons, and outer hexagon angle,
        find the minimum side length and center of outer hexagon that contains all.
        Uses LP: minimize D such that for each inner hex vertex, 
        |dot(v - c, n_k)| <= D - apothem_inner for all 6 normals n_k.
        Wait, actually: the outer hexagon has apothem = S * sqrt(3)/2.
        A point p is inside the outer hexagon centered at c with side S and angle a iff
        for all k: dot(p - c, n_k) <= S * sqrt(3)/2, where n_k are the 6 outward normals.
        So we need: for all inner vertices v, for all k: dot(v - c, n_k) <= D
        where D = S * sqrt(3)/2.
        This is a feasibility problem: find c and D minimizing D.
        """
        oa = math.radians(outer_angle)
        normals = []
        for k in range(6):
            ang = oa + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        # Collect all vertices of all inner hexagons
        all_verts = []
        for i in range(N):
            cx, cy = positions[i]
            a = angles[i]
            verts = hex_vertices(cx, cy, 1.0, a)
            all_verts.extend(verts.tolist())
        
        all_verts = np.array(all_verts)
        nv = len(all_verts)
        
        # LP: variables are (cx, cy, D)
        # Minimize D
        # Constraints: for each vertex v and each normal n:
        #   dot(v - c, n) <= D  =>  -v.x*n.x - v.y*n.y + cx*n.x + cy*n.y - D <= 0
        #   -dot(v - c, n) <= D  =>  v.x*n.x + v.y*n.y - cx*n.x - cy*n.y - D <= 0
        c_obj = np.array([0.0, 0.0, 1.0])
        A_ub = []
        b_ub = []
        
        for v in all_verts:
            for nx, ny in normals:
                # -dot(v-c, n) <= D  =>  (v-c).n <= D  =>  v.n - c.n <= D
                # => -cx*nx - cy*ny - D <= -v.n  =>  cx*nx + cy*ny + D >= v.n
                # Let's write as: -cx*nx - cy*ny - D <= -v.x*nx - v.y*ny
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-(v[0]*nx + v[1]*ny))
                # dot(v-c, n) <= D  =>  v.n - c.n <= D  =>  -cx*nx - cy*ny - D <= -v.n
                # Wait, that's the same. Let me redo.
                # (v-c).n <= D  =>  v.n - c.n <= D  =>  -c.n - D <= -v.n
                # So: -cx*nx - cy*ny - 1*D <= -(v.x*nx + v.y*ny)
                # And: c.n - v.n <= D  =>  cx*nx + cy*ny - D <= v.n
                # So: cx*nx + cy*ny - 1*D <= v.x*nx + v.y*ny
                A_ub.append([nx, ny, -1.0])
                b_ub.append(v[0]*nx + v[1]*ny)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                D = res.x[2]
                cx = res.x[0]
                cy = res.x[1]
                S = D / APOTHEM
                return S, (cx, cy), D
            else:
                return float('inf'), None, None
        except Exception:
            return float('inf'), None, None

    def objective(params, outer_angle):
        """Objective function for optimization: given flat array of positions and angles,
        compute the outer hexagon side length."""
        # params: [x0, y0, a0, x1, y1, a1, ..., x10, y10, a10]
        positions = []
        angles = []
        for i in range(N):
            positions.append((params[3*i], params[3*i+1]))
            angles.append(params[3*i+2])
        
        S, center, D = compute_outer_size(positions, angles, outer_angle)
        # Add penalty if infeasible (hexagons overlap or outside)
        # Actually, compute_outer_size already ensures containment.
        # We need to check disjointness separately.
        # For now, just return S (the LP handles containment).
        # Disjointness will be checked after.
        return S

    def check_disjointness(positions, angles):
        """Check if all hexagons are pairwise disjoint."""
        for i in range(N):
            for j in range(i+1, N):
                sep = min_separation_hexagons(
                    positions[i][0], positions[i][1], angles[i],
                    positions[j][0], positions[j][1], angles[j]
                )
                if sep < -1e-6:
                    return False
        return True

    def penalty_objective(params, outer_angle):
        """Objective with penalty for overlapping hexagons."""
        positions = []
        angles = []
        for i in range(N):
            positions.append((params[3*i], params[3*i+1]))
            angles.append(params[3*i+2])
        
        S, center, D = compute_outer_size(positions, angles, outer_angle)
        
        # Penalty for overlaps
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                sep = min_separation_hexagons(
                    positions[i][0], positions[i][1], angles[i],
                    positions[j][0], positions[j][1], angles[j]
                )
                if sep < 0:
                    penalty += (-sep) ** 2 * 100.0
        
        return S + penalty

    # Strategy: 
    # 1. Start with a reasonable initial configuration (hexagonal lattice)
    # 2. Optimize positions and angles using gradient-free methods
    # 3. Try multiple outer angles
    # 4. Use the LP to compute the optimal outer hexagon for given inner positions

    best_S = float('inf')
    best_positions = None
    best_angles = None
    best_outer_center = None
    best_outer_angle = 0.0

    # Initial configuration: hexagonal lattice
    # Place hexagons in a compact arrangement
    # For N=11, a good starting point is a hexagonal cluster
    # Center + 6 surrounding + 4 more
    
    def make_initial_config():
        """Create an initial hexagonal lattice configuration."""
        # Axial coordinates for a compact 11-hexagon cluster
        # Center at (0,0), then ring of 6, then 4 more
        axial_coords = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (-2, 0), (0, 2), (0, -2)
        ]
        
        positions = []
        angles = [0.0] * N
        
        for q, r in axial_coords:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            positions.append((x, y))
        
        return positions, angles

    # Try different outer angles
    outer_angles_to_try = [0.0, 15.0, 30.0, 45.0, 60.0]

    for outer_angle_deg in outer_angles_to_try:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        positions, angles = make_initial_config()
        
        # Flatten to parameter vector
        params = []
        for i in range(N):
            params.extend([positions[i][0], positions[i][1], angles[i]])
        
        # Optimize using Nelder-Mead with penalty
        try:
            res = minimize(
                penalty_objective, 
                np.array(params), 
                args=(outer_angle_deg,),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            
            if res.fun < 1e10:
                # Extract positions and angles
                opt_positions = []
                opt_angles = []
                for i in range(N):
                    opt_positions.append((res.x[3*i], res.x[3*i+1]))
                    opt_angles.append(res.x[3*i+2])
                
                # Check feasibility
                S, center, D = compute_outer_size(opt_positions, opt_angles, outer_angle_deg)
                
                if S < float('inf') and check_disjointness(opt_positions, opt_angles):
                    if S < best_S:
                        best_S = S
                        best_positions = opt_positions
                        best_angles = opt_angles
                        best_outer_center = center
                        best_outer_angle = outer_angle_deg
                        
        except Exception:
            pass

    # If the main optimization didn't find anything, try a simpler approach
    if best_positions is None:
        # Fall back to the lattice configuration with LP-optimized outer
        positions, angles = make_initial_config()
        for outer_angle_deg in [0.0, 15.0, 30.0]:
            S, center, D = compute_outer_size(positions, angles, outer_angle_deg)
            if S < best_S:
                best_S = S
                best_positions = positions
                best_angles = angles
                best_outer_center = center
                best_outer_angle = outer_angle_deg

    # Final polish: try to improve with more iterations if time allows
    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN - 20 and best_positions is not None:
        try:
            params = []
            for i in range(N):
                params.extend([best_positions[i][0], best_positions[i][1], best_angles[i]])
            
            res = minimize(
                penalty_objective,
                np.array(params),
                args=(best_outer_angle,),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            
            if res.fun < 1e10:
                opt_positions = []
                opt_angles = []
                for i in range(N):
                    opt_positions.append((res.x[3*i], res.x[3*i+1]))
                    opt_angles.append(res.x[3*i+2])
                
                S, center, D = compute_outer_size(opt_positions, opt_angles, best_outer_angle)
                if S < float('inf') and check_disjointness(opt_positions, opt_angles):
                    if S < best_S:
                        best_S = S
                        best_positions = opt_positions
                        best_angles = opt_angles
                        best_outer_center = center
        except Exception:
            pass

    # Construct the output
    if best_positions is None:
        # Absolute fallback
        positions, angles = make_initial_config()
        S, center, D = compute_outer_size(positions, angles, 0.0)
        if center is None:
            cx = float(np.mean([p[0] for p in positions]))
            cy = float(np.mean([p[1] for p in positions]))
            center = (cx, cy)
            # Compute S manually
            max_proj = 0
            for k in range(6):
                ang = math.pi / 6.0 + k * math.pi / 3.0
                nx, ny = math.cos(ang), math.sin(ang)
                for p in positions:
                    proj = abs((p[0] - cx) * nx + (p[1] - cy) * ny)
                    if proj > max_proj:
                        max_proj = proj
            S = (max_proj + APOTHEM) / APOTHEM
        best_S = S
        best_positions = positions
        best_angles = angles
        best_outer_center = center
        best_outer_angle = 0.0

    # Add tiny safety margin
    SCALE = 1.0 + 1e-10
    cx, cy = best_outer_center
    
    inner = []
    for i in range(N):
        x = cx + (best_positions[i][0] - cx) * SCALE
        y = cy + (best_positions[i][1] - cy) * SCALE
        inner.append([float(x), float(y), float(best_angles[i])])
    
    # Recompute S with the scaled positions
    S_final, center_final, _ = compute_outer_size(best_positions, best_angles, best_outer_angle)
    if center_final is None:
        S_final = best_S
        center_final = (cx, cy)
    else:
        cx, cy = center_final
        # Re-scale positions relative to new center
        inner = []
        for i in range(N):
            x = cx + (best_positions[i][0] - cx) * SCALE
            y = cy + (best_positions[i][1] - cy) * SCALE
            inner.append([float(x), float(y), float(best_angles[i])])
    
    # Apply safety margin to side length
    side = float(S_final * (1.0 + 1e-9))
    
    center_out = [float(cx), float(cy)]
    rotation = float(best_outer_angle)
    
    return inner, center_out, side, rotation
