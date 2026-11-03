import math
import numpy as np
import time
import random

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def hexagon_vertices(cx, cy, side, angle_deg=0.0):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    verts = []
    for i in range(6):
        ang = math.radians(angle_deg) + i * math.pi / 3.0
        x = cx + side * math.cos(ang)
        y = cy + side * math.sin(ang)
        verts.append((x, y))
    return np.array(verts)

def polygon_overlap(p1, p2, eps=1e-9):
    """Check if two convex polygons overlap using SAT. Returns True if they overlap."""
    n1 = len(p1)
    n2 = len(p2)
    
    for i in range(n1):
        # Edge from p1[i] to p1[(i+1)%n1]
        dx = p1[(i+1)%n1][0] - p1[i][0]
        dy = p1[(i+1)%n1][1] - p1[i][1]
        # Normal
        nx = -dy
        ny = dx
        norm = math.sqrt(nx*nx + ny*ny)
        if norm < 1e-12:
            continue
        nx /= norm
        ny /= norm
        
        # Project both polygons onto this normal
        proj1 = p1[:, 0] * nx + p1[:, 1] * ny
        proj2 = p2[:, 0] * nx + p2[:, 1] * ny
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 + eps or max2 < min1 + eps:
            return False  # Separating axis found
    
    for i in range(n2):
        dx = p2[(i+1)%n2][0] - p2[i][0]
        dy = p2[(i+1)%n2][1] - p2[i][1]
        nx = -dy
        ny = dx
        norm = math.sqrt(nx*nx + ny*ny)
        if norm < 1e-12:
            continue
        nx /= norm
        ny /= norm
        
        proj1 = p1[:, 0] * nx + p1[:, 1] * ny
        proj2 = p2[:, 0] * nx + p2[:, 1] * ny
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 + eps or max2 < min1 + eps:
            return False  # Separating axis found
    
    return True  # No separating axis, polygons overlap

def point_in_hexagon(px, py, cx, cy, side, angle_deg=0.0):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with given side and rotation."""
    # Transform to local coordinates
    local_x = px - cx
    local_y = py - cy
    
    # Rotate by -angle
    ang = -math.radians(angle_deg)
    lx = local_x * math.cos(ang) - local_y * math.sin(ang)
    ly = local_x * math.sin(ang) + local_y * math.cos(ang)
    
    # In local coords, hexagon has vertices at angles 0, 60, 120, 180, 240, 300 degrees
    # The hexagon is the intersection of 6 half-planes
    # For a regular hexagon with side s, the apothem is s*sqrt(3)/2
    apothem = side * math.sqrt(3) / 2.0
    
    # The 6 normals point to the midpoints of edges, at angles 30, 90, 150, 210, 270, 330 degrees
    for k in range(6):
        ang_normal = math.pi / 6.0 + k * math.pi / 3.0
        nx = math.cos(ang_normal)
        ny = math.sin(ang_normal)
        proj = lx * nx + ly * ny
        if proj > apothem + EPS:
            return False
    
    return True

def all_vertices_in_hexagon(inner_verts, cx, cy, side, angle_deg=0.0):
    """Check if all inner hexagon vertices are inside the outer hexagon."""
    for v in inner_verts:
        if not point_in_hexagon(v[0], v[1], cx, cy, side, angle_deg):
            return False
    return True

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0
    
    random.seed(42)
    
    best_score = float('inf')
    best_result = None
    
    # Try multiple random restarts
    for restart in range(5):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        # Initialize positions in a circle
        positions = []
        angles = []
        for i in range(N):
            ang = 2 * math.pi * i / N
            r = 2.5
            x = r * math.cos(ang)
            y = r * math.sin(ang)
            positions.append([x, y])
            angles.append(random.uniform(0, 60))  # Random initial rotation
        
        positions = np.array(positions)
        angles = np.array(angles)
        
        # Start with a large outer hexagon and shrink it
        outer_side = 8.0
        outer_center = [0.0, 0.0]
        outer_angle = 0.0
        
        # Simulation parameters
        dt = 0.01
        repulsion_strength = 1.0
        iterations = 2000
        
        for it in range(iterations):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            # Compute forces
            forces = np.zeros_like(positions)
            
            # Repulsion between overlapping hexagons
            for i in range(N):
                verts_i = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
                for j in range(i+1, N):
                    verts_j = hexagon_vertices(positions[j, 0], positions[j, 1], 1.0, angles[j])
                    
                    if polygon_overlap(verts_i, verts_j, eps=0.001):
                        # Compute separation direction
                        dx = positions[j, 0] - positions[i, 0]
                        dy = positions[j, 1] - positions[i, 1]
                        dist = math.sqrt(dx*dx + dy*dy)
                        if dist < 1e-9:
                            dx = random.uniform(-1, 1)
                            dy = random.uniform(-1, 1)
                            dist = math.sqrt(dx*dx + dy*dy)
                        
                        # Push apart
                        fx = dx / dist
                        fy = dy / dist
                        forces[i, 0] -= repulsion_strength * fx
                        forces[i, 1] -= repulsion_strength * fy
                        forces[j, 0] += repulsion_strength * fx
                        forces[j, 1] += repulsion_strength * fy
            
            # Attraction to center (to keep them compact)
            for i in range(N):
                forces[i, 0] -= 0.01 * positions[i, 0]
                forces[i, 1] -= 0.01 * positions[i, 1]
            
            # Apply forces with damping
            damping = 0.9
            positions += dt * forces
            forces *= damping
            
            # Slowly shrink the outer hexagon
            outer_side *= 0.9995
            if outer_side < 2.0:
                outer_side = 2.0
        
        # Evaluate the final configuration
        # Find the minimum outer hexagon side that contains all inner hexagons
        # We need to find the optimal center, side, and rotation
        
        # Collect all vertices of inner hexagons
        all_verts = []
        for i in range(N):
            verts = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
            all_verts.append(verts)
        all_verts = np.vstack(all_verts)
        
        # For a given rotation, find the minimum enclosing hexagon
        # The hexagon is defined by 6 half-planes
        # We want to minimize the side length such that all points are inside
        
        best_side_for_config = float('inf')
        best_center_for_config = None
        best_rotation_for_config = 0.0
        
        # Try different rotations
        for rot_deg in range(0, 60, 5):
            rot_rad = math.radians(rot_deg)
            
            # The 6 normals of the outer hexagon
            normals = []
            for k in range(6):
                ang = rot_rad + math.pi / 6.0 + k * math.pi / 3.0
                normals.append((math.cos(ang), math.sin(ang)))
            
            # For a hexagon with center (cx, cy) and side s, the constraint is:
            # For each normal n_k: |dot(p - c, n_k)| <= s * apothem_factor
            # where apothem_factor = sqrt(3)/2
            # Actually, the distance from center to each edge is s * sqrt(3)/2
            
            # We want to find (cx, cy) that minimizes max_k max_i |dot(p_i - c, n_k)|
            # This is a convex optimization problem
            
            # Use scipy to solve
            from scipy.optimize import minimize
            
            def objective(params):
                cx, cy = params
                max_proj = 0.0
                for nx, ny in normals:
                    projs = (all_verts[:, 0] - cx) * nx + (all_verts[:, 1] - cy) * ny
                    max_proj = max(max_proj, np.max(np.abs(projs)))
                return max_proj
            
            # Initial guess: centroid
            cx0 = np.mean(all_verts[:, 0])
            cy0 = np.mean(all_verts[:, 1])
            
            try:
                res = minimize(objective, [cx0, cy0], method='Nelder-Mead',
                             options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10})
                if res.fun < best_side_for_config:
                    # The max projection D relates to side by: D = side * sqrt(3)/2
                    # So side = D / (sqrt(3)/2) = 2*D/sqrt(3)
                    side_candidate = res.fun / APOTHEM
                    if side_candidate < best_side_for_config:
                        best_side_for_config = side_candidate
                        best_center_for_config = (res.x[0], res.x[1])
                        best_rotation_for_config = rot_deg
            except Exception:
                pass
        
        if best_center_for_config is not None and best_side_for_config < best_score:
            # Verify no overlaps and all inside
            valid = True
            cx, cy = best_center_for_config
            side = best_side_for_config
            rot = best_rotation_for_config
            
            # Check overlaps
            for i in range(N):
                verts_i = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
                for j in range(i+1, N):
                    verts_j = hexagon_vertices(positions[j, 0], positions[j, 1], 1.0, angles[j])
                    if polygon_overlap(verts_i, verts_j, eps=0.0):
                        valid = False
                        break
                if not valid:
                    break
            
            # Check all vertices inside outer hexagon
            if valid:
                for i in range(N):
                    verts_i = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
                    for v in verts_i:
                        if not point_in_hexagon(v[0], v[1], cx, cy, side, rot):
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                best_score = best_side_for_config
                inner = []
                for i in range(N):
                    inner.append([float(positions[i, 0]), float(positions[i, 1]), float(angles[i])])
                best_result = (inner, [float(cx), float(cy)], float(side), float(rot))
    
    if best_result is None:
        # Fallback: simple grid arrangement
        inner = []
        # Place hexagons in a hexagonal pattern
        pos = [(0, 0)]
        for i in range(1, N):
            # Simple spiral
            r = math.sqrt(i)
            ang = i * 2.4
            x = r * math.cos(ang)
            y = r * math.sin(ang)
            pos.append((x, y))
        
        for x, y in pos:
            inner.append([float(x), float(y), 0.0])
        
        # Compute bounding hexagon
        all_verts = []
        for x, y in pos:
            verts = hexagon_vertices(x, y, 1.0, 0.0)
            all_verts.append(verts)
        all_verts = np.vstack(all_verts)
        
        cx = np.mean(all_verts[:, 0])
        cy = np.mean(all_verts[:, 1])
        
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = (all_verts[:, 0] - cx) * nx + (all_verts[:, 1] - cy) * ny
            max_proj = max(max_proj, np.max(np.abs(projs)))
        
        side = max_proj / APOTHEM
        return inner, [float(cx), float(cy)], float(side), 0.0
    
    return best_result
