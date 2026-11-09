import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Unit hexagon vertices (flat-top, centered at origin)
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees
    def unit_hex_verts(angle_rad=0.0):
        angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0 + angle_rad
        return np.column_stack([np.cos(angles), np.sin(angles)])
    
    def hex_verts(cx, cy, angle_deg):
        angle_rad = angle_deg * np.pi / 180.0
        v = unit_hex_verts(angle_rad)
        return v + np.array([cx, cy])
    
    def hex_containment_dist(verts, outer_center, outer_side, outer_angle_deg):
        """Minimum signed distance from any vertex to outer hexagon boundary.
        Positive means inside, negative means outside."""
        outer_angle_rad = outer_angle_deg * np.pi / 180.0
        # Outer hexagon vertices
        outer_verts = outer_side * unit_hex_verts(outer_angle_rad) + np.array(outer_center)
        
        # For a regular hexagon, containment can be checked by distance to each edge
        # Edge normals point inward
        min_dist = float('inf')
        for i in range(6):
            v1 = outer_verts[i]
            v2 = outer_verts[(i + 1) % 6]
            edge = v2 - v1
            # Normal pointing inward (toward center)
            normal = np.array([-edge[1], edge[0]])
            normal = normal / np.linalg.norm(normal)
            # Check if normal points toward center
            center_to_v1 = np.array(outer_center) - v1
            if np.dot(normal, center_to_v1) < 0:
                normal = -normal
            # Distance from each vertex to this edge (positive if on center side)
            for v in verts:
                d = np.dot(v - v1, normal)
                min_dist = min(min_dist, d)
        return min_dist
    
    def polygon_overlap(verts1, verts2):
        """Check if two convex polygons overlap using SAT."""
        def get_edges(verts):
            edges = []
            for i in range(len(verts)):
                v1 = verts[i]
                v2 = verts[(i + 1) % len(verts)]
                edge = v2 - v1
                edges.append(edge)
            return edges
        
        def project(verts, axis):
            projections = np.dot(verts, axis)
            return projections.min(), projections.max()
        
        all_edges = get_edges(verts1) + get_edges(verts2)
        for edge in all_edges:
            axis = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(axis)
            if norm < 1e-10:
                continue
            axis = axis / norm
            min1, max1 = project(verts1, axis)
            min2, max2 = project(verts2, axis)
            if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
                return False  # Separating axis found
        return True  # Overlap
    
    def separation_distance(verts1, verts2):
        """Minimum distance between two convex polygons (positive if separated)."""
        def get_edges(verts):
            edges = []
            for i in range(len(verts)):
                v1 = verts[i]
                v2 = verts[(i + 1) % len(verts)]
                edge = v2 - v1
                edges.append(edge)
            return edges
        
        def project_range(verts, axis):
            projections = np.dot(verts, axis)
            return projections.min(), projections.max()
        
        all_edges = get_edges(verts1) + get_edges(verts2)
        min_sep = -float('inf')
        for edge in all_edges:
            axis = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(axis)
            if norm < 1e-10:
                continue
            axis = axis / norm
            min1, max1 = project_range(verts1, axis)
            min2, max2 = project_range(verts2, axis)
            # Separation on this axis
            sep = min(max1 - min2, max2 - min1)
            if sep > min_sep:
                min_sep = sep
        return min_sep
    
    # Initial compact arrangement: honeycomb-like
    # Place hexagons in rows with offsets
    def initial_positions():
        positions = []
        angles = []
        # Try a 3-4-4 arrangement or similar compact layout
        # For 11 hexagons, try: row of 4, row of 4, row of 3
        # Or better: use a hexagonal packing pattern
        
        # Flat-top hexagons: width = 2*sqrt(3), height = 2
        # For flat-top, adjacent in same row: dx = 2*sqrt(3), dy = 0
        # Adjacent in next row: dx = sqrt(3), dy = 1.5 (for pointy-top)
        # For flat-top: adjacent row offset: dx = 0, dy = 2, but with horizontal shift
        
        # Actually for flat-top hexagons:
        # Width (horizontal) = 2*sqrt(3) ≈ 3.464
        # Height (vertical) = 2
        # To pack, offset rows by sqrt(3) horizontally and 1.5 vertically? No...
        
        # Let me think: flat-top hexagon has vertices at (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5,-sqrt(3)/2), (0.5,-sqrt(3)/2)
        # So width = 2, height = sqrt(3)
        # Wait, let me recalculate. With my unit_hex_verts:
        # angles 0,60,120,180,240,300
        # (1,0), (0.5, 0.866), (-0.5, 0.866), (-1,0), (-0.5,-0.866), (0.5,-0.866)
        # So width = 2 (from x=-1 to x=1), height = sqrt(3) ≈ 1.732
        
        # For packing flat-top hexagons:
        # Same row: shift by 2 in x (since width is 2, but they interlock)
        # Actually adjacent flat-top hexagons share an edge, so center-to-center distance for horizontal neighbors = 2*sqrt(3)/2 * ... 
        
        # Let me just use a grid and let the optimizer fix it
        # Start with a reasonable guess
        
        # Place in a 3x4 grid roughly
        sqrt3 = np.sqrt(3)
        # For flat-top hexagons packed in honeycomb:
        # Row 0: x = 0, 3.464, 6.928, 10.392 (4 hexagons)
        # Row 1: x = 1.732, 5.196, 8.660 (3 hexagons), y offset = 1.5
        # Row 2: x = 0, 3.464, 6.928 (3 hexagons), y offset = 3.0
        # Total: 4+3+4 = 11
        
        cols = 4
        rows = 3
        idx = 0
        for r in range(rows):
            for c in range(cols):
                if idx >= N:
                    break
                x = c * 2 * sqrt3
                y = r * 1.5
                if r % 2 == 1:
                    x += sqrt3  # offset every other row
                positions.append([x, y])
                angles.append(0.0)
                idx += 1
            if idx >= N:
                break
        
        # Center the arrangement
        if positions:
            px = np.array([p[0] for p in positions])
            py = np.array([p[1] for p in positions])
            cx = (px.max() + px.min()) / 2
            cy = (py.max() + py.min()) / 2
            for i in range(len(positions)):
                positions[i][0] -= cx
                positions[i][1] -= cy
        
        return positions, angles
    
    positions, angles = initial_positions()
    
    # Variables: for each hex: x, y, angle_deg, plus outer_side, outer_angle_deg
    # Total: N*3 + 2 = 35 variables
    n_vars = N * 3 + 2
    
    def unpack(x):
        pos = []
        ang = []
        for i in range(N):
            pos.append([x[3*i], x[3*i+1]])
            ang.append(x[3*i+2])
        outer_side = x[N*3]
        outer_angle = x[N*3+1]
        return pos, ang, outer_side, outer_angle
    
    def objective(x):
        pos, ang, outer_side, outer_angle = unpack(x)
        return outer_side
    
    def constraints(x):
        pos, ang, outer_side, outer_angle = unpack(x)
        
        cons = []
        
        # Containment: each inner hex must be inside outer hex
        for i in range(N):
            vi = hex_verts(pos[i][0], pos[i][1], ang[i])
            
            def containment_constr(x, i=i):
                pos, ang, os, oa = unpack(x)
                vi = hex_verts(pos[i][0], pos[i][1], ang[i])
                return hex_containment_dist(vi, (0.0, 0.0), os, oa)
            
            cons.append({'type': 'ineq', 'fun': containment_constr})
        
        # Non-overlap: pairs of inner hexagons must not overlap
        for i in range(N):
            for j in range(i+1, N):
                def nonoverlap_constr(x, i=i, j=j):
                    pos, ang, os, oa = unpack(x)
                    vi = hex_verts(pos[i][0], pos[i][1], ang[i])
                    vj = hex_verts(pos[j][0], pos[j][1], ang[j])
                    # Return separation distance (positive if separated)
                    return separation_distance(vi, vj)
                
                cons.append({'type': 'ineq', 'fun': nonoverlap_constr})
        
        return cons
    
    # Bounds: outer_side > 0, angles in [-180, 180]
    bounds = []
    for i in range(N):
        bounds.append((-50, 50))  # x
        bounds.append((-50, 50))  # y
        bounds.append((-180, 180))  # angle
    bounds.append((0.5, 50))  # outer_side
    bounds.append((-180, 180))  # outer_angle
    
    # Initial point
    x0 = []
    for i in range(N):
        x0.append(positions[i][0])
        x0.append(positions[i][1])
        x0.append(angles[i])
    # Estimate initial outer side
    max_extent = 0
    for p in positions:
        max_extent = max(max_extent, abs(p[0]), abs(p[1]))
    x0.append(max_extent + 3)  # outer_side
    x0.append(0.0)  # outer_angle
    
    best_result = None
    best_side = float('inf')
    
    # Try multiple restarts with different initial configurations
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        # Perturb initial positions for diversity
        if restart > 0:
            x0_perturbed = list(x0)
            for i in range(N):
                x0_perturbed[3*i] += np.random.uniform(-2, 2)
                x0_perturbed[3*i+1] += np.random.uniform(-2, 2)
                x0_perturbed[3*i+2] += np.random.uniform(-30, 30)
            x0_perturbed[N*3] = max_extent + 3 + np.random.uniform(-1, 2)
        else:
            x0_perturbed = x0
        
        try:
            result = minimize(objective, x0_perturbed, method='SLSQP',
                            bounds=bounds, constraints=constraints,
                            options={'maxiter': 100, 'ftol': 1e-6})
            
            if result.success or result.fun < best_side:
                pos, ang, outer_side, outer_angle = unpack(result.x)
                # Verify validity
                valid = True
                for i in range(N):
                    vi = hex_verts(pos[i][0], pos[i][1], ang[i])
                    if hex_containment_dist(vi, (0.0, 0.0), outer_side, outer_angle) < -1e-6:
                        valid = False
                        break
                if valid:
                    for i in range(N):
                        for j in range(i+1, N):
                            vi = hex_verts(pos[i][0], pos[i][1], ang[i])
                            vj = hex_verts(pos[j][0], pos[j][1], ang[j])
                            if separation_distance(vi, vj) < -1e-6:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        best_result = result.x
                        best_side = outer_side
        except Exception as e:
            pass
        
        if time.time() > deadline - 20:
            break
    
    if best_result is not None:
        pos, ang, outer_side, outer_angle = unpack(best_result)
        inner = [[pos[i][0], pos[i][1], ang[i]] for i in range(N)]
        return inner, (0.0, 0.0), outer_side, outer_angle
    else:
        # Fallback: return the initial arrangement with a safe outer side
        inner = [[positions[i][0], positions[i][1], angles[i]] for i in range(N)]
        return inner, (0.0, 0.0), max_extent + 5.0, 0.0

# EVOLVE-BLOCK-END
