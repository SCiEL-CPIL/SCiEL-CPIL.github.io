"""Packing 12 unit hexagons inside a regular hexagon using Voronoi-based optimization."""
import numpy as np
from scipy.spatial import Voronoi
import time

N = 12

def _hexagon_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx, cy) with given side length and rotation."""
    angle_rad = np.radians(angle_deg)
    angles = np.pi / 3 * np.arange(6) + angle_rad
    verts = np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])
    return verts

def _polygon_area(verts):
    """Compute area of a polygon given its vertices (counterclockwise)."""
    n = len(verts)
    if n < 3:
        return 0.0
    x = verts[:, 0]
    y = verts[:, 1]
    return 0.5 * np.abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))

def _polygon_contains_point(verts, px, py):
    """Check if point (px, py) is inside polygon verts using ray casting."""
    n = len(verts)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = verts[i]
        xj, yj = verts[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def _point_in_polygon(verts, p):
    return _polygon_contains_point(verts, p[0], p[1])

def _polygon_intersection(verts1, verts2):
    """Compute intersection polygon of two convex polygons using Sutherland-Hodgman."""
    # Clip subject (verts1) by clip (verts2)
    def clip_edge(subject, a, b):
        """Clip subject polygon by edge from a to b (inward normal)."""
        if len(subject) == 0:
            return []
        
        def inside(p):
            # Inside if cross product is >= 0 (assuming CCW clip polygon)
            return np.cross(b - a, p - a) >= -1e-10
        
        def intersect(p, q):
            # Find intersection of line pq with line ab
            d1 = np.cross(b - a, p - a)
            d2 = np.cross(b - a, q - a)
            if abs(d1 - d2) < 1e-15:
                return p
            t = d1 / (d1 - d2)
            return p + t * (q - p)
        
        output = []
        for i in range(len(subject)):
            curr = subject[i]
            prev = subject[i - 1]
            if inside(curr):
                if not inside(prev):
                    output.append(intersect(prev, curr))
                output.append(curr)
            elif inside(prev):
                output.append(intersect(prev, curr))
        return output
    
    subject = [v.copy() for v in verts1]
    for i in range(len(verts2)):
        a = verts2[i]
        b = verts2[(i + 1) % len(verts2)]
        subject = clip_edge(subject, a, b)
        if len(subject) == 0:
            return None
    return np.array(subject)

def _polygon_overlap_area(verts1, verts2):
    """Compute area of intersection of two convex polygons."""
    inter = _polygon_intersection(verts1, verts2)
    if inter is None or len(inter) < 3:
        return 0.0
    return _polygon_area(inter)

def _hexagon_distance(verts1, verts2):
    """Compute minimum distance between two convex hexagons (0 if overlapping)."""
    # Check if they overlap
    # Use SAT-like approach: for each edge of each polygon, project all vertices
    min_dist = float('inf')
    
    for verts in [verts1, verts2]:
        for i in range(len(verts)):
            a = verts[i]
            b = verts[(i + 1) % len(verts)]
            edge = b - a
            # Normal to edge (pointing outward for CCW polygon)
            normal = np.array([-edge[1], edge[0]])
            normal = normal / (np.linalg.norm(normal) + 1e-15)
            
            # Project all vertices of both polygons onto this normal
            proj1 = np.dot(verts1, normal)
            proj2 = np.dot(verts2, normal)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Check for separation
            if max1 < min2 - 1e-10:
                return 0.0  # Overlapping or touching
            if max2 < min1 - 1e-10:
                return 0.0
            
            # Gap along this axis
            gap = max(0, min2 - max1, min1 - max2)
            # The actual distance is at least this gap, but we need the true minimum
            # For convex polygons, the minimum distance is the minimum over all edge pairs
            # This is an approximation; for better accuracy, we compute vertex-edge distances
    
    # Compute minimum distance as min of vertex-to-edge distances
    min_dist = float('inf')
    for v1 in verts1:
        for i in range(len(verts2)):
            a = verts2[i]
            b = verts2[(i + 1) % len(verts2)]
            edge = b - a
            edge_len = np.linalg.norm(edge)
            if edge_len < 1e-15:
                continue
            t = np.dot(v1 - a, edge) / (edge_len * edge_len)
            t = np.clip(t, 0, 1)
            closest = a + t * edge
            d = np.linalg.norm(v1 - closest)
            min_dist = min(min_dist, d)
    
    for v2 in verts2:
        for i in range(len(verts1)):
            a = verts1[i]
            b = verts1[(i + 1) % len(verts1)]
            edge = b - a
            edge_len = np.linalg.norm(edge)
            if edge_len < 1e-15:
                continue
            t = np.dot(v2 - a, edge) / (edge_len * edge_len)
            t = np.clip(t, 0, 1)
            closest = a + t * edge
            d = np.linalg.norm(v2 - closest)
            min_dist = min(min_dist, d)
    
    return min_dist

def _min_distance_to_hexagon_boundary(centers, outer_center, outer_side, outer_angle):
    """Compute minimum distance from any inner hexagon vertex to outer hexagon boundary."""
    outer_verts = _hexagon_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    
    min_dist = float('inf')
    for i in range(N):
        inner_verts = _hexagon_vertices(centers[i, 0], centers[i, 1], 1.0, 0.0)
        for v in inner_verts:
            # Distance from point v to outer hexagon (negative if outside)
            # Find minimum distance to any edge of outer hexagon
            min_point_dist = float('inf')
            for j in range(6):
                a = outer_verts[j]
                b = outer_verts[(j + 1) % 6]
                edge = b - a
                edge_len_sq = np.dot(edge, edge)
                if edge_len_sq < 1e-15:
                    continue
                t = np.dot(v - a, edge) / edge_len_sq
                t = np.clip(t, 0, 1)
                closest = a + t * edge
                d = np.linalg.norm(v - closest)
                min_point_dist = min(min_point_dist, d)
            # Check if point is inside outer hexagon
            if not _polygon_contains_point(outer_verts, v[0], v[1]):
                min_dist = min(min_dist, -min_point_dist)  # Negative means outside
            else:
                min_dist = min(min_dist, min_point_dist)
    
    return min_dist

def _check_validity(centers, outer_center, outer_side, outer_angle):
    """Check if packing is valid: no overlaps between inner hexagons, all inside outer."""
    # Check inner hexagons don't overlap
    for i in range(N):
        for j in range(i + 1, N):
            vi = _hexagon_vertices(centers[i, 0], centers[i, 1], 1.0, 0.0)
            vj = _hexagon_vertices(centers[j, 0], centers[j, 1], 1.0, 0.0)
            # Check if they overlap using SAT
            overlap = False
            for verts in [vi, vj]:
                for k in range(6):
                    a = verts[k]
                    b = verts[(k + 1) % 6]
                    edge = b - a
                    normal = np.array([-edge[1], edge[0]])
                    normal = normal / (np.linalg.norm(normal) + 1e-15)
                    proj1 = np.dot(vi, normal)
                    proj2 = np.dot(vj, normal)
                    if np.max(proj1) < np.min(proj2) - 1e-8 or np.max(proj2) < np.min(proj1) - 1e-8:
                        overlap = True
                        break
                if overlap:
                    break
            if not overlap:
                continue
            # They overlap, check if actually intersecting (not just touching)
            # Compute intersection area
            inter_area = _polygon_overlap_area(vi, vj)
            if inter_area > 1e-6:
                return False
    
    # Check all inner hexagons are inside outer
    outer_verts = _hexagon_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    for i in range(N):
        inner_verts = _hexagon_vertices(centers[i, 0], centers[i, 1], 1.0, 0.0)
        for v in inner_verts:
            if not _polygon_contains_point(outer_verts, v[0], v[1]):
                return False
    
    return True

def _compute_cost(centers, outer_center, outer_side, outer_angle):
    """Compute cost: outer side length + penalties for invalidity."""
    # Penalty for inner hexagons overlapping
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            vi = _hexagon_vertices(centers[i, 0], centers[i, 1], 1.0, 0.0)
            vj = _hexagon_vertices(centers[j, 0], centers[j, 1], 1.0, 0.0)
            inter_area = _polygon_overlap_area(vi, vj)
            penalty += max(0, inter_area)
    
    # Penalty for inner hexagons outside outer
    outer_verts = _hexagon_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    for i in range(N):
        inner_verts = _hexagon_vertices(centers[i, 0], centers[i, 1], 1.0, 0.0)
        for v in inner_verts:
            if not _polygon_contains_point(outer_verts, v[0], v[1]):
                # Compute how far outside
                min_dist = float('inf')
                for j in range(6):
                    a = outer_verts[j]
                    b = outer_verts[(j + 1) % 6]
                    edge = b - a
                    edge_len_sq = np.dot(edge, edge)
                    if edge_len_sq < 1e-15:
                        continue
                    t = np.dot(v - a, edge) / edge_len_sq
                    t = np.clip(t, 0, 1)
                    closest = a + t * edge
                    d = np.linalg.norm(v - closest)
                    min_dist = min(min_dist, d)
                penalty += max(0, min_dist) ** 2
    
    return outer_side + 10.0 * penalty

def construct():
    """Construct a valid packing of 12 unit hexagons inside a regular hexagon."""
    np.random.seed(42)
    
    # Initialize with a reasonable guess: hexagonal lattice arrangement
    # Place hexagons in a honeycomb-like pattern
    centers = np.zeros((N, 2))
    
    # Hexagonal grid positions for 12 hexagons
    # Row 0: 3 hexagons
    # Row 1: 4 hexagons  
    # Row 2: 3 hexagons
    # Row 3: 2 hexagons
    # Actually, let's use a different arrangement
    # Use a 4-3-4-1 or similar pattern
    
    # Better: use a circular/hexagonal arrangement
    # Center + 6 around it + 5 more
    spacing = 2.0 * np.sqrt(3) / 2 * 1.0  # Distance between centers in hex lattice for unit hexagons
    
    # For unit hexagons, center-to-center distance in hex lattice is 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3)
    d = np.sqrt(3)
    
    # Position 0: center
    centers[0] = [0.0, 0.0]
    
    # Positions 1-6: hexagonal ring around center
    for i in range(6):
        angle = np.pi / 3 * i
        centers[i + 1] = [d * np.cos(angle), d * np.sin(angle)]
    
    # Positions 7-11: second ring
    for i in range(5):
        angle = np.pi / 3 * i + np.pi / 6
        centers[i + 7] = [2 * d * np.cos(angle), 2 * d * np.sin(angle)]
    
    outer_center = np.array([0.0, 0.0])
    outer_side = 5.0
    outer_angle = 0.0
    
    # Optimize using gradient descent
    best_cost = _compute_cost(centers, outer_center, outer_side, outer_angle)
    best_centers = centers.copy()
    best_outer_center = outer_center.copy()
    best_outer_side = outer_side
    best_outer_angle = outer_angle
    
    # Simulated annealing / random perturbation approach
    for iteration in range(5000):
        # Try to reduce outer side
        if iteration % 100 == 0:
            # Try reducing outer side
            new_side = outer_side * 0.98
            if new_side > 3.0:
                cost = _compute_cost(centers, outer_center, new_side, outer_angle)
                if cost < best_cost:
                    outer_side = new_side
                    best_cost = cost
                    best_centers = centers.copy()
                    best_outer_center = outer_center.copy()
                    best_outer_side = outer_side
                    best_outer_angle = outer_angle
        
        # Perturb centers
        if iteration % 5 == 0:
            idx = np.random.randint(N)
            perturb = np.random.normal(0, 0.1, 2)
            new_centers = centers.copy()
            new_centers[idx] += perturb
            cost = _compute_cost(new_centers, outer_center, outer_side, outer_angle)
            if cost < best_cost:
                centers = new_centers
                best_cost = cost
                best_centers = centers.copy()
                best_outer_center = outer_center.copy()
                best_outer_side = outer_side
                best_outer_angle = outer_angle
        
        # Perturb outer center
        if iteration % 10 == 0:
            perturb = np.random.normal(0, 0.05, 2)
            new_oc = outer_center + perturb
            cost = _compute_cost(centers, new_oc, outer_side, outer_angle)
            if cost < best_cost:
                outer_center = new_oc
                best_cost = cost
                best_centers = centers.copy()
                best_outer_center = outer_center.copy()
                best_outer_side = outer_side
                best_outer_angle = outer_angle
    
    # Final validation and adjustment
    # Ensure validity by expanding outer if needed
    for _ in range(100):
        if _check_validity(best_centers, best_outer_center, best_outer_side, best_outer_angle):
            break
        best_outer_side *= 1.02
    
    # Format inner hexagons
    inner = []
    for i in range(N):
        inner.append([float(best_centers[i, 0]), float(best_centers[i, 1]), 0.0])
    
    return inner, (float(best_outer_center[0]), float(best_outer_center[1])), float(best_outer_side), float(best_outer_angle)
