"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 12

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def hex_corners(cx, cy, side, angle_deg):
    """Return the 6 corners of a regular hexagon centered at (cx,cy) with given side and rotation."""
    angle = np.radians(angle_deg)
    # Flat-top hexagon: vertices at angles 0, 60, 120, 180, 240, 300 degrees from center
    # But we want pointy-top or flat-top? Let's use standard: vertices at 0, 60, ..., 300
    angles = np.linspace(0, 2*np.pi, 7)[:-1] + angle
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack([x, y])

def hex_corners_flat(cx, cy, side, angle_deg):
    """Flat-top hexagon: vertices at 30, 90, 150, 210, 270, 330 degrees."""
    angle = np.radians(angle_deg)
    angles = np.linspace(0, 2*np.pi, 7)[:-1] + angle + np.pi/6
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack([x, y])

def check_disjoint(corners_list):
    """Check if all hexagons are pairwise disjoint (no overlapping interiors).
    Using separation axis theorem for convex polygons."""
    n = len(corners_list)
    for i in range(n):
        for j in range(i+1, n):
            if not _separated(corners_list[i], corners_list[j]):
                return False
    return True

def _separated(poly1, poly2):
    """Check if two convex polygons are separated (no intersection)."""
    # Try all edges from both polygons as potential separating axes
    all_edges = []
    for poly in [poly1, poly2]:
        for k in range(len(poly)):
            p1 = poly[k]
            p2 = poly[(k+1) % len(poly)]
            edge = p2 - p1
            # Normal to edge
            normal = np.array([-edge[1], edge[0]])
            if np.linalg.norm(normal) > 1e-12:
                normal = normal / np.linalg.norm(normal)
                all_edges.append(normal)
    
    for normal in all_edges:
        proj1 = np.dot(poly1, normal)
        proj2 = np.dot(poly2, normal)
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        # If there's a gap, they're separated
        if max1 < min2 or max2 < min1:
            return True
    return False

def contains_in_outer(inner_corners, outer_corners):
    """Check if all inner hexagons are contained within the outer hexagon."""
    # For each inner corner, check if it's inside the outer hexagon
    for corners in inner_corners:
        for point in corners:
            if not _point_in_convex_polygon(point, outer_corners):
                return False
    return True

def _point_in_convex_polygon(point, polygon):
    """Check if point is inside convex polygon using cross products."""
    n = len(polygon)
    for i in range(n):
        p1 = polygon[i]
        p2 = polygon[(i+1) % n]
        edge = p2 - p1
        normal = np.array([edge[1], -edge[0]])  # Right normal
        v = point - p1
        if np.dot(normal, v) < -1e-9:  # Point is on the wrong side
            return False
    return True

def construct():
    """Place 12 unit hexagons inside a regular hexagon, minimizing outer side length."""
    # We'll use a grid-based approach with local optimization
    # Start with a reasonable initial configuration
    
    # For 12 hexagons, a good arrangement is 3 rows of 4, or similar
    # Let's try a honeycomb-like packing
    
    # Initial guess: place hexagons in a grid pattern
    # Hexagonal side = 1, so distance between centers in honeycomb = 2*sqrt(3)/2 * 2 = sqrt(3) * 2?
    # Actually for flat-top hexagons of side 1:
    # - Horizontal spacing: 1.5 (for adjacent columns in same row)
    # - Vertical spacing: sqrt(3) (for rows)
    # - Column offset: sqrt(3)/2
    
    # Let's try a different approach: use optimization
    
    # Parameterize: 12 hexagons, each with (x, y, angle)
    # Outer hexagon: center (0,0) for simplicity, side length s, rotation 0
    
    def objective(params):
        # params: [x1,y1,a1, x2,y2,a2, ..., x12,y22,a12, s]
        # Last param is outer side length
        positions = params[:36].reshape(12, 3)
        outer_side = params[36]
        
        if outer_side < 0:
            return 1e10
        
        # Compute inner corners
        inner_corners = []
        for x, y, angle_deg in positions:
            corners = hex_corners_flat(x, y, 1.0, angle_deg)
            inner_corners.append(corners)
        
        # Check disjointness
        if not check_disjoint(inner_corners):
            # Penalty for overlap
            penalty = 0
            for i in range(12):
                for j in range(i+1, 12):
                    # Compute minimum distance between two convex polygons
                    # Approximate by checking if any vertices are inside the other
                    for p in inner_corners[i]:
                        if _point_in_convex_polygon(p, inner_corners[j]):
                            penalty += 10.0
                    for p in inner_corners[j]:
                        if _point_in_convex_polygon(p, inner_corners[i]):
                            penalty += 10.0
            return penalty + outer_side
        
        # Compute outer corners
        outer_corners = hex_corners_flat(0.0, 0.0, outer_side, 0.0)
        
        # Check containment
        if not contains_in_outer(inner_corners, outer_corners):
            # Penalty for being outside
            penalty = 0
            for corners in inner_corners:
                for point in corners:
                    if not _point_in_convex_polygon(point, outer_corners):
                        # Distance to boundary approximation
                        penalty += 10.0
            return penalty + outer_side
        
        return outer_side
    
    # Initial guess
    # Place hexagons in a honeycomb pattern
    positions = []
    # 3 rows x 4 columns arrangement
    for row in range(3):
        for col in range(4):
            x = col * 1.5 + (0.75 if row % 2 == 1 else 0)
            y = row * np.sqrt(3)
            positions.append([x, y, 0.0])
    
    # Center the arrangement
    cx = np.mean([p[0] for p in positions])
    cy = np.mean([p[1] for p in positions])
    for p in positions:
        p[0] -= cx
        p[1] -= cy
    
    # Initial outer side estimate
    max_dist = max(np.linalg.norm([p[0], p[1]]) for p in positions)
    outer_side_init = max_dist + 2.0
    
    x0 = np.array(positions).flatten()
    x0 = np.append(x0, outer_side_init)
    
    # Optimize
    try:
        result = minimize(objective, x0, method='L-BFGS-B', 
                          options={'maxiter': 500, 'ftol': 1e-6})
        
        if result.success or result.fun < outer_side_init:
            params = result.x
            positions_opt = params[:36].reshape(12, 3)
            outer_side = params[36]
            
            # Final validation
            inner_corners = []
            for x, y, angle_deg in positions_opt:
                corners = hex_corners_flat(x, y, 1.0, angle_deg)
                inner_corners.append(corners)
            
            if check_disjoint(inner_corners):
                outer_corners = hex_corners_flat(0.0, 0.0, outer_side, 0.0)
                if contains_in_outer(inner_corners, outer_corners):
                    inner_list = [[p[0], p[1], p[2]] for p in positions_opt]
                    return inner_list, [0.0, 0.0], outer_side, 0.0
    except:
        pass
    
    # Fallback: return a valid but larger configuration
    # Use the initial honeycomb with enough margin
    max_dist = max(np.linalg.norm([p[0], p[1]]) for p in positions)
    outer_side = max_dist + 2.5
    
    inner_list = [[p[0], p[1], p[2]] for p in positions]
    return inner_list, [0.0, 0.0], outer_side, 0.0

# EVOLVE-BLOCK-END
