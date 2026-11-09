"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Try to pack 12 unit hexagons in a honeycomb-like arrangement
    # and optimize the outer hexagon size
    
    # Unit hexagon vertices (side length 1, centered at origin, pointy-top)
    def hex_vertices(cx, cy, angle_deg, side=1.0):
        """Get vertices of a regular hexagon with given center, angle, side."""
        angle_rad = np.radians(angle_deg)
        # Pointy-top hexagon: vertices at angles 90, 150, 210, 270, 330, 30
        base_angles = np.array([90, 150, 210, 270, 330, 30]) * np.pi / 180
        angles = base_angles + angle_rad
        r = side  # distance from center to vertex = side length for regular hexagon
        vx = cx + r * np.cos(angles)
        vy = cy + r * np.sin(angles)
        return np.column_stack([vx, vy])
    
    def outer_hex_half_width(point, outer_center, outer_side, outer_angle_deg):
        """
        For a point, compute the maximum over 6 directions of the 
        projection relative to the outer hexagon's apothem.
        Returns the minimum outer side length needed so that the point is inside.
        """
        ox, oy = outer_center
        oa = np.radians(outer_angle_deg)
        # Outer hexagon normals (pointing outward from center to each side)
        # For pointy-top hexagon, sides are perpendicular to directions 0, 60, 120, 180, 240, 300
        normal_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180 + oa
        px, py = point
        dx = px - ox
        dy = py - oy
        # Project point onto each normal direction
        projections = np.array([dx * np.cos(na) + dy * np.sin(na) for na in normal_angles])
        # The apothem of outer hexagon with side L is L * sqrt(3)/2
        # So we need: max(|projection|) <= L * sqrt(3)/2
        # Thus L >= max(|projection|) / (sqrt(3)/2)
        max_proj = np.max(np.abs(projections))
        return max_proj / (np.sqrt(3) / 2)
    
    def objective(params):
        """
        params: [x0, y0, a0, x1, y1, a1, ..., x11, y11, a11, ox, oy, outer_angle]
        First 3*N: positions and rotations of inner hexagons
        Last 3: outer hexagon center and rotation
        Returns: the outer side length needed
        """
        inner_params = params[:3*N]
        outer_params = params[3*N:]
        ox, oy, outer_angle = outer_params
        
        max_L = 0.0
        for i in range(N):
            xi = inner_params[3*i]
            yi = inner_params[3*i+1]
            ai = inner_params[3*i+2]
            verts = hex_vertices(xi, yi, ai)
            for j in range(6):
                L_needed = outer_hex_half_width(verts[j], (ox, oy), 1.0, outer_angle)
                if L_needed > max_L:
                    max_L = L_needed
        return max_L
    
    # Initial guess: honeycomb arrangement
    # For unit hexagons in honeycomb, center distance = sqrt(3)
    # Arrange in a roughly hexagonal cluster
    sqrt3 = np.sqrt(3)
    
    # Try a 2-3-3-2-2 arrangement or similar
    # Let's try: center + 6 around + 5 more in a ring
    # Or simpler: rows of 2,3,3,2,2 = 12
    
    # Row-based honeycomb:
    # Row 0: 2 hexagons
    # Row 1: 3 hexagons  
    # Row 2: 3 hexagons
    # Row 3: 2 hexagons
    # Row 4: 2 hexagons
    # That's 12. But this isn't very symmetric.
    
    # Better: use a more symmetric arrangement
    # Central hexagon, then 6 in first ring, then 5 in second ring (partial)
    
    # Let's just use optimization with a reasonable initial guess
    # Initial: all hexagons in a line or small cluster
    
    np.random.seed(42)
    
    # Initial guess: honeycomb positions
    # Generate honeycomb positions for 12 hexagons
    # Axial coordinates for honeycomb
    # Let's place them in a compact hexagonal shape
    
    # For now, start with a simple arrangement and let optimization work
    # Place in a 3x4 grid with hexagonal offsets
    positions = []
    # Row 0: 4 hexagons
    for j in range(4):
        x = j * sqrt3
        y = 0
        positions.append([x, y, 0.0])
    # Row 1: 4 hexagons, offset
    for j in range(4):
        x = j * sqrt3 + sqrt3/2
        y = 1.5
        positions.append([x, y, 0.0])
    # Row 2: 4 hexagons
    for j in range(4):
        x = j * sqrt3
        y = 3.0
        positions.append([x, y, 0.0])
    
    # Center the arrangement
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    for p in positions:
        p[0] -= cx
        p[1] -= cy
    
    # Initial params
    params0 = []
    for p in positions:
        params0.extend([p[0], p[1], p[2]])
    params0.extend([0.0, 0.0, 0.0])  # outer center and angle
    
    # Optimize
    result = minimize(objective, params0, method='Nelder-Mead', 
                      options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8})
    
    best_params = result.x
    inner = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        a = best_params[3*i+2]
        inner.append([x, y, a])
    
    ox = best_params[3*N]
    oy = best_params[3*N+1]
    outer_angle = best_params[3*N+2]
    
    # Compute the actual outer side length
    L = objective(best_params)
    
    return inner, (ox, oy), L, outer_angle
# EVOLVE-BLOCK-END
