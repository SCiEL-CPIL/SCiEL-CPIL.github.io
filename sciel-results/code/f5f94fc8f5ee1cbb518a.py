import numpy as np
import time

N = 12

def _hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    a = np.radians(angle_deg)
    angles = a + np.pi/6 + np.arange(6) * np.pi/3  # pointy-top orientation (vertices at 30, 90, ...)
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    sqrt3 = np.sqrt(3)
    
    # Generate candidate configurations
    # Config 1: 3 rows, 4 cols honeycomb
    # Config 2: 4 rows, 3 cols honeycomb
    # Config 3: 3 rows, 4 cols with slight shift to optimize bounding box?
    
    candidates = []
    
    # Helper to generate honeycomb grid
    def gen_honeycomb(rows, cols, spacing_x, spacing_y, shift_x):
        pts = []
        for r in range(rows):
            for c in range(cols):
                x = c * spacing_x + (r % 2) * shift_x
                y = r * spacing_y
                pts.append([x, y, 0.0])
        # Center
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        cx_off = (min(xs) + max(xs)) / 2
        cy_off = (min(ys) + max(ys)) / 2
        for p in pts:
            p[0] -= cx_off
            p[1] -= cy_off
        return pts

    # Use a slight expansion to ensure no overlap (epsilon)
    eps = 1e-6
    sx = sqrt3 * (1 + eps)
    sy = 1.5 * (1 + eps)
    shift = (sqrt3 / 2) * (1 + eps)
    
    # 3x4 Grid
    inner_34 = gen_honeycomb(3, 4, sx, sy, shift)
    candidates.append(inner_34)
    
    # 4x3 Grid
    inner_43 = gen_honeycomb(4, 3, sx, sy, shift)
    candidates.append(inner_43)
    
    # Try a 2x6 grid?
    inner_26 = gen_honeycomb(2, 6, sx, sy, shift)
    candidates.append(inner_26)
    
    best_side = float('inf')
    best_inner = None
    best_center = (0,0)
    best_angle = 0.0
    
    for inner in candidates:
        if time.time() > deadline - 10:
            break
            
        # Get all vertices
        all_pts = []
        for x, y, ang in inner:
            verts = _hex_vertices(x, y, 1.0, ang)
            all_pts.extend(verts.tolist())
        pts = np.array(all_pts)
        
        # Evaluate two main orientations for enclosing hex
        # Orientation 0: Pointy-Top (aligned with inner)
        # Orientation 30: Flat-Top (rotated 30 deg)
        
        # For each orientation, we can analytically find the min side S
        # assuming center is at (0,0) because we centered the points.
        # However, the optimal center might not be (0,0) if the shape is asymmetric?
        # The honeycomb grids are centrally symmetric, so (0,0) is optimal for center.
        
        # Check Orientation 0 (Pointy-Top)
        # Constraints:
        # S >= 2*|x|/sqrt(3)
        # S >= |x/sqrt(3) + y|
        # S >= |x/sqrt(3) - y|
        
        s0_x = np.max(np.abs(pts[:, 0])) * 2 / sqrt3
        s0_y1 = np.max(np.abs(pts[:, 0] / sqrt3 + pts[:, 1]))
        s0_y2 = np.max(np.abs(pts[:, 0] / sqrt3 - pts[:, 1]))
        s0 = max(s0_x, s0_y1, s0_y2)
        
        if s0 < best_side:
            best_side = s0
            best_inner = inner
            best_center = (0.0, 0.0)
            best_angle = 0.0 # Pointy-Top is angle 0 in our vertex definition?
            # Our _hex_vertices uses angles = a + pi/6.
            # If a=0, vertices at 30, 90... -> Pointy-Top.
            # So enclosing hex angle 0 is Pointy-Top.
            
        # Check Orientation 30 (Flat-Top)
        # Rotate points by -30 deg? Or just change constraints?
        # If enclosing hex is Flat-Top, it is rotated 30 deg relative to Pointy-Top.
        # Constraints for Flat-Top (side S) centered at 0:
        # S >= 2*|y|/sqrt(3)
        # S >= |sqrt(3)*x + y| / sqrt(3)
        # S >= |sqrt(3)*x - y| / sqrt(3)
        
        s1_y = np.max(np.abs(pts[:, 1])) * 2 / sqrt3
        s1_x1 = np.max(np.abs(sqrt3 * pts[:, 0] + pts[:, 1])) / sqrt3
        s1_x2 = np.max(np.abs(sqrt3 * pts[:, 0] - pts[:, 1])) / sqrt3
        s1 = max(s1_y, s1_x1, s1_x2)
        
        if s1 < best_side:
            best_side = s1
            best_inner = inner
            best_center = (0.0, 0.0)
            best_angle = 30.0 # Flat-Top corresponds to angle 30 in our definition?
            # If a=30, angles = 30+30=60, 120, 180, 240, 300, 0.
            # Vertices at 0, 60, 120... -> Flat-Top.
            # So enclosing hex angle 30 is Flat-Top.

    # Add a tiny safety margin to the side length to avoid boundary issues
    best_side += 1e-5
    
    # Return best result
    return best_inner, best_center, best_side, best_angle
