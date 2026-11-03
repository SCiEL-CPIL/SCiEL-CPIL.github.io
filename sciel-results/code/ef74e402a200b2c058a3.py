import numpy as np
import time
import math
from scipy.optimize import linprog

N = 11
_SQRT3 = math.sqrt(3)
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # safety margin before 150s kill

def _hex_vertices(cx, cy, angle_deg):
    """Vertices of a unit regular hexagon centered at (cx,cy) with rotation angle_deg."""
    a0 = math.radians(angle_deg)
    angles = a0 + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def _min_enclosing_hex_side(verts, outer_angle_deg):
    """
    Find minimum side length of a regular hexagon (with vertex at outer_angle_deg)
    that contains all points in verts (numpy array of shape (M,2)).
    Returns (cx, cy, side) or None.
    """
    # Normals to the 6 sides of the outer hexagon
    # Vertices at outer_angle + 60k => edge normals at outer_angle + 30 + 60k
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    M = len(verts)
    if M == 0:
        return 0.0, 0.0, 0.0
    
    # Variables: [cx, cy, R], minimize R
    # Constraints for each vertex v and normal n:
    #   (v - c).n <= R  =>  -n.c - R <= -v.n
    #   (c - v).n <= R  =>   n.c - R <=  v.n
    
    # v_dot_n: (M, 6)
    v_dot_n = verts @ normals.T
    
    A_rows = []
    b_vals = []
    
    for k in range(6):
        nx, ny = normals[k]
        v_dots = v_dot_n[:, k]
        
        # Row 1: -nx*cx - ny*cy - R <= -v_dot
        A_rows.append(np.column_stack([np.full(M, -nx), np.full(M, -ny), np.full(M, -1.0)]))
        b_vals.append(-v_dots)
        
        # Row 2: nx*cx + ny*cy - R <= v_dot
        A_rows.append(np.column_stack([np.full(M, nx), np.full(M, ny), np.full(M, -1.0)]))
        b_vals.append(v_dots)
    
    A = np.vstack(A_rows)
    b = np.concatenate(b_vals)
    
    # R >= 0
    A = np.vstack([A, [[0.0, 0.0, -1.0]]])
    b = np.append(b, 0.0)
    
    c_obj = [0.0, 0.0, 1.0]
    try:
        res = linprog(c_obj, A_ub=A, b_ub=b,
                      bounds=[(None, None), (None, None), (0, None)],
                      method='highs')
        if res.success:
            cx, cy, R = res.x
            side = R / (_SQRT3/2)
            return cx, cy, side
    except Exception:
        pass
    return None

def _build_hex_lattice_positions():
    """
    Build 11 positions on a hexagonal lattice.
    Uses axial coordinates for flat-top hexagons.
    Returns list of (x, y) tuples.
    """
    # For flat-top hexagons with side length 1:
    # Adjacent center distance = sqrt(3)
    # Axial to pixel: x = 1.5 * q, y = sqrt(3) * (r + q/2)
    # But we need to scale so that adjacent centers are sqrt(3) apart.
    # Actually, the standard axial to pixel for size s (circumradius) is:
    # x = s * 3/2 * q
    # y = s * sqrt(3) * (r + q/2)
    # With s=1, adjacent distance = sqrt((3/2)^2 + (sqrt(3)/2)^2) = sqrt(9/4 + 3/4) = sqrt(3). Correct.
    
    scale = 1.0
    positions_axial = [
        (0, 0),      # center
        (1, 0),      # ring 1
        (0, 1),
        (-1, 1),
        (-1, 0),
        (0, -1),
        (1, -1),
        # ring 2 - pick 4 to make compact shape
        (2, 0),
        (1, 1),
        (-2, 1),
        (-1, -1),
    ]
    
    positions = []
    for q, r in positions_axial:
        x = scale * 1.5 * q
        y = scale * _SQRT3 * (r + q/2)
        positions.append((x, y))
    
    return positions

def _build_row_positions():
    """
    Alternative layout: rows of hexagons.
    """
    scale = 1.0
    positions = []
    # Row 1: y = -1.5, x = -sqrt(3)/2, 0, sqrt(3)/2  (3 hexes)
    for k in range(-1, 2):
        x = k * _SQRT3 * scale
        y = -1.5 * scale
        positions.append((x, y))
    # Row 2: y = 0, x = -sqrt(3)/2, 0, sqrt(3)/2, sqrt(3)  (4 hexes)
    for k in range(-1, 3):
        x = (k - 0.5) * _SQRT3 * scale
        y = 0.0
        positions.append((x, y))
    # Row 3: y = 1.5, x = -sqrt(3)/2, 0, sqrt(3)/2  (3 hexes)
    for k in range(-1, 2):
        x = k * _SQRT3 * scale
        y = 1.5 * scale
        positions.append((x, y))
    
    return positions

def _compute_all_vertices(positions, inner_angle):
    """Compute all 6*N vertices for given positions and inner angle."""
    all_verts = []
    for px, py in positions:
        verts = _hex_vertices(px, py, inner_angle)
        all_verts.append(verts)
    return np.vstack(all_verts)

def _evaluate_configuration(positions, inner_angle, outer_angle):
    """
    Evaluate a configuration. Returns (side, center) or None if infeasible.
    """
    all_verts = _compute_all_vertices(positions, inner_angle)
    res = _min_enclosing_hex_side(all_verts, outer_angle)
    if res is not None:
        cx, cy, side = res
        return side, (cx, cy)
    return None

def _refine_with_optimization(positions, inner_angle, outer_angle):
    """
    Try to refine positions using scipy.optimize to reduce outer side.
    This is a structural change: we allow positions to move.
    """
    from scipy.optimize import minimize
    
    def objective(params):
        # params: 2*N values (x, y) for each hexagon
        pos = [(params[2*i], params[2*i+1]) for i in range(N)]
        # Check for overlaps (simple distance check)
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i][0] - pos[j][0]
                dy = pos[i][1] - pos[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < _SQRT3 - 0.1:  # allow small tolerance
                    return 1000.0 + (10.0 - dist)**2
        all_verts = _compute_all_vertices(pos, inner_angle)
        res = _min_enclosing_hex_side(all_verts, outer_angle)
        if res is None:
            return 1000.0
        return res[2]
    
    x0 = np.array([pos[0] for pos in positions] + [pos[1] for pos in positions])
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-6})
        if result.fun < 100.0:  # valid solution
            pos = [(result.x[2*i], result.x[2*i+1]) for i in range(N)]
            all_verts = _compute_all_vertices(pos, inner_angle)
            res = _min_enclosing_hex_side(all_verts, outer_angle)
            if res is not None:
                return res[2], res[0], res[1], pos
    except Exception:
        pass
    
    return None

def construct():
    global _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        # Fallback: return a valid but loose configuration
        positions = _build_hex_lattice_positions()
        inner = [[pos[0], pos[1], 0.0] for pos in positions]
        all_verts = _compute_all_vertices(positions, 0.0)
        res = _min_enclosing_hex_side(all_verts, 0.0)
        if res is not None:
            return inner, (res[0], res[1]), res[2], 0.0
        return inner, (0.0, 0.0), 10.0, 0.0
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    best_inner_angle = 0.0
    
    # Generate multiple layouts
    layouts = [
        _build_hex_lattice_positions(),
        _build_row_positions(),
    ]
    
    # Try different inner angles
    inner_angles = [0.0, 30.0]
    
    # Coarse search over outer angles
    outer_angles_coarse = np.arange(0, 30.1, 2.0)
    
    for layout_idx, positions in enumerate(layouts):
        if time.time() > _DEADLINE - 15.0:
            break
        
        for inner_angle in inner_angles:
            if time.time() > _DEADLINE - 15.0:
                break
            
            best_coarse_side = float('inf')
            best_coarse_angle = 0.0
            best_coarse_center = (0.0, 0.0)
            
            for outer_angle in outer_angles_coarse:
                if time.time() > _DEADLINE - 15.0:
                    break
                
                result = _evaluate_configuration(positions, inner_angle, outer_angle)
                if result is not None:
                    side, center = result
                    if side < best_coarse_side:
                        best_coarse_side = side
                        best_coarse_angle = outer_angle
                        best_coarse_center = center
            
            # Fine search around best coarse angle
            if best_coarse_side < float('inf'):
                fine_angles = np.arange(max(0, best_coarse_angle - 2.0), 
                                        min(30.1, best_coarse_angle + 2.0), 0.1)
                for outer_angle in fine_angles:
                    if time.time() > _DEADLINE - 15.0:
                        break
                    
                    result = _evaluate_configuration(positions, inner_angle, outer_angle)
                    if result is not None:
                        side, center = result
                        if side < best_coarse_side:
                            best_coarse_side = side
                            best_coarse_angle = outer_angle
                            best_coarse_center = center
            
            if best_coarse_side < best_side:
                best_side = best_coarse_side
                best_inner = [[pos[0], pos[1], inner_angle] for pos in positions]
                best_center = best_coarse_center
                best_outer_angle = best_coarse_angle
                best_inner_angle = inner_angle
        
        # Try optimization refinement on the best configuration for this layout
        if time.time() > _DEADLINE - 30.0:
            continue
        
        # Refine using optimization
        refinement = _refine_with_optimization(positions, best_inner_angle, best_outer_angle)
        if refinement is not None:
            side, cx, cy, pos = refinement
            if side < best_side:
                best_side = side
                best_inner = [[pos[i][0], pos[i][1], best_inner_angle] for i in range(N)]
                best_center = (cx, cy)
    
    if best_inner is None:
        # Final fallback
        positions = _build_hex_lattice_positions()
        best_inner = [[pos[0], pos[1], 0.0] for pos in positions]
        all_verts = _compute_all_vertices(positions, 0.0)
        res = _min_enclosing_hex_side(all_verts, 0.0)
        if res is not None:
            best_center = (res[0], res[1])
            best_side = res[2]
            best_outer_angle = 0.0
    
    return best_inner, best_center, best_side, best_outer_angle

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
