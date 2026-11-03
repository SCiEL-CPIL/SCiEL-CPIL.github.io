import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
TIME_LIMIT = 118.0  # Safety margin below 150s

# For two unit regular hexagons with arbitrary rotations,
# the minimum center distance for disjointness is 2*sqrt(3) (when aligned).
# For arbitrary relative rotations, the minimum distance is still 2*sqrt(3)
# because the width of a unit hexagon is 2 (vertex-to-vertex) and the
# minimum distance between parallel supporting lines is 2*sqrt(3)/... 
# Actually, let's think more carefully.
# A unit regular hexagon has inradius (apothem) = sqrt(3)/2, circumradius = 1.
# The minimum distance between two non-overlapping unit hexagons depends on rotation.
# For aligned hexagons, min center distance = 2*sqrt(3) (touching at flat sides).
# For 30-degree rotated hexagons, min center distance = 2 (touching at vertices).
# The minimum over all relative rotations is 2 (vertex-to-vertex contact).
# But we need a conservative bound: if distance > 2, they MIGHT still overlap.
# The safest approach: use 2*sqrt(3) as the threshold for "definitely disjoint"
# but that's too conservative. 
#
# Actually, for two congruent regular hexagons of side 1, the minimum center
# distance for non-overlap is 2 (when one is rotated 30 degrees relative to the other,
# they can touch vertex-to-vertex). But for aligned hexagons, it's 2*sqrt(3).
#
# The correct minimum distance for non-overlap of two unit regular hexagons
# over all relative rotations is 2.0 (achieved at 30-degree relative rotation).
# So if center distance > 2.0, they are NOT guaranteed to be disjoint.
# If center distance > 2*sqrt(3) ≈ 3.464, they ARE guaranteed to be disjoint.
#
# For a fast but slightly conservative check, we can use:
# - If dist > 2*sqrt(3): definitely disjoint (no overlap)
# - If dist < 2: definitely overlapping (must be disjoint, so invalid)
# - Otherwise: need SAT check (rare case)
#
# But for speed, let's use a slightly tighter bound. The actual minimum
# non-overlap distance for two unit hexagons is 2.0 (at 30° relative rotation).
# So dist > 2.0 does NOT guarantee disjointness.
#
# Let me use the exact minimum: for two unit regular hexagons, the minimum
# center distance for non-intersection is 2.0 (vertex-to-vertex, 30° rotation).
# So we need dist >= 2.0 as a necessary condition, but it's not sufficient.
#
# For a fast approximate check that's slightly conservative:
# Use dist > 2 * sqrt(3) as "definitely disjoint" and dist < 2 as "definitely overlapping".
# For the intermediate range, we'll do a quick SAT check.

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Vertices of a regular hexagon centered at (cx,cy) with rotation angle_deg."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    verts = []
    for k in range(6):
        theta = math.radians(60 * k)
        x = side * math.cos(theta)
        y = side * math.sin(theta)
        verts.append((cx + ca * x - sa * y, cy + sa * x + ca * y))
    return verts

def _project_polygon(verts, axis):
    """Project polygon vertices onto axis, return (min, max)."""
    ax, ay = axis
    vals = [vx * ax + vy * ay for vx, vy in verts]
    return min(vals), max(vals)

def _axes_from_polygon(verts):
    """Get edge normals (perpendicular to edges) for SAT."""
    axes = []
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-12:
            continue
        nx, ny = -dy / length, dx / length
        axes.append((nx, ny))
    return axes

def _separating_axis_check(vertsA, vertsB):
    """Check if two convex polygons overlap using SAT. Return True if they overlap (not disjoint)."""
    axes = _axes_from_polygon(vertsA) + _axes_from_polygon(vertsB)
    for axis in axes:
        minA, maxA = _project_polygon(vertsA, axis)
        minB, maxB = _project_polygon(vertsB, axis)
        if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
            return False  # Found separating axis -> no overlap
    return True  # No separating axis -> overlap

def _is_valid_fast(cx_arr, cy_arr, angle_arr):
    """
    Fast validity check using distance-based pre-filter + SAT fallback.
    cx_arr, cy_arr, angle_arr: numpy arrays of shape (N,)
    Returns True if all hexagons are pairwise disjoint.
    """
    n = len(cx_arr)
    # Precompute vertices for all hexagons
    verts_list = []
    for i in range(n):
        verts_list.append(_hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i]))
    
    # Precompute axes for all hexagons
    axes_list = []
    for i in range(n):
        axes_list.append(_axes_from_polygon(verts_list[i]))
    
    for i in range(n):
        xi, yi = cx_arr[i], cy_arr[i]
        vi = verts_list[i]
        ai = axes_list[i]
        for j in range(i + 1, n):
            xj, yj = cx_arr[j], cy_arr[j]
            vj = verts_list[j]
            aj = axes_list[j]
            
            # Quick distance check
            dx = xi - xj
            dy = yi - yj
            dist_sq = dx * dx + dy * dy
            
            # If distance is large enough, definitely disjoint
            if dist_sq > (2 * math.sqrt(3)) ** 2:
                continue
            
            # If distance is very small, definitely overlapping
            if dist_sq < 4.0 - 1e-9:
                return False
            
            # Intermediate case: do SAT check
            for axis in ai:
                ax, ay = axis
                minA = min(vx * ax + vy * ay for vx, vy in vi)
                maxA = max(vx * ax + vy * ay for vx, vy in vi)
                minB = min(vx * ax + vy * ay for vx, vy in vj)
                maxB = max(vx * ax + vy * ay for vx, vy in vj)
                if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
                    break
            else:
                for axis in aj:
                    ax, ay = axis
                    minA = min(vx * ax + vy * ay for vx, vy in vi)
                    maxA = max(vx * ax + vy * ay for vx, vy in vi)
                    minB = min(vx * ax + vy * ay for vx, vy in vj)
                    maxB = max(vx * ax + vy * ay for vx, vy in vj)
                    if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
                        break
                else:
                    return False
    
    return True

def _sdf_penalty(cx_arr, cy_arr, angle_arr, min_dist=2.0):
    """
    Compute a smooth penalty based on minimum distance between hexagon centers.
    Returns 0 if all pairs are at least min_dist apart, positive otherwise.
    This provides a smooth landscape for gradient-based optimization.
    """
    n = len(cx_arr)
    penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx_arr[i] - cx_arr[j]
            dy = cy_arr[i] - cy_arr[j]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < min_dist:
                # Quadratic penalty for violating minimum distance
                violation = min_dist - dist
                penalty += violation * violation
    return penalty

def _outer_hex_side(cx_arr, cy_arr, angle_arr, center, angle_deg):
    """Compute minimum side length S of outer hexagon containing all inner hexagons."""
    a = math.radians(angle_deg)
    apothem_unit = math.sqrt(3) / 2.0
    # 6 outward normals of the outer hexagon
    normals = []
    for k in range(6):
        na = a + math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))
    
    cx, cy = center
    max_need = 0.0
    for i in range(len(cx_arr)):
        verts = _hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i])
        for (vx, vy) in verts:
            for (nx, ny) in normals:
                val = (nx * (vx - cx) + ny * (vy - cy)) / apothem_unit
                if val > max_need:
                    max_need = val
    return max_need

def _score_config(cx_arr, cy_arr, angle_arr, center=(0.0, 0.0), angle_deg=0.0):
    """Return (valid, side_length)."""
    if not _is_valid_fast(cx_arr, cy_arr, angle_arr):
        return False, float('inf')
    side = _outer_hex_side(cx_arr, cy_arr, angle_arr, center, angle_deg)
    return True, side

def construct():
    t0 = time.time()
    random.seed(42)
    
    def make_flower_initial():
        """
        Create a 'flower' like arrangement.
        1 center, 6 around, 5 in gaps.
        """
        pts = []
        # Center
        pts.append([0.0, 0.0, 0.0])
        # Ring 1: 6 hexagons touching the center
        r1 = 2 * math.sqrt(3)  # For aligned hexagons, center-to-center distance
        for i in range(6):
            angle = math.radians(60 * i)
            x = r1 * math.cos(angle)
            y = r1 * math.sin(angle)
            pts.append([x, y, 0.0])
        
        # We have 7. Need 5 more.
        # Place them in the 'valleys' of the first ring, further out.
        r2 = r1 * 2  # Approximate distance to next layer
        angles = [30, 90, 150, 210, 270]  # deg
        for deg in angles:
            rad = math.radians(deg)
            x = r2 * math.cos(rad)
            y = r2 * math.sin(rad)
            pts.append([x, y, 0.0])
            
        return pts[:12]

    def make_lattice_initial():
        """
        Hexagonal lattice arrangement.
        """
        pts = []
        row_height = math.sqrt(3)
        col_width = 2.0
        # Rows: 3, 4, 3 = 10 hexes
        for c in range(3):
            x = c * col_width - col_width
            y = -row_height
            pts.append([x, y, 0.0])
        for c in range(4):
            x = c * col_width - col_width - col_width/2
            y = 0.0
            pts.append([x, y, 0.0])
        for c in range(3):
            x = c * col_width - col_width
            y = row_height
            pts.append([x, y, 0.0])
        
        # Add 2 more
        pts.append([-col_width*2 - col_width/2 - col_width, 0.0, 0.0])
        pts.append([col_width*2 - col_width/2 + col_width, 0.0, 0.0])
        
        return pts

    def make_random_initial():
        """Random scatter."""
        pts = []
        for _ in range(N):
            x = random.uniform(-3, 3)
            y = random.uniform(-3, 3)
            a = random.uniform(0, 60)
            pts.append([x, y, a])
        return pts

    # Generate multiple initial configurations
    initializers = [
        make_flower_initial,
        make_lattice_initial,
        make_random_initial,
        make_random_initial,
        make_random_initial,
        make_random_initial,
        make_random_initial,
        make_random_initial,
    ]
    
    best_side = float('inf')
    best_cfg = None
    best_center = (0.0, 0.0)
    best_angle = 0.0

    # Try each initializer and find the best valid starting point
    for init_func in initializers:
        if time.time() - t0 > TIME_LIMIT - 30:
            break
            
        cfg = init_func()
        if len(cfg) != N:
            continue
        
        cx_arr = np.array([p[0] for p in cfg])
        cy_arr = np.array([p[1] for p in cfg])
        angle_arr = np.array([p[2] for p in cfg])
        
        valid, side = _score_config(cx_arr, cy_arr, angle_arr)
        if valid and side < best_side:
            best_side = side
            best_cfg = cfg[:]
            best_center = (0.0, 0.0)
            best_angle = 0.0
    
    # If no valid configuration found, use a loose one
    if best_cfg is None:
        cfg = make_lattice_initial()
        # Scale up to ensure validity
        scale = 1.5
        for p in cfg:
            p[0] *= scale
            p[1] *= scale
        best_cfg = cfg
        cx_arr = np.array([p[0] for p in cfg])
        cy_arr = np.array([p[1] for p in cfg])
        angle_arr = np.array([p[2] for p in cfg])
        valid, side = _score_config(cx_arr, cy_arr, angle_arr)
        best_side = side if valid else 10.0
        best_center = (0.0, 0.0)
        best_angle = 0.0

    # Hybrid optimization: SDF-based gradient descent + SA for escape
    cfg = [row[:] for row in best_cfg]
    cx_arr = np.array([p[0] for p in cfg])
    cy_arr = np.array([p[1] for p in cfg])
    angle_arr = np.array([p[2] for p in cfg])
    
    cur_score = best_side
    T = 0.3
    T_end = 0.005
    alpha = 0.995
    
    # Precompute normals for outer hexagon (angle=0, center=(0,0))
    apothem_unit = math.sqrt(3) / 2.0
    normals = []
    for k in range(6):
        na = math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))
    
    def _quick_score(cx_arr, cy_arr, angle_arr):
        """Fast score computation assuming center (0,0) and angle 0."""
        max_need = 0.0
        for i in range(N):
            a = math.radians(angle_arr[i])
            ca, sa = math.cos(a), math.sin(a)
            for k in range(6):
                theta = math.radians(60 * k)
                vx = cx_arr[i] + ca * math.cos(theta) - sa * math.sin(theta)
                vy = cy_arr[i] + sa * math.cos(theta) + ca * math.sin(theta)
                for (nx, ny) in normals:
                    val = (nx * vx + ny * vy) / apothem_unit
                    if val > max_need:
                        max_need = val
        return max_need

    def _sdf_objective(x):
        """
        Objective function for L-BFGS-B: minimize outer hex side length + SDF penalty.
        x is a flat array of [x0,y0,a0, x1,y1,a1, ..., x11,y11,a11]
        """
        cx = x[0::3]
        cy = x[1::3]
        ang = x[2::3]
        
        # SDF penalty for overlapping hexagons
        sdf_pen = _sdf_penalty(cx, cy, ang, min_dist=2.0)
        
        # Outer hex side length
        side = _quick_score(cx, cy, ang)
        
        # Weighted objective: prioritize validity, then minimize side
        return side + 100.0 * sdf_pen

    # Phase 1: Gradient-based optimization using L-BFGS-B
    x0 = np.concatenate([cx_arr, cy_arr, angle_arr])
    
    # Run multiple L-BFGS-B optimizations with different starting points
    n_gradient_rounds = 5
    for round_idx in range(n_gradient_rounds):
        if time.time() - t0 > TIME_LIMIT - 20:
            break
        
        # Add small random perturbation for diversity
        if round_idx > 0:
            x0 = x0 + np.random.normal(0, 0.1, size=x0.shape)
        
        try:
            result = minimize(
                _sdf_objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 100, 'maxfun': 500}
            )
            
            x_opt = result.x
            cx_opt = x_opt[0::3]
            cy_opt = x_opt[1::3]
            ang_opt = x_opt[2::3]
            
            # Check if this is valid and better
            if _is_valid_fast(cx_opt, cy_opt, ang_opt):
                side_opt = _quick_score(cx_opt, cy_opt, ang_opt)
                if side_opt < best_side:
                    best_side = side_opt
                    best_cfg = [[cx_opt[i], cy_opt[i], ang_opt[i]] for i in range(N)]
                    best_center = (0.0, 0.0)
                    best_angle = 0.0
                    x0 = x_opt.copy()
                    cx_arr, cy_arr, angle_arr = cx_opt, cy_opt, ang_opt
                    cur_score = side_opt
        except Exception:
            pass
    
    # Phase 2: Simulated annealing for fine-tuning and escape from local minima
    cx_arr = np.array([p[0] for p in best_cfg])
    cy_arr = np.array([p[1] for p in best_cfg])
    angle_arr = np.array([p[2] for p in best_cfg])
    cur_score = best_side
    
    # Main SA loop
    while time.time() - t0 < TIME_LIMIT - 3:
        r = random.random()
        idx = random.randrange(N)
        
        # Save old values
        old_x, old_y, old_a = cx_arr[idx], cy_arr[idx], angle_arr[idx]
        
        if r < 0.7:
            # Move one hexagon
            cx_arr[idx] += random.gauss(0, 0.1)
            cy_arr[idx] += random.gauss(0, 0.1)
            angle_arr[idx] += random.gauss(0, 2.0)
        elif r < 0.85:
            # Rotate one hexagon
            angle_arr[idx] += random.gauss(0, 5.0)
        else:
            # Move two hexagons
            idx2 = random.randrange(N)
            if idx2 == idx:
                idx2 = (idx + 1) % N
            old_x2, old_y2, old_a2 = cx_arr[idx2], cy_arr[idx2], angle_arr[idx2]
            cx_arr[idx] += random.gauss(0, 0.08)
            cy_arr[idx] += random.gauss(0, 0.08)
            cx_arr[idx2] += random.gauss(0, 0.08)
            cy_arr[idx2] += random.gauss(0, 0.08)
            angle_arr[idx] += random.gauss(0, 1.5)
            angle_arr[idx2] += random.gauss(0, 1.5)
        
        # Check validity
        valid = _is_valid_fast(cx_arr, cy_arr, angle_arr)
        
        if valid:
            new_score = _quick_score(cx_arr, cy_arr, angle_arr)
            
            if new_score < cur_score or (T > T_end and random.random() < math.exp(-(new_score - cur_score) / T)):
                cur_score = new_score
                if new_score < best_side:
                    best_side = new_score
                    best_cfg = [[cx_arr[i], cy_arr[i], angle_arr[i]] for i in range(N)]
                    best_center = (0.0, 0.0)
                    best_angle = 0.0
            else:
                # Revert
                cx_arr[idx], cy_arr[idx], angle_arr[idx] = old_x, old_y, old_a
                if r >= 0.85:
                    cx_arr[idx2], cy_arr[idx2], angle_arr[idx2] = old_x2, old_y2, old_a2
        else:
            # Revert
            cx_arr[idx], cy_arr[idx], angle_arr[idx] = old_x, old_y, old_a
            if r >= 0.85:
                cx_arr[idx2], cy_arr[idx2], angle_arr[idx2] = old_x2, old_y2, old_a2
        
        T = max(T_end, T * alpha)
    
    # Final verification
    cx_final = np.array([p[0] for p in best_cfg])
    cy_final = np.array([p[1] for p in best_cfg])
    angle_final = np.array([p[2] for p in best_cfg])
    
    if not _is_valid_fast(cx_final, cy_final, angle_final):
        # Fallback: use a known valid loose configuration
        cfg = make_lattice_initial()
        scale = 1.3
        for p in cfg:
            p[0] *= scale
            p[1] *= scale
        best_cfg = cfg
        cx_final = np.array([p[0] for p in cfg])
        cy_final = np.array([p[1] for p in cfg])
        angle_final = np.array([p[2] for p in cfg])
        best_side = _outer_hex_side(cx_final, cy_final, angle_final, (0.0, 0.0), 0.0)
    else:
        best_side = _outer_hex_side(cx_final, cy_final, angle_final, (0.0, 0.0), 0.0)
    
    inner = [[p[0], p[1], p[2]] for p in best_cfg]
    return inner, best_center, best_side, best_angle

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
