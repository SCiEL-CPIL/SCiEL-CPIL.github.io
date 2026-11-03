import math
import time
import numpy as np
from scipy.optimize import linprog

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 8.0
EPS = 1e-9

# Precompute the 6 unit normals for hexagon side directions (angle 0 outer)
NORMALS_0 = np.array([
    [math.cos(math.pi/6 + k*math.pi/3), math.sin(math.pi/6 + k*math.pi/3)]
    for k in range(6)
])

# Axial to cartesian
def axial_to_cartesian(axial_list):
    pts = []
    for q, r in axial_list:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        pts.append((x, y))
    return np.array(pts)

# Candidate axial configurations (same as before, but we'll also generate random ones)
BASE_CANDIDATES = [
    [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)],
    [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(-2,1),(2,-1)],
    [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)],
    [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3)],
    [(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(5,1)],
    [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(-2,2),(-1,3),(1,-2)],
    [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3)],
    [(0,0),(1,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3),(1,4)],
    [(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3),(3,1)],
    [(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3)],
    [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3)],
    [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3)],
    [(0,0),(1,0),(2,0),(3,0),(4,0),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)],
    [(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,2)],
    [(0,0),(1,0),(2,0),(1,1),(2,1),(3,1),(0,2),(1,2),(1,3),(2,3),(3,3)],
    [(0,0),(1,0),(2,0),(0,1),(1,1),(1,2),(2,2),(0,3),(1,3),(2,3),(1,4)],
    [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2)],
    [(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)],
    [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)],
    [(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2)],
    [(0,0),(1,0),(0,1),(1,1),(0,2),(1,2),(2,2),(1,3),(2,3),(0,4),(1,4)],
    [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2),(0,3),(1,3)],
    [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3)],
    [(0,0),(1,0),(2,0),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3),(2,3)],
    [(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(5,1)],
    [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3),(3,2)],
    [(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3),(3,3)],
    [(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(0,2),(1,2),(2,2),(3,2),(1,3)],
    [(0,0),(1,0),(2,0),(0,1),(1,1),(1,2),(2,2),(0,3),(1,3),(2,3),(1,4)],
    [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2),(1,3)],
    [(0,0),(1,0),(2,0),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3),(2,3),(3,3)],
]

def check_overlap_fast(pts, angles_deg):
    """
    Check if any two hexagons overlap.
    pts: (N, 2) array of centers
    angles_deg: (N,) array of rotation angles in degrees
    Two unit hexagons with rotations a1, a2 overlap iff there exists a direction
    such that the projection of the center difference is less than the sum of apothems
    in that direction... Actually for regular hexagons, the Minkowski sum is a hexagon
    (if same orientation) or a 12-gon (if different). 
    
    Simplest exact check: for each pair, check if the vector d = p_i - p_j is outside
    the Minkowski sum of the two hexagons. For regular hexagons of side 1, the
    Minkowski sum of two hexagons with rotations a1, a2 is a convex polygon.
    
    For same orientation (a1 = a2), the Minkowski sum is a hexagon of side 2, apothem sqrt(3).
    Overlap iff min over 6 normals |d . n_k| < sqrt(3).
    
    For different orientations, it's more complex. Let's handle the general case:
    A point p is inside a regular hexagon centered at c with rotation a and side 1 iff
    for all k=0..5: | (p-c) . n_k(a) | <= apothem, where n_k(a) = (cos(a + pi/6 + k*pi/3), sin(...))
    
    Two hexagons H1(c1, a1) and H2(c2, a2) are disjoint iff
    the convex sets H1 and H2 don't intersect.
    
    A practical check: H1 and H2 overlap iff there exists a direction u such that
    the projection of H1 on u and projection of H2 on u overlap, AND the same for
    perpendicular... No, that's not sufficient.
    
    Better: Use the separating axis theorem. Two convex polygons are disjoint iff
    there exists an axis (normal to an edge of either polygon) such that their
    projections on that axis are disjoint.
    
    For two regular hexagons, the candidate axes are the 6 normals of H1 and 6 of H2.
    """
    n_pts = len(pts)
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            d = pts[i] - pts[j]
            a1 = math.radians(angles_deg[i])
            a2 = math.radians(angles_deg[j])
            
            # Normals for hexagon i
            n1 = np.array([
                [math.cos(a1 + math.pi/6 + k*math.pi/3), math.sin(a1 + math.pi/6 + k*math.pi/3)]
                for k in range(6)
            ])
            # Normals for hexagon j
            n2 = np.array([
                [math.cos(a2 + math.pi/6 + k*math.pi/3), math.sin(a2 + math.pi/6 + k*math.pi/3)]
                for k in range(6)
            ])
            
            all_normals = np.vstack([n1, n2])
            
            # Project d onto each normal
            projs = np.abs(d @ all_normals.T)
            
            # For each axis, the "gap" between the two hexagons is:
            # gap = |d . n| - (apothem1 + apothem2) = |d . n| - 2*apothem
            # If for ANY axis, gap > 0, they are separated (disjoint).
            # They overlap iff for ALL axes, gap <= 0, i.e., |d.n| <= 2*apothem for all axes.
            # Wait, that's the condition for them to potentially overlap.
            # Actually: two convex sets are disjoint iff there exists a separating axis.
            # The separating axis must be normal to an edge of one of the polygons.
            # So if for ALL 12 candidate axes, the projections overlap (|d.n| <= 2*apothem),
            # then the hexagons overlap (or touch).
            
            if np.all(projs <= 2 * APOTHEM + 1e-9):
                return False  # Overlap detected
    return True


def enclosing_hexagon_side(pts, outer_angle):
    """
    Compute the minimum side length of a regular hexagon with given rotation
    (outer_angle) that encloses all points in pts.
    Returns (side_length, center) or (inf, None).
    """
    normals = np.array([
        [math.cos(outer_angle + math.pi/6 + k*math.pi/3), math.sin(outer_angle + math.pi/6 + k*math.pi/3)]
        for k in range(6)
    ])
    
    n_pts = len(pts)
    A_ub = []
    b_ub = []
    
    for nx, ny in normals:
        for px, py in pts:
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-(px * nx + py * ny))
            A_ub.append([nx, ny, -1.0])
            b_ub.append(px * nx + py * ny)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    c_obj = np.array([0.0, 0.0, 1.0])
    bounds = [(None, None), (None, None), (0, None)]
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status == 0:
            D = res.x[2]
            cx, cy = res.x[0], res.x[1]
            S = (D + APOTHEM) / APOTHEM
            return S, (cx, cy)
        else:
            return float('inf'), None
    except Exception:
        return float('inf'), None


def min_enclosing_hexagon(pts):
    """
    Find the minimum enclosing regular hexagon over all 6 equivalent orientations.
    Returns (side_length, center, best_outer_angle).
    """
    best_S = float('inf')
    best_center = None
    best_angle = 0.0
    
    for k in range(6):
        angle = k * math.pi / 6.0
        S, center = enclosing_hexagon_side(pts, angle)
        if S < best_S:
            best_S = S
            best_center = center
            best_angle = angle
    
    # Also try a few intermediate angles
    for k in range(12):
        angle = k * math.pi / 36.0
        S, center = enclosing_hexagon_side(pts, angle)
        if S < best_S:
            best_S = S
            best_center = center
            best_angle = angle
    
    return best_S, best_center, best_angle


def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0
    
    rng = np.random.default_rng(42)
    
    best_S = float('inf')
    best_pts = None
    best_angles = None
    best_center = None
    best_outer_angle = 0.0
    
    all_configs = []
    
    # Add base candidates
    for axial in BASE_CANDIDATES:
        pts = axial_to_cartesian(axial)
        angles = np.zeros(N)
        all_configs.append((pts, angles))
    
    # Generate random configurations with jitter
    for trial in range(200):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        # Pick a random base and add jitter
        base_idx = rng.integers(0, len(BASE_CANDIDATES))
        base_axial = BASE_CANDIDATES[base_idx]
        pts = axial_to_cartesian(base_axial).copy()
        
        # Add small random jitter
        jitter_scale = rng.uniform(0.0, 0.15)
        pts += rng.normal(0, jitter_scale, size=pts.shape)
        
        # Random angles (0 or small rotations)
        if rng.random() < 0.3:
            angles = rng.choice([0.0, 30.0, 60.0], size=N)
        else:
            angles = np.zeros(N)
        
        all_configs.append((pts, angles))
    
    # Also try some configurations with larger structural changes
    for trial in range(100):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        # Random axial coordinates near a hexagonal pattern
        # Start with a 3x3 block and perturb
        q_coords = rng.integers(-2, 3, size=N)
        r_coords = rng.integers(-2, 3, size=N)
        axial = list(zip(q_coords, r_coords))
        pts = axial_to_cartesian(axial)
        angles = np.zeros(N)
        all_configs.append((pts, angles))
    
    # Evaluate each configuration
    for pts, angles in all_configs:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        # Quick overlap check
        if not check_overlap_fast(pts, angles):
            continue
        
        # Find minimum enclosing hexagon
        S, center, outer_angle = min_enclosing_hexagon(pts)
        
        if S < best_S:
            best_S = S
            best_pts = pts.copy()
            best_angles = angles.copy()
            best_center = center
            best_outer_angle = outer_angle
    
    # If we have a good result, try to polish it with local search
    if best_pts is not None and time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        # Local search: try small perturbations of the best configuration
        current_best = best_S
        for polish_iter in range(50):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            improved = False
            for dim in range(2 * N):
                if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                    break
                
                trial_pts = best_pts.copy()
                step = rng.uniform(0.01, 0.05) * rng.choice([-1, 1])
                trial_pts[dim // 2, dim % 2] += step
                
                if not check_overlap_fast(trial_pts, best_angles):
                    continue
                
                S_trial, center_trial, angle_trial = min_enclosing_hexagon(trial_pts)
                
                if S_trial < current_best - 1e-10:
                    current_best = S_trial
                    best_pts = trial_pts
                    best_center = center_trial
                    best_outer_angle = angle_trial
                    best_S = current_best
                    improved = True
            
            if not improved:
                break
    
    # Fallback
    if best_pts is None or best_S == float('inf'):
        pts = axial_to_cartesian(BASE_CANDIDATES[0])
        angles = np.zeros(N)
        best_S, best_center, best_outer_angle = min_enclosing_hexagon(pts)
        best_pts = pts
        best_angles = angles
    
    # Final output
    cx, cy = best_center
    angle_deg = math.degrees(best_outer_angle)
    
    # Scale slightly to ensure no overlap
    SCALE = 1.0 + 1e-12
    pts_scaled = best_pts.copy()
    pts_scaled = cx + (pts_scaled - cx) * SCALE
    
    # Recompute S with scaled points
    normals = np.array([
        [math.cos(best_outer_angle + math.pi/6 + k*math.pi/3), math.sin(best_outer_angle + math.pi/6 + k*math.pi/3)]
        for k in range(6)
    ])
    max_proj = 0
    for nx, ny in normals:
        projs = (pts_scaled[:, 0] - cx) * nx + (pts_scaled[:, 1] - cy) * ny
        max_proj = max(max_proj, np.max(projs), -np.min(projs))
    
    D_scaled = max_proj
    S_scaled = (D_scaled + APOTHEM) / APOTHEM
    
    inner = []
    for i in range(N):
        inner.append([float(pts_scaled[i, 0]), float(pts_scaled[i, 1]), float(best_angles[i])])
    
    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
