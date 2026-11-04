import time
import numpy as np
from scipy.optimize import minimize

N = 11
START_TIME = time.time()
DEADLINE = START_TIME + 115.0  # Safety margin for 120s limit

def _hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy)."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def _check_overlap(poly1, poly2):
    """
    Check if two convex polygons overlap using Separating Axis Theorem (SAT).
    Returns True if they overlap, False otherwise.
    """
    # Get all edges from both polygons
    edges = []
    for i in range(len(poly1)):
        p1 = poly1[i]
        p2 = poly1[(i+1)%len(poly1)]
        edge = p2 - p1
        edges.append(edge)
    for i in range(len(poly2)):
        p1 = poly2[i]
        p2 = poly2[(i+1)%len(poly2)]
        edge = p2 - p1
        edges.append(edge)
        
    # For each edge, get the normal axis
    for edge in edges:
        # Normal to the edge
        axis = np.array([-edge[1], edge[0]])
        # Project both polygons onto this axis
        proj1 = np.dot(poly1, axis)
        proj2 = np.dot(poly2, axis)
        
        # Check for separation
        max1 = np.max(proj1)
        min1 = np.min(proj1)
        max2 = np.max(proj2)
        min2 = np.min(proj2)
        
        if max1 < min2 or max2 < min1:
            return False # Separating axis found, no overlap
            
    return True # No separating axis, they overlap

def _min_outer_side(inner_positions, inner_angles, outer_center, outer_angle):
    """Compute minimum outer side length to contain all inner hexagons."""
    cx, cy = outer_center
    pos = np.asarray(inner_positions, dtype=np.float64)
    ang = np.radians(np.asarray(inner_angles, dtype=np.float64))
    
    # Generate all vertices for all hexagons
    base_angles = np.array([0, 1, 2, 3, 4, 5]) * np.pi / 3
    total_angles = ang[:, None] + base_angles[None, :]
    
    offsets_x = np.cos(total_angles)
    offsets_y = np.sin(total_angles)
    
    verts_x = pos[:, 0:1] + offsets_x
    verts_y = pos[:, 1:2] + offsets_y
    
    all_verts_x = verts_x.ravel()
    all_verts_y = verts_y.ravel()
    
    # Transform to outer hexagon coordinate system
    dx = all_verts_x - cx
    dy = all_verts_y - cy
    
    a_outer = np.radians(outer_angle)
    ca, sa = np.cos(a_outer), np.sin(a_outer)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    
    sqrt3 = np.sqrt(3)
    
    # Support function for regular hexagon centered at origin
    c1 = np.abs(lx)
    c2 = np.abs(lx + ly / sqrt3)
    c3 = np.abs(lx - ly / sqrt3)
    
    required = np.maximum(c1, np.maximum(c2, c3))
    return float(np.max(required))

def _objective(params):
    """
    Objective function for optimization.
    params: [x0, y0, a0, x1, y1, a1, ..., x10, y10, a10]
    Returns: outer_side_length + penalty_for_overlap
    """
    pos = params[0:2*N].reshape(N, 2)
    ang = params[2*N:3*N]
    
    # Compute outer side length
    side = _min_outer_side(pos, ang, (0.0, 0.0), 0.0)
    
    # Compute overlap penalty using SAT
    penalty = 0.0
    # We only need to check pairs that are "close" to save time, but for correctness we check all.
    # To speed up, we can pre-filter by bounding box or center distance.
    for i in range(N):
        for j in range(i+1, N):
            # Quick check: if center distance is large, no overlap possible
            # Max distance between centers for unit hexagons to overlap is < 2 * circumradius = 2
            # Actually, max distance is 2*sqrt(3) ~ 3.46 for edge sharing, but overlap requires < 2*apothem? No.
            # Overlap is possible if distance < 2 * circumradius = 2? No, circumradius is 1.
            # If distance > 2, they definitely don't overlap.
            dx = pos[i,0] - pos[j,0]
            dy = pos[i,1] - pos[j,1]
            dist_sq = dx*dx + dy*dy
            if dist_sq < 4.0: # 2^2
                v1 = _hex_vertices(pos[i,0], pos[i,1], 1.0, ang[i])
                v2 = _hex_vertices(pos[j,0], pos[j,1], 1.0, ang[j])
                if _check_overlap(v1, v2):
                    # Penalty: we want to push them apart.
                    # A simple penalty is the overlap depth, but SAT doesn't give depth directly.
                    # We can use a fixed large penalty if they overlap, or try to estimate depth.
                    # For Nelder-Mead, a binary penalty might be too harsh.
                    # Let's use a penalty proportional to how much they are "inside" each other.
                    # Approximation: if they overlap, add a penalty based on the distance to the threshold.
                    # Threshold for no overlap is roughly 2*apothem = sqrt(3) ~ 1.732? No.
                    # Let's just add a large constant penalty for any overlap.
                    penalty += 100.0
    
    return side + penalty

def _initial_configuration():
    """
    Generate a valid initial configuration.
    We use the 'flower' pattern.
    """
    sqrt3 = np.sqrt(3)
    pos = np.zeros((N, 2))
    ang = np.zeros(N)
    
    # 0: Center
    pos[0] = [0, 0]
    
    # 1-6: Ring 1
    for i in range(6):
        a = i * np.pi / 3
        pos[i+1] = [2*sqrt3 * np.cos(a), 2*sqrt3 * np.sin(a)]
        
    # 7-10: Ring 2 (4 hexagons)
    ring2_angles = [np.pi/6, 5*np.pi/6, 7*np.pi/6, 11*np.pi/6] # 30, 150, 210, 330
    for i in range(4):
        a = ring2_angles[i]
        pos[i+7] = [6.0 * np.cos(a), 6.0 * np.sin(a)]
        
    return pos, ang

def construct():
    global START_TIME, DEADLINE
    
    # Initialize with the standard flower pattern
    pos0, ang0 = _initial_configuration()
    params0 = np.concatenate([pos0.ravel(), ang0])
    
    # Initial objective
    best_params = params0.copy()
    best_obj = _objective(best_params)
    
    # Multi-start optimization using Nelder-Mead
    # Nelder-Mead is derivative-free and robust to non-smoothness (like SAT penalties)
    starts = [params0.copy()]
    
    # Add some random perturbations
    for _ in range(10):
        p = params0.copy()
        # Perturb positions slightly
        p[0:2*N] += np.random.normal(0, 0.5, (2*N,))
        # Perturb angles slightly
        p[2*N:3*N] += np.random.normal(0, 10.0, (N,))
        starts.append(p)
    
    # Optimization loop
    for start in starts:
        if time.time() > DEADLINE:
            break
        
        try:
            # Use Nelder-Mead
            # maxiter is per function call, but Nelder-Mead can be slow.
            # We set a moderate maxiter to avoid hanging on a single start.
            res = minimize(_objective, start, method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
            
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
        
        # Check time
        if time.time() > DEADLINE:
            break
    
    # Extract best solution
    pos_best = best_params[0:2*N].reshape(N, 2)
    ang_best = best_params[2*N:3*N]
    
    # Verify validity (no overlap)
    # If the penalty was high, the solution might still have overlaps.
    # We need to check and potentially fix or reject.
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            dx = pos_best[i,0] - pos_best[j,0]
            dy = pos_best[i,1] - pos_best[j,1]
            dist_sq = dx*dx + dy*dy
            if dist_sq < 4.0:
                v1 = _hex_vertices(pos_best[i,0], pos_best[i,1], 1.0, ang_best[i])
                v2 = _hex_vertices(pos_best[j,0], pos_best[j,1], 1.0, ang_best[j])
                if _check_overlap(v1, v2):
                    valid = False
                    break
        if not valid:
            break
    
    if not valid:
        # Fallback to initial configuration
        pos_best = pos0
        ang_best = ang0
    
    # Calculate the actual side length without penalty
    side_best = _min_outer_side(pos_best, ang_best, (0.0, 0.0), 0.0)
    
    inner_list = [[float(pos_best[i,0]), float(pos_best[i,1]), float(ang_best[i])] for i in range(N)]
    
    return inner_list, (0.0, 0.0), float(side_best), 0.0

if __name__ == "__main__":
    construct()
