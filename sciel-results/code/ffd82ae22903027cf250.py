# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    th = a + np.deg2rad(60.0) * np.arange(6)
    return np.column_stack((cx + s * np.cos(th), cy + s * np.sin(th)))

def _edge_normals(V):
    n = np.empty((6, 2))
    for i in range(6):
        e = V[(i + 1) % 6] - V[i]
        L = np.hypot(e[0], e[1])
        n[i] = np.array([e[1], -e[0]]) / L
    return n

def _gap_and_margin(A, B):
    """Return (best_margin, active_normal). best_margin>0 means separated."""
    nA = _edge_normals(A)
    nB = _edge_normals(B)
    best = -np.inf
    best_n = nA[0]
    for n in nA:
        loA, hiA = (A @ n).min(), (A @ n).max()
        loB, hiB = (B @ n).min(), (B @ n).max()
        sep = max(loB - hiA, loA - hiB)
        if sep > best:
            best = sep
            best_n = n
    for n in nB:
        loA, hiA = (A @ n).min(), (A @ n).max()
        loB, hiB = (B @ n).min(), (B @ n).max()
        sep = max(loB - hiA, loA - hiB)
        if sep > best:
            best = sep
            best_n = n
    return best, best_n

def _all_valid(inner, outer, side):
    """Full validity check. inner (11,3), outer (3,), side float."""
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)) or not np.isfinite(side):
        return False
    if side < np.sqrt(11.0) - 1e-9:
        return False
    # separation
    Vs = [_hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2]) for i in range(11)]
    for i in range(11):
        for j in range(i + 1, 11):
            m, _ = _gap_and_margin(Vs[i], Vs[j])
            if m <= 1e-6:
                return False
    # containment
    cx, cy, ca = outer[0], outer[1], outer[2]
    OV = _hex_vertices(cx, cy, side, ca)
    for V in Vs:
        for i in range(6):
            p1 = OV[i]
            p2 = OV[(i + 1) % 6]
            cross = (p2[0] - p1[0]) * (V[:, 1] - p1[1]) - (p2[1] - p1[1]) * (V[:, 0] - p1[0])
            if np.any(cross < -1e-6):
                return False
    # edge length check after shifting by outer center
    for V in Vs:
        Vs2 = V - np.array([cx, cy])
        for i in range(6):
            L = np.hypot(*(Vs2[(i + 1) % 6] - Vs2[i]))
            if abs(L - 1.0) > 1e-8:
                return False
    return True

def _min_side_for_positions(positions, angle=0.0):
    """
    Calculate the minimal side length of a regular hexagon (fixed orientation)
    that encloses the given positions (centers of unit hexagons).
    We assume the outer hexagon is centered at the centroid of the positions for simplicity 
    in the objective, but the final reconstruction can optimize center/angle if needed.
    However, for the local search, we fix the outer hexagon orientation to 0 and center 
    to the centroid to reduce dimensionality, or we can just compute the required side 
    length assuming the hexagon is centered at (0,0) and rotated to fit best?
    
    Actually, the task allows the outer hexagon to be at any (x,y,angle).
    To make the objective smooth and low-dimensional, we will:
    1. Fix the outer hexagon orientation to 0 (or optimize it separately later).
    2. Fix the outer hexagon center to the centroid of the inner hexagons.
    3. Compute the required side length.
    
    If the cluster is not symmetric, this might be suboptimal, but it's a good start.
    A better approach: The minimal enclosing hexagon for a set of points with fixed orientation 
    is determined by the max projection on each of the 6 normals.
    side = max_k ( max_i (n_k . p_i) - min_i (n_k . p_i) ) / sqrt(3)
    This formula assumes the hexagon is centered such that it is tight on both sides for the 
    active normal, which is optimal for that orientation.
    """
    # Normals for a hexagon with angle 0 (flat top? or pointy top?)
    # Vertices at 0, 60, 120...
    # Normals are at 30, 90, 150...
    normals = np.array([
        [np.cos(np.deg2rad(30 + 60*i)), np.sin(np.deg2rad(30 + 60*i))]
        for i in range(6)
    ])
    
    proj = positions @ normals.T
    ranges = proj.max(axis=0) - proj.min(axis=0)
    max_range = ranges.max()
    return max_range / np.sqrt(3.0)

def _objective(params, target_side):
    """
    params: (22,) array of x,y for 11 hexagons.
    Returns a penalty value.
    """
    positions = params.reshape(11, 2)
    
    # 1. Overlap penalty
    # We want gap > 0.
    # Approximate gap by distance between centers minus 2*apothem?
    # No, hexagons are not circles. But for a smooth surrogate, 
    # we can use the distance between centers.
    # Two unit hexagons with parallel edges and aligned centers are separated by 
    # distance d. The gap is d - 2*apothem? No, apothem is s*sqrt(3)/2.
    # If aligned, the distance between parallel edges is d. 
    # The "radius" in the normal direction is apothem.
    # So gap = d - 2*apothem.
    # This is a lower bound on the actual gap if orientations are aligned.
    # Since we fix all inner hexagons to angle 0, this is exact for the 6 primary directions.
    apothem = np.sqrt(3.0) / 2.0
    min_gap = np.inf
    for i in range(11):
        for j in range(i + 1, 11):
            d = np.linalg.norm(positions[i] - positions[j])
            gap = d - 2.0 * apothem
            if gap < min_gap:
                min_gap = gap
    
    overlap_penalty = 0.0
    if min_gap < 0:
        overlap_penalty = -min_gap * 100.0 # Strong penalty for overlap
    
    # 2. Size penalty
    # We want the resulting side length to be <= target_side.
    current_side = _min_side_for_positions(positions)
    size_penalty = 0.0
    if current_side > target_side:
        size_penalty = (current_side - target_side) * 10.0
    
    return overlap_penalty + size_penalty

def hexagon_packing_11():
    t0 = time.time()
    deadline = t0 + 590.0  # Safety margin

    best_val = -np.inf
    best_result = None

    # Initial guess: A compact cluster
    # Use the "Flower" + 4 extensions from the seed, but converted to cartesian
    def axial_to_cartesian(q, r):
        x = 1.5 * q
        y = np.sqrt(3) * (r + q / 2.0)
        return np.array([x, y])
    
    coords_A = [
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (1,1), (-1,1), (0,2), (0,-2)
    ]
    
    init_positions = np.array([axial_to_cartesian(q, r) for q, r in coords_A])
    # Center it
    init_positions -= init_positions.mean(axis=0)
    
    # We will run a series of local optimizations with decreasing target_side.
    # Start with a loose target and tighten it.
    
    # Estimate a loose side
    initial_side = _min_side_for_positions(init_positions)
    
    # We want to go below initial_side.
    # Let's try to optimize for a target_side that is slightly smaller than the current best.
    # We'll use a loop that tries to reduce the side.
    
    current_positions = init_positions.copy()
    current_side = initial_side
    
    # Define a range of target sides to try
    # We know side >= sqrt(11) ~ 3.31
    # initial_side is likely around 4-5.
    
    num_restarts = 10
    rng = np.random.default_rng(42)
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        if restart == 0:
            pos = init_positions.copy()
        else:
            # Add noise
            pos = init_positions + rng.normal(0, 0.5, size=(11, 2))
            
        # Tighten the target side gradually
        # Start with a target that is 5% smaller than the current best known side
        # Or just start with a reasonable guess.
        
        # Let's use a binary search on the target side? 
        # Or just optimize for a fixed small target and see if we can satisfy it.
        
        # Better: Use a penalty method where the target is fixed to a very small value,
        # and the optimizer will push the positions to minimize the side.
        # But if we set target to 0, the size penalty will always be large.
        
        # Alternative: Objective is simply the side length, plus a large penalty for overlap.
        # This is cleaner.
        
        def obj_no_target(p):
            positions = p.reshape(11, 2)
            # Overlap penalty
            apothem = np.sqrt(3.0) / 2.0
            min_gap = np.inf
            for i in range(11):
                for j in range(i + 1, 11):
                    d = np.linalg.norm(positions[i] - positions[j])
                    gap = d - 2.0 * apothem
                    if gap < min_gap:
                        min_gap = gap
            
            penalty = 0.0
            if min_gap < 0:
                penalty = -min_gap * 1000.0 # Very strong penalty to ensure separation
            
            side = _min_side_for_positions(positions)
            return side + penalty
        
        res = minimize(obj_no_target, pos.flatten(), method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
        
        if res.fun < 1000: # Check if it's not just a huge penalty
            final_pos = res.x.reshape(11, 2)
            
            # Reconstruct inner
            # Angle 0 for all
            inner = np.column_stack((final_pos[:, 0], final_pos[:, 1], np.zeros(11)))
            
            # Determine outer
            # Center at centroid
            cx, cy = final_pos.mean(axis=0)
            # Angle 0
            side = _min_side_for_positions(final_pos)
            outer = np.array([cx, cy, 0.0])
            
            # Validate
            if _all_valid(inner, outer, side):
                val = 1.0 / side
                if val > best_val:
                    best_val = val
                    best_result = (inner, outer, side)
                    # Update current best positions for next restart if we want to exploit
                    # But Nelder-Mead is local, so we keep the global best.
                    
    if best_result is None:
        # Fallback
        inner = np.array([
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
        ], dtype=float)
        outer = np.array([0, 0, 0], dtype=float)
        return inner, outer, 8.0

    return best_result

# EVOLVE-BLOCK-END
