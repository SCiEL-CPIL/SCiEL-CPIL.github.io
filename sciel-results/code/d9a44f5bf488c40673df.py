"""Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length."""
import math
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 12

# EVOLVE-BLOCK-START

def _hex_vertices(cx, cy, angle_deg=0.0, side=1.0):
    """
    Returns vertices of a regular hexagon with center (cx, cy), 
    side length 'side', and rotation 'angle_deg' (degrees).
    Orientation: 0 degrees means a vertex is at angle 0 (pointing right).
    """
    a = math.radians(angle_deg)
    verts = np.empty((6, 2))
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts[i, 0] = cx + side * math.cos(theta)
        verts[i, 1] = cy + side * math.sin(theta)
    return verts

def _sat_gap(v1, v2, tol=1e-9):
    """
    Calculates the maximum gap between two convex polygons v1 and v2.
    If gap > 0, they are separated by at least 'gap'.
    If gap < 0, they overlap by at least '-gap'.
    """
    max_gap = -float('inf')
    # Check axes from v1
    for i in range(6):
        e1 = v1[(i+1)%6] - v1[i]
        n1 = np.array([-e1[1], e1[0]])
        norm1 = np.linalg.norm(n1)
        if norm1 > 1e-9:
            n1 = n1 / norm1
            proj1 = np.dot(v1, n1)
            proj2 = np.dot(v2, n1)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            if max1 < min2:
                gap = min2 - max1
            elif max2 < min1:
                gap = min1 - max2
            else:
                gap = -max(0.0, min(max1, max2) - max(min1, min2))
                
            if gap > max_gap:
                max_gap = gap

    # Check axes from v2
    for i in range(6):
        e2 = v2[(i+1)%6] - v2[i]
        n2 = np.array([-e2[1], e2[0]])
        norm2 = np.linalg.norm(n2)
        if norm2 > 1e-9:
            n2 = n2 / norm2
            proj1 = np.dot(v1, n2)
            proj2 = np.dot(v2, n2)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            if max1 < min2:
                gap = min2 - max1
            elif max2 < min1:
                gap = min1 - max2
            else:
                gap = -max(0.0, min(max1, max2) - max(min1, min2))
                
            if gap > max_gap:
                max_gap = gap
                
    return max_gap

def compute_outer_side(centers, angles, tol=1e-9):
    """
    Computes the minimum side length of a regular hexagon centered at origin
    with rotation 0 (vertices at 0, 60, ...) to contain all inner hexagons.
    """
    # Normals for the outer hexagon (perpendicular to edges)
    # Vertices of outer hex (side S) are at angles 0, 60, ...
    # Edges are between these.
    # Normal to edge between 0 and 60 is at 30 deg.
    # Normals are at 30, 90, 150, 210, 270, 330 degrees.
    normal_angles = [math.pi/6 + k * math.pi/3 for k in range(6)]
    normals = np.array([[math.cos(a), math.sin(a)] for a in normal_angles])
    
    max_proj = 0.0
    
    for i in range(N):
        cx, cy = centers[i]
        ang = angles[i]
        verts = _hex_vertices(cx, cy, ang, 1.0)
        
        # Center the cluster at origin for this calculation
        # Note: In the objective, we pass centered centers.
        # Here we assume centers are already centered or we center them.
        # For robustness, let's center the vertices relative to the global centroid.
        # But this function is called from objective where we handle centering.
        # Let's assume input centers are already centered at (0,0) effectively.
        
        # Project all vertices onto all normals
        # verts shape (6,2), normals shape (6,2)
        # We want max dot product for each normal, then max over normals.
        projs = verts @ normals.T # shape (6, 6)
        max_per_normal = np.max(projs, axis=0)
        max_proj = max(max_proj, np.max(max_per_normal))
        
    # Side length S relates to max projection (which is distance to edge, i.e., inradius)
    # inradius = S * sqrt(3) / 2
    # So S = 2 * inradius / sqrt(3)
    return 2.0 * max_proj / math.sqrt(3.0)

def objective(params):
    """
    Objective function for optimization.
    params: 2N floats, x1, y1, ..., xN, yN
    Returns: outer side length + penalty for overlaps
    """
    start_time = time.time()
    # Hard timeout guard inside objective to prevent infinite loops in DE
    if time.time() > 110.0:
        return 1e10

    centers = np.array(params).reshape(N, 2)
    angles = np.zeros(N) # Assume all aligned
    
    # 1. Center the cluster at the origin
    centroid = np.mean(centers, axis=0)
    centers_centered = centers - centroid
    
    # 2. Compute vertices for centered hexagons
    verts_list = []
    for i in range(N):
        cx, cy = centers_centered[i]
        verts = _hex_vertices(cx, cy, angles[i], 1.0)
        verts_list.append(verts)
        
    # 3. Compute minimal outer side length
    # We need to project the centered vertices onto the outer hexagon normals
    normal_angles = [math.pi/6 + k * math.pi/3 for k in range(6)]
    normals = np.array([[math.cos(a), math.sin(a)] for a in normal_angles])
    
    max_proj = 0.0
    for verts in verts_list:
        projs = verts @ normals.T
        max_per_normal = np.max(projs, axis=0)
        max_proj = max(max_proj, np.max(max_per_normal))
        
    side = 2.0 * max_proj / math.sqrt(3.0)
    
    # 4. Compute overlap penalty
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            gap = _sat_gap(verts_list[i], verts_list[j])
            if gap < 0:
                # Quadratic penalty for overlap
                penalty += (-gap) ** 2
                
    return side + penalty

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Define bounds for positions
    # The cluster of 12 hexagons is roughly 5x5 units.
    # So coordinates should be within [-3, 3] relative to centroid.
    bounds = [(-3.0, 3.0)] * (2 * N)
    
    best_result = None
    best_score = float('inf')
    
    # Strategy 1: Differential Evolution (Global Search)
    # Use a smaller population and maxiter to stay within time limits
    try:
        de_res = differential_evolution(
            objective, 
            bounds, 
            seed=42,
            maxiter=50,
            popsize=15,
            tol=1e-4,
            polish=True # Uses L-BFGS-B for polishing
        )
        
        if de_res.fun < best_score:
            best_score = de_res.fun
            best_result = de_res.x
            
    except Exception as e:
        print(f"DE failed: {e}")
        
    # Strategy 2: Local Refinement from Lattice Initializations
    # If DE didn't run or finished early, or to ensure we have a good local min.
    
    def get_lattice_block(rows, cols):
        centers = []
        for r in range(rows):
            for c in range(cols):
                x = c * 2.0 + (1.0 if r % 2 == 1 else 0.0)
                y = r * math.sqrt(3)
                centers.append([x, y])
        # Center
        cx_mean = np.mean([c[0] for c in centers])
        cy_mean = np.mean([c[1] for c in centers])
        return np.array([[c[0]-cx_mean, c[1]-cy_mean] for c in centers]).flatten()

    # Candidate initializations
    inits = [
        get_lattice_block(2, 6),
        get_lattice_block(3, 4),
        get_lattice_block(4, 3),
        get_lattice_block(6, 2),
        # Add a random perturbation to break symmetry if needed
    ]
    
    for init in inits:
        if time.time() > deadline - 20.0:
            break
            
        # Run local minimization
        try:
            res = minimize(
                objective, 
                init, 
                method='Nelder-Mead', 
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x
        except Exception as e:
            pass

    if best_result is None:
        # Fallback: Return a known valid configuration (3x4 lattice)
        centers = get_lattice_block(3, 4).reshape(N, 2)
        # Calculate side
        angles = [0.0] * N
        # Center is 0,0
        side = compute_outer_side(centers, angles)
        inner = []
        for i in range(N):
            inner.append([float(centers[i][0]), float(centers[i][1]), 0.0])
        return inner, (0.0, 0.0), float(side * 1.01), 0.0

    # Final Refinement: Use L-BFGS-B on the best result found
    # L-BFGS-B requires smooth objective, but our penalty is smooth (quadratic).
    # However, the 'max' function in side calculation is not differentiable at ties.
    # Nelder-Mead or Powell is safer. Let's use Powell for final polish.
    try:
        final_res = minimize(
            objective, 
            best_result, 
            method='Powell', 
            options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8}
        )
        if final_res.fun < best_score:
            best_score = final_res.fun
            best_result = final_res.x
    except Exception:
        pass

    # Extract final parameters
    final_centers = best_result.reshape(N, 2)
    
    # Center the final configuration
    centroid = np.mean(final_centers, axis=0)
    final_centers_centered = final_centers - centroid
    
    angles = [0.0] * N
    
    # Calculate final side length
    side = compute_outer_side(final_centers_centered, angles)
    
    # Check validity (no overlaps)
    verts_list = [_hex_vertices(final_centers_centered[i][0], final_centers_centered[i][1], 0.0, 1.0) for i in range(N)]
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            gap = _sat_gap(verts_list[i], verts_list[j])
            if gap < -1e-5:
                valid = False
    
    if not valid:
        # If invalid, add a small buffer to side length to ensure validity?
        # No, the task requires disjoint hexagons.
        # If the optimizer failed to find a valid config, we should return the best valid one.
        # For now, we assume the penalty was strong enough.
        # If not, we might return a slightly larger side to "push" them apart?
        # The score is the side length. If we return a config that overlaps, it's invalid (score 0).
        # So we must ensure validity.
        
        # Fallback: Use the 3x4 lattice which is guaranteed valid
        print("Final config invalid, falling back to lattice")
        centers = get_lattice_block(3, 4).reshape(N, 2)
        centroid = np.mean(centers, axis=0)
        centers = centers - centroid
        side = compute_outer_side(centers, angles)
        final_centers_centered = centers

    inner = []
    for i in range(N):
        inner.append([float(final_centers_centered[i][0]), float(final_centers_centered[i][1]), 0.0])
    
    # Add a tiny epsilon to side length to account for numerical precision in evaluator
    final_side = float(side * 1.0001 + 1e-6)
    
    return inner, (0.0, 0.0), final_side, 0.0

# EVOLVE-BLOCK-END
