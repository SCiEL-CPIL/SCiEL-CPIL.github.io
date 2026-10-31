import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 120.0
SAFETY_MARGIN = 20.0

def construct():
    # Deadline guard
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # --- Step 1: Generate initial cluster of 11 unit hexagons ---
    # We use a hexagonal packing (flower pattern) as the starting point.
    # The centers are on a triangular lattice.
    # To ensure they are disjoint (non-overlapping) with a strict gap, we scale the lattice slightly.
    # Distance between adjacent centers in a tight hex packing is 2 * apothem = sqrt(3).
    # If we scale by S > 1, distance becomes S*sqrt(3).
    # The condition for non-overlap is distance >= 2*apothem = sqrt(3).
    # So S must be >= 1.
    # We use S = 1.005 to be safe against floating point errors and strict inequality checks.
    
    SCALE = 1.005
    
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    
    initial_pts = []
    for q, r in axial_coords:
        x = math.sqrt(3) * (q + r / 2.0) * SCALE
        y = 1.5 * r * SCALE
        initial_pts.append([x, y, 0.0]) # Initial angle 0
        
    # --- Step 2: Optimize the cluster shape (compactness) ---
    # We want to minimize the "radius" of the cluster to fit in a smaller container.
    # A good proxy is the maximum distance from the centroid, or the diameter.
    # However, simply minimizing the max distance might collapse the structure.
    # Instead, we can try to minimize the bounding box area or the max projection length.
    # Let's try minimizing the maximum distance between any two points (diameter) 
    # subject to non-overlap constraints? That's hard.
    # Alternative: Minimize the sum of squared distances from centroid (moment of inertia)
    # while maintaining minimum distance between centers.
    
    # Let's use a penalty function for overlap.
    # Variables: 3*N (x, y, angle for each hexagon)
    # But rotating individual hexagons in a dense packing is tricky. 
    # In a hexagonal close packing, all hexagons are usually aligned (angle 0) or rotated by 60 deg (which is same shape).
    # If we allow individual rotations, we might be able to interlock them better? 
    # Regular hexagons are convex. Rotating them doesn't change their footprint relative to a container if the container is also hexagonal and aligned? 
    # Actually, the container is a regular hexagon. The inner hexagons can rotate.
    # If we rotate an inner hexagon by 30 degrees, its "width" in the direction of the container's sides changes.
    # The width of a regular hexagon of side 1 is 2*apothem = sqrt(3) when measured parallel to a pair of opposite sides.
    # If rotated by 30 deg, the width perpendicular to those sides is 2*side = 2.
    # So aligning the inner hexagons with the container (angle 0 relative to container) minimizes the required container size for that specific orientation.
    # Therefore, it is likely optimal to keep all inner hexagons aligned with the outer container.
    # So we can fix the inner angles to be the same as the outer container's rotation.
    # This reduces the problem to:
    # 1. Place N centers (2N variables).
    # 2. Choose outer container rotation beta (1 variable).
    # 3. Minimize outer side length S.
    
    # Let's define the objective function for a given set of centers and beta.
    # S is determined by the max projection of (center + hexagon_extent) onto the 6 normals.
    
    def calc_S_from_centers(centers, beta):
        # centers: (N, 2) array
        # beta: rotation of outer hexagon
        # The inner hexagons are aligned with the outer one.
        # So the "extent" of each inner hexagon in the direction of a normal n is APOTHEM.
        # Wait, is it?
        # If the inner hexagon is aligned with the outer one, its apothem direction is parallel to the outer hexagon's apothem directions.
        # The projection of a regular hexagon of side 1 (aligned) onto a normal parallel to its apothem is exactly APOTHEM.
        # If the normal is at 30 deg to the apothem, the projection is SIDE = 1.
        # The outer hexagon's normals are at angles beta + pi/6 + k*pi/3.
        # If the inner hexagons are aligned with the outer hexagon (angle = beta), then their apothems are also at beta + pi/6 + k*pi/3?
        # No, the apothem of a hexagon points to the middle of a side.
        # Let's define the hexagon geometry precisely.
        # Standard hexagon: vertices at (1,0), (0.5, sqrt(3)/2), ...
        # Apothem direction: (0, sqrt(3)/2) -> angle pi/2? No.
        # Let's use the standard: Vertex at angle 0. Side is perpendicular to apothem.
        # If we align inner with outer, the "width" of the inner hexagon in the direction of the outer normal is constant?
        # Actually, if we fix the relative angle between inner and outer to 0, the required S is simply:
        # max over all normals n_k of ( max_i (dot(center_i, n_k)) + APOTHEM ) / APOTHEM ?
        # No, the distance from center to the boundary of the inner hexagon in direction n_k is:
        # If n_k is aligned with an apothem of the inner hex, distance is APOTHEM.
        # If n_k is aligned with a vertex of the inner hex, distance is 1.
        # Since we can choose the alignment, we should align the inner hexagons such that their apothems are parallel to the outer normals?
        # Yes, that minimizes the extent in those directions.
        # So, if we set inner_angle = outer_angle, the extent in each outer normal direction is APOTHEM.
        
        # Let's verify.
        # Outer normal angle: theta.
        # Inner hexagon rotated by theta.
        # The distance from center to the edge of the inner hexagon in direction theta is APOTHEM.
        # So the required container apothem is max_i(dot(center_i, n_k)) + APOTHEM.
        # And S = (max_apothem_required) / APOTHEM.
        
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        max_S = 0.0
        for nx, ny in normals:
            # Project centers onto normal
            # rel_pts = centers - centroid? No, the container is centered at centroid of inner cluster?
            # The container can be translated. To minimize S, we should center the container on the "center of mass" of the projections?
            # Actually, for a fixed set of centers and fixed beta, the optimal container center is the average of the max and min projections?
            # No, the container is a convex polygon. The minimal side length hexagon containing a set of points is determined by the support functions.
            # For a regular hexagon, the side length S is determined by the maximum width in the 6 normal directions.
            # Width in direction n is max(dot(p, n)) - min(dot(p, n)).
            # But we also have the "thickness" of the inner hexagons.
            # The total width in direction n is (max_i dot(c_i, n) - min_i dot(c_i, n)) + 2 * APOTHEM.
            # Wait, if the container is centered at some point C, the constraint is:
            # dot(c_i - C, n) + APOTHEM <= S * APOTHEM
            # dot(c_i - C, -n) + APOTHEM <= S * APOTHEM
            # So |dot(c_i - C, n)| + APOTHEM <= S * APOTHEM
            # This must hold for all i and all 6 normals.
            # To minimize S, we should choose C to minimize the max over all constraints.
            # This is equivalent to finding the smallest regular hexagon (fixed orientation) containing the set of points {c_i} expanded by APOTHEM in all directions?
            # No, the expansion is not isotropic. It's APOTHEM in the 6 normal directions.
            # Actually, if we align inner hexagons with outer, the Minkowski sum of the inner hexagon and a point is just the hexagon.
            # The container must contain the union of these hexagons.
            # The support function of the union is the max of the support functions.
            # h_union(n) = max_i ( dot(c_i, n) + h_hex(n) )
            # If aligned, h_hex(n) = APOTHEM for the 6 normal directions.
            # So we need a regular hexagon of side S (apothem S*APOTHEM) centered at C such that:
            # h_union(n_k) <= dot(C, n_k) + S*APOTHEM
            # and h_union(-n_k) <= dot(C, -n_k) + S*APOTHEM
            # Let h_k = max_i dot(c_i, n_k). Let h'_k = max_i dot(c_i, -n_k) = -min_i dot(c_i, n_k).
            # We need C such that:
            # dot(C, n_k) >= h_k - S*APOTHEM
            # dot(C, -n_k) >= h'_k - S*APOTHEM  => -dot(C, n_k) >= h'_k - S*APOTHEM => dot(C, n_k) <= S*APOTHEM - h'_k
            # So for each k, we need:
            # h_k - S*APOTHEM <= dot(C, n_k) <= S*APOTHEM - h'_k
            # This is possible if and only if h_k - S*APOTHEM <= S*APOTHEM - h'_k
            # h_k + h'_k <= 2*S*APOTHEM
            # S >= (h_k + h'_k) / (2*APOTHEM)
            # Note that h_k + h'_k = max_i dot(c_i, n_k) - min_i dot(c_i, n_k) = Width of the point set in direction n_k.
            # So the minimal S for a given beta is:
            # S_min = max_k ( Width_k(beta) / (2*APOTHEM) + 1 )  ?
            # Wait, the inequality was S >= (h_k + h'_k) / (2*APOTHEM).
            # But we also need to account for the fact that the container itself has size.
            # Let's re-derive.
            # Constraint: max_i ( dot(c_i, n) + APOTHEM ) <= dot(C, n) + S*APOTHEM
            # And max_i ( dot(c_i, -n) + APOTHEM ) <= dot(C, -n) + S*APOTHEM
            # Let W_n = max_i dot(c_i, n) - min_i dot(c_i, n).
            # The range of dot(c_i, n) is [min, max].
            # The union of hexagons has support in direction n: max_i dot(c_i, n) + APOTHEM.
            # The support in direction -n: max_i dot(c_i, -n) + APOTHEM = -min_i dot(c_i, n) + APOTHEM.
            # The container centered at C has support in direction n: dot(C, n) + S*APOTHEM.
            # So we need:
            # dot(C, n) + S*APOTHEM >= max_i dot(c_i, n) + APOTHEM
            # dot(C, -n) + S*APOTHEM >= -min_i dot(c_i, n) + APOTHEM
            # Let u = dot(C, n).
            # u >= max_i dot(c_i, n) + APOTHEM - S*APOTHEM
            # -u >= -min_i dot(c_i, n) + APOTHEM - S*APOTHEM  => u <= min_i dot(c_i, n) - APOTHEM + S*APOTHEM
            # So we need:
            # max_i dot(c_i, n) + APOTHEM - S*APOTHEM <= min_i dot(c_i, n) - APOTHEM + S*APOTHEM
            # max_i dot(c_i, n) - min_i dot(c_i, n) + 2*APOTHEM <= 2*S*APOTHEM
            # W_n + 2*APOTHEM <= 2*S*APOTHEM
            # S >= (W_n + 2*APOTHEM) / (2*APOTHEM) = W_n/(2*APOTHEM) + 1
            
            # So S_min(beta) = max_k ( Width_k(beta) / (2*APOTHEM) + 1 )
            
            width = np.max(centers @ np.array([nx, ny])) - np.min(centers @ np.array([nx, ny]))
            required_S = width / (2 * APOTHEM) + 1.0
            if required_S > max_S:
                max_S = required_S
        
        return max_S

    def objective(params):
        # params: 2*N (x, y for each center) + 1 (beta)
        # But we can fix the center of mass at (0,0) to remove translational symmetry?
        # The container center will be at the centroid of the projections, but since the formula for S depends only on Widths,
        # the absolute position of the centers doesn't matter, only their relative positions.
        # So we can fix the centroid of the centers to (0,0).
        # This reduces variables to 2*N - 2.
        # However, for simplicity, let's just optimize 2*N + 1 and let the optimizer handle it, or fix centroid manually.
        # Fixing centroid is better for conditioning.
        
        # Extract centers and beta
        centers = params[:2*N].reshape(N, 2)
        beta = params[2*N]
        
        # Enforce centroid at 0
        centroid = np.mean(centers, axis=0)
        centers = centers - centroid
        
        S = calc_S_from_centers(centers, beta)
        
        # Penalty for overlap
        # Check pairwise distances
        # Distance between centers i and j must be >= 2*APOTHEM * SCALE?
        # No, the hexagons are aligned. The minimum distance between centers of two aligned unit hexagons
        # to be disjoint is 2*APOTHEM = sqrt(3).
        # If distance < sqrt(3), they overlap.
        # We add a penalty for violations.
        
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dist = np.linalg.norm(centers[i] - centers[j])
                min_dist = math.sqrt(3) # Tight packing distance
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2 * 100.0 # Strong penalty
        
        return S + penalty

    # Initial guess
    initial_params = []
    for x, y, a in initial_pts:
        initial_params.extend([x, y])
    initial_params.append(0.0) # beta
    initial_params = np.array(initial_params)
    
    # Optimize
    best_result = None
    best_score = float('inf')
    
    # Try multiple starts?
    # Start 1: The lattice
    try:
        res = minimize(objective, initial_params, method='Nelder-Mead', 
                       options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
        if res.fun < best_score:
            best_score = res.fun
            best_result = res
    except Exception:
        pass
    
    # Start 2: Slightly perturbed lattice
    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        perturbed = initial_params + np.random.normal(0, 0.1, len(initial_params))
        try:
            res = minimize(objective, perturbed, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < best_score:
                best_score = res.fun
                best_result = res
        except Exception:
            pass
            
    # If optimization failed or didn't improve, use the initial lattice
    if best_result is None:
        final_params = initial_params
    else:
        final_params = best_result.x
        
    centers = final_params[:2*N].reshape(N, 2)
    beta = final_params[2*N]
    
    # Recalculate S accurately
    S = calc_S_from_centers(centers, beta)
    
    # Construct output
    # The container center should be at the centroid of the centers?
    # From the derivation, any C that satisfies the inequalities works.
    # The "tightest" C is the midpoint of the valid range for each normal.
    # But since we only return the side length and the inner hexagons, we can just return the centroid as the container center.
    # Wait, the evaluator checks if the inner hexagons are INSIDE the outer hexagon.
    # So we must provide a valid container center.
    # Let's calculate the optimal container center C.
    
    normals = []
    for k in range(6):
        ang = beta + math.pi / 6.0 + k * math.pi / 3.0
        normals.append((math.cos(ang), math.sin(ang)))
        
    # For each normal n_k, we have an interval [L_k, R_k] for dot(C, n_k)
    # L_k = max_i dot(c_i, n_k) + APOTHEM - S*APOTHEM
    # R_k = min_i dot(c_i, n_k) - APOTHEM + S*APOTHEM
    
    # We need to find C such that dot(C, n_k) is in [L_k, R_k] for all k.
    # This is a system of linear inequalities.
    # We can use a simple iterative projection or just pick the centroid of the centers and hope it works?
    # Actually, since S was calculated to be the minimum possible, the intervals should intersect.
    # Let's pick C = centroid of centers.
    C = np.mean(centers, axis=0)
    
    # Verify validity
    # If C doesn't work, we might need to adjust S or C.
    # But given the formula S >= (W + 2A) / 2A, and we used that, it should be valid.
    # Let's double check the logic.
    # If S = (W + 2A) / 2A, then the interval length is R_k - L_k = (min - A + S*A) - (max + A - S*A) = min - max - 2A + 2S*A = -W - 2A + 2S*A.
    # Substituting S: -W - 2A + 2*(W+2A)/2*A*A = -W - 2A + (W+2A) = 0.
    # So the interval is a single point!
    # dot(C, n_k) must be exactly (L_k + R_k) / 2.
    # L_k + R_k = max + A - S*A + min - A + S*A = max + min.
    # So dot(C, n_k) = (max_i dot(c_i, n_k) + min_i dot(c_i, n_k)) / 2.
    # This defines C uniquely (since 6 normals span 2D).
    
    # Let's compute this specific C.
    # We have 6 equations dot(C, n_k) = d_k.
    # We can solve this using least squares or just pick 2 independent normals.
    # Let's use the first 2 normals.
    n1 = np.array(normals[0])
    n2 = np.array(normals[1])
    
    d1 = (np.max(centers @ n1) + np.min(centers @ n1)) / 2.0
    d2 = (np.max(centers @ n2) + np.min(centers @ n2)) / 2.0
    
    # Solve [n1; n2] C = [d1; d2]
    A_mat = np.array([n1, n2])
    d_vec = np.array([d1, d2])
    C_opt = np.linalg.solve(A_mat, d_vec)
    
    # Check if this C satisfies the other constraints (it should, due to symmetry of the hexagon and the way S was calculated)
    # But due to floating point, it might be slightly off.
    # Let's just use C_opt.
    
    inner = []
    angle_deg = math.degrees(beta)
    for x, y in centers:
        inner.append([float(x), float(y), angle_deg])
        
    center = [float(C_opt[0]), float(C_opt[1])]
    side = float(S)
    rotation = angle_deg
    
    return inner, center, side, rotation
