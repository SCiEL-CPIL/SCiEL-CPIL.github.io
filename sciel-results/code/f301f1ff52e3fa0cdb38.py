# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize

def hexagon_packing_11():
    """
    Finds a packing of 11 unit regular hexagons inside a minimal regular hexagon.
    Strategy: Symmetry-reduced continuous optimization of a 'flower' configuration.
    Model: 1 center, 6 in inner ring, 4 in outer ring.
    Variables: r1 (inner radius), r2 (outer radius), theta1 (inner offset), theta2 (outer offset).
    Optimization: Minimize outer hex side length S, subject to non-overlap.
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin for 600s limit

    SQRT3 = np.sqrt(3.0)
    APOTHEM = SQRT3 / 2.0

    # Base vertices for unit hexagon at (0,0) with angle 0
    # Vertices at angles 0, 60, 120, 180, 240, 300
    base_verts = np.array([
        [1.0, 0.0],
        [0.5, SQRT3/2.0],
        [-0.5, SQRT3/2.0],
        [-1.0, 0.0],
        [-0.5, -SQRT3/2.0],
        [0.5, -SQRT3/2.0]
    ])

    def get_hex_vertices(cx, cy, angle_deg):
        """Get vertices of a unit hexagon."""
        theta = np.deg2rad(angle_deg)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        return base_verts @ R.T + np.array([cx, cy])

    def get_hex_normals(cx, cy, angle_deg):
        """Get outward normals of edges for a unit hexagon."""
        # Normals are perpendicular to edges.
        # Edge midpoints are at radius sqrt(3)/2, angles 30, 90, ... relative to hex angle.
        # So normals are at angles angle_deg + 30 + 60*k.
        angles = np.deg2rad(angle_deg + 30 + 60 * np.arange(6))
        return np.array([np.cos(angles), np.sin(angles)]).T

    def build_configuration(r1, r2, t1_deg, t2_deg):
        """
        Build centers and angles for the 11 hexagons based on symmetry parameters.
        r1: radius of inner ring (6 hexagons)
        r2: radius of outer ring (4 hexagons)
        t1_deg: angular offset for inner ring
        t2_deg: angular offset for outer ring
        All inner hexagons have angle 0 (aligned with outer).
        """
        centers = np.zeros((11, 2))
        angles = np.zeros(11)
        
        # 1. Center
        centers[0] = [0.0, 0.0]
        angles[0] = 0.0
        
        # 2. Inner Ring (6 hexagons)
        # Placed at angles t1, t1+60, ..., t1+300
        for i in range(6):
            a = np.deg2rad(t1_deg + 60 * i)
            centers[1 + i] = [r1 * np.cos(a), r1 * np.sin(a)]
            angles[1 + i] = 0.0
            
        # 3. Outer Ring (4 hexagons)
        # Placed at angles t2, t2+90, t2+180, t2+270
        # This configuration is often optimal for filling the "corners" of the hexagon
        for i in range(4):
            a = np.deg2rad(t2_deg + 90 * i)
            centers[7 + i] = [r2 * np.cos(a), r2 * np.sin(a)]
            angles[7 + i] = 0.0

        return centers, angles

    def compute_separation_violation(centers, angles):
        """
        Compute the maximum overlap depth between any pair of hexagons.
        Returns 0 if valid, >0 if intersecting.
        Uses SAT: For each pair, find the minimum gap over all 12 normals.
        If min_gap < 0, they overlap. We want to maximize the minimum gap (or minimize negative gap).
        """
        n = len(centers)
        max_violation = 0.0
        
        # Precompute vertices and normals
        verts = []
        norms = []
        for i in range(n):
            verts.append(get_hex_vertices(centers[i][0], centers[i][1], angles[i]))
            norms.append(get_hex_normals(centers[i][0], centers[i][1], angles[i]))
            
        for i in range(n):
            for j in range(i+1, n):
                v1 = verts[i]
                v2 = verts[j]
                
                min_gap = float('inf')
                
                # Check normals from hex i
                for k in range(6):
                    n_vec = norms[i][k]
                    p1 = np.dot(v1, n_vec)
                    p2 = np.dot(v2, n_vec)
                    
                    min1, max1 = np.min(p1), np.max(p1)
                    min2, max2 = np.min(p2), np.max(p2)
                    
                    # Gap is negative if overlapping
                    # If intervals [min1, max1] and [min2, max2] overlap:
                    # gap = min(max1, max2) - max(min1, min2)
                    # If separated, gap > 0.
                    # We want to know the "worst" (most negative) gap for this pair.
                    gap = min(max1, max2) - max(min1, min2)
                    if gap < min_gap:
                        min_gap = gap
                        
                # Check normals from hex j
                for k in range(6):
                    n_vec = norms[j][k]
                    p1 = np.dot(v1, n_vec)
                    p2 = np.dot(v2, n_vec)
                    
                    min1, max1 = np.min(p1), np.max(p1)
                    min2, max2 = np.min(p2), np.max(p2)
                    
                    gap = min(max1, max2) - max(min1, min2)
                    if gap < min_gap:
                        min_gap = gap
                
                # If min_gap < 0, there is an overlap of depth -min_gap
                if min_gap < max_violation:
                    max_violation = min_gap
                    
        # Return the violation magnitude (positive if invalid)
        return max(0.0, -max_violation)

    def compute_outer_side(centers, angles):
        """
        Compute the side length of the smallest enclosing regular hexagon (centered at origin, angle 0).
        """
        max_proj = 0.0
        for i in range(len(centers)):
            v = get_hex_vertices(centers[i][0], centers[i][1], angles[i])
            # Outer hex normals are fixed at 30, 90, 150, 210, 270, 330 degrees
            for k in range(6):
                ang = np.deg2rad(30 + 60 * k)
                n_vec = np.array([np.cos(ang), np.sin(ang)])
                projs = np.dot(v, n_vec)
                max_proj = max(max_proj, np.max(projs))
        
        S = max_proj / APOTHEM
        return S

    def objective(params):
        """
        Objective function for optimization.
        params: [r1, r2, t1, t2]
        Returns S + penalty * violation
        """
        r1, r2, t1, t2 = params
        
        # Basic bounds to avoid weird geometries
        if r1 < 1.5 or r1 > 3.0:
            return 100.0
        if r2 < 2.5 or r2 > 5.0:
            return 100.0
            
        centers, angles = build_configuration(r1, r2, t1, t2)
        
        S = compute_outer_side(centers, angles)
        violation = compute_separation_violation(centers, angles)
        
        # Penalty: If violation > 0, add a large penalty
        # We want to minimize S, but S is only valid if violation == 0.
        # Using a smooth penalty helps gradient-based methods.
        penalty = 10.0 * violation**2 + 5.0 * violation
        return S + penalty

    def evaluate_final(centers, angles):
        """
        Strict validation and return S if valid.
        """
        # Check separation strictly
        violation = compute_separation_violation(centers, angles)
        if violation > 1e-6:
            return None
            
        # Check containment strictly
        S = compute_outer_side(centers, angles)
        
        # Verify all inner vertices are inside outer hex
        outer_verts = []
        for k in range(6):
            ang = np.deg2rad(60 * k)
            outer_verts.append([S * np.cos(ang), S * np.sin(ang)])
        outer_verts = np.array(outer_verts)
        
        # Check cross product condition for each inner vertex
        for i in range(11):
            v = get_hex_vertices(centers[i][0], centers[i][1], angles[i])
            for p in v:
                # Check if p is inside outer hex
                # For each edge (p1, p2) of outer hex, cross(p2-p1, p-p1) should have same sign as cross(p2-p1, center-p1)
                # Since outer hex is convex and centered at 0, and p is inside, 
                # the cross product with the edge vector should be non-negative (assuming CCW winding).
                # Outer hex vertices: (S,0), (S/2, S*sqrt(3)/2), ... CCW.
                is_inside = True
                for k in range(6):
                    p1 = outer_verts[k]
                    p2 = outer_verts[(k+1)%6]
                    edge = p2 - p1
                    vec = p - p1
                    cross = edge[0]*vec[1] - edge[1]*vec[0]
                    # For CCW polygon, interior points have cross > 0
                    if cross < -1e-6:
                        is_inside = False
                        break
                if not is_inside:
                    return None
                    
        return S

    best_S = float('inf')
    best_params = None
    
    # Initial guesses for optimization
    # r1 ~ sqrt(3) ~ 1.732
    # r2 ~ 2*sqrt(3) ~ 3.464
    # t1 ~ 0 or 30
    # t2 ~ 30 or 90 (to fit in corners)
    
    initial_guesses = [
        [1.732, 3.464, 0.0, 30.0],
        [1.732, 3.464, 30.0, 0.0],
        [1.75, 3.5, 0.0, 45.0],
        [1.70, 3.40, 15.0, 15.0],
        [1.732, 3.6, 0.0, 90.0],
    ]
    
    bounds = [
        (1.6, 2.0),   # r1
        (3.0, 4.0),   # r2
        (-10, 10),    # t1 (degrees)
        (-45, 45)     # t2 (degrees)
    ]
    
    for guess in initial_guesses:
        if time.time() > deadline - 60:
            break
            
        try:
            # Use Powell for robustness without gradients (expensive but safe)
            # Or L-BFGS-B if we had gradients. Let's use Nelder-Mead or Powell.
            # Powell is good for box-constrained problems.
            res = scipy.optimize.minimize(objective, guess, method='Powell', 
                                          bounds=bounds, 
                                          options={'maxiter': 500, 'maxfev': 5000, 'xtol': 1e-8, 'ftol': 1e-8})
            
            if res.success or res.fun < best_S:
                params = res.x
                centers, angles = build_configuration(*params)
                S = evaluate_final(centers, angles)
                
                if S is not None and S < best_S:
                    best_S = S
                    best_params = params
                    
        except Exception as e:
            pass

    # If optimization failed or didn't find valid, try a few manual tweaks around the best guess
    if best_S == float('inf'):
        # Fallback to a known valid configuration (the seed's or a simple one)
        # Let's construct a valid one manually:
        # Center (0,0)
        # 6 at r=sqrt(3), angles 0..300
        # 4 at r=3.5, angles 30, 120, 210, 300 (Tetrahedral projection?)
        # Let's just use the grid search fallback from the previous version if needed,
        # but for now, let's try a specific valid config.
        
        # Config: 1 center, 6 ring (r=1.732), 4 outer (r=3.5, angles 30, 120, 210, 300)
        # This might overlap. 
        # Safer: 1 center, 6 ring (r=1.732), 4 outer (r=3.8, angles 0, 90, 180, 270)
        
        centers = np.zeros((11, 2))
        angles = np.zeros(11)
        centers[0] = [0,0]
        for i in range(6):
            a = np.deg2rad(60*i)
            centers[1+i] = [1.732*np.cos(a), 1.732*np.sin(a)]
        for i in range(4):
            a = np.deg2rad(90*i)
            centers[7+i] = [3.8*np.cos(a), 3.8*np.sin(a)]
            
        S = evaluate_final(centers, angles)
        if S is not None:
            best_S = S
            best_params = [1.732, 3.8, 0.0, 0.0] # Approx params

    # Final Refinement: Local search around best_params
    if best_params is not None:
        # Do a small local search to polish
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            # Perturb parameters slightly
            noise = np.random.normal(0, 0.01, 4)
            new_params = np.clip(best_params + noise, bounds[0][0], bounds[0][1])
            # Handle angle bounds separately
            new_params[2] = (best_params[2] + noise[2] + 180) % 360 - 180
            new_params[3] = (best_params[3] + noise[3] + 180) % 360 - 180
            
            centers, angles = build_configuration(*new_params)
            S = evaluate_final(centers, angles)
            if S is not None and S < best_S:
                best_S = S
                best_params = new_params

    # Construct final output
    if best_params is None:
        # Emergency fallback: Return a very large hexagon with hexagons far apart
        # This should not happen, but ensures return type.
        centers = np.zeros((11, 2))
        angles = np.zeros(11)
        for i in range(11):
            centers[i] = [i*2, 0]
        S = 25.0
    else:
        centers, angles = build_configuration(*best_params)

    inner_hex_data = np.zeros((11, 3))
    for i in range(11):
        inner_hex_data[i, 0] = centers[i, 0]
        inner_hex_data[i, 1] = centers[i, 1]
        inner_hex_data[i, 2] = angles[i]
        
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, best_S

# EVOLVE-BLOCK-END
