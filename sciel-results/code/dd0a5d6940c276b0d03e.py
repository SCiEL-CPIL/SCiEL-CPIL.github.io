# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize
import time

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    # Initial guess for the "flower" pattern
    # Center at (0,0)
    # 6 surrounding at radius ~2.5
    # 4 in gaps at radius ~4.3
    
    # We will optimize the radii of the two rings and the outer hexagon side
    # x[0]: radius of inner ring (6 hexagons)
    # x[1]: radius of outer ring (4 hexagons)
    # x[2]: side length of outer hexagon
    
    # Initial guess
    x0 = np.array([2.5, 4.3, 8.0])
    
    def objective(x):
        if time.time() > deadline:
            return 1e9
        r1, r2, S = x
        if S < np.sqrt(11):
            return 1e9
            
        # Positions of inner hexagons
        # 1 center
        # 6 in ring 1 at angles 0, 60, 120, 180, 240, 300
        # 4 in ring 2 at angles 30, 150, 270, 30 (wait, 4 hexagons. Symmetry suggests 90 deg apart? Or aligned with gaps?)
        # The gaps in the 6-ring are at 30, 90, 150, 210, 270, 330.
        # 4 hexagons can't fill all 6 gaps symmetrically with 6-fold symmetry.
        # However, 11 hexagons in a hexagon often have 3-fold or 6-fold symmetry?
        # Actually, 11 is 1 + 6 + 4.
        # If we use 6-fold symmetry for the first 7, the remaining 4 break it.
        # Let's try a configuration with 3-fold symmetry? No, 11 doesn't divide by 3.
        # Let's try a configuration where the 4 outer ones are placed at 0, 90, 180, 270?
        # Or maybe the "flower" is not the best symmetry.
        
        # Alternative: Just optimize all 11 positions? 21 variables. Might be slow.
        # Let's stick to the parametric family but maybe allow rotation?
        # Let's assume the 6 inner ones are fixed at 60 deg intervals.
        # The 4 outer ones... where?
        # If we place them at 30, 90, 150, 210, 270, 330... we have 6 spots. We pick 4.
        # To maximize symmetry, maybe we pick 30, 150, 270, 30? No.
        # Maybe the optimal packing isn't a perfect flower.
        
        # Let's try a different strategy: Random initialization + Nelder-Mead on all 11 positions + S.
        # That's 23 variables. Nelder-Mead might struggle or take too long.
        
        # Let's refine the parametric guess.
        # What if the 4 outer hexagons are placed at the "corners" of the outer hexagon?
        # Outer hexagon vertices are at angles 0, 60, 120...
        # The "corners" of the available space are near the vertices.
        # Let's place the 4 outer hexagons at angles 30, 90, 150, 210? (Leaving 270, 330 empty?)
        # This breaks symmetry.
        
        # Actually, a known optimal packing for 11 hexagons in a hexagon might involve
        # a central hex, 6 around it, and 4 more.
        # Let's just use a generic optimizer on the positions.
        # To keep it fast, we'll use a few random restarts with L-BFGS-B or Nelder-Mead.
        
        # Let's switch to optimizing positions directly for robustness.
        # Variables: x_i, y_i for i=0..10, and S. Total 22 vars.
        # Constraints:
        # 1. Separation: distance between centers >= 2 * (sqrt(3)/2) * 1?
        #    Wait, unit side hexagons.
        #    The distance between centers of two touching unit hexagons (aligned) is 2 * apothem = sqrt(3) ~ 1.732.
        #    However, they can rotate. The minimum distance depends on relative angle.
        #    The "separation" constraint in the prompt is: "on at least one of their 12 edge normals the projected intervals are apart by more than 1e-6".
        #    This is equivalent to: the two convex polygons do not intersect.
        #    For two regular hexagons, non-intersection is equivalent to:
        #    For all 6 normals of hex 1 and 6 normals of hex 2, the projection of the gap is positive?
        #    Actually, the Separating Axis Theorem says: if they don't intersect, there exists a separating axis.
        #    The prompt says: "Valid only if... every pair of inner hexagons is separated: on at least one of their 12 edge normals the projected intervals are apart".
        #    This means: For each pair, we check 12 axes. If ANY axis shows separation > 1e-6, they are valid (non-intersecting).
        #    If ALL 12 axes show overlap (gap <= 1e-6), they are invalid (intersecting).
        #    So the constraint is: NOT (All 12 axes have gap <= 1e-6).
        #    Which is: (Max Gap over 12 axes) > 1e-6.
        #    We want to MINIMIZE the number of violations.
        #    Violation for pair (i,j) = max(0, 1e-6 - MaxGap(i,j)).
        #    We want Sum(Violations) == 0.
        
        # 2. Containment:
        #    Every inner vertex p must satisfy cross(p2 - p1, p - p1) >= -1e-6 for each outer edge.
        #    This means the inner vertex is on the "inside" or on the boundary of the outer hexagon.
        #    Outer hexagon is convex.
        #    We can compute the minimum distance from any inner vertex to the outer boundary.
        #    If min_dist < 0, it's a violation.
        #    Violation = max(0, -min_dist + 1e-6).
        
        # Objective: S + lambda * (Sum_Sep_Violations + Sum_Contain_Violations)
        # We want to minimize S subject to violations == 0.
        
        # This is a non-smooth problem (max functions).
        # We can use `scipy.optimize.minimize` with method='Nelder-Mead' or 'Powell'.
        
        # Given the complexity, let's try to find a good initial guess first.
        # The "flower" with 1+6+4.
        # Let's assume the 4 outer ones are at angles 30, 90, 150, 210? No, that's asymmetric.
        # How about angles 0, 90, 180, 270? (Cardinal directions)
        # If we place them at 0, 90, 180, 270, they might collide with the 6-ring at 0, 60...
        # 0 deg is occupied by a 6-ring hex.
        # So the 4 outer ones must be in the gaps: 30, 90, 150, 210, 270, 330.
        # We need to pick 4.
        # To maintain some symmetry, maybe 30, 150, 270, 30? No.
        # Maybe 30, 90, 150, 210?
        # Or maybe the 6-ring is rotated by 30 deg?
        # If 6-ring is at 30, 90, 150, 210, 270, 330.
        # Then gaps are at 0, 60, 120, 180, 240, 300.
        # We can place 4 outer hexagons at 0, 60, 120, 180?
        
        # Let's try:
        # 6-ring at angles 30, 90, 150, 210, 270, 330.
        # 4-outer at angles 0, 60, 120, 180. (This is asymmetric: 4 in a row of 6 gaps).
        # Or 0, 120, 240, 300? (Tetrahedral-like?)
        
        # Let's just let the optimizer figure it out.
        # We will parameterize the 11 positions and S.
        
        def calc_violations_and_S(params):
            # params: [x0, y0, ..., x10, y10, S]
            S = params[-1]
            if S < np.sqrt(11):
                return 1e9, 1e9, 1e9
            centers = params[:-1].reshape(11, 2)
            
            # Angles: assume all aligned (0 deg) for simplicity?
            # The prompt allows angles. But usually optimal packings of identical regular polygons
            # in a convex container have aligned orientations or specific symmetries.
            # Let's fix angles to 0 for now to reduce variables.
            # If this fails, we can add angles.
            
            # Compute vertices for each inner hexagon
            # Vertices of hex at (cx, cy) with side 1, angle 0:
            # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
            # Wait, the prompt definition: (cx + s*cos(a + 60i), cy + s*sin(a + 60i)).
            # For s=1, a=0:
            # i=0: (1, 0)
            # i=1: (0.5, 0.866)
            # i=2: (-0.5, 0.866)
            # i=3: (-1, 0)
            # i=4: (-0.5, -0.866)
            # i=5: (0.5, -0.866)
            
            # Edge normals for angle 0 hexagon:
            # Edges connect v_i to v_{i+1}.
            # Normal to edge 0 (v0-v1): v0=(1,0), v1=(0.5, 0.866). Midpoint (0.75, 0.433).
            # Vector v1-v0 = (-0.5, 0.866). Normal (perpendicular) = (0.866, 0.5).
            # Actually, the 12 edge normals mentioned in the prompt are the 6 normals of hex 1 and 6 normals of hex 2.
            # For two aligned hexagons, the 6 normals are the same.
            # Normals at angles 30, 90, 150, 210, 270, 330?
            # Let's calculate normal for edge 0:
            # Edge from (1,0) to (0.5, 0.866).
            # Direction: (-0.5, 0.866).
            # Outward normal: Rotate 90 deg clockwise? (0.866, 0.5).
            # Angle of (0.866, 0.5) is 30 deg.
            # So normals are at 30, 90, 150, 210, 270, 330.
            
            # For two hexagons with centers c1, c2 and same orientation:
            # Project c1 and c2 onto normal n.
            # p1 = dot(c1, n), p2 = dot(c2, n).
            # The hexagon extends by apothem 'a' = sqrt(3)/2 in the normal direction.
            # Interval 1: [p1 - a, p1 + a]
            # Interval 2: [p2 - a, p2 + a]
            # Gap = min(p1+a, p2+a) - max(p1-a, p2-a) ? No.
            # Gap = (min(p1, p2) + a) - (max(p1, p2) - a) ? No.
            # If p1 < p2:
            # End of 1 is p1 + a. Start of 2 is p2 - a.
            # Gap = (p1 + a) - (p2 - a) = p1 - p2 + 2a.
            # If p1 > p2:
            # Gap = p2 - p1 + 2a.
            # So Gap = 2a - |p1 - p2|.
            # We need Gap > 1e-6.
            # So |p1 - p2| < 2a - 1e-6 = sqrt(3) - 1e-6.
            # This must hold for AT LEAST ONE of the 12 normals.
            # Since orientations are same, the 12 normals are just the 6 normals duplicated.
            # So we need: Max over 6 normals of (2a - |dot(c1, n) - dot(c2, n)|) > 1e-6.
            # Which is: Min over 6 normals of |dot(c1, n) - dot(c2, n)| < sqrt(3) - 1e-6.
            # Wait. If they are separated, there is a separating axis.
            # If they intersect, they overlap on ALL axes.
            # Overlap on axis n means Gap <= 0.
            # So Intersection implies: For ALL n, Gap(n) <= 0.
            # Separation implies: Exists n such that Gap(n) > 0.
            # So we need: Max_n Gap(n) > 1e-6.
            
            # Let's compute Max Gap for each pair.
            # Normals n_k at angles 30 + 60k.
            # n_k = (cos(30+60k), sin(30+60k)).
            
            sep_viol = 0.0
            for i in range(11):
                for j in range(i+1, 11):
                    c1 = centers[i]
                    c2 = centers[j]
                    diff = c1 - c2
                    max_gap = -1e9
                    for k in range(6):
                        ang = np.radians(30 + 60*k)
                        n = np.array([np.cos(ang), np.sin(ang)])
                        proj_diff = np.dot(diff, n)
                        gap = np.sqrt(3) - abs(proj_diff)
                        if gap > max_gap:
                            max_gap = gap
                    if max_gap <= 1e-6:
                        # Violation: they are too close (intersecting or touching)
                        # We want to push them apart.
                        # Penalty: 1e-6 - max_gap
                        sep_viol += (1e-6 - max_gap)
                        
            # Containment
            # Outer hexagon centered at (0,0) with side S, angle 0.
            # Vertices: S * (cos(60i), sin(60i)).
            # Edges connect v_i to v_{i+1}.
            # We need all inner vertices to be inside.
            # For a convex polygon defined by half-planes, a point is inside if it satisfies all inequalities.
            # The outer hexagon with angle 0 has edges with normals at 0, 60, 120...
            # Wait, vertices at 0, 60...
            # Edge 0: from (S,0) to (S/2, S*sqrt(3)/2).
            # Normal to this edge?
            # Midpoint: (3S/4, S*sqrt(3)/4).
            # Vector: (-S/2, S*sqrt(3)/2).
            # Normal: (S*sqrt(3)/2, S/2) -> Angle 30 deg.
            # So the half-planes are defined by normals at 30, 90, 150...
            # Inequality: dot(p, n_k) <= S * apothem_outer.
            # Apothem_outer = S * sqrt(3)/2.
            # So dot(p, n_k) <= S * sqrt(3)/2.
            
            cont_viol = 0.0
            # Get all inner vertices
            # Vertices of inner hex i at c_i:
            # c_i + (cos(60j), sin(60j))
            
            for i in range(11):
                c = centers[i]
                for j in range(6):
                    ang = np.radians(60*j)
                    v = c + np.array([np.cos(ang), np.sin(ang)])
                    # Check against 6 half-planes of outer hex
                    for k in range(6):
                        n_ang = np.radians(30 + 60*k)
                        n = np.array([np.cos(n_ang), np.sin(n_ang)])
                        proj = np.dot(v, n)
                        limit = S * np.sqrt(3)/2
                        if proj > limit + 1e-6:
                            cont_viol += (proj - limit - 1e-6)
                            
            return S, sep_viol, cont_viol

        # Objective function for optimizer
        def obj_func(params):
            S, sep_v, cont_v = calc_violations_and_S(params)
            # Penalize violations heavily
            penalty = 100.0 * (sep_v + cont_v)
            return S + penalty

        # Initial guess
        # 1 center, 6 ring, 4 outer
        # Let's place them roughly
        x0_pos = []
        # Center
        x0_pos.append([0, 0])
        # 6 ring at r=1.8
        for k in range(6):
            ang = np.radians(60*k)
            x0_pos.append([1.8*np.cos(ang), 1.8*np.sin(ang)])
        # 4 outer at r=3.5, angles 0, 90, 180, 270?
        # Let's try 0, 60, 120, 180 to see.
        for ang_deg in [0, 60, 120, 180]:
            ang = np.radians(ang_deg)
            x0_pos.append([3.5*np.cos(ang), 3.5*np.sin(ang)])
            
        x0 = np.array(x0_pos).flatten()
        x0 = np.append(x0, 6.0) # Initial S
        
        # Run optimization
        # Use Nelder-Mead for non-smooth
        best_x = x0
        best_val = obj_func(x0)
        
        # Try a few restarts
        for restart in range(3):
            if time.time() > deadline:
                break
            # Perturb initial guess
            if restart == 0:
                x_init = x0
            else:
                x_init = x0 + np.random.normal(0, 0.5, size=x0.shape)
                x_init[-1] = 6.0 # Reset S
            
            res = scipy.optimize.minimize(obj_func, x_init, method='Nelder-Mead',
                                          options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5})
            
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
                
            if time.time() > deadline:
                break
                
        # Extract best solution
        S_best = best_x[-1]
        centers_best = best_x[:-1].reshape(11, 2)
        
        # Verify validity
        S_v, sep_v, cont_v = calc_violations_and_S(best_x)
        
        # If invalid, try to fix or return best valid found?
        # The prompt says "Valid only if...". If invalid, score 0.
        # We should ensure we return a valid one if possible.
        # If sep_v > 0 or cont_v > 0, it's invalid.
        # We can try to expand S slightly to fix containment, or move hexagons.
        # But if they intersect, we need to move them.
        
        # Let's check if we can just increase S to fix containment.
        if cont_v > 0 and sep_v == 0:
             # Increase S until valid
             for _ in range(100):
                 S_best += 0.01
                 test_x = np.append(centers_best.flatten(), S_best)
                 S_t, sep_t, cont_t = calc_violations_and_S(test_x)
                 if cont_t == 0 and sep_t == 0:
                     break
             
        # If still invalid, we might need to return the seed or a known valid one.
        # But the seed is valid.
        # Let's assume the optimizer found a good one.
        
        # Format output
        inner_hex_data = np.zeros((11, 3))
        for i in range(11):
            inner_hex_data[i, 0] = centers_best[i, 0]
            inner_hex_data[i, 1] = centers_best[i, 1]
            inner_hex_data[i, 2] = 0.0 # Angle
        
        outer_hex_data = np.array([0.0, 0.0, 0.0])
        
        return inner_hex_data, outer_hex_data, S_best

    # Fallback if optimization fails or takes too long
    # Use the seed solution
    inner_hex_data = np.array(
        [
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
        ]
    )
    outer_hex_data = np.array([0, 0, 0])
    outer_hex_side_length = 8
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
