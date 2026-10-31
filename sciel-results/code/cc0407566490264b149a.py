import math
import time
import numpy as np
import itertools

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # 1. Define the feasible region for centers.
    # We want to pack N=11 unit hexagons inside a regular hexagon of side S.
    # The center of a unit hexagon must be at least the inradius (apothem) away from the boundary.
    # For a hexagon of side S, the apothem is S * sqrt(3)/2.
    # The region of valid centers is a hexagon of side S - 1 (since apothem(S-1) = S*sqrt(3)/2 - sqrt(3)/2).
    # Wait, let's verify.
    # Distance from center to side is apothem.
    # Inner hexagon has apothem a_in = sqrt(3)/2.
    # Outer hexagon has apothem a_out = S * sqrt(3)/2.
    # The center of the inner hex must be at least a_in away from the outer boundary.
    # So the distance from the outer center to the inner center, projected on the normal, must be <= a_out - a_in.
    # This defines a smaller hexagon of "effective side" S' such that a_out(S') = a_out(S) - a_in.
    # S' * sqrt(3)/2 = S * sqrt(3)/2 - sqrt(3)/2  => S' = S - 1.
    # So the centers must lie within a regular hexagon of side S-1, centered at the origin.
    
    # We need to find the minimum S such that we can place N points in a hexagon of side S-1
    # such that pairwise distances are >= 2 * apothem = sqrt(3).
    
    # Let R = S - 1 be the side length of the center-hexagon.
    # We want to minimize S = R + 1.
    # So we minimize R.
    
    # Strategy:
    # 1. Generate a dense triangular lattice of points.
    # 2. Filter points that fit inside a hexagon of side R.
    # 3. Find a subset of N points from this lattice that are pairwise distance >= sqrt(3).
    #    Note: On a triangular lattice with spacing d, the distance between nearest neighbors is d.
    #    If we set d = sqrt(3), nearest neighbors touch. This is valid (disjoint interiors).
    #    So we can use a lattice with spacing exactly sqrt(3).
    #    Any two points on this lattice that are not nearest neighbors are further apart.
    #    Actually, we just need to ensure no two selected points are closer than sqrt(3).
    #    On a lattice with spacing sqrt(3), the minimum distance is sqrt(3). So ANY subset is valid!
    #    Wait, is that true?
    #    Yes, if the lattice spacing is exactly the minimum allowed distance, then any two distinct points
    #    are at least that distance apart.
    #    So the problem reduces to: Find the smallest R such that a regular hexagon of side R
    #    contains at least N points from a triangular lattice of spacing sqrt(3).
    
    # Let's verify the lattice orientation.
    # Standard triangular lattice basis:
    # v1 = (d, 0)
    # v2 = (d/2, d * sqrt(3)/2)
    # where d = sqrt(3).
    
    d = math.sqrt(3)
    v1 = np.array([d, 0.0])
    v2 = np.array([d/2.0, d * math.sqrt(3)/2.0])
    
    # Generate a large set of lattice points.
    # We need to cover a hexagon of side R.
    # R is expected to be around 3 or 4 for N=11.
    # Let's generate points for i, j in range [-10, 10].
    
    max_idx = 15
    points = []
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            p = i * v1 + j * v2
            points.append(p)
    points = np.array(points)
    
    # We want to find the minimum R such that we can pick N points from 'points'
    # that all lie inside a hexagon of side R centered at the origin.
    
    # Function to check if a point is inside a regular hexagon of side R centered at origin.
    # A regular hexagon with vertices at (R, 0), (R/2, R*sqrt(3)/2), etc.
    # The condition for (x, y) to be inside is:
    # max( |x|, |x/2 + y*sqrt(3)/2|, |-x/2 + y*sqrt(3)/2| ) <= R
    # Or more simply, the support function of the hexagon.
    # The normals to the sides are at angles 30, 90, 150, 210, 270, 330 degrees.
    # Let's use the formula:
    # A point (x,y) is in hex(R) if and only if:
    # |x| <= R
    # |x/2 + y*sqrt(3)/2| <= R
    # |-x/2 + y*sqrt(3)/2| <= R
    
    def is_in_hexagon(p, R):
        x, y = p
        return (abs(x) <= R) and (abs(x/2.0 + y*math.sqrt(3)/2.0) <= R) and (abs(-x/2.0 + y*math.sqrt(3)/2.0) <= R)
    
    # Vectorized check
    def get_points_in_hexagon(R):
        mask = (
            (np.abs(points[:, 0]) <= R) &
            (np.abs(points[:, 0]/2.0 + points[:, 1]*math.sqrt(3)/2.0) <= R) &
            (np.abs(-points[:, 0]/2.0 + points[:, 1]*math.sqrt(3)/2.0) <= R)
        )
        return points[mask]
    
    # We search for the smallest R.
    # Since the lattice is discrete, R will take specific values.
    # However, we can just iterate R from a lower bound upwards.
    # Lower bound: Area of hexagon side R is (3*sqrt(3)/2) * R^2.
    # Area of N hexagons is N * (3*sqrt(3)/2) * 1^2.
    # So R^2 >= N => R >= sqrt(11) ~ 3.3.
    # Let's try R from 3.0 to 5.0 in small steps.
    
    best_R = float('inf')
    best_subset = None
    
    # We will iterate R. For each R, we get the list of points inside.
    # If count >= N, we have a valid configuration.
    # To minimize R, we just find the first R where count >= N.
    # But wait, the points must be a subset of the lattice.
    # Since ANY subset of the lattice is valid (due to spacing), we just need the count to be >= N.
    # Is it possible that a smaller R allows N points if we shift the lattice?
    # The problem allows translation of the inner hexagons.
    # So we can shift the entire lattice.
    # To find the optimal shift, we can try a few shifts or optimize the center.
    # However, the lattice is periodic. The "density" is constant.
    # The number of points in a convex shape is roughly Area / CellArea.
    # To maximize the number of points for a given R, we should center the hexagon on a lattice point
    # or on the center of a triangle of lattice points, depending on the shape.
    # For a hexagon, centering on a lattice point is usually good.
    
    # Let's try to optimize the center of the hexagon relative to the lattice.
    # We can define the hexagon center (cx, cy) and check how many lattice points fall inside.
    # We want to maximize the count for a given R, or find min R for count N.
    
    # Let's do a grid search over the center (cx, cy) in the fundamental domain of the lattice [0, d]x[0, d*sqrt(3)].
    # Actually, due to symmetry, we can just try a few offsets.
    # Or, we can use the fact that the optimal packing of equal circles (or hexagons) in a container
    # often aligns with the container's symmetry.
    
    # Let's try centering the hexagon at the origin and shifting the lattice.
    # Equivalent to shifting the hexagon relative to the lattice.
    
    # We will iterate R. For each R, we try several offsets (dx, dy) in [0, d) x [0, d*sqrt(3)).
    # For each offset, we count points. If count >= N, we found a solution.
    
    # To be efficient, we can precompute the lattice points.
    # Then for a given R and offset, we check which points are in hexagon centered at offset.
    
    # Let's refine the search range for R.
    # sqrt(11) = 3.316.
    # Let's check R = 3.3, 3.4, 3.5, etc.
    
    # We need to return the coordinates of the inner hexagons.
    # If we use a lattice point p, the hexagon center is p - offset (if we shifted the hexagon by offset).
    # Or if we shifted the lattice by offset, the points are p + offset.
    # Let's say we fix the lattice at origin. We move the hexagon center to (cx, cy).
    # The points inside are those p such that p is in hex(cx, cy, R).
    # The hexagon centers are p.
    # The outer hexagon center is (cx, cy).
    # The outer hexagon side is S = R + 1.
    # The outer hexagon angle is 0 (aligned with lattice).
    # The inner hexagons are at p, angle 0.
    
    # We want to minimize S.
    
    # Let's iterate R from 3.3 to 4.0 with step 0.01.
    # For each R, try a grid of centers in the fundamental domain.
    
    found_solution = False
    best_S = float('inf')
    best_centers = None
    best_outer_center = (0.0, 0.0)
    
    # Optimization: The number of points in the hexagon changes discretely as R increases.
    # We can just check specific R values corresponding to the distances of lattice points from the center.
    # But iterating R is easier to implement correctly.
    
    # To speed up, we can sort lattice points by distance from origin.
    # But we are shifting the center.
    
    # Let's just do a coarse-to-fine search.
    
    # Range of centers to try (fundamental domain of lattice)
    # Lattice basis: v1=(d,0), v2=(d/2, d*sqrt(3)/2)
    # Fundamental domain is a parallelogram.
    # We can just try a grid of centers in [-d, d] x [-d*sqrt(3), d*sqrt(3)] with step 0.1.
    # Or better, use the lattice symmetry.
    
    # Let's try centers at:
    # 1. Origin (0,0)
    # 2. Center of the triangle formed by (0,0), v1, v2. Centroid is (v1+v2)/3.
    # 3. Midpoint of v1.
    # 4. Midpoint of v2.
    # And their symmetric equivalents.
    
    candidate_centers = [
        (0.0, 0.0),
        (v1[0]/3.0, v1[1]/3.0 + v2[1]/3.0), # Centroid of 0, v1, v2
        (v1[0]/2.0, 0.0), # Midpoint v1
        (v2[0]/2.0, v2[1]/2.0), # Midpoint v2
        (v1[0]/3.0 + v2[0]/3.0, v2[1]/3.0), # Centroid of 0, v1, v2? No, that's (v1+v2)/3.
        # Let's just generate a small grid of centers in the fundamental domain.
    ]
    
    # Generate grid of centers in [0, d] x [0, d*sqrt(3)]
    # Actually, the fundamental domain is spanned by v1 and v2.
    # Let's just sample a few points.
    
    centers_to_try = []
    for i in range(4):
        for j in range(4):
            cx = i * v1[0]/4.0 + j * v2[0]/4.0
            cy = i * v1[1]/4.0 + j * v2[1]/4.0
            centers_to_try.append((cx, cy))
    
    # Add origin and centroid
    centers_to_try.append((0.0, 0.0))
    centroid = (v1 + v2) / 3.0
    centers_to_try.append((centroid[0], centroid[1]))
    
    # Search R
    # We know R must be at least sqrt(11) * (something).
    # Let's start R from 3.3.
    
    for R in np.arange(3.3, 4.5, 0.01):
        if time.time() > deadline - 10:
            break
            
        for cx, cy in centers_to_try:
            # Check points in hexagon centered at (cx, cy) with side R
            # p in hex(cx, cy, R) <=> p - (cx, cy) in hex(0, 0, R)
            shifted = points - np.array([cx, cy])
            mask = (
                (np.abs(shifted[:, 0]) <= R) &
                (np.abs(shifted[:, 0]/2.0 + shifted[:, 1]*math.sqrt(3)/2.0) <= R) &
                (np.abs(-shifted[:, 0]/2.0 + shifted[:, 1]*math.sqrt(3)/2.0) <= R)
            )
            count = np.sum(mask)
            
            if count >= N:
                # We found a valid R and center.
                # Get the N points.
                # Which N? Any N will do, as they are all valid.
                # But to be robust, we pick the first N.
                indices = np.where(mask)[0]
                selected_points = points[indices[:N]]
                
                S = R + 1.0
                if S < best_S:
                    best_S = S
                    best_centers = selected_points
                    best_outer_center = (cx, cy)
                    found_solution = True
                    # Since we iterate R upwards, the first valid R for a given center is minimal for that center.
                    # But we should check all centers for this R to see if a smaller S is possible?
                    # No, S depends only on R. So the first R that works for ANY center is the global minimum.
                    # We can break out of loops.
                    break
        
        if found_solution:
            break
    
    if not found_solution:
        # Fallback: Use a larger R or a different strategy if the lattice approach failed.
        # This shouldn't happen for N=11.
        # Let's just return a linear arrangement as fallback.
        inner = []
        for i in range(N):
            inner.append([i * d, 0.0, 0.0])
        # Compute S for this
        # Max x is (N-1)*d. Max projection on x-axis is (N-1)*d.
        # Apothem of outer hex must cover this.
        # S * sqrt(3)/2 >= (N-1)*d + sqrt(3)/2 (inradius of inner)
        # S * sqrt(3)/2 >= (10)*sqrt(3) + sqrt(3)/2
        # S >= 10.5
        best_S = 10.5
        best_centers = np.array([[i*d, 0.0] for i in range(N)])
        best_outer_center = (0.0, 0.0)

    # Construct the return values
    inner_list = []
    for p in best_centers:
        # The center of the inner hex is p.
        # The angle is 0.
        inner_list.append([float(p[0]), float(p[1]), 0.0])
    
    return inner_list, best_outer_center, float(best_S), 0.0

# EVOLVE-BLOCK-END
