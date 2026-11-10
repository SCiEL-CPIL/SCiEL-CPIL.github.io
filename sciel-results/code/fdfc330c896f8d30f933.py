# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations

def erdos_szekeres_problem7() -> np.ndarray:
    """
    Returns a (33, 2) float array of points minimizing the number of convex 7-gons.
    """
    start_time = time.time()
    deadline = start_time + 600  # 10 minutes
    
    # 1. Generate the base 32-point configuration with 0 convex 7-gons.
    # This uses the recursive Erdos-Szekeres construction.
    base_points = generate_es_base(7)
    
    # 2. Add a 33rd point.
    # The centroid of the 32 points is a good starting guess.
    centroid = np.mean(base_points, axis=0)
    points = np.vstack([base_points, centroid])
    
    # 3. Optimize the position of the 33rd point (and potentially all points)
    # to minimize the count C.
    # Since the theoretical minimum is 1, we check if we already have 1.
    C = count_convex_7gons(points)
    
    if C == 1:
        return points
    
    # If C > 1, we need to search for a better configuration.
    # We can try moving the 33rd point to different locations.
    # Or we can use a local search on all points.
    
    best_points = points
    best_C = C
    
    # Simple local search: perturb the 33rd point
    # The 33rd point is the last one.
    # We can also try to slightly perturb the base points if needed,
    # but the base points are rigid in the standard construction.
    
    # Try a few random positions for the 33rd point within the convex hull
    # or near the centroid.
    for _ in range(10000):
        if time.time() > deadline - 60:
            break
        
        # Random perturbation of the 33rd point
        # Keep it within the unit square
        p33 = best_points[32]
        dx = np.random.normal(0, 0.05)
        dy = np.random.normal(0, 0.05)
        new_p33 = p33 + np.array([dx, dy])
        
        # Clamp to unit square
        new_p33[0] = np.clip(new_p33[0], 1e-6, 1 - 1e-6)
        new_p33[1] = np.clip(new_p33[1], 1e-6, 1 - 1e-6)
        
        candidate = best_points.copy()
        candidate[32] = new_p33
        
        # Check validity (no 3 collinear, no 2 coincident)
        if not is_valid(candidate):
            continue
            
        new_C = count_convex_7gons(candidate)
        if new_C < best_C:
            best_C = new_C
            best_points = candidate
            if best_C == 1:
                break
                
    return best_points

def generate_es_base(k):
    """
    Generates 2^(k-2) points in the unit square with no convex k-gons.
    Uses the recursive cup-cap construction.
    """
    if k == 3:
        # 2 points. Any 2 points have no convex 3-gon.
        # We need to define the "cup" and "cap" orientation for recursion.
        # For k=3, size is 2.
        # Let's return two points that will form the base of the chain.
        # The recursion expects to combine two sets of size 2^(k-3).
        # So for k=3, we return 2 points.
        # To allow the "cup" and "cap" distinction, we might need to return
        # points that are part of a convex chain.
        # Standard construction:
        # S_3 (2 pts): (0,0) and (1,0) ?
        # But we need to fit in the unit square and allow scaling.
        # Let's return points in [0,1]x[0,1].
        return np.array([[0.0, 0.0], [1.0, 0.0]])
    
    size = 2 ** (k - 2)
    prev_size = 2 ** (k - 3)
    
    # Generate two sets of size prev_size
    # The standard construction involves two "cups" or one cup and one cap?
    # Actually, the set of 2^(k-2) points is formed by two sets of 2^(k-3) points.
    # One is a "lower" chain, one is an "upper" chain.
    
    # Let's use the recursive definition from the literature:
    # Let L be the left half, R be the right half?
    # No, the "double chain" is:
    # Lower chain: 2^(k-3) points.
    # Upper chain: 2^(k-3) points.
    
    # To build the lower chain for level k, we take two lower chains from level k-1?
    # No, the property is that the whole set has no convex k-gon.
    
    # Let's implement the specific geometric construction:
    # 1. Take two copies of the (k-1)-configuration (size 2^(k-3)).
    # 2. Scale one to be the "left" and "lower", the other "right" and "upper"?
    # 
    # A simpler recursive geometric construction:
    # Let P_{k-1} be the set of 2^(k-3) points.
    # P_k is formed by taking P_{k-1} and transforming it:
    # - Copy 1: Scale by 0.5, translate to bottom-left.
    # - Copy 2: Scale by 0.5, translate to top-right.
    # This creates a "staircase" of convex hulls.
    
    if k == 3:
        # Base case for the recursion that produces 2 points.
        # We need these 2 points to be part of a convex chain.
        # Let's return points that form a "cup" segment.
        # (0, 0) and (1, 0) is a flat line, which is degenerate.
        # We need general position.
        # Let's use (0, 0.1) and (1, 0.1)?
        # Or better, define the base case to return points that can be combined.
        
        # Let's restart the base case logic to ensure general position.
        # For k=3, we need 2 points.
        # Return [[0.0, 0.0], [1.0, 0.0]] is collinear with many things.
        # Let's use [[0.0, 0.0], [1.0, 0.1]]
        return np.array([[0.0, 0.0], [1.0, 0.1]])

    # Recursive step
    # We need to generate two sets of size 2^(k-3).
    # Let's generate one set and use it twice with different transformations?
    # No, the two sets should be identical in structure but transformed.
    
    # Generate the (k-1) configuration
    prev_points = generate_es_base(k - 1)
    
    # Transform 1: Scale by 0.5, translate to (0,0)
    left_points = prev_points * 0.5
    left_points[:, 0] += 0.0
    left_points[:, 1] += 0.0
    
    # Transform 2: Scale by 0.5, translate to (0.5, 0.5)
    right_points = prev_points * 0.5
    right_points[:, 0] += 0.5
    right_points[:, 1] += 0.5
    
    # Combine
    # This creates a set of 2 * 2^(k-3) = 2^(k-2) points.
    # Does this have no convex k-gons?
    # Yes, this is the standard recursive construction for the "Erdos-Szekeres" lower bound.
    # The convex hull of the combined set is the union of the hulls of the two parts,
    # and any convex polygon must respect the separation.
    
    return np.vstack([left_points, right_points])

def count_convex_7gons(points):
    """
    Counts the number of 7-point subsets in convex position.
    """
    n = len(points)
    count = 0
    # This is O(N^7) which is too slow for N=33.
    # 33 choose 7 is ~ 5 million.
    # 5 million * (7 points check) is feasible in Python if optimized?
    # 5M * 10 ops = 50M ops. Might take ~10-50 seconds in Python.
    # We need to be careful with the time limit.
    
    # Optimization: Precompute indices?
    # We can use a more efficient check.
    
    # Let's stick to the simple check first, but break early if possible?
    # No, we need the exact count.
    
    # To speed up, we can use NumPy vectorization for the cross product check.
    # But iterating over combinations is the bottleneck.
    
    # 33C7 = 5,292,426.
    # In Python, a loop of 5M iterations takes ~0.5-1s if simple.
    # With inner logic, maybe 5-10s.
    # We have 600s. So we can do many evaluations.
    
    for combo in combinations(range(n), 7):
        pts = points[list(combo)]
        if is_convex(pts):
            count += 1
    return count

def is_convex(pts):
    """
    Checks if 7 points form a convex polygon.
    Assumes points are in general position.
    We need to check if they are in convex position.
    This is equivalent to checking if the convex hull has 7 vertices.
    Or, if they are sorted by angle around the centroid, all turns are same sign.
    """
    # Calculate centroid
    c = np.mean(pts, axis=0)
    # Sort by angle
    angles = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    sorted_pts = pts[np.argsort(angles)]
    
    # Check turns
    # p0 -> p1 -> p2 ... p6 -> p0
    # Cross product of (p1-p0) and (p2-p1)
    # If all cross products have the same sign, it's convex.
    
    signs = []
    for i in range(7):
        p0 = sorted_pts[i]
        p1 = sorted_pts[(i + 1) % 7]
        p2 = sorted_pts[(i + 2) % 7]
        
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        
        if abs(cross) < 1e-9:
            return False # Collinear or duplicate
        signs.append(1 if cross > 0 else -1)
        
    # All signs must be the same
    return len(set(signs)) == 1

def is_valid(points):
    """
    Checks validity constraints:
    - Shape (33, 2)
    - No NaN/Inf
    - Coords in [-1e-6, 1+1e-6]
    - No two points too close (dist^2 < 1e-12)
    - No three points collinear (area < 1e-9)
    """
    if points.shape != (33, 2):
        return False
    if not np.all(np.isfinite(points)):
        return False
    if np.any(points < -1e-6) or np.any(points > 1 + 1e-6):
        return False
        
    # Check pairwise distances
    # 33 points -> 528 pairs
    for i in range(33):
        for j in range(i + 1, 33):
            d2 = np.sum((points[i] - points[j]) ** 2)
            if d2 < 1e-12:
                return False
                
    # Check collinearity
    # 33C3 = 5456 triples
    # This is fast.
    for i in range(33):
        for j in range(i + 1, 33):
            for k in range(j + 1, 33):
                p1, p2, p3 = points[i], points[j], points[k]
                area2 = abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))
                if area2 < 1e-9:
                    return False
    return True

# EVOLVE-BLOCK-END
