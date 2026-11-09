"""Program for no-isosceles set task using bitset-based distance tracking."""
import numpy as np

N = 64

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a set of grid points in [0, N)^2 with no three points forming an isosceles triangle.
    
    Approach: Greedy with bitset-based squared distance tracking.
    For each candidate point, we check if adding it would create an isosceles triangle.
    Three points A, B, C form an isosceles triangle if any two pairwise squared distances are equal.
    So for a new point P, and existing points A, B:
      - d(P,A) == d(P,B)  => P is apex
      - d(P,A) == d(A,B)  => A is apex  
      - d(P,B) == d(A,B)  => B is apex
    
    We track:
      - For each existing point A, the set of squared distances from A to other points in S.
      - When adding P, we need to ensure that for every pair (A,B) in S:
        d(P,A) != d(P,B), d(P,A) != d(A,B), d(P,B) != d(A,B)
    
    Efficient approach:
      - Maintain for each point A in S, a set (or bitset) of squared distances from A to other points in S.
      - When considering P, compute d(P, A) for all A in S.
      - Check 1: All d(P, A) must be distinct among themselves (no two equal).
      - Check 2: For each A in S, d(P, A) must not be in the distance set of A (distances from A to other points in S).
    
    Since N=64, max squared distance is 2*(63)^2 = 7938. We can use Python integers as bitsets.
    
    Greedy strategy: Try points in a fixed order, add if valid.
    To maximize size, we might want a good ordering. Let's try scanning the grid in a pattern.
    """
    N_local = N
    # Max squared distance
    max_d2 = 2 * (N_local - 1) ** 2
    
    # We'll use a list of points and for each point, a bitset of squared distances to other points in the set
    points = []
    dist_bitsets = []  # dist_bitsets[i] has bit j set if there's a point in points at squared distance j from points[i]
    
    # Precompute all possible grid points
    # Try a greedy approach with a specific ordering
    # Order: sort by some heuristic. Let's try row by row, or maybe a space-filling curve.
    # For simplicity, let's try all points in lexicographic order first, but that might not be optimal.
    # Alternative: try points in order of increasing "density" of neighbors? 
    # Let's just do a simple greedy: iterate through all grid points, add if valid.
    
    # To improve, we can try multiple orderings and keep the best.
    # But for speed, let's start with a single good ordering.
    
    # Generate all candidate points
    candidates = [(x, y) for x in range(N_local) for y in range(N_local)]
    
    # Try a few orderings and keep the best result
    best_points = []
    
    orderings = [
        lambda x, y: (x, y),           # lexicographic
        lambda x, y: (x + y, x),       # diagonal
        lambda x, y: (y, x),           # transposed lexicographic
        lambda x, y: (max(x,y), min(x,y)),  # symmetric
    ]
    
    for order_func in orderings:
        ordered = sorted(candidates, key=lambda p: order_func(p[0], p[1]))
        
        points = []
        dist_bitsets = []
        
        for (px, py) in ordered:
            # Check if we can add (px, py)
            valid = True
            new_dists = []  # squared distances from (px,py) to each existing point
            
            for i, (ax, ay) in enumerate(points):
                d2 = (px - ax) ** 2 + (py - ay) ** 2
                new_dists.append(d2)
                
                # Check 2: d2 must not be in dist_bitsets[i]
                # dist_bitsets[i] has bit j set if distance j exists
                if (dist_bitsets[i] >> d2) & 1:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check 1: all new_dists must be distinct
            # Use a set for this (small list)
            if len(set(new_dists)) != len(new_dists):
                continue
            
            # Also need to check: for each pair (i, j) in existing points,
            # d(P, A_i) != d(A_i, A_j) and d(P, A_j) != d(A_i, A_j)
            # This is already covered by Check 2 above! Because dist_bitsets[i] contains d(A_i, A_j) for all j != i.
            # So if d(P, A_i) == d(A_i, A_j) for some j, then d2 is in dist_bitsets[i].
            
            # All checks passed, add the point
            points.append((px, py))
            
            # Update dist_bitsets
            # For the new point, its dist_bitset is the set of all new_dists
            new_bitset = 0
            for d2 in new_dists:
                new_bitset |= (1 << d2)
            dist_bitsets.append(new_bitset)
            
            # For each existing point i, add the new distance d2 to its bitset
            for i, d2 in enumerate(new_dists):
                dist_bitsets[i] |= (1 << d2)
        
        if len(points) > len(best_points):
            best_points = points
    
    return best_points
# EVOLVE-BLOCK-END
