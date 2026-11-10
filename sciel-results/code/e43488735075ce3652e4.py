import numpy as np
import time
from itertools import combinations, product

def kissing_number11() -> np.ndarray:
    d = 11
    points = []
    
    # E8 part: 240 vectors in first 8 dims, 0 in last 3
    # E8 min norm vectors have four +1 and four -1
    # Generate all combinations of 4 indices for +1
    # The remaining 4 are -1
    # There are C(8,4) = 70 combinations for +1 positions.
    # But each vector is counted twice? No.
    # If we choose 4 for +1, the other 4 are -1.
    # This generates 70 vectors.
    # Wait, E8 has 240 min norm vectors.
    # The 240 vectors are:
    # 1. 112 vectors with two +1/2, two -1/2, four 0? No, that's not integer.
    # The integer representation of E8 is:
    # All (x1,...,x8) in Z^8 with sum xi even.
    # OR all (x1,...,x8) in (Z+1/2)^8 with sum xi even? No.
    # The "even" lattice D8 has 128 min norm vectors (weight 2, +/-1).
    # The "half-integer" lattice has 112 min norm vectors (weight 4, +/-1/2).
    # Total 128 + 112 = 240.
    # But we need INTEGER vectors.
    # So we can only use the D8 part?
    # D8 min norm vectors: permutations of (+-1, +-1, 0, 0, 0, 0, 0, 0).
    # Count: Choose 2 positions for non-zero: C(8,2) = 28.
    # Signs: 4 options (+-1, +-1).
    # Total 28 * 4 = 112?
    # Wait, D8 is vectors in Z^8 with even sum.
    # Min norm squared is 2.
    # Vectors: two non-zero entries, each +/-1.
    # Sum is +/-2 or 0. Always even.
    # So all such vectors are in D8.
    # Count: 8*7 (ordered pairs of distinct indices) * 4 (signs) / 2?
    # Let's list:
    # Indices i, j.
    # (1, 0, ..., 1, 0)
    # (1, 0, ..., -1, 0)
    # (-1, 0, ..., 1, 0)
    # (-1, 0, ..., -1, 0)
    # Number of pairs (i,j) with i!=j is 8*7 = 56.
    # For each pair, 4 sign combinations.
    # But (i,j) and (j,i) generate the same set of vectors?
    # No, vector is defined by values at indices.
    # So 56 * 4 = 224?
    # Let's check.
    # Vector with 1 at i, 1 at j.
    # Vector with 1 at j, 1 at i. Same vector.
    # So we should count unordered pairs?
    # If we use combinations of indices, C(8,2) = 28.
    # For each pair, 4 vectors.
    # Total 112.
    # So D8 has 112 min norm vectors.
    # E8 has 240. The other 112 are half-integers.
    # Since we need integers, we can only use the 112 from D8.
    # So E8 integer part gives 112 points.
    
    # So k = 112 (from D8) + 12 (from A3) = 124.
    # This is better than 48, but less than 240.
    
    # Can we use the half-integer part by scaling?
    # If we scale E8 by 2, the half-integers become integers.
    # Min norm squared becomes 2 * 4 = 8?
    # Original min norm squared 2.
    # Half-integers: (1/2, 1/2, 1/2, 1/2, 0...). Norm sq = 4 * 1/4 = 1?
    # No, E8 min norm is 2.
    # Vectors like (1/2, 1/2, 1/2, 1/2, 0, 0, 0, 0) have norm squared 1?
    # Wait, standard E8 min norm is 2.
    # The vectors (1/2, ..., 1/2) have norm squared 8 * 1/4 = 2.
    # Yes.
    # So if we scale by 2, min norm squared is 8.
    # Integer vectors from D8 part: norm squared 2 * 4 = 8.
    # Integer vectors from half-int part: norm squared 2 * 4 = 8.
    # All 240 vectors become integers with norm squared 8.
    # Min dist squared:
    # Original min dist squared 2.
    # Scaled min dist squared 2 * 4 = 8.
    # Condition: Min dist >= Max norm.
    # 8 >= 8. Holds.
    # So we can use all 240 E8 vectors, scaled by 2.
    
    # Now, can we add A3 vectors?
    # A3 vectors have norm squared 2.
    # If we add them to the scaled E8 vectors (norm 8),
    # Max norm becomes 8.
    # We need Min dist >= 8.
    # Dist between E8 vector (norm 8) and A3 vector (norm 2):
    # Supports are disjoint.
    # Dist squared = 8 + 2 = 10.
    # 10 >= 8. Holds.
    # Dist between A3 vectors:
    # Min dist squared in A3 is 2.
    # But we require Min dist >= 8.
    # 2 < 8. Fails.
    # So we cannot add A3 vectors if we scale E8 by 2.
    
    # So we have two options:
    # 1. Unscaled E8 (112 pts, norm 2) + A3 (12 pts, norm 2).
    #    Max norm 2. Min dist 2.
    #    k = 112 + 12 = 124.
    # 2. Scaled E8 (240 pts, norm 8) + No A3.
    #    Max norm 8. Min dist 8.
    #    k = 240.
    
    # 240 > 124.
    # So scaled E8 is better.
    
    # Can we add anything to scaled E8?
    # We need vectors v such that:
    # 1. ||v||^2 <= 8
    # 2. ||v - u||^2 >= 8 for all u in E8_scaled
    # 3. ||v - w||^2 >= 8 for all other v, w
    
    # Since E8_scaled occupies the first 8 dimensions,
    # any vector with non-zero components in the last 3 dimensions
    # will have distance squared >= 8 (from E8 part) + ||v_last3||^2.
    # So cross distance is >= 8.
    # We just need to pack vectors in the last 3 dimensions
    # such that their mutual distances are >= 8 and norms <= 8.
    
    # In 3D, with max norm squared 8 and min dist squared 8.
    # How many integer vectors can we fit?
    # If we pick v1, v2.
    # ||v1||^2 <= 8, ||v2||^2 <= 8.
    # ||v1 - v2||^2 >= 8.
    
    # Try v1 = (0,0,0)? No, zero vector not allowed.
    # Try v1 = (2,0,0). Norm 4.
    # v2 must be dist >= 8 from (2,0,0).
    # (2,0,0) to (x,y,z). (x-2)^2 + y^2 + z^2 >= 8.
    # Also x^2+y^2+z^2 <= 8.
    
    # Let's try to find a set.
    # (0, 2, 0) dist to (2,0,0) is sqrt(4+4)=sqrt(8). Sq 8. OK.
    # (0, 0, 2) dist to (2,0,0) is sqrt(4+4)=sqrt(8). Sq 8. OK.
    # (0, 0, 2) dist to (0,2,0) is sqrt(4+4)=sqrt(8). Sq 8. OK.
    # (-2, 0, 0) dist to (2,0,0) is 4. Sq 16 >= 8. OK.
    # But norm of (-2,0,0) is 4 <= 8.
    
    # Let's check the set { (2,0,0), (-2,0,0), (0,2,0), (0,-2,0), (0,0,2), (0,0,-2) }.
    # Norms are 4.
    # Distances:
    # (2,0,0) to (-2,0,0): 16.
    # (2,0,0) to (0,2,0): 8.
    # (2,0,0) to (0,0,2): 8.
    # (2,0,0) to (0,-2,0): 8.
    # (2,0,0) to (0,0,-2): 8.
    # All >= 8.
    # So we can add 6 vectors.
    
    # Can we add more?
    # (1,1,2)? Norm 1+1+4=6.
    # Dist to (2,0,0): (1)^2 + (1)^2 + (2)^2 = 6 < 8. Fail.
    
    # What about (2,2,0)? Norm 8.
    # Dist to (2,0,0): 4 < 8. Fail.
    
    # So 6 vectors in the 3D part seems good.
    # Total k = 240 + 6 = 246.
    
    # This is close to 252? No, 246 < 252.
    # Wait, my previous 252 estimate was based on unscaled E8 (112) + A3 (12) = 124.
    # And scaled E8 (240) + 6 = 246.
    # So 246 is the best so far.
    
    # Can we get more than 6 in 3D with norm <= 8 and dist >= 8?
    # Kissing number of 3D is 12.
    # If we scale the 3D lattice so that min dist is 8?
    # FCC min dist 2. Scale by 2 -> min dist 4.
    # Scale by sqrt(8)/2? Not integer.
    # If we use integer vectors, min dist 8 is large.
    # The 6 vectors along axes seem optimal for this constraint.
    
    # So 246 is a strong candidate.
    
    points_e8 = []
    # Generate 240 E8 vectors
    # Part 1: D8 (112 vectors)
    # Two non-zero entries, +/-1.
    for i in range(8):
        for j in range(i+1, 8):
            for si in [-1, 1]:
                for sj in [-1, 1]:
                    v = [0]*8
                    v[i] = si
                    v[j] = sj
                    points_e8.append(v)
                    
    # Part 2: Half-integers (112 vectors)
    # Four non-zero entries, +/-1/2.
    # To make them integers, we will scale the whole E8 by 2 later.
    # So we can just store the half-integers as integers by multiplying by 2?
    # If we store (1,1,1,1,0,0,0,0), it represents (1/2, 1/2, 1/2, 1/2, 0,0,0,0) scaled by 2.
    # Yes.
    for comb in combinations(range(8), 4):
        for signs in product([-1, 1], repeat=4):
            v = [0]*8
            for idx, s in zip(comb, signs):
                v[idx] = s
            points_e8.append(v)
            
    # Scale by 2
    points_e8_scaled = []
    for v in points_e8:
        v_scaled = [2*x for x in v]
        # Pad to 11 dims
        v_scaled.extend([0, 0, 0])
        points_e8_scaled.append(v_scaled)
        
    # Add 6 vectors in last 3 dims
    points_extra = [
        [0,0,0,0,0,0,0,0, 2, 0, 0],
        [0,0,0,0,0,0,0,0, -2, 0, 0],
        [0,0,0,0,0,0,0,0, 0, 2, 0],
        [0,0,0,0,0,0,0,0, 0, -2, 0],
        [0,0,0,0,0,0,0,0, 0, 0, 2],
        [0,0,0,0,0,0,0,0, 0, 0, -2]
    ]
    
    all_points = points_e8_scaled + points_extra
    return np.array(all_points, dtype=np.int64)
