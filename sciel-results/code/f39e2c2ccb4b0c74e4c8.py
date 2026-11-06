import time
import itertools

def construct():
    # Define the candidate set of 20 points
    points = [
        [0, 0], [1, 0], [2, 0],
        [3, 1], [4, 1], [5, 1],
        [6, 2], [7, 2], [8, 2],
        [0, 3], [1, 3], [2, 3],
        [3, 4], [4, 4], [5, 4],
        [6, 5], [7, 5], [8, 5],
        [100, 100],
        [100, -100]
    ]
    
    # Verify no four collinear
    # We check all lines defined by pairs of points
    # If any line contains 4 or more points, it's invalid
    n = len(points)
    is_valid = True
    # To speed up, we can just check all pairs and count
    # But with 20 points, it's fast enough to check all 4-combinations?
    # No, checking all pairs and counting is O(N^2 * N) = O(N^3) ~ 8000 ops. Very fast.
    
    # Better: For each pair, define a line, count how many points are on it.
    # Use a set of normalized line equations to avoid duplicates?
    # Or just check all 4-subsets? 4845 subsets. For each, check collinearity.
    # Collinearity check is O(1). Total ~ 5000 ops. Very fast.
    
    for combo in itertools.combinations(points, 4):
        p1, p2, p3, p4 = combo
        # Check if p1, p2, p3 are collinear
        # (x2-x1)(y3-y1) == (x3-x1)(y2-y1)
        if (p2[0]-p1[0])*(p3[1]-p1[1]) == (p3[0]-p1[0])*(p2[1]-p1[1]):
            # Now check if p4 is on the same line
            # Line through p1, p2
            # (x4-x1)(y2-y1) == (x2-x1)(y4-y1)
            if (p4[0]-p1[0])*(p2[1]-p1[1]) == (p2[0]-p1[0])*(p4[1]-p1[1]):
                is_valid = False
                break
        # Also need to check if p1, p2, p4 collinear and p3 on line?
        # The condition "no 4 collinear" means no 4 points lie on a single line.
        # If p1,p2,p3 are collinear, and p4 is also on that line, then all 4 are collinear.
        # If p1,p2,p3 are NOT collinear, they don't form a line with 3 points, so they can't be part of a 4-collinear set 
        # UNLESS the line is defined by p1,p2 and p3 is not on it, but p4 is? No, then only 2 points on that line.
        # A 4-collinear set means there exists a line L such that 4 points are on L.
        # So we just need to check if any 4 points are on a line.
        # My check above:
        # 1. Check if p1,p2,p3 collinear.
        # 2. If yes, check if p4 is on that line.
        # This covers the case where the line is defined by p1,p2,p3.
        # But what if the line is defined by p1,p2,p4?
        # Then p1,p2,p3 are NOT collinear (since p3 is not on the line).
        # So my check would miss it.
        
        # Correct check:
        # For a set of 4 points, they are 4-collinear if and only if:
        # - p1,p2,p3 collinear AND p4 on that line
        # - OR p1,p2,p4 collinear AND p3 on that line
        # - OR p1,p3,p4 collinear AND p2 on that line
        # - OR p2,p3,p4 collinear AND p1 on that line
        
        # Let's rewrite the check properly.
        # Actually, simpler:
        # Count points on line defined by p1,p2.
        # If count >= 4, invalid.
        # But we are iterating 4-combos.
        # Let's just check all 4 combinations for collinearity of all 4.
        
    # Let's restart the validation logic cleanly
    is_valid = True
    for combo in itertools.combinations(points, 4):
        p1, p2, p3, p4 = combo
        # Check if all 4 are collinear
        # They are collinear if the area of triangle (p1,p2,p3) is 0 AND (p1,p2,p4) is 0
        # Area(p1,p2,p3) = (x2-x1)(y3-y1) - (x3-x1)(y2-y1)
        area123 = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1])
        area124 = (p2[0]-p1[0])*(p4[1]-p1[1]) - (p4[0]-p1[0])*(p2[1]-p1[1])
        if area123 == 0 and area124 == 0:
            is_valid = False
            break
            
    if not is_valid:
        # Fallback to parabola if invalid
        return [[x, x * x] for x in range(-10, 10)]
        
    return points
