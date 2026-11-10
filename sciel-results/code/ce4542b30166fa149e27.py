import numpy as np
import time
import heapq
from collections import defaultdict

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using greedy complement maximization.
    
    A Nikodym set N satisfies: for every point x in F_49^2, there exists a
    direction v (either (1,m) for m in F_49, or (0,1)) such that all points
    on the line {x + t*v : t in F_49} except x itself belong to N.
    
    We maximize the complement S = F_49^2 \ N by greedily adding points to S
    while maintaining the Nikodym property.
    """
    start_time = time.time()
    DEADLINE = 600  # seconds
    
    p = 7
    q = 49  # field size
    
    # Precompute field operations
    # alpha^2 = 3, so (a+b*alpha)*(c+d*alpha) = (ac+bd*3) + (ad+bc)*alpha
    def field_add(a, b):
        """Add two field elements represented as (a0, a1) + (b0, b1)"""
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
    
    def field_mul(a, b):
        """Multiply two field elements: (a0+a1*alpha)*(b0+b1*alpha)"""
        a0, a1 = a
        b0, b1 = b
        # (a0*b0 + a1*b1*3) + (a0*b1 + a1*b0)*alpha
        return ((a0 * b0 + a1 * b1 * 3) % p, (a0 * b1 + a1 * b0) % p)
    
    def field_sub(a, b):
        """Subtract: (a0, a1) - (b0, b1)"""
        return ((a[0] - b[0]) % p, (a[1] - b[1]) % p)
    
    # Generate all points in F_49^2
    # Point is (x, y) where x, y are field elements (a, b)
    # Encode point as index: x_a * 49 + x_b * 7 + y_a * 7 + y_b... 
    # Actually: index = ((x_a * 7 + x_b) * 7 + y_a) * 7 + y_b
    # Or simpler: point_idx = x_a * 49 + x_b * 7 + y_a * 7 + y_b
    # Let's use: idx = (x_a * 7 + x_b) * 49 + (y_a * 7 + y_b)
    
    # Precompute all points
    all_points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    all_points.append(((xa, xb), (ya, yb)))
    
    n_points = len(all_points)  # 2401
    
    # Precompute lines for each point and direction
    # Directions: (1, m) for m in F_49 (49 directions) + (0, 1) (1 direction) = 50 directions
    # For each point and direction, compute the line (set of points)
    
    # We'll work with a boolean array: in_S[idx] = True if point is in complement S
    in_S = np.zeros(n_points, dtype=bool)
    
    # For each point, track which directions can "cover" it
    # A point x is covered by direction v if all other points on line(x,v) are in N
    # i.e., all other points on line(x,v) are NOT in S
    
    # Precompute for each point and direction: the list of other points on the line
    # line_points[point_idx][dir_idx] = list of other point indices on that line
    
    # Directions: index 0..48 for (1, m) where m = (m_a, m_b), index 49 for (0,1)
    directions = []
    for ma in range(p):
        for mb in range(p):
            directions.append((1, 0, ma, mb))  # (vx_a, vx_b, vy_a, vy_b)
    directions.append((0, 0, 0, 1))  # vertical direction (0, 1)
    
    n_dirs = len(directions)  # 50
    
    # For each point, for each direction, precompute the other points on the line
    # This is expensive but necessary for fast updates
    
    # Actually, let's think differently. For a point x and direction v,
    # the line is {x + t*v : t in F_49}. The other points are x + t*v for t != 0.
    
    # Let's precompute: for each point x and direction v, the set of other points
    # But storing 2401 * 50 * 49 integers is ~5.9M integers, which is fine
    
    # Alternative: compute on the fly but cache
    
    # Let's use a different approach: 
    # For each direction v, group points into lines (parallel lines)
    # For direction (1, m): lines are determined by y - m*x (the "intercept")
    # For direction (0, 1): lines are determined by x (vertical lines)
    
    # For each direction, we have q parallel lines, each with q points
    # A point x is "covered" by direction v if all q-1 other points on its line are in N
    
    # Let's precompute for each direction the line assignment of each point
    # line_assignment[dir_idx][point_idx] = line_id (0 to q-1)
    
    line_assignment = np.zeros((n_dirs, n_points), dtype=np.int32)
    
    for dir_idx, (va, vb, wc, wd) in enumerate(directions):
        # Direction vector is (va + vb*alpha, wc + wd*alpha)
        # For (1, m) where m = (wc, wd): 
        #   The line through x with direction (1, m) consists of points
        #   (x + t, y + t*m) for t in F_49
        #   The "intercept" is y - x*m (constant along the line)
        # For (0, 1):
        #   The line through x with direction (0, 1) consists of points
        #   (x, y + t) for t in F_49
        #   The "intercept" is x (constant along the line)
        
        if dir_idx < 49:
            # Direction (1, m)
            ma, mb = wc, wd
            for pt_idx, ((xa, xb), (ya, yb)) in enumerate(all_points):
                # intercept = y - x*m
                # x*m = (xa + xb*alpha) * (ma + mb*alpha) = (xa*ma + xb*mb*3) + (xa*mb + xb*ma)*alpha
                xm_a = (xa * ma + xb * mb * 3) % p
                xm_b = (xa * mb + xb * ma) % p
                # intercept = (ya - xm_a, yb - xm_b)
                ia = (ya - xm_a) % p
                ib = (yb - xm_b) % p
                line_assignment[dir_idx, pt_idx] = ia * p + ib
        else:
            # Direction (0, 1): vertical lines, intercept = x
            for pt_idx, ((xa, xb), (ya, yb)) in enumerate(all_points):
                line_assignment[dir_idx, pt_idx] = xa * p + xb
    
    # For each direction and line, store the list of points on that line
    # lines[dir_idx][line_id] = list of point indices
    lines = [dict() for _ in range(n_dirs)]
    for dir_idx in range(n_dirs):
        for pt_idx in range(n_points):
            lid = line_assignment[dir_idx, pt_idx]
            if lid not in lines[dir_idx]:
                lines[dir_idx][lid] = []
            lines[dir_idx][lid].append(pt_idx)
    
    # For each point, for each direction, store the other points on the line
    # other_points[pt_idx][dir_idx] = list of other point indices
    # But this is 2401 * 50 * 49 entries... let's compute lazily
    
    # Instead, let's track for each point and direction: how many of the other
    # points on the line are in S. If this count is 0, the point is covered by that direction.
    
    # count_in_S[pt_idx][dir_idx] = number of other points on line(pt_idx, dir_idx) that are in S
    count_in_S = np.zeros((n_points, n_dirs), dtype=np.int32)
    
    # Initially, no points are in S, so all counts are 0, and all points are covered
    # by all directions.
    
    # Greedy: try to add points to S. A point can be added to S if after adding it,
    # every point is still covered by at least one direction.
    
    # When we add point p to S, for each direction d, the points on the same line
    # as p (other than p) will have their count_in_S increased by 1.
    # If any of those points had count_in_S == 0 for ALL directions before, and now
    # for direction d it becomes 1, we need to check if it's still covered by some other direction.
    
    # This is complex. Let's use a simpler approach: 
    # For each candidate point, check if adding it to S maintains validity.
    # To speed up, we can use a priority queue based on some heuristic.
    
    # Actually, let's use a different strategy:
    # Start with S = empty. For each point, determine if it CAN be in S
    # (i.e., if there's no direction where all other points on the line are already in S).
    # Wait, that's not quite right either.
    
    # Let me reconsider. The condition is:
    # For every point x, there exists a direction v such that for all t != 0,
    # x + t*v is NOT in S (i.e., is in N).
    
    # So x can be in S only if for every direction v, there exists some t != 0
    # such that x + t*v is also in S. Wait no, that's the opposite.
    
    # x can be in S if and only if for every direction v, it is NOT the case that
    # all other points on line(x,v) are in N. I.e., for every direction v, there
    # exists some other point on line(x,v) that is also in S.
    
    # No wait. Let me re-read the condition.
    
    # "every point x of F_49^2 must lie on a line {x + t*v : t in F_49}, with v = (1, m)
    # for some m in F_49 or v = (0, 1), whose points other than x all belong to N"
    
    # So for each x, there must exist a direction v such that ALL other points on
    # line(x,v) are in N (not in S).
    
    # So x can be in S if and only if there exists a direction v such that all
    # other points on line(x,v) are in N.
    
    # This means: when considering adding x to S, we need to check if there's a
    # direction v where none of the other points on line(x,v) are in S.
    
    # Let's implement this check.
    
    def can_add_to_S(pt_idx, in_S_arr):
        """Check if pt_idx can be added to S given current in_S state."""
        for dir_idx in range(n_dirs):
            lid = line_assignment[dir_idx, pt_idx]
            line_pts = lines[dir_idx][lid]
            # Check if any other point on this line is in S
            all_in_N = True
            for other_pt in line_pts:
                if other_pt != pt_idx and in_S_arr[other_pt]:
                    all_in_N = False
                    break
            if all_in_N:
                return True
        return False
    
    # This check is O(q * q) = O(q^2) per point per direction, so O(q^3) per point.
    # Total: O(q^3 * n_points) = O(q^5) which is 49^5 ≈ 282M. Too slow.
    
    # Let's optimize: precompute for each point and direction whether all other
    # points on the line are in N, and update incrementally.
    
    # all_in_N[pt_idx][dir_idx] = True if all other points on line(pt_idx, dir_idx) are in N
    all_in_N = np.ones((n_points, n_dirs), dtype=bool)
    
    # When we add a point p to S, for each direction d, for each other point q on
    # the same line as p in direction d, all_in_N[q][d] becomes False.
    
    # Let's implement the greedy algorithm with this tracking.
    
    in_S_arr = in_S  # numpy array
    # all_in_N is already initialized to all True
    
    # We need to be able to efficiently find which points can be added to S.
    # A point can be added if any of its all_in_N[pt][dir] is True.
    
    # Let's maintain a count: num_covered_dirs[pt_idx] = number of directions where all_in_N[pt_idx][dir] is True
    num_covered_dirs = np.full(n_points, n_dirs, dtype=np.int32)
    
    # A point can be added to S if num_covered_dirs[pt_idx] > 0
    
    # When we add point p to S:
    # For each direction d, for each other point q on the same line as p in direction d:
    #   if all_in_N[q][d] was True, set it to False and decrement num_covered_dirs[q]
    
    # Let's precompute the "neighbors" for each point and direction
    # neighbors[pt_idx][dir_idx] = list of other points on the same line
    # This is 2401 * 50 * 49 ≈ 5.9M entries, each a list of up to 48 ints
    # That's about 5.9M * 48 * 4 bytes ≈ 1.1GB. Too much.
    
    # Instead, let's compute on the fly using the lines structure.
    
    # Let's use a priority queue. The "cost" of adding a point is the number of
    # other points it will "block" (i.e., points that will lose coverage).
    # We want to add points that block the fewest other points.
    
    # Actually, let's just do a simple greedy: repeatedly find a point that can be
    # added to S and add it. Use a priority queue to prioritize points that are
    # "easier" to add (fewer covered directions, meaning they're more constrained).
    
    # Wait, we want to maximize |S|. So we should add as many points as possible.
    # The greedy approach: while there exists a point that can be added, add it.
    # The order might matter for the final size.
    
    # Let's try: repeatedly pick a random point that can be added, and add it.
    # Or: pick the point with the fewest covered directions (most constrained).
    
    # Let's implement with a simple loop:
    
    added = 0
    max_iterations = 1000000  # safety limit
    
    # Use a set of points that can currently be added
    # Actually, let's just iterate through all points repeatedly
    
    while time.time() - start_time < DEADLINE - 30:
        found = False
        for pt_idx in range(n_points):
            if in_S_arr[pt_idx]:
                continue
            if num_covered_dirs[pt_idx] > 0:
                # This point can be added to S
                in_S_arr[pt_idx] = True
                added += 1
                found = True
                
                # Update: for each direction, for each other point on the same line,
                # if all_in_N[other][dir] was True, set to False and decrement num_covered_dirs
                for dir_idx in range(n_dirs):
                    lid = line_assignment[dir_idx, pt_idx]
                    line_pts = lines[dir_idx][lid]
                    for other_pt in line_pts:
                        if other_pt == pt_idx:
                            continue
                        if all_in_N[other_pt, dir_idx]:
                            all_in_N[other_pt, dir_idx] = False
                            num_covered_dirs[other_pt] -= 1
                
                break  # restart the loop after adding a point
        
        if not found:
            break
    
    # Build the Nikodym set N = all points not in S
    points = []
    for pt_idx in range(n_points):
        if not in_S_arr[pt_idx]:
            (xa, xb), (ya, yb) = all_points[pt_idx]
            points.append([[xa, xb], [ya, yb]])
    
    if not points:
        # Should not happen, but just in case
        points = [[[0, 0], [0, 0]]]
    
    return np.array(points, dtype=np.int64)
