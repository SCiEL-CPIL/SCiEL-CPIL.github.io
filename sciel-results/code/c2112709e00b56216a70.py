# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using a greedy removal strategy with
    witness tracking. Start from the full space and greedily remove points
    (adding to the complement C) while maintaining the Nikodym property.
    
    The key insight: for each point x, we need at least one direction v such
    that ALL other points on the line through x in direction v are in N.
    As we remove points from N, we track which (x, v) pairs remain valid.
    A point can only be removed if it still has at least one valid line.
    
    We use a priority queue to prefer removing points that are "least useful"
    - i.e., points that appear in the fewest valid lines of other points.
    """
    start_time = time.time()
    deadline = start_time + 600.0
    
    p = 7
    q = p * p  # 49
    total_points = q * q  # 2401
    
    # Precompute field operations
    # F_49 = F_7[alpha] with alpha^2 = 3
    k = 3  # alpha^2 = 3
    
    def add_f49(a, b, c, d):
        """Add (a + b*alpha) + (c + d*alpha)"""
        return ((a + c) % p, (b + d) % p)
    
    def mul_f49(a, b, c, d):
        """Multiply (a + b*alpha) * (c + d*alpha)"""
        # (a + b*alpha)(c + d*alpha) = ac + ad*alpha + bc*alpha + bd*alpha^2
        # = (ac + bd*k) + (ad + bc)*alpha
        return ((a * c + b * d * k) % p, (a * d + b * c) % p)
    
    def neg_f49(a, b):
        """Negate (a + b*alpha)"""
        return ((-a) % p, (-b) % p)
    
    # Enumerate all points: index = xa * q + xb * p + ya * p + yb
    # Point (xa, xb, ya, yb) represents x = xa + xb*alpha, y = ya + yb*alpha
    
    # All possible directions: (1, m) for m in F_49, and (0, 1)
    # Direction (1, m) where m = (ma, mb)
    # Direction (0, 1) is the vertical direction
    
    # Precompute all lines for each point and direction
    # For point x and direction v, the line is {x + t*v : t in F_49}
    # The other points on the line are {x + t*v : t in F_49, t != 0}
    
    # We need to track for each point x, which directions v still have all
    # other points on the line in N.
    
    # Strategy: 
    # 1. Start with N = all points, C = empty
    # 2. For each point x, count how many directions v are "fully valid" 
    #    (all other points on the line through x in direction v are in N)
    # 3. Greedily remove points from N (add to C) that have the minimum 
    #    number of valid directions, but only if they have at least 1 valid direction
    # 4. When removing a point, update the validity of lines for all other points
    
    # This is expensive, so let's use a smarter approach:
    # 
    # Alternative: Use the structure of the problem more directly.
    # A point x is "safe" to remove if there exists a direction v such that
    # all points on the line through x in direction v (except x) are in N.
    # 
    # Let's define: for each point x and direction v, let L(x, v) be the set
    # of other points on the line through x in direction v.
    # x can be removed if there exists v such that L(x, v) ⊆ N.
    #
    # We'll maintain:
    # - in_N[x]: whether point x is in N
    # - valid_count[x]: number of directions v for which L(x, v) ⊆ N
    #
    # When we remove a point y from N:
    # - For each point x and direction v where y ∈ L(x, v), we need to check
    #   if L(x, v) is still ⊆ N. If not, valid_count[x] decreases.
    #
    # This is still expensive. Let's use a different approach:
    #
    # Key observation: The Nikodym property is local in a sense. Let's try
    # a simpler greedy: iterate through points in some order, and for each
    # point, check if it can be removed. If yes, remove it.
    #
    # To make this efficient, we'll precompute the line structure.
    
    # Point indexing: idx = ((xa * p + xb) * p + ya) * p + yb
    def point_idx(xa, xb, ya, yb):
        return ((xa * p + xb) * p + ya) * p + yb
    
    def idx_to_point(idx):
        yb = idx % p
        idx //= p
        ya = idx % p
        idx //= p
        xb = idx % p
        idx //= p
        xa = idx
        return xa, xb, ya, yb
    
    # Precompute for each point and each direction, the list of other points on the line
    # Directions: (1, m) for m in F_49 (49 directions) + (0, 1) (1 direction) = 50 directions
    
    # For direction (1, m) where m = (ma, mb):
    # Line through x = (xa, xb, ya, yb) in direction (1, m):
    # Point at parameter t = (ta, tb): 
    #   x_coord = x + t * 1 = (xa + ta, xb + tb)
    #   y_coord = y + t * m = (ya + ta*ma + tb*mb*k, yb + ta*mb + tb*ma)
    #   Wait, let me redo this.
    #   x + t * (1, m) = (x + t, y + t*m)
    #   where t = (ta, tb), so:
    #   x + t = (xa + ta, xb + tb)
    #   y + t*m = y + (ta, tb) * (ma, mb)
    #   (ta, tb) * (ma, mb) = (ta*ma + tb*mb*k, ta*mb + tb*ma)
    #   So y + t*m = (ya + ta*ma + tb*mb*k, yb + ta*mb + tb*ma)
    
    # For direction (0, 1):
    # x + t*(0, 1) = (x, y + t) = (xa, xb, ya + ta, yb + tb)
    
    # Let's precompute lines. This will be a big memory usage but let's try.
    # 2401 points * 50 directions * 48 other points per line = too much.
    # Instead, let's compute on the fly or use a more compact representation.
    
    # Better approach: For each direction, group points into lines.
    # For direction (1, m): lines are determined by the "slope-intercept" 
    # In F_q^2, a line with direction (1, m) can be parameterized as:
    # (x, y) = (x0 + t, y0 + t*m) for t in F_q
    # The line is determined by (x0, y0 - x0*m) or equivalently by the 
    # "intercept" y - x*m at x = 0.
    
    # For direction (0, 1): lines are vertical, determined by x coordinate.
    
    # Let's use this structure. For each direction, we can partition points into lines.
    
    # Precompute direction list
    directions = []
    for ma in range(p):
        for mb in range(p):
            directions.append((1, ma, mb))  # (dx, m_a, m_b) where direction is (1, m)
    directions.append((0, 0, 1))  # vertical direction (0, 1)
    
    num_dirs = len(directions)  # 50
    
    # For each direction, precompute the line_id for each point and the points on each line
    # line_id for direction (1, m): the "intercept" y - x*m at x=0
    # Actually, for direction (1, m), the line through (x0, y0) is {(x0 + t, y0 + t*m) : t in F_q}
    # Two points (x1, y1) and (x2, y2) are on the same line iff y1 - x1*m = y2 - x2*m
    # So the line is determined by c = y - x*m.
    
    # For direction (0, 1), the line is determined by x.
    
    # Let's build: for each direction d, a mapping from point_idx to line_idx,
    # and for each line, the list of points on it.
    
    # This is 50 * 2401 entries for the mapping, and 50 * (2401/49) = 50 * 49 = 2450 lines total.
    
    # Let's build this structure.
    
    # point_lines[d][p_idx] = line_idx
    # lines[d][line_idx] = list of point indices on that line
    
    point_lines = []
    lines_by_dir = []
    
    for d_idx, (dx, ma, mb) in enumerate(directions):
        # Map from line_key to line_idx
        line_map = {}
        line_points = []
        
        for xa in range(p):
            for xb in range(p):
                for ya in range(p):
                    for yb in range(p):
                        p_idx = point_idx(xa, xb, ya, yb)
                        
                        if dx == 1:
                            # Line key: y - x*m
                            # y - x*m = (ya, yb) - (xa, xb)*(ma, mb)
                            # (xa, xb)*(ma, mb) = (xa*ma + xb*mb*k, xa*mb + xb*ma)
                            xma_a = (xa * ma + xb * mb * k) % p
                            xma_b = (xa * mb + xb * ma) % p
                            key_a = (ya - xma_a) % p
                            key_b = (yb - xma_b) % p
                            line_key = (key_a, key_b)
                        else:
                            # Vertical: line key is x
                            line_key = (xa, xb)
                        
                        if line_key not in line_map:
                            line_map[line_key] = len(line_points)
                            line_points.append([])
                        line_idx = line_map[line_key]
                        line_points[line_idx].append(p_idx)
        
        # Build point_lines for this direction
        pl = np.zeros(total_points, dtype=np.int32)
        for xa in range(p):
            for xb in range(p):
                for ya in range(p):
                    for yb in range(p):
                        p_idx = point_idx(xa, xb, ya, yb)
                        if dx == 1:
                            xma_a = (xa * ma + xb * mb * k) % p
                            xma_b = (xa * mb + xb * ma) % p
                            key_a = (ya - xma_a) % p
                            key_b = (yb - xma_b) % p
                            line_key = (key_a, key_b)
                        else:
                            line_key = (xa, xb)
                        pl[p_idx] = line_map[line_key]
        
        point_lines.append(pl)
        lines_by_dir.append(line_points)
        
        if time.time() > deadline - 30:
            # Not enough time, return full set
            all_points = []
            for xa in range(p):
                for xb in range(p):
                    for ya in range(p):
                        for yb in range(p):
                            all_points.append([[xa, xb], [ya, yb]])
            return np.array(all_points, dtype=np.int64)
    
    # Now, greedy removal with witness tracking
    # in_N[p_idx] = True if point is in N
    in_N = np.ones(total_points, dtype=bool)
    
    # For each point, count how many directions have all other points on the line in N
    # valid_dir_count[p_idx] = number of directions v such that all other points on line through p_idx in dir v are in N
    
    # Initialize: for each point and direction, check if all other points on the line are in N
    # Since initially all points are in N, all directions are valid for all points.
    valid_dir_count = np.full(total_points, num_dirs, dtype=np.int32)
    
    # For efficient updates, we need to know: when we remove a point y,
    # which (x, v) pairs are affected? Those where y is on the line through x in direction v
    # and y != x.
    
    # Precompute: for each point y, for each direction d, which points x have y on their line in direction d?
    # This is the same as: for each direction d, for each line L, for each y in L, 
    # all other points x in L have y on their line in direction d.
    
    # Let's build: affected[d][y] = list of (x, d) pairs where y is on the line through x in direction d
    # Actually, for each direction d and line L, if we remove y from L, then for all x in L (x != y),
    # the line through x in direction d may no longer be fully in N.
    
    # We need to track, for each (x, d), how many points on the line through x in direction d are NOT in N.
    # Let's call this missing_count[x, d]. Initially 0.
    # When we remove y, for each d and each x such that y is on the line through x in direction d (y != x),
    # missing_count[x, d] increases by 1.
    # valid_dir_count[x] decreases when missing_count[x, d] goes from 0 to 1.
    
    # This is a 2D array of size 2401 * 50 = 120050, which is fine.
    
    missing_count = np.zeros((total_points, num_dirs), dtype=np.int32)
    
    # Precompute: for each direction d and point y, the list of points x (x != y) such that y is on the line through x in direction d
    # This is: for each direction d, for each line L, for each y in L, x = all other points in L
    
    # Let's build this as: neighbors[d][y] = list of x such that y is on line through x in direction d
    # But this is symmetric: x is on line through y in direction d iff y is on line through x in direction d
    # So neighbors[d][y] = all other points on the line containing y in direction d
    
    neighbors = []
    for d_idx in range(num_dirs):
        nb = []
        for y in range(total_points):
            line_idx = point_lines[d_idx][y]
            line_pts = lines_by_dir[d_idx][line_idx]
            # All other points on this line
            others = [x for x in line_pts if x != y]
            nb.append(others)
        neighbors.append(nb)
    
    if time.time() > deadline - 30:
        all_points = []
        for xa in range(p):
            for xb in range(p):
                for ya in range(p):
                    for yb in range(p):
                        all_points.append([[xa, xb], [ya, yb]])
        return np.array(all_points, dtype=np.int64)
    
    # Greedy removal:
    # Use a priority queue: points with smallest valid_dir_count first
    # But we need to be careful: a point can only be removed if valid_dir_count[x] >= 1
    
    import heapq
    
    # Priority queue: (valid_dir_count, point_idx)
    # We'll use a lazy deletion approach
    
    # Initialize heap
    heap = [(valid_dir_count[i], i) for i in range(total_points)]
    heapq.heapify(heap)
    
    # Track which points have been processed (removed from N)
    removed = np.zeros(total_points, dtype=bool)
    
    removal_order = []
    
    while time.time() < deadline - 10:
        # Get the point with minimum valid_dir_count
        while heap:
            count, p_idx = heap[0]
            if removed[p_idx]:
                heapq.heappop(heap)
                continue
            if count != valid_dir_count[p_idx]:
                heapq.heappop(heap)
                continue
            break
        
        if not heap:
            break
        
        count, p_idx = heap[0]
        
        if count < 1:
            # No valid direction, cannot remove
            break
        
        # Remove p_idx from N
        heapq.heappop(heap)
        removed[p_idx] = True
        in_N[p_idx] = False
        removal_order.append(p_idx)
        
        # Update: for each direction d, for each x such that p_idx is on the line through x in direction d (x != p_idx)
        for d_idx in range(num_dirs):
            for x in neighbors[d_idx][p_idx]:
                if removed[x]:
                    continue
                old_missing = missing_count[x, d_idx]
                missing_count[x, d_idx] = old_missing + 1
                if old_missing == 0 and missing_count[x, d_idx] == 1:
                    valid_dir_count[x] -= 1
                    heapq.heappush(heap, (valid_dir_count[x], x))
        
        # Check if any point now has 0 valid directions
        # We don't need to check immediately; the heap will handle it when we pop
    
    # Build the result: N = all points not removed
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    p_idx = point_idx(xa, xb, ya, yb)
                    if not removed[p_idx]:
                        points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
