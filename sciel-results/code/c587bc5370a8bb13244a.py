# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using efficient greedy removal.
    Returns an array of points in N where N is the Nikodym set.
    Uses O(1) amortized validity checks for faster search.
    """
    p = 7
    k = 3  # alpha^2 = 3
    
    # Precompute all points as indices 0..2400
    # Point index = xa * 49 + xb * 7 + ya * 7 + yb (flattened)
    total_points = p ** 4  # 2401
    
    # Precompute line memberships for each point and direction
    # Directions: m in F_49 (49 directions) + vertical (1 direction) = 50 directions
    # Each line has 49 points
    
    # For each point, store which lines it belongs to
    # Lines are indexed by (point_on_line, direction)
    
    # Build line structure: for each direction d (0..49), and each "base point" on that direction's lines
    # Actually, let's think differently: for each direction, partition all 2401 points into 49 lines of 49 points each
    
    # Direction representation:
    # d=0: vertical, v=(0,1)
    # d=1..49: v=(1,m) where m is encoded from F_49 elements
    
    # Encode F_49 element (a,b) as a*7+b, so m_idx = ma*7+mb
    # For direction d (1..49): m = decode(d-1)
    
    def decode_m(m_idx):
        """Decode direction index to (ma, mb)"""
        return (m_idx // 7, m_idx % 7)
    
    def add_field(a, b, c, d):
        """Add (a+b*alpha) + (c+d*alpha) in F_49"""
        return ((a + c) % p, (b + d) % p)
    
    def mul_field(a, b, c, d):
        """Multiply (a+b*alpha) * (c+d*alpha) in F_49"""
        # (a+b*alpha)(c+d*alpha) = ac + ad*alpha + bc*alpha + bd*alpha^2
        # = (ac + bd*k) + (ad + bc)*alpha
        return ((a * c + b * d * k) % p, (a * d + b * c) % p)
    
    # Precompute all lines
    # For each direction, compute which line each point belongs to and which other points are on that line
    # lines[point_idx] = list of (other_point_idx) for each direction
    # Actually, we need: for each point and each direction, the set of other points on the same line
    
    # Let's precompute: line_membership[direction][point_idx] = frozenset of other points on same line
    # This is memory intensive: 50 * 2401 sets of 48 elements
    
    # Alternative: for each direction, group points into lines
    # For vertical (d=0): line determined by (xa, xb), all y values
    # For d>0: line through point with direction (1,m)
    
    # Let's use a different approach: precompute for each point, for each direction, 
    # the "line_id" and maintain counts of how many points in N are on each line
    
    # Simpler: precompute the 48 other points for each (point, direction) pair
    # other_points[point_idx][dir_idx] = array of 48 point indices
    
    other_points = np.zeros((total_points, 50, 48), dtype=np.int32)
    
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    point_idx = (xa * p + xb) * p * p + ya * p + yb
                    # Actually let's use: point_idx = xa * 49 + xb * 7 + ya * 7 + yb
                    # Wait, that's not right either. Let me use:
                    # point_idx = ((xa * p + xb) * p + ya) * p + yb
                    # = xa * 343 + xb * 49 + ya * 7 + yb
                    point_idx = xa * p**3 + xb * p**2 + ya * p + yb
                    
                    for d in range(50):
                        if d == 0:
                            # Vertical: v = (0, 1)
                            # Line: all points with same (xa, xb), varying y
                            for j in range(48):
                                # j goes 0..47, skip when it equals current y
                                yj = (yb + j + 1) % p  # This doesn't work for 48 points
                                # Better: iterate all y != yb
                                pass
                        else:
                            ma, mb = decode_m(d - 1)
                            # v = (1, m) where m = ma + mb*alpha
                            # Line: points (xa + t, ya + t*m) for t in F_49
                            # t = ta + tb*alpha
                            count = 0
                            for ta in range(p):
                                for tb in range(p):
                                    if ta == 0 and tb == 0:
                                        continue
                                    # new_x = (xa, xb) + (ta, tb)
                                    nx_a = (xa + ta) % p
                                    nx_b = (xb + tb) % p
                                    # new_y = (ya, yb) + (ta, tb) * (ma, mb)
                                    pm_a, pm_b = mul_field(ta, tb, ma, mb)
                                    ny_a = (ya + pm_a) % p
                                    ny_b = (yb + pm_b) % p
                                    other_idx = nx_a * p**3 + nx_b * p**2 + ny_a * p + ny_b
                                    other_points[point_idx, d, count] = other_idx
                                    count += 1
                        if d == 0:
                            # Fix vertical case
                            count = 0
                            for y2a in range(p):
                                for y2b in range(p):
                                    if y2a == ya and y2b == yb:
                                        continue
                                    other_idx = xa * p**3 + xb * p**2 + y2a * p + y2b
                                    other_points[point_idx, d, count] = other_idx
                                    count += 1
    
    # Now use greedy removal with efficient tracking
    # State: in_N[point_idx] = True/False
    # For each line (identified by direction and a representative point or line_id), 
    # track count of points still in N
    
    # Actually, let's track: for each point and direction, how many of the 48 other points are in N
    # line_count[point_idx][dir_idx] = number of other points on that line that are in N
    
    # Initialize: all points in N
    in_N = np.ones(total_points, dtype=bool)
    
    # line_count[point_idx][dir_idx] = number of the 48 other points that are in N
    # Initially all 48 are in N, so line_count = 48 for all
    line_count = np.full((total_points, 50), 48, dtype=np.int32)
    
    # min_line_count[point_idx] = minimum over all directions of line_count[point_idx][d]
    # A point can be removed if min_line_count[point_idx] == 0 (i.e., some line has no other points in N)
    # Wait, that's not right. We need: for point x to be removed, for EVERY direction d,
    # the line through x in direction d must have at least one OTHER point in N.
    # So x can be removed if for all d, line_count[x][d] > 0.
    
    # Let me redefine: line_count[x][d] = number of other points on line through x in dir d that are in N
    # x can be removed if min_d line_count[x][d] >= 1
    
    min_line_count = np.min(line_count, axis=1)  # shape (total_points,)
    
    # Count how many points have min_line_count >= 1 (removable)
    removable_count = np.sum(min_line_count >= 1)
    
    # Greedy removal: repeatedly remove a point that is removable
    # To maximize complement, we want to remove as many points as possible
    # Strategy: always remove a point that causes the least damage
    # i.e., a point whose removal makes the fewest other points non-removable
    
    # For efficiency, use random restarts with greedy
    best_complement = 0
    best_in_N = in_N.copy()
    
    deadline = time.time() + 580  # Safety margin before 630s limit
    
    np.random.seed(42)
    
    for restart in range(200):
        if time.time() > deadline - 30:
            break
        
        # Reset
        in_N = np.ones(total_points, dtype=bool)
        line_count = np.full((total_points, 50), 48, dtype=np.int32)
        min_line_count = np.full(total_points, 48, dtype=np.int32)
        
        # Random order of points
        order = np.random.permutation(total_points)
        
        removed = 0
        for point_idx in order:
            if time.time() > deadline - 30:
                break
            
            if not in_N[point_idx]:
                continue
            
            # Check if removable: min_line_count[point_idx] >= 1
            if min_line_count[point_idx] < 1:
                continue
            
            # Remove this point
            in_N[point_idx] = False
            removed += 1
            
            # Update line_count for all points on lines through point_idx
            # For each direction d, the line through point_idx in dir d
            # All other points on that line lose one from their count for direction d
            for d in range(50):
                for j in range(48):
                    other_idx = other_points[point_idx, d, j]
                    if in_N[other_idx]:
                        line_count[other_idx, d] -= 1
                        # Update min_line_count for other_idx
                        old_min = min_line_count[other_idx]
                        new_min = min(old_min, line_count[other_idx, d])
                        if new_min != old_min:
                            min_line_count[other_idx] = new_min
        
        complement = total_points - removed
        if complement > best_complement:
            best_complement = complement
            best_in_N = in_N.copy()
    
    # Extract points in best_in_N
    points = []
    for idx in range(total_points):
        if best_in_N[idx]:
            xa = idx // (p**3)
            xb = (idx // (p**2)) % p
            ya = (idx // p) % p
            yb = idx % p
            points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
