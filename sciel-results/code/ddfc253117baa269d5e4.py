# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using a greedy approach with fast validity checking.
    
    The complement S of the Nikodym set N must satisfy:
    - For every x in S: there exists a line through x such that L ∩ S = {x}
    - For every x in N: there exists a line through x such that L ∩ S = ∅
    
    We use integer-indexed field operations, precomputed line membership, and
    incremental tracking of line counts to enable fast validity checks.
    """
    start_time = time.time()
    DEADLINE = 580  # safety margin before 630s kill
    
    p = 7
    q = 49
    k = 3  # alpha^2 = 3
    
    # Precompute field operations using integer indices 0..48
    # idx = a * p + b represents a + b*alpha
    add_tbl = np.zeros((q, q), dtype=np.int64)
    sub_tbl = np.zeros((q, q), dtype=np.int64)
    mul_tbl = np.zeros((q, q), dtype=np.int64)
    
    for i in range(q):
        a0, a1 = i // p, i % p
        for j in range(q):
            b0, b1 = j // p, j % p
            r0, r1 = (a0 + b0) % p, (a1 + b1) % p
            add_tbl[i, j] = r0 * p + r1
            r0, r1 = (a0 - b0) % p, (a1 - b1) % p
            sub_tbl[i, j] = r0 * p + r1
            r0 = (a0 * b0 + a1 * b1 * k) % p
            r1 = (a0 * b1 + a1 * b0) % p
            mul_tbl[i, j] = r0 * p + r1
    
    total_points = q * q  # 2401
    
    # Build all lines: 49 vertical + 49*49 slanted = 2450 lines
    lines = []
    # Vertical lines: x = c
    for c in range(q):
        pts = []
        for y in range(q):
            pts.append(c * q + y)
        lines.append(pts)
    
    # Slanted lines: y = m*x + b
    for m in range(q):
        for b in range(q):
            pts = []
            for x in range(q):
                y = add_tbl[mul_tbl[m, x], b]
                pts.append(x * q + y)
            lines.append(pts)
    
    num_lines = len(lines)
    
    # Precompute: for each point, which lines contain it
    point_lines = [[] for _ in range(total_points)]
    for lid, L in enumerate(lines):
        for pt in L:
            point_lines[pt].append(lid)
    
    # For each line, precompute the set of points
    line_points = [set(L) for L in lines]
    
    # Initial state: S = empty (trivially valid, N = all points)
    S_mask = np.zeros(total_points, dtype=bool)
    line_S_count = np.zeros(num_lines, dtype=np.int32)
    
    # Track for each point how many "good" lines it has
    # For x in S: good line = line through x with S-count == 1
    # For x in N: good line = line through x with S-count == 0
    point_good_lines = np.zeros(total_points, dtype=np.int32)
    
    # Initialize: all points are in N, all lines have S-count 0
    # So every point has all its lines as "good" (count 0)
    for pt in range(total_points):
        point_good_lines[pt] = len(point_lines[pt])
    
    def check_validity():
        """Check if current S is a valid Nikodym complement."""
        for x in range(total_points):
            if point_good_lines[x] == 0:
                return False
        return True
    
    # Greedy: try to add points to S one by one
    # Priority: try all points, but be smart about ordering
    # A point p can be added to S if:
    # 1. For p itself: at least one line through p will have S-count become 1 (was 0)
    # 2. For all other points: they still have at least one good line
    
    # Better: just try adding and check validity incrementally
    
    # Candidate ordering: try points that are "easy" to add first
    # Points with fewer lines are easier (fewer points affected)
    # But all points have the same number of lines (50: 1 vertical + 49 slanted)
    # Actually: vertical line + 49 slanted = 50 lines per point
    
    # Let's try a different strategy: start with a known valid S and grow it
    # Or: try adding points in a specific order that maximizes success
    
    # For now, try all points in order, with backtracking if needed
    added_count = 0
    max_added = 2401
    
    # Sort candidates: try points that might be "easier" to add
    # Heuristic: points where adding them affects lines that are "less critical"
    # For simplicity, just try all points in index order
    
    candidates = list(range(total_points))
    
    for pt in candidates:
        if time.time() - start_time > DEADLINE:
            break
        if added_count >= max_added:
            break
        
        if S_mask[pt]:
            continue
        
        # Try adding pt to S
        S_mask[pt] = True
        affected_lines = point_lines[pt]
        for lid in affected_lines:
            line_S_count[lid] += 1
        
        # Update point_good_lines for affected points
        # For each line lid through pt, the S-count changed from old to new
        # Points on lid that are in N: they lose this line as a "good" line if count was 0
        # Points on lid that are in S: they lose this line as a "good" line if count was 1
        
        # Actually, let's just recompute point_good_lines for affected points
        # Affected points: all points on any of the affected_lines
        affected_points = set()
        for lid in affected_lines:
            for p2 in lines[lid]:
                affected_points.add(p2)
        
        # Recompute good lines for affected points
        for p2 in affected_points:
            good_count = 0
            for lid2 in point_lines[p2]:
                if S_mask[p2]:
                    if line_S_count[lid2] == 1:
                        good_count += 1
                else:
                    if line_S_count[lid2] == 0:
                        good_count += 1
            point_good_lines[p2] = good_count
        
        # Check validity
        if check_validity():
            added_count += 1
        else:
            # Revert
            S_mask[pt] = False
            for lid in affected_lines:
                line_S_count[lid] -= 1
            # Recompute point_good_lines for affected points
            for p2 in affected_points:
                good_count = 0
                for lid2 in point_lines[p2]:
                    if S_mask[p2]:
                        if line_S_count[lid2] == 1:
                            good_count += 1
                    else:
                        if line_S_count[lid2] == 0:
                            good_count += 1
                point_good_lines[p2] = good_count
    
    # Construct N = all_points - S
    N_points = []
    for pt in range(total_points):
        if not S_mask[pt]:
            x = pt // q
            y = pt % q
            xa, xb = x // p, x % p
            ya, yb = y // p, y % p
            N_points.append([[xa, xb], [ya, yb]])
    
    if len(N_points) == 0:
        # Fallback: return all points (trivially valid, score 0)
        for xa in range(p):
            for xb in range(p):
                for ya in range(p):
                    for yb in range(p):
                        N_points.append([[xa, xb], [ya, yb]])
    
    return np.array(N_points, dtype=np.int64)

# EVOLVE-BLOCK-END
