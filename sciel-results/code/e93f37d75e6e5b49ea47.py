# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 with q = 49, p = 7.
    
    Strategy: Model the Nikodym condition as a hypergraph independent set problem.
    Vertices are points. For each point x and each of the 50 directions through x,
    there is a hyperedge consisting of the other 48 points on that line.
    A valid Nikodym set N means: for every point x, there exists at least one
    direction d such that NO point on the punctured line (x + t*v_d, t != 0) is 
    in the excluded set S = complement of N.
    
    Equivalently: S is a set of points such that for every point x, there is at 
    least one direction d where the punctured line has zero points in S.
    
    We want to maximize |S| (minimize |N|).
    
    Approach: Vectorized iterative peeling heuristic (2-core decomposition style).
    Start with all points as candidates for S. Iteratively remove points that
    violate the constraint, prioritizing removal of points that appear in the 
    fewest "bad" hyperedges.
    """
    p = 7
    q = 49
    k = 3  # alpha^2 = 3
    start_time = time.time()
    DEADLINE = 580.0  # safety margin before 630s kill
    
    # Precompute field multiplication table
    # Element i represents (i//p, i%p) = a + b*alpha
    mul_tbl = np.zeros((q, q), dtype=np.int64)
    for i in range(q):
        a0, a1 = i // p, i % p
        for j in range(q):
            b0, b1 = j // p, j % p
            mul_tbl[i, j] = ((a0*b0 + a1*b1*k) % p) * p + ((a0*b1 + a1*b0) % p)
    
    # Total points: 2401, indexed as (x_idx * q + y_idx) where x_idx, y_idx in 0..48
    total = q * q
    
    # Build all 2450 lines (49 vertical + 49*49 slanted)
    # Each line is an array of 49 point indices
    lines = []
    
    # Vertical lines: x = c, varying y
    for c in range(q):
        line = np.array([c * q + y for y in range(q)], dtype=np.int64)
        lines.append(line)
    
    # Slanted lines: y = m*x + b
    for m in range(q):
        for b in range(q):
            line = np.zeros(q, dtype=np.int64)
            for x_idx in range(q):
                y_val = (mul_tbl[m, x_idx] + b) % q
                line[x_idx] = x_idx * q + y_val
            lines.append(line)
    
    num_lines = len(lines)  # 2450
    
    # For each point, precompute which lines pass through it and the positions
    # point_to_lines[pt] = list of (line_idx, position_in_line)
    point_to_lines = [[] for _ in range(total)]
    for li, line in enumerate(lines):
        for pos, pt in enumerate(line):
            point_to_lines[pt].append((li, pos))
    
    # Each point lies on exactly 50 lines (49 slanted + 1 vertical)
    # For each point, the 50 directions correspond to 50 lines through it
    
    # Build the hypergraph structure:
    # For each point x and each direction (line through x), the hyperedge is
    # the set of other 48 points on that line.
    # 
    # Constraint: For point x to be "satisfied", at least one of its 50 lines
    # must have zero points in S among the other 48 points.
    #
    # Equivalently: point x is "unsatisfied" if ALL 50 lines through x have
    # at least one other point in S.
    
    # Peeling approach:
    # Start with S = all points.
    # For each point, count how many of its 50 lines have all 48 other points in S.
    # Actually, we need: for point x, a line L through x is "empty of S" (other than x)
    # if none of the other 48 points on L are in S.
    #
    # Let's think differently. Let s[pt] = 1 if pt is in S, 0 otherwise.
    # For point x and line L through x, define:
    #   empty_count[L, x] = number of points on L (other than x) that are NOT in S
    #   = 48 - (number of points on L other than x that ARE in S)
    #
    # Point x is satisfied if there exists a line L through x with empty_count[L,x] = 48
    # (i.e., all other points on L are not in S).
    #
    # Wait, that's the opposite of what we want. We want the OTHER points to BE in N,
    # i.e., NOT in S. So we need: exists line L through x such that all 48 other 
    # points on L are in N (not in S).
    
    # So: point x is satisfied iff exists line L through x such that 
    #     sum_{y in L, y != x} s[y] = 0
    
    # Peeling: start with s[pt] = 1 for all pt.
    # For each point x, compute bad_lines[x] = number of lines L through x where 
    #     sum_{y in L, y != x} s[y] > 0
    # Point x is unsatisfied if bad_lines[x] = 50 (all lines have some other point in S).
    # 
    # Actually, we need x to be satisfied, meaning at least one line has zero other points in S.
    # So x is UNSATISFIED if for ALL 50 lines L through x, there's at least one y != x on L with s[y] = 1.
    
    # Greedy removal: if a point x is unsatisfied, we must either:
    #   - Remove x from S (set s[x] = 0), OR
    #   - Remove some other point y from S that lies on all 50 lines through x
    #     (but there's no single point on all 50 lines through x, since different lines 
    #     only intersect at x)
    # 
    # Wait, that's not right either. The constraint is on the EXISTENCE of a good direction.
    # If x is in S, we still need the Nikodym property to hold for x.
    # The Nikodym property says: for every point x (whether in N or S), there exists
    # a direction v such that all other points on the line are in N.
    
    # So the constraint applies to ALL points, not just those in N.
    # Point x is satisfied iff exists direction v such that all 48 other points on 
    # the line through x with direction v are in N (i.e., not in S).
    
    # This is a global constraint. Let me reformulate:
    # We choose S (the excluded set). For every point x in F_49^2:
    #   exists line L through x such that (L \ {x}) ∩ S = ∅
    
    # This means: for every x, at least one of the 50 lines through x has its 
    # other 48 points all outside S.
    
    # Peeling heuristic:
    # 1. Start with S = all points (s[pt] = 1 for all pt)
    # 2. For each point x, compute for each of its 50 lines L through x:
    #      count_s[L, x] = |{y in L : y != x, s[y] = 1}|
    #    x is satisfied if min over L of count_s[L,x] = 0
    # 3. Find unsatisfied points (all 50 lines have count_s > 0)
    # 4. For unsatisfied points, we need to remove points from S to make at least one
    #    line through x have count_s = 0.
    # 5. Greedy: remove the point y from S that appears in the most "critical" lines
    #    (lines that are the last to become empty for their points).
    
    # Alternative simpler approach: 
    # Since we want to MAXIMIZE |S|, start with S = all, and remove points that
    # are "forced" to be removed.
    
    # A point y is forced to be removed from S if removing y is necessary to satisfy
    # some point x. But this is complex.
    
    # Simpler: use the dual view.
    # For each line L, let e[L] = number of points of L in S.
    # For point x on line L, the constraint "L is good for x" means e[L] <= 1 
    # (only x itself might be in S, or even x might not be in S).
    # Actually, the constraint is: all OTHER points on L are not in S.
    # So for line L and point x on L: good iff e[L \ {x}] = 0, i.e., e[L] <= s[x].
    # If s[x] = 1: good iff e[L] = 1 (only x is in S on this line)
    # If s[x] = 0: good iff e[L] = 0 (no points of L are in S)
    
    # Point x is satisfied iff exists L through x such that:
    #   if s[x] = 1: e[L] = 1
    #   if s[x] = 0: e[L] = 0
    
    # This is still complex. Let me use a different strategy.
    
    # Strategy: Randomized local search.
    # 1. Start with a random S (or a structured initial S).
    # 2. Check validity.
    # 3. If invalid, find violated points and try to fix by adding/removing points.
    # 4. If valid, try to add more points to S while maintaining validity.
    
    # For efficiency, use numpy vectorization.
    
    # Precompute: for each point x, the list of 50 lines through it.
    # And for each line, the 49 points on it.
    
    # Let's use a bitset representation for S.
    # s_bits: boolean array of size 2401
    
    # For each line L, compute e[L] = sum of s over points in L.
    # For point x on line L, the line L is "good for x" if:
    #   e[L] - s[x] == 0  (all other points on L are not in S)
    
    # Point x is satisfied if exists L through x: e[L] - s[x] == 0
    
    # Let's implement this with numpy.
    
    # line_points: array of shape (2450, 49) with point indices
    line_points = np.array(lines, dtype=np.int64)  # (2450, 49)
    
    # For each point, which lines contain it and at what position
    # point_lines: array of shape (2401, 50) with line indices
    # point_line_pos: array of shape (2401, 50) with positions in line
    point_lines = np.zeros((total, 50), dtype=np.int64)
    point_line_pos = np.zeros((total, 50), dtype=np.int64)
    
    for li in range(num_lines):
        for pos, pt in enumerate(line_points[li]):
            # Find where to insert in point_lines[pt]
            # Since each point is on exactly 50 lines, we can use a counter
            pass
    
    # More efficient: build from the lines
    line_counter = np.zeros(total, dtype=np.int64)
    for li in range(num_lines):
        for pos in range(q):
            pt = line_points[li, pos]
            idx = line_counter[pt]
            point_lines[pt, idx] = li
            point_line_pos[pt, idx] = pos
            line_counter[pt] += 1
    
    # Verify each point is on exactly 50 lines
    assert np.all(line_counter == 50), f"Points on wrong number of lines: {line_counter[:10]}"
    
    def check_validity(s):
        """Check if S (boolean array) gives a valid Nikodym set.
        Returns True if valid, False otherwise."""
        # e[L] = sum of s over points in line L
        e = np.sum(s[line_points], axis=1)  # shape (2450,)
        
        # For each point x and each line L through x:
        # good if e[L] - s[x] == 0
        # x is satisfied if any of its 50 lines is good
        
        # For all points and all 50 lines:
        # e_for_point[x, j] = e[point_lines[x, j]]
        # s_for_point[x, j] = s[x] (same for all j)
        # good[x, j] = (e_for_point[x, j] - s[x] == 0)
        # satisfied[x] = any(good[x, :])
        
        # Vectorized:
        # e_for_all = e[point_lines]  # shape (2401, 50)
        # good = (e_for_all == s[:, None])  # since e[L] - s[x] == 0 means e[L] == s[x]
        # satisfied = np.any(good, axis=1)
        
        e_for_all = e[point_lines]  # (2401, 50)
        # s[x] is scalar per x, broadcast to (2401, 50)
        good = (e_for_all == s[:, None])
        satisfied = np.any(good, axis=1)
        
        return np.all(satisfied)
    
    def count_violations(s):
        """Return number of unsatisfied points."""
        e = np.sum(s[line_points], axis=1)
        e_for_all = e[point_lines]
        good = (e_for_all == s[:, None])
        satisfied = np.any(good, axis=1)
        return np.sum(~satisfied)
    
    # Initial solution: start with a structured set
    # Use the unital-based construction from the current program as initial guess
    
    # Unital: 2*xa + ya^2 - k*yb^2 = 0
    U_mask = np.zeros(total, dtype=bool)
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (2*xa + ya*ya - k*yb*yb) % p == 0:
                        pt = (xa*p+xb)*q + (ya*p+yb)
                        U_mask[pt] = True
    
    # Special y-coordinates
    y_special_mask = np.zeros(q, dtype=bool)
    for u in range(1, p):
        for v in range(p):
            y_special_mask[u*u % p * p + v] = True
    
    # Initial S: points in U where y is not special
    s = np.zeros(total, dtype=bool)
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    pt = (xa*p+xb)*q + (ya*p+yb)
                    if U_mask[pt] and not y_special_mask[ya*p+yb]:
                        s[pt] = True
    
    # Check initial validity
    initial_violations = count_violations(s)
    
    best_s = s.copy()
    best_size = np.sum(s)
    
    # Local search: try to add points to S while maintaining validity
    # And remove points that cause violations
    
    # Phase 1: Fix any violations by removing points
    violations = count_violations(s)
    while violations > 0 and time.time() - start_time < DEADLINE - 60:
        # Find violated points
        e = np.sum(s[line_points], axis=1)
        e_for_all = e[point_lines]
        good = (e_for_all == s[:, None])
        satisfied = np.any(good, axis=1)
        violated_pts = np.where(~satisfied)[0]
        
        if len(violated_pts) == 0:
            break
        
        # For each violated point x, we need to find a line L through x
        # and remove points from S on L (other than x) to make e[L] == s[x]
        
        # Greedy: for each violated point, pick the line L through x with 
        # minimum e[L] (easiest to fix), and remove points from S on L
        
        fixed_any = False
        for x in violated_pts[:100]:  # Limit per iteration for speed
            if time.time() - start_time > DEADLINE - 60:
                break
            
            # Get the 50 lines through x
            lines_through_x = point_lines[x]
            positions = point_line_pos[x]
            
            # For each line, compute how many points need to be removed
            # e[L] - s[x] = number of other points on L in S
            # We need to reduce this to 0
            
            best_line = -1
            best_removals = np.inf
            best_removal_pts = None
            
            for j in range(50):
                li = lines_through_x[j]
                pos = positions[j]
                other_pts = np.delete(line_points[li], pos)
                # Points in S on other_pts
                in_s = s[other_pts]
                num_in_s = np.sum(in_s)
                
                if num_in_s < best_removals:
                    best_removals = num_in_s
                    best_line = li
                    best_removal_pts = other_pts[in_s]
            
            if best_line >= 0 and best_removals > 0:
                # Remove these points from S
                s[best_removal_pts] = False
                fixed_any = True
                best_size = np.sum(s)
        
        if not fixed_any:
            break
        
        violations = count_violations(s)
    
    if np.sum(s) > best_size:
        best_s = s.copy()
        best_size = np.sum(s)
    
    # Phase 2: Try to add points to S
    # For each point not in S, check if adding it maintains validity
    
    rng = np.random.default_rng(42)
    
    for iteration in range(50):
        if time.time() - start_time > DEADLINE - 30:
            break
        
        # Try random additions
        added = 0
        candidates = np.where(~s)[0]
        rng.shuffle(candidates)
        
        for pt in candidates[:500]:
            if time.time() - start_time > DEADLINE - 30:
                break
            
            # Try adding pt to S
            s[pt] = True
            if count_violations(s) == 0:
                added += 1
            else:
                s[pt] = False
        
        if added > 0:
            best_s = s.copy()
            best_size = np.sum(s)
    
    # Phase 3: Multi-start with different random initializations
    for attempt in range(5):
        if time.time() - start_time > DEADLINE - 30:
            break
        
        # Random initial S with density 0.3
        s_random = np.random.default_rng(attempt * 1000 + 7).random(total) < 0.3
        
        # Fix violations
        for fix_iter in range(100):
            if time.time() - start_time > DEADLINE - 30:
                break
            
            e = np.sum(s_random[line_points], axis=1)
            e_for_all = e[point_lines]
            good = (e_for_all == s_random[:, None])
            satisfied = np.any(good, axis=1)
            violated_pts = np.where(~satisfied)[0]
            
            if len(violated_pts) == 0:
                break
            
            fixed_any = False
            for x in violated_pts[:50]:
                lines_through_x = point_lines[x]
                positions = point_line_pos[x]
                
                best_line = -1
                best_removals = np.inf
                best_removal_pts = None
                
                for j in range(50):
                    li = lines_through_x[j]
                    pos = positions[j]
                    other_pts = np.delete(line_points[li], pos)
                    in_s = s_random[other_pts]
                    num_in_s = np.sum(in_s)
                    
                    if num_in_s < best_removals:
                        best_removals = num_in_s
                        best_line = li
                        best_removal_pts = other_pts[in_s]
                
                if best_line >= 0 and best_removals > 0:
                    s_random[best_removal_pts] = False
                    fixed_any = True
            
            if not fixed_any:
                break
        
        # Try to add points
        candidates = np.where(~s_random)[0]
        rng.shuffle(candidates)
        for pt in candidates[:300]:
            if time.time() - start_time > DEADLINE - 30:
                break
            s_random[pt] = True
            if count_violations(s_random) == 0:
                pass
            else:
                s_random[pt] = False
        
        if np.sum(s_random) > best_size:
            best_s = s_random.copy()
            best_size = np.sum(s_random)
    
    # Final validity check
    if not check_validity(best_s):
        # Fallback to a known valid small set
        # Just return all points (trivially valid)
        best_s = np.ones(total, dtype=bool)
    
    # Convert to output format
    points = []
    for pt in range(total):
        if not best_s[pt]:
            x_idx = pt // q
            y_idx = pt % q
            xa, xb = x_idx // p, x_idx % p
            ya, yb = y_idx // p, y_idx % p
            points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
