import numpy as np
import time

def nikodym_set() -> np.ndarray:
    start_time = time.time()
    deadline = 580.0
    
    p = 7
    k = 3  # alpha^2 = 3 mod 7
    
    # Field element: (a, b) representing a + b*alpha
    # Precompute field multiplication table
    # idx = a + 7*b for a,b in 0..6
    def fidx(a, b):
        return a + 7 * b
    
    def fadd(i, j):
        a1, b1 = i % 7, i // 7
        a2, b2 = j % 7, j // 7
        return fidx((a1+a2)%7, (b1+b2)%7)
    
    def fmul(i, j):
        a1, b1 = i % 7, i // 7
        a2, b2 = j % 7, j // 7
        return fidx((a1*a2 + k*b1*b2)%7, (a1*b2 + b1*a2)%7)
    
    # Precompute addition and multiplication tables
    add_table = np.zeros((49, 49), dtype=np.int32)
    mul_table = np.zeros((49, 49), dtype=np.int32)
    for i in range(49):
        for j in range(49):
            add_table[i, j] = fadd(i, j)
            mul_table[i, j] = fmul(i, j)
    
    # All points: 2401 points, each is (x_idx, y_idx) where x_idx, y_idx in 0..48
    # Point index = x_idx * 49 + y_idx
    # x_idx = xa + 7*xb, y_idx = ya + 7*yb
    
    # Build all lines
    # Direction (0,1): for each x_idx, line = {(x_idx, y_idx) : y_idx in 0..48}
    # Direction (1,m): for each m and each "intercept", line = {(x_idx, y_idx) : y_idx = m*x_idx + c}
    
    # Precompute all lines as arrays of point indices
    all_lines = []  # list of numpy arrays of point indices
    
    # Vertical lines: direction (0,1)
    for x_idx in range(49):
        line = np.array([x_idx * 49 + y_idx for y_idx in range(49)], dtype=np.int32)
        all_lines.append(line)
    
    # Slanted lines: direction (1,m) for m in F_49
    for m in range(49):
        # y = m*x + c for each c in F_49
        for c in range(49):
            line = np.zeros(49, dtype=np.int32)
            for x_idx in range(49):
                y_idx = fmul(m, x_idx)
                y_idx = fadd(y_idx, c)
                line[x_idx] = x_idx * 49 + y_idx
            all_lines.append(line)
    
    # Total lines: 49 + 49*49 = 49 + 2401 = 2450
    num_lines = len(all_lines)
    
    # For each point, store which lines pass through it
    # point_lines[pt] = list of line indices
    point_lines = [[] for _ in range(2401)]
    for li, line in enumerate(all_lines):
        for pt in line:
            point_lines[pt].append(li)
    
    # Convert to numpy arrays for efficiency
    point_lines_arr = []
    for pl in point_lines:
        point_lines_arr.append(np.array(pl, dtype=np.int32))
    
    # Start with empty S (complement), try to grow it
    # S is represented as a boolean array of size 2401
    S = np.zeros(2401, dtype=bool)
    
    # For checking if adding point p to S maintains validity:
    # For every point x, there must exist a line L through x with L ∩ S ⊆ {x}
    # When we add p to S, the only points whose validity might be affected are those
    # that share a line with p (other than p itself), because for those points,
    # the line through them and p now has an extra S-point.
    
    # Actually, more precisely: for point x, a line L through x is "valid" if L ∩ S ⊆ {x}.
    # Adding p to S invalidates a line L if p ∈ L and p ≠ x. So for each x that shares
    # a line with p, we need to check if x still has at least one valid line.
    
    # Precompute: for each point x, the set of lines through x
    # For each line, the set of points on it
    
    # Let's precompute line membership as a 2D structure
    # line_points[li] = array of points on line li
    line_points = all_lines  # already have this
    
    # For efficient checking, let's precompute for each point x, 
    # the count of valid lines through x (lines where all other points are not in S)
    
    # valid_line_count[x] = number of lines through x with no other S-points
    
    def compute_valid_line_counts(S_arr):
        """For each point, count lines through it with no other S-points."""
        counts = np.zeros(2401, dtype=np.int32)
        for x in range(2401):
            for li in point_lines_arr[x]:
                line = line_points[li]
                # Check if any other point on this line is in S
                # line contains x; check if any other point is in S
                other_in_S = np.sum(S_arr[line] & (line != x))
                if other_in_S == 0:
                    counts[x] += 1
        return counts
    
    # Initial: S is empty, so every line is valid for every point
    # valid_line_count[x] = len(point_lines[x]) for all x
    
    valid_counts = np.array([len(pl) for pl in point_lines_arr], dtype=np.int32)
    
    # Greedily add points to S
    # For each candidate point p (not in S), check if adding it keeps all valid_counts >= 1
    # We need to efficiently check the impact
    
    # When adding p to S:
    # For each line L containing p, for each other point x on L:
    #   if L was a valid line for x (i.e., L ∩ S ⊆ {x} before adding p),
    #   then after adding p, L is no longer valid for x (since p ∈ L, p ≠ x)
    #   So valid_counts[x] decreases by 1 for each such x
    
    # We need valid_counts[x] >= 1 for all x after the change
    
    # Let's try a greedy approach:
    # 1. Start with S = {}
    # 2. For each point p (in some order), try adding p to S
    # 3. Check if all points still have valid_count >= 1
    # 4. If yes, keep p in S; if no, skip p
    
    # To make this efficient, we'll process points in a smart order
    # and use the fact that the field is small (2401 points, 2450 lines)
    
    # Let's precompute for each line, the list of points on it (excluding a given point)
    # Actually, let's just do it directly
    
    # Order points by some heuristic - maybe random or by degree
    # Let's try multiple random orders
    
    best_S = np.zeros(2401, dtype=bool)
    best_size = 0
    
    np.random.seed(42)
    
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() - start_time > deadline - 10:
            break
        
        S_trial = np.zeros(2401, dtype=bool)
        valid_counts_trial = valid_counts.copy()
        
        # Random order of points
        order = np.random.permutation(2401)
        
        for p in order:
            if time.time() - start_time > deadline - 5:
                break
            
            # Check if adding p to S maintains validity
            # Find all lines through p
            lines_through_p = point_lines_arr[p]
            
            # For each line through p, find other points on that line
            # and check if that line is currently valid for those points
            
            # For each line L through p:
            #   For each x in L, x != p:
            #     If L is currently a valid line for x (i.e., before adding p, L ∩ S ⊆ {x}),
            #     then after adding p, L is no longer valid for x
            #     So we need to decrement valid_counts_trial[x]
            #     But we need to check if valid_counts_trial[x] would go below 1
            
            # More precisely: for each x sharing a line with p (x != p),
            # count how many lines through x and p are currently valid for x.
            # After adding p, those lines become invalid for x.
            # So valid_counts_trial[x] decreases by that count.
            
            # Let's compute the decrement for each affected point
            # affected_points: points that share at least one line with p (other than p)
            
            # For each line through p, get all points on that line except p
            affected = {}
            for li in lines_through_p:
                line = line_points[li]
                for x in line:
                    if x == p:
                        continue
                    # Check if this line is currently valid for x
                    # A line L is valid for x if L ∩ S_trial ⊆ {x}
                    # i.e., no point in L other than x is in S_trial
                    # Since p is not yet in S_trial, we check if any other point in L\{x} is in S_trial
                    # But we're about to add p, so after adding p, L will contain p (which is in S)
                    # So L becomes invalid for x if p ∈ L and p != x, which is always true here
                    # But we only decrement if L was valid for x before adding p
                    
                    # L was valid for x iff no point in L\{x} is in S_trial
                    # Check: is there any point in L\{x}\{p} that is in S_trial?
                    # If yes, L was already invalid for x, so adding p doesn't change validity
                    # If no, L was valid for x, and adding p makes it invalid
                    
                    # For efficiency, just check if L\{x} intersects S_trial
                    # L\{x} includes p and other points
                    # We know p is not in S_trial yet
                    # So check if (L\{x}\{p}) intersects S_trial
                    
                    # This is getting complex. Let's just do a direct check.
                    
                    # Simpler: for each x, count how many lines through both x and p
                    # are currently valid for x. After adding p, those become invalid.
                    
                    if x not in affected:
                        affected[x] = 0
                    
                    # Check if line li is currently valid for x
                    # li is valid for x iff no point in line[li]\{x} is in S_trial
                    # We can check: np.sum(S_trial[line[li]] & (line[li] != x)) == 0
                    # But this is O(49) per check, and we have many checks
                    
                    # Let's precompute: for each line, which points are in S_trial
                    # Actually, let's just do the check directly but efficiently
                    
                    # For line li, check if any point other than x is in S_trial
                    # We can do: np.any(S_trial[line[li]] & (line[li] != x))
                    # But we need to exclude p from consideration since p is not in S_trial yet
                    # Actually p is not in S_trial, so we just check all of line[li]\{x}
                    
                    other_in_S = np.sum(S_trial[line[li]] & (line[li] != x))
                    if other_in_S == 0:
                        # Line li is currently valid for x
                        affected[x] += 1
            
            # Now check if adding p would make any point have valid_count < 1
            # For each affected point x, new valid_count = valid_counts_trial[x] - affected[x]
            # We need new valid_count >= 1 for all x
            
            can_add = True
            for x, dec in affected.items():
                if valid_counts_trial[x] - dec < 1:
                    can_add = False
                    break
            
            if can_add:
                # Add p to S
                S_trial[p] = True
                for x, dec in affected.items():
                    valid_counts_trial[x] -= dec
        
        # End of restart
        size = np.sum(S_trial)
        if size > best_size:
            best_size = size
            best_S = S_trial.copy()
        
        if time.time() - start_time > deadline - 5:
            break
    
    # Verify the best_S is valid
    # For each point x, check if there's a line through x with no other S-points
    def verify_S(S_arr):
        for x in range(2401):
            found = False
            for li in point_lines_arr[x]:
                line = line_points[li]
                # Check if any other point on line is in S
                other_in_S = np.sum(S_arr[line] & (line != x))
                if other_in_S == 0:
                    found = True
                    break
            if not found:
                return False
        return True
    
    if not verify_S(best_S):
        # Fallback: use empty S (N = all points, which is trivially valid)
        best_S = np.zeros(2401, dtype=bool)
        best_size = 0
    
    # Convert S to N (complement)
    # N = all points not in S
    N_points = []
    for i in range(2401):
        if not best_S[i]:
            x_idx = i // 49
            y_idx = i % 49
            xa, xb = x_idx % 7, x_idx // 7
            ya, yb = y_idx % 7, y_idx // 7
            N_points.append([[xa, xb], [ya, yb]])
    
    if len(N_points) == 0:
        # Shouldn't happen, but fallback
        N_points = [[[0, 0], [0, 0]]]
    
    return np.array(N_points, dtype=np.int64)

