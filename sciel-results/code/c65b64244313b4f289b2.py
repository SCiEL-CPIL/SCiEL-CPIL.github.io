import numpy as np
import time

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    n_points = q * q  # 2401
    deadline = time.time() + 580.0  # Safety margin
    
    # Field F_49 = F_7[alpha], alpha^2 = 3
    # Element (a, b) represents a + b*alpha
    def add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
    
    def sub(a, b):
        return ((a[0] - b[0]) % p, (a[1] - b[1]) % p)
    
    def mul(a, b):
        return ((a[0]*b[0] + a[1]*b[1]*3) % p,
                (a[0]*b[1] + a[1]*b[0]) % p)
    
    def pt_to_idx(pt):
        xa, xb = pt[0]
        ya, yb = pt[1]
        return xa * 343 + xb * 49 + ya * 7 + yb
    
    def idx_to_pt(idx):
        xa = idx // 343
        idx %= 343
        xb = idx // 49
        idx %= 49
        ya = idx // 7
        yb = idx % 7
        return [[xa, xb], [ya, yb]]
    
    # Directions: (1, m) for m in F_49, plus (0, 1)
    directions = []
    for ma in range(p):
        for mb in range(p):
            directions.append(((1, 0), (ma, mb)))
    directions.append(((0, 0), (1, 0)))
    num_dirs = len(directions)  # 50
    
    # Precompute line membership: line_members[d_idx, x_idx, t_idx] = point index
    # For each direction d and point x, line through x in direction d:
    #   {x + t*v_d : t in F_49}
    # t_idx 0..48 maps to field elements
    
    # Build mapping from t_idx to field element
    t_elements = []
    for ta in range(p):
        for tb in range(p):
            t_elements.append((ta, tb))
    
    # line_members[d_idx, x_idx, t_idx] = idx of point x + t*v_d
    line_members = np.zeros((num_dirs, n_points, q), dtype=np.int32)
    
    for d_idx, (vx, vy) in enumerate(directions):
        for x_idx in range(n_points):
            x_pt = idx_to_pt(x_idx)
            px, py = x_pt[0], x_pt[1]
            for t_idx, (ta, tb) in enumerate(t_elements):
                tvx = mul((ta, tb), vx)
                tvy = mul((ta, tb), vy)
                nx = add(px, tvx)
                ny = add(py, tvy)
                n_idx = pt_to_idx([[nx[0], nx[1]], [ny[0], ny[1]]])
                line_members[d_idx, x_idx, t_idx] = n_idx
    
    # For each point x and direction d, the "other points" on the line (excluding x itself)
    # other_points[d_idx, x_idx, j] = point index, j=0..47
    other_points = np.zeros((num_dirs, n_points, q-1), dtype=np.int32)
    
    for d_idx, (vx, vy) in enumerate(directions):
        for x_idx in range(n_points):
            j = 0
            for t_idx in range(q):
                if t_elements[t_idx] == (0, 0):
                    continue
                other_points[d_idx, x_idx, j] = line_members[d_idx, x_idx, t_idx]
                j += 1
    
    # Build the initial complement S using the unital construction
    # A point (x, y) is in S if it's on the unital but not in special y-positions
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(((u * u) % p, v))
    
    S = np.zeros(n_points, dtype=bool)
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    in_unital = (2 * xa + ya * ya - yb * yb * 3) % p == 0
                    in_y_special = (ya, yb) in y_special
                    if in_unital and not in_y_special:
                        S[pt_to_idx([[xa, xb], [ya, yb]])] = True
    
    # Precompute: for each point x and direction d, which other points are on the line
    # We need: for point x, direction d, the set of other points on the line through x in dir d
    # Constraint: for every point x (in S or not) and direction d,
    #   all other points on line(x, d) must be in N (i.e., NOT in S)
    # Equivalently: for every x not in S and direction d,
    #   all other points on line(x, d) must be in N
    # And for every x in S and direction d,
    #   all other points on line(x, d) must be in N
    
    # Energy function: count violations
    # A violation occurs when there exists x (any point) and direction d such that
    # some other point on line(x, d) is in S
    
    # Precompute: for each point x and direction d, the list of other points
    # violation_count[x, d] = number of other points on line(x,d) that are in S
    # Total violations = sum over all x, d of [violation_count[x,d] > 0]
    # Actually, we need: for each x, d, if ANY other point is in S, that's a violation
    
    # Let's precompute: for each x, d, the other points as a flat array
    # Then for a given S, violations[x,d] = sum(S[other_points[d,x,:]])
    # A point x,d is violated if violations[x,d] > 0
    
    # For efficiency, precompute the other_points as a 2D array per (x,d)
    # But that's 2401*50*48 = 5.7M entries. We can compute on the fly with numpy.
    
    # Actually, let's precompute a 3D array: others[d, x, :] = other points
    # We already have other_points[d_idx, x_idx, :]
    
    # Energy: for each x in 0..2400, d in 0..49:
    #   if sum(S[other_points[d,x,:]]) > 0: violation
    # Total energy = number of (x,d) pairs with violations
    
    def compute_energy(S_arr):
        """Compute number of violated (x, d) pairs."""
        # For each d, x: check if any other point on line is in S
        # other_points[d, x, :] has shape (48,)
        # S_arr[other_points[d, x, :]] gives boolean array of length 48
        # Sum > 0 means violation
        
        # Vectorized: for each d, S_arr[other_points[d, :, :]] has shape (2401, 48)
        # Sum over axis 1 > 0 gives violations for that d
        total_violations = 0
        for d in range(num_dirs):
            # other_points[d, :, :] has shape (2401, 48)
            # S_arr[other_points[d, :, :]] has shape (2401, 48)
            line_has_S = np.sum(S_arr[other_points[d, :, :]], axis=1) > 0
            total_violations += np.sum(line_has_S)
        return total_violations
    
    # Verify initial S is valid
    initial_energy = compute_energy(S)
    assert initial_energy == 0, f"Initial S has {initial_energy} violations"
    
    best_S = S.copy()
    best_size = np.sum(S)
    current_energy = 0
    
    # Simulated annealing
    np.random.seed(42)
    T_start = 5.0
    T_end = 0.01
    n_steps = 500000
    
    # Precompute all other_points for fast access
    # other_points has shape (num_dirs, n_points, q-1) = (50, 2401, 48)
    
    # For a move: flip S[x] for some x
    # Delta energy: for each d, check if line(x, d) had 0 or 1 S-point among others
    # If S[x] was True (removing x from S):
    #   For each d: the other points on line(x,d) don't change
    #   But x itself is no longer in S, so lines through OTHER points that include x may change
    #   Actually, the constraint is about other points on the line, not x itself
    #   So flipping S[x] only affects:
    #   1. Lines through x: other points unchanged, so no effect on violations for x
    #   2. Lines through other points y where x is on line(y, d): 
    #      if x was in S and we remove it, violations for (y,d) may decrease
    #      if x was not in S and we add it, violations for (y,d) may increase
    
    # This is complex. Let's use a simpler approach: recompute energy from scratch
    # but only for affected points.
    
    # Actually, let's just do full recompute but with early termination if energy > current
    
    # For speed, let's precompute for each x, the list of (y, d) pairs where x is an
    # "other point" on line(y, d). That is, x = y + t*v_d for some t != 0.
    
    # reverse_map[x] = list of (y, d) such that x is on line(y, d) and x != y
    # For each x, d: the line through x in direction d contains points x + t*v_d
    # x is on line(y, d) iff y = x + t'*v_d for some t' != 0
    # So for fixed x and d, y = x + t'*v_d for t' in F_49 \ {0}
    # That's 48 values of y for each d, so 48*50 = 2400 entries per x
    
    # Let's precompute: for each x, a list of (y, d) pairs
    # But that's 2401 * 2400 = 5.7M pairs, which is a lot
    
    # Alternative: for a flip of x, the affected (y,d) pairs are:
    #   For each d, y in {x + t*v_d : t in F_49\{0}} (48 values)
    #   So 48*50 = 2400 affected (y,d) pairs
    # For each affected (y,d), we need to check if line(y,d) has any S-point among others
    # This requires checking 48 points per (y,d), so 2400*48 = 115200 checks per flip
    # That's too slow for 500K steps
    
    # Let's use a different strategy: maintain violation counts incrementally
    # violation_count[y, d] = number of S-points on line(y,d) among others
    # When we flip x:
    #   For each d, for each y in {x + t*v_d : t != 0}:
    #     if x was in S and we remove it: violation_count[y,d] -= 1
    #     if x was not in S and we add it: violation_count[y,d] += 1
    #   A (y,d) is violated if violation_count[y,d] > 0
    #   Total energy = number of (y,d) with violation_count[y,d] > 0
    
    # Precompute: for each x and d, the 48 y values
    # y_values[x, d, :] = the 48 points y such that x is on line(y, d)
    # y = x + t*v_d for t in F_49\{0}
    
    y_values = np.zeros((n_points, num_dirs, q-1), dtype=np.int32)
    for x_idx in range(n_points):
        x_pt = idx_to_pt(x_idx)
        px, py = x_pt[0], x_pt[1]
        for d_idx, (vx, vy) in enumerate(directions):
            j = 0
            for t_idx in range(q):
                ta, tb = t_elements[t_idx]
                if ta == 0 and tb == 0:
                    continue
                tvx = mul((ta, tb), vx)
                tvy = mul((ta, tb), vy)
                ny = add(px, tvx)
                ny2 = add(py, tvy)
                n_idx = pt_to_idx([[ny[0], ny[1]], [ny2[0], ny2[1]]])
                y_values[x_idx, d_idx, j] = n_idx
                j += 1
    
    # Initialize violation_count
    violation_count = np.zeros((n_points, num_dirs), dtype=np.int32)
    for x_idx in range(n_points):
        if S[x_idx]:
            for d_idx in range(num_dirs):
                # x is in S, so for each y in y_values[x_idx, d_idx, :],
                # x is on line(y, d_idx), so violation_count[y, d_idx] += 1
                y_list = y_values[x_idx, d_idx, :]
                violation_count[y_list, d_idx] += 1
    
    current_energy = np.sum(violation_count > 0)
    assert current_energy == 0, f"Initial energy {current_energy}"
    
    # Simulated annealing
    T = T_start
    cooling_rate = (T_end / T_start) ** (1.0 / n_steps)
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        # Choose a random point to flip
        x_idx = np.random.randint(n_points)
        
        # Compute delta energy
        was_in_S = S[x_idx]
        
        # For each d, the 48 y values in y_values[x_idx, d, :]
        # If was_in_S: removing x decreases violation_count[y, d] by 1 for each y
        # If not was_in_S: adding x increases violation_count[y, d] by 1 for each y
        
        delta_energy = 0
        for d_idx in range(num_dirs):
            y_list = y_values[x_idx, d_idx, :]
            old_vc = violation_count[y_list, d_idx]
            old_violated = (old_vc > 0).astype(np.int32)
            
            if was_in_S:
                new_vc = old_vc - 1
                new_violated = (new_vc > 0).astype(np.int32)
            else:
                new_vc = old_vc + 1
                new_violated = (new_vc > 0).astype(np.int32)
            
            delta_energy += np.sum(new_violated - old_violated)
        
        # Metropolis acceptance
        if delta_energy <= 0 or np.random.rand() < np.exp(-delta_energy / T):
            # Apply the flip
            S[x_idx] = not was_in_S
            for d_idx in range(num_dirs):
                y_list = y_values[x_idx, d_idx, :]
                if was_in_S:
                    violation_count[y_list, d_idx] -= 1
                else:
                    violation_count[y_list, d_idx] += 1
            current_energy += delta_energy
            
            if current_energy == 0 and np.sum(S) > best_size:
                best_size = np.sum(S)
                best_S = S.copy()
        
        T *= cooling_rate
    
    S = best_S
    final_energy = compute_energy(S)
    assert final_energy == 0, f"Final S has {final_energy} violations"
    
    # Return the complement: points NOT in S
    points = []
    for i in range(n_points):
        if not S[i]:
            points.append(idx_to_pt(i))
    
    return np.array(points, dtype=np.int64)
