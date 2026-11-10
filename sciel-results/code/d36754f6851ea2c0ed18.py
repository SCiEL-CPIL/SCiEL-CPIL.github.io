# EVOLVE-BLOCK-START
import numpy as np

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using a greedy maximization of the excluded
    set S. The field is F_49 = F_7[alpha] with alpha^2 = 3.

    Returns:
        points: np.ndarray of shape (k, 2, 2) integer array. Each of the k points
        is [[x_a, x_b], [y_a, y_b]] with the F_49 coordinates x = (x_a, x_b) and
        y = (y_a, y_b), all entries in [0, 7).
    """
    p = 7
    k_const = 3  # alpha^2 = 3

    # Precompute all 49 elements of F_49 as (a, b) pairs
    all_elements = [(a, b) for a in range(p) for b in range(p)]
    element_to_idx = {el: i for i, el in enumerate(all_elements)}
    
    # Precompute multiplication table
    # mult[i][j] = index of all_elements[i] * all_elements[j]
    mult = np.zeros((p*p, p*p), dtype=np.int64)
    for i, (a1, b1) in enumerate(all_elements):
        for j, (a2, b2) in enumerate(all_elements):
            # (a1 + b1*alpha) * (a2 + b2*alpha) = (a1*a2 + b1*b2*alpha^2) + (a1*b2 + a2*b1)*alpha
            ra = (a1 * a2 + b1 * b2 * k_const) % p
            rb = (a1 * b2 + a2 * b1) % p
            mult[i][j] = element_to_idx[(ra, rb)]

    # Precompute addition table
    # add[i][j] = index of all_elements[i] + all_elements[j]
    add = np.zeros((p*p, p*p), dtype=np.int64)
    for i, (a1, b1) in enumerate(all_elements):
        for j, (a2, b2) in enumerate(all_elements):
            ra = (a1 + a2) % p
            rb = (b1 + b2) % p
            add[i][j] = element_to_idx[(ra, rb)]

    # Precompute subtraction table
    # sub[i][j] = index of all_elements[i] - all_elements[j]
    sub = np.zeros((p*p, p*p), dtype=np.int64)
    for i, (a1, b1) in enumerate(all_elements):
        for j, (a2, b2) in enumerate(all_elements):
            ra = (a1 - a2) % p
            rb = (b1 - b2) % p
            sub[i][j] = element_to_idx[(ra, rb)]

    # There are 49 non-zero slopes m in F_49. We index slopes from 1 to 48 (element index 1..48),
    # but we also need to handle the vertical direction separately.
    # Slopes are non-zero elements of F_49. Index 0 is the zero element.
    # So slopes correspond to element indices 1..48.
    
    # For each point (x, y) in F_49^2, and each slope m (non-zero),
    # the line is y = m*x + c. So c = y - m*x.
    # Precompute for each point index (x_idx, y_idx) and slope index (1..48), the intercept c_idx.
    # Point index: x_idx * 49 + y_idx
    
    # slope_intercept[point_idx][slope_idx] = c_idx
    # slope_idx is 1..48. We'll use an array of shape (2401, 49) where index 0 is unused or for m=0 (not a valid slope).
    # Actually, let's just use 1..48.
    
    # vertical_intercept[point_idx] = x_idx (since line is x = c_v)
    
    # Let's create arrays for counts.
    # slope_counts[slope_idx][c_idx] = number of excluded points on that line so far
    # v_counts[x_idx] = number of excluded points on vertical line x = x_idx so far

    all_points = []
    for x_idx in range(49):
        for y_idx in range(49):
            all_points.append((x_idx, y_idx))
            
    # We want to greedily add points to S (excluded set) to maximize |S|.
    # A point x can be added to S if there exists a line through x (with a valid direction)
    # such that NO other point on that line is currently in S.
    # Directions are: v = (1, m) for m in F_49 \ {0}, and v = (0, 1).
    
    # For a point (x, y):
    # - For each m in F_49 \ {0}, line is y = m*x + c, c = y - m*x.
    #   We can add to S if slope_counts[m_idx][c_idx] == 0.
    # - For vertical, line is x = x.
    #   We can add to S if v_counts[x_idx] == 0.
    
    # Greedy strategy:
    # 1. Start with empty S.
    # 2. For each point (in some order), check if it can be added.
    # 3. If it can, add it and update counts.
    # To improve, we can shuffle the order of points and try multiple times, keeping the best S.
    
    best_s_size = -1
    best_s_points = None
    rng = np.random.default_rng(42)
    
    # Try a few random orderings
    for trial in range(10):
        s_counts_m = np.zeros((49, 49), dtype=np.int64) # [slope_idx (1..48 used), c_idx]
        s_counts_v = np.zeros(49, dtype=np.int64) # [x_idx]
        s_points = []
        
        # Shuffle points
        order = np.arange(2401)
        rng.shuffle(order)
        
        for pt_idx in order:
            x_idx = pt_idx // 49
            y_idx = pt_idx % 49
            
            can_add = False
            
            # Check vertical
            if s_counts_v[x_idx] == 0:
                can_add = True
                add_vert = True
            else:
                add_vert = False
                # Check slopes
                for m_idx in range(1, 49):
                    # c = y - m*x
                    # c_idx = sub[y_idx, m_mult_x]
                    # m*x = mult[m_idx, x_idx]
                    # c_idx = sub[y_idx, mult[m_idx, x_idx]]
                    mx_idx = mult[m_idx, x_idx]
                    c_idx = sub[y_idx, mx_idx]
                    if s_counts_m[m_idx, c_idx] == 0:
                        can_add = True
                        add_vert = False
                        add_m_idx = m_idx
                        add_c_idx = c_idx
                        break
                else:
                    can_add = False
            
            if can_add:
                s_points.append(pt_idx)
                if add_vert:
                    s_counts_v[x_idx] += 1
                else:
                    s_counts_m[add_m_idx, add_c_idx] += 1
        
        s_size = len(s_points)
        if s_size > best_s_size:
            best_s_size = s_size
            best_s_points = s_points[:]
            
    # Now, best_s_points contains the indices of points in S (excluded set).
    # We need to return the points NOT in S.
    s_set = set(best_s_points)
    
    nikodym_points = []
    for pt_idx in range(2401):
        if pt_idx not in s_set:
            x_idx = pt_idx // 49
            y_idx = pt_idx % 49
            xa, xb = all_elements[x_idx]
            ya, yb = all_elements[y_idx]
            nikodym_points.append([[xa, xb], [ya, yb]])
            
    return np.array(nikodym_points, dtype=np.int64)

# EVOLVE-BLOCK-END
