# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using a corrected greedy algorithm.
    """
    p_prime = 7
    q = 49
    N_points = q * q
    num_lines = q + 1 # 49 slopes + 1 vertical

    # Field arithmetic
    # F_49 = F_7[alpha], alpha^2 = 3
    # Elements are (a, b) representing a + b*alpha
    # We represent elements as integers 0..48
    # idx = a * 7 + b
    
    def add(e1, e2):
        a1, b1 = divmod(e1, 7)
        a2, b2 = divmod(e2, 7)
        a = (a1 + a2) % 7
        b = (b1 + b2) % 7
        return a * 7 + b

    def sub(e1, e2):
        a1, b1 = divmod(e1, 7)
        a2, b2 = divmod(e2, 7)
        a = (a1 - a2) % 7
        b = (b1 - b2) % 7
        return a * 7 + b

    def mul(e1, e2):
        a1, b1 = divmod(e1, 7)
        a2, b2 = divmod(e2, 7)
        # (a1 + b1*alpha) * (a2 + b2*alpha) = a1*a2 + (a1*b2 + b1*a2)*alpha + b1*b2*alpha^2
        # alpha^2 = 3
        a = (a1 * a2 + 3 * b1 * b2) % 7
        b = (a1 * b2 + b1 * a2) % 7
        return a * 7 + b

    def inv(e):
        if e == 0:
            return 0
        # (a + b*alpha)^-1 = (a - b*alpha) / (a^2 - 3b^2)
        a, b = divmod(e, 7)
        denom = (a * a - 3 * b * b) % 7
        # In F_7, inverse of d is d^5 (since 7 is prime, x^-1 = x^(7-2) = x^5)
        denom_inv = pow(denom, 5, 7)
        # Numerator: a - b*alpha
        na = a
        nb = (-b) % 7
        return mul(na * 7 + nb, denom_inv)

    def div(e1, e2):
        return mul(e1, inv(e2))

    # Precompute points and coordinates
    # Point index i: 0..2400
    # i = xa * 343 + xb * 49 + ya * 7 + yb
    # xa, xb, ya, yb in 0..6
    
    coords = np.zeros((N_points, 4), dtype=np.int32)
    for i in range(N_points):
        yb = i % 7
        ya = (i // 7) % 7
        xb = (i // 49) % 7
        xa = (i // 343) % 7
        coords[i] = [xa, xb, ya, yb]
        
    # Convert coords to field elements
    # x = xa + xb*alpha -> idx_x = xa*7 + xb
    # y = ya + yb*alpha -> idx_y = ya*7 + yb
    x_elems = coords[:, 0] * 7 + coords[:, 1]
    y_elems = coords[:, 2] * 7 + coords[:, 3]

    # Precompute line indices for each pair (x, p)
    # line_idx[x, p] is the index (0..49) of the line passing through x and p
    # 0..48: slopes m = 0..48
    # 49: vertical
    
    line_idx = np.zeros((N_points, N_points), dtype=np.int8)
    
    for x in range(N_points):
        x_elem = x_elems[x]
        y_elem = y_elems[x]
        for p in range(N_points):
            if p == x:
                line_idx[x, p] = -1
                continue
            p_x_elem = x_elems[p]
            p_y_elem = y_elems[p]
            
            dx = sub(p_x_elem, x_elem)
            dy = sub(p_y_elem, y_elem)
            
            if dx == 0:
                line_idx[x, p] = 49
            else:
                m = div(dy, dx)
                line_idx[x, p] = m

    # State
    is_valid_line = np.ones((N_points, num_lines), dtype=bool)
    valid_count = np.full(N_points, num_lines, dtype=np.int32)
    in_S = np.zeros(N_points, dtype=bool)
    
    rng = np.random.RandomState(42)
    candidates = np.arange(N_points)
    rng.shuffle(candidates)
    
    start_time = time.time()
    deadline = start_time + 500
    
    for p_idx in candidates:
        if time.time() > deadline:
            break
            
        # Tentatively add p_idx
        # For each x != p_idx, check if line_idx[x, p_idx] was valid
        # If yes, mark it invalid and decrement valid_count[x]
        
        affected = []
        
        # We iterate over all x != p_idx
        # To speed up, we can use numpy where possible, but the logic depends on is_valid_line
        # Let's just loop. 2400 iterations is small.
        
        min_count_after = num_lines # Track min to see if we need to revert
        
        for x in range(N_points):
            if x == p_idx:
                continue
            l_idx = line_idx[x, p_idx]
            if is_valid_line[x, l_idx]:
                # This line becomes invalid
                is_valid_line[x, l_idx] = False
                valid_count[x] -= 1
                affected.append((x, l_idx))
                if valid_count[x] < min_count_after:
                    min_count_after = valid_count[x]
        
        if min_count_after > 0:
            # Commit
            in_S[p_idx] = True
        else:
            # Revert
            for x, l_idx in affected:
                is_valid_line[x, l_idx] = True
                valid_count[x] += 1
    
    # Construct result N
    # N is complement of S
    n_points = np.sum(~in_S)
    
    result_list = []
    for i in range(N_points):
        if not in_S[i]:
            xa, xb, ya, yb = coords[i]
            result_list.append([
                [xa, xb],
                [ya, yb]
            ])
            
    if len(result_list) == 0:
        return np.empty((0, 2, 2), dtype=np.int64)
        
    result = np.array(result_list, dtype=np.int64)
    
    return result

# EVOLVE-BLOCK-END
