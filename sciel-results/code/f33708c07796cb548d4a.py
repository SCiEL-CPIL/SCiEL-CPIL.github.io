import numpy as np
import time


def nikodym_set() -> np.ndarray:
    """
    Construct a small Nikodym set in F_49^2.
    Strategy:
    1. Start with a known valid Nikodym set based on the Hermitian unital construction.
    2. Greedily add points to the excluded set S (remove from N) to maximize complement size.
       Use numpy vectorization for fast validity checks.
    """
    p = 7
    q = p * p  # 49
    k = 3  # alpha^2 = 3, quadratic non-residue mod 7
    n_points = q * q  # 2401
    
    # Directions: (1, m) for m in F_q, and (0, 1).
    directions = []
    for ma in range(p):
        for mb in range(p):
            directions.append(((1, 0), (ma, mb)))
    directions.append(((0, 0), (0, 1)))
    n_dirs = len(directions)  # 50
    
    # Precompute field operations
    def f_mul(a, b):
        return ((a[0]*b[0] + a[1]*b[1]*k) % p, (a[0]*b[1] + a[1]*b[0]) % p)
        
    def f_add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
        
    # Field elements indexed 0..48
    field_elements = []
    for a in range(p):
        for b in range(p):
            field_elements.append((a, b))
            
    def get_coords(i):
        xa = i // 343
        xb = (i // 49) % 7
        ya = (i // 7) % 7
        yb = i % 7
        return xa, xb, ya, yb
        
    def get_index(xa, xb, ya, yb):
        return xa * 343 + xb * 49 + ya * 7 + yb

    # Precompute line members: line_members[i, d, t] = index of point at distance t from i in direction d
    line_members = np.zeros((n_points, n_dirs, q-1), dtype=np.int32)
    
    for i in range(n_points):
        xa, xb, ya, yb = get_coords(i)
        pt = ((xa, xb), (ya, yb))
        for d in range(n_dirs):
            v = directions[d]
            for t_idx in range(1, q):
                t = field_elements[t_idx]
                tv_x = f_mul(t, v[0])
                tv_y = f_mul(t, v[1])
                new_x = f_add(pt[0], tv_x)
                new_y = f_add(pt[1], tv_y)
                nxa, nxb = new_x
                nya, nyb = new_y
                j = get_index(nxa, nxb, nya, nyb)
                line_members[i, d, t_idx-1] = j

    # Step 1: Construct initial S using the unital method.
    # U = {(x, y) : Trace(x) + Norm(y) = 0}
    # Trace((a,b)) = 2a, Norm((a,b)) = a^2 - 3*b^2
    # Special y set Y = {(u^2 mod p, v) : u, v in F_p}
    
    y_special = np.zeros((p, p), dtype=bool)
    for u in range(p):
        for v in range(p):
            y_special[(u * u) % p, v] = True
            
    S = np.zeros(n_points, dtype=bool)
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    in_unital = (2 * xa + ya * ya - 3 * yb * yb) % p == 0
                    in_y_special = y_special[ya, yb]
                    if in_unital and not in_y_special:
                        idx = get_index(xa, xb, ya, yb)
                        S[idx] = True
                        
    # Initialize validity tracking
    # is_valid[x, d] = True if L(x, d) \ {x} is fully in N (no point of S on that line)
    is_valid = np.ones((n_points, n_dirs), dtype=bool)
    valid_dirs = np.zeros(n_points, dtype=np.int32)
    
    for x in range(n_points):
        for d in range(n_dirs):
            if np.any(S[line_members[x, d, :]]):
                is_valid[x, d] = False
            else:
                valid_dirs[x] += 1
                
    # Verify initial set is valid
    if np.any(valid_dirs == 0):
        # Fallback: if construction failed, return full set
        result = np.zeros((n_points, 2, 2), dtype=np.int64)
        for idx in range(n_points):
            xa, xb, ya, yb = get_coords(idx)
            result[idx, 0, 0] = xa
            result[idx, 0, 1] = xb
            result[idx, 1, 0] = ya
            result[idx, 1, 1] = yb
        return result

    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    # Greedy addition to S with multiple passes
    # Try several passes with different orderings to escape local optima
    
    best_complement = int(np.sum(~S))
    best_S = S.copy()
    best_is_valid = is_valid.copy()
    best_valid_dirs = valid_dirs.copy()
    
    n_passes = 5
    for pass_num in range(n_passes):
        if time.time() > deadline:
            break
            
        if pass_num == 0:
            order = list(range(n_points))
        else:
            order = list(range(n_points))
            np.random.shuffle(order)
            
        for p_idx in order:
            if time.time() > deadline:
                    break
                    
            if S[p_idx]:
                continue
                
            # Check if adding p_idx to S is valid
            # For each direction d, check if any x in line_members[p_idx, d] has valid_dirs[x]==1 and is_valid[x,d]
            valid = True
            for d in range(n_dirs):
                members = line_members[p_idx, d, :]
                # Find x in members that are currently valid for direction d
                x_valid_mask = is_valid[members, d]
                if np.any(x_valid_mask):
                    x_valid = members[x_valid_mask]
                    if np.any(valid_dirs[x_valid] == 1):
                        valid = False
                        break
                        
            if valid:
                S[p_idx] = True
                # Update is_valid and valid_dirs
                for d in range(n_dirs):
                    members = line_members[p_idx, d, :]
                    x_valid_mask = is_valid[members, d]
                    if np.any(x_valid_mask):
                        is_valid[members[x_valid_mask], d] = False
                        valid_dirs[members[x_valid_mask]] -= 1
                        
        complement = int(np.sum(~S))
        if complement > best_complement:
            best_complement = complement
            best_S = S.copy()
            best_is_valid = is_valid.copy()
            best_valid_dirs = valid_dirs.copy()
            
        # Check if we can continue
        if time.time() > deadline - 60:
            break
            
    # Use the best result found
    S = best_S
    is_valid = best_is_valid
    valid_dirs = best_valid_dirs
    
    # Construct N
    N_indices = np.where(~S)[0]
    
    # Create result array
    result = np.zeros((len(N_indices), 2, 2), dtype=np.int64)
    for idx, i in enumerate(N_indices):
        xa, xb, ya, yb = get_coords(int(i))
        result[idx, 0, 0] = xa
        result[idx, 0, 1] = xb
        result[idx, 1, 0] = ya
        result[idx, 1, 1] = yb
        
    return result
