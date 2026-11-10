import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    # Field F_49 = F_7[alpha] with alpha^2 = 3
    # alpha^2 = 3 is a non-residue in F_7?
    # Squares in F_7: 0, 1, 4, 2 (3^2=9=2), 2 (4^2=16=2), 1 (5^2=25=4), 4 (6^2=36=1)
    # Residues: {0, 1, 2, 4}. Non-residues: {3, 5, 6}.
    # So alpha^2 = 3 works.
    
    # Map (a, b) to index a * p + b
    # Point (x, y) where x = xa + xb*alpha, y = ya + yb*alpha
    # Index = (xa * p + xb) * q + (ya * p + yb)
    # Total points: q * q = 2401
    
    # Precompute field operations
    # add, sub, mul in F_49
    # (a, b) + (c, d) = (a+c, b+d)
    # (a, b) - (c, d) = (a-c, b-d)
    # (a, b) * (c, d) = (ac + bd*3, ad + bc)
    
    def add_f(v1, v2):
        return ((v1[0] + v2[0]) % p, (v1[1] + v2[1]) % p)
        
    def sub_f(v1, v2):
        return ((v1[0] - v2[0]) % p, (v1[1] - v2[1]) % p)
        
    def mul_f(v1, v2):
        a, b = v1
        c, d = v2
        return ((a * c + b * d * 3) % p, (a * d + b * c) % p)
        
    # Inverse in F_49
    # Norm(a, b) = a^2 - 3b^2
    # Inv(a, b) = (a, -b) / Norm
    def inv_f(v):
        a, b = v
        norm = (a * a - 3 * b * b) % p
        if norm == 0:
            return None
        inv_norm = pow(norm, p - 2, p)
        return ((a * inv_norm) % p, (-b * inv_norm) % p)
        
    # Slopes: 0..6 (elements of F_7) and 7 (infinity)
    # Actually, slopes are in F_49.
    # There are q = 49 finite slopes and 1 infinite slope. Total 50.
    # Let's represent slopes as indices 0..48 for finite, 49 for infinite.
    
    # Precompute lines
    # lines[x_idx][m_idx] = list of point indices on the line through x with slope m
    # x_idx in 0..2400
    # m_idx in 0..49
    
    # To save memory and time, we can compute on the fly or precompute neighbors.
    # 2401 * 50 * 49 is large but manageable if we just store the neighbors for update.
    # Actually, we just need to know which points are on the line to update their forbidden sets.
    
    # Let's precompute for each point x and slope m, the list of other points on that line.
    # This takes 2401 * 50 * 49 ~ 5.8M entries.
    # Each entry is an int. 5.8M * 4 bytes = 23 MB. Fine.
    
    lines = np.zeros((q * q, q + 1, q - 1), dtype=np.int32)
    # Initialize with -1
    
    for xa in range(p):
        for xb in range(p):
            x_val = (xa, xb)
            x_idx = (xa * p + xb) * q
            for m_idx in range(q + 1):
                if m_idx == q: # Infinite slope (vertical line in affine plane? No, x=const)
                    # Line: x = x_val
                    # Points: (x_val, y) for all y
                    count = 0
                    for ya in range(p):
                        for yb in range(p):
                            y_val = (ya, yb)
                            y_idx = (ya * p + yb)
                            z_idx = x_idx + y_idx
                            if z_idx != (x_idx + 0): # exclude x itself? No, x is (x_val, 0) ? No.
                                # x_idx corresponds to (x_val, 0) in y?
                                # No, x_idx is index of (x_val, 0) if y=0?
                                # My indexing: idx = x_part * q + y_part
                                # So x_idx = (xa*p+xb)*q + 0
                                # z_idx = (xa*p+xb)*q + (ya*p+yb)
                                # If ya=0, yb=0, z_idx == x_idx.
                                pass
                    # This loop is getting messy. Let's generate all points first.
                    
    # Better approach:
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    points.append(((xa, xb), (ya, yb)))
                    
    # Index mapping
    # idx = (xa * p + xb) * q + (ya * p + yb)
    
    # Precompute lines
    # lines[x_idx][m_idx] is an array of indices of points on the line, excluding x_idx.
    
    # Slopes representation:
    # m in F_49 is (ma, mb).
    # Line: y = m * (x - x0) + y0
    # For each x' in F_49, y' = m * (x' - x0) + y0
    # If m is infinity, x' = x0.
    
    lines = []
    for x_idx in range(q * q):
        lines.append([])
        for m_idx in range(q + 1):
            pts_on_line = []
            # Get x0, y0
            x_part = x_idx // q
            y_part = x_idx % q
            xa, xb = divmod(x_part, p)
            ya, yb = divmod(y_part, p)
            x0 = (xa, xb)
            y0 = (ya, yb)
            
            if m_idx == q:
                # Vertical line: x = x0
                for x_prime_idx in range(q):
                    if x_prime_idx == x_part:
                        continue
                    # All y' for this x'
                    for y_prime_idx in range(q):
                        z_idx = x_prime_idx * q + y_prime_idx
                        pts_on_line.append(z_idx)
            else:
                ma, mb = divmod(m_idx, p)
                m_val = (ma, mb)
                for x_prime_idx in range(q):
                    if x_prime_idx == x_part:
                        continue
                    xa_p, xb_p = divmod(x_prime_idx, p)
                    x_prime = (xa_p, xb_p)
                    dx = sub_f(x_prime, x0)
                    dy = add_f(mul_f(m_val, dx), y0)
                    ya_p, yb_p = dy
                    y_prime_idx = ya_p * p + yb_p
                    z_idx = x_prime_idx * q + y_prime_idx
                    pts_on_line.append(z_idx)
            lines[x_idx].append(np.array(pts_on_line, dtype=np.int32))
            
    # Greedy algorithm
    def run_greedy(seed):
        random.seed(seed)
        # forbidden[x_idx] is a set of slopes that are forbidden for x_idx
        # We can use a bitmask since there are 50 slopes.
        # 50 bits fits in a 64-bit int.
        forbidden = np.zeros(q * q, dtype=np.int64)
        in_S = np.zeros(q * q, dtype=bool)
        # available_count[x_idx] = number of available slopes
        available_count = np.full(q * q, q + 1, dtype=np.int32)
        
        # Order of processing
        order = list(range(q * q))
        random.shuffle(order)
        
        S_size = 0
        for x_idx in order:
            if in_S[x_idx]:
                continue
            # Check if any slope is available
            if available_count[x_idx] == 0:
                continue
            # Pick a random available slope
            # Find available slopes
            avail_mask = ~forbidden[x_idx] & ((1 << (q + 1)) - 1)
            # Pick one
            # Get list of set bits
            # Simple way: iterate
            chosen_m = -1
            temp_mask = avail_mask
            while temp_mask:
                # Get lowest set bit
                lsb = temp_mask & -temp_mask
                # Index of lsb
                # This is slow in Python loop.
                # Alternative: just pick random index and check
                pass
            
            # Faster: precompute random permutation of slopes?
            # Or just pick random m and check if forbidden.
            while True:
                m = random.randint(0, q)
                if not (forbidden[x_idx] & (1 << m)):
                    chosen_m = m
                    break
            
            # Add x_idx to S with slope chosen_m
            in_S[x_idx] = True
            S_size += 1
            forbidden[x_idx] |= (1 << chosen_m) # Mark chosen slope as used/forbidden for itself? No, it's assigned.
            
            # Update neighbors
            neighbors = lines[x_idx][chosen_m]
            for z_idx in neighbors:
                if in_S[z_idx]:
                    continue
                # Add chosen_m to forbidden[z_idx]
                forbidden[z_idx] |= (1 << chosen_m)
                available_count[z_idx] -= 1
                # If available_count becomes 0, it's dead.
                
        return S_size, in_S
        
    best_S_size = 0
    best_in_S = None
    
    start_time = time.time()
    deadline = start_time + 590 # 10s margin
    
    seed = 0
    while time.time() < deadline:
        S_size, in_S = run_greedy(seed)
        if S_size > best_S_size:
            best_S_size = S_size
            best_in_S = in_S.copy()
            # print(f"New best: {S_size} at seed {seed}")
        seed += 1
        
    if best_in_S is None:
        # Fallback to baseline if greedy fails (should not happen)
        # Reconstruct baseline logic quickly or return empty?
        # Baseline is better than empty.
        # Let's just return a valid set from the first run if possible.
        pass
        
    # Construct N from best_in_S
    # N = all points NOT in S
    N_indices = np.where(~best_in_S)[0]
    
    # Convert indices to points
    points = []
    for idx in N_indices:
        x_part = idx // q
        y_part = idx % q
        xa, xb = divmod(x_part, p)
        ya, yb = divmod(y_part, p)
        points.append([[xa, xb], [ya, yb]])
        
    return np.array(points, dtype=np.int64)

nikodym_set()
