import numpy as np
import random

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    k = 3 # alpha^2 = 3
    
    # Field elements as (a, b)
    F_q_elems = [(a, b) for a in range(p) for b in range(p)]
    zero = (0, 0)
    
    def add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
        
    def sub(a, b):
        return ((a[0] - b[0]) % p, (a[1] - b[1]) % p)
        
    def mul(a, b):
        a0, a1 = a
        b0, b1 = b
        return ((a0*b0 + a1*b1*k) % p, (a0*b1 + a1*b0) % p)
        
    def inv(a):
        a0, a1 = a
        norm = (a0*a0 + a1*a1*k) % p
        if norm == 0:
            return None
        norm_inv = pow(norm, p-2, p)
        a_p = (a0, (-a1) % p)
        return mul(a_p, (norm_inv, 0))

    # Generate all points
    all_points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    all_points.append(((xa, xb), (ya, yb)))
                    
    # Map point to index
    point_to_idx = {pt: i for i, pt in enumerate(all_points)}
    
    # Directions: (1, m) for m in F_q, and (0, 1)
    # Represent directions as vectors v
    # v = (1, m) -> (1, 0) + m*(0, 1) ? No, v is a vector in F_q^2.
    # v = (1, m_a + m_b alpha) is a point in F_q^2.
    # The line is { x + t*v | t in F_q }.
    # If v = (0, 1), line is vertical.
    # If v = (1, m), line is "slope m".
    
    # Precompute direction vectors
    directions = []
    for m in F_q_elems:
        # v = (1, m)
        directions.append(((1, 0), m)) # (v_x, v_y) where v_x=(1,0), v_y=m
    directions.append(((0, 1), (0, 0))) # v = (0, 1)
    
    # Helper to check if point y is on line through z with direction v
    # Line: z + t*v. y = z + t*v => y-z = t*v.
    # If v_x = (1,0), t = (y_x - z_x) * inv((1,0)) = y_x - z_x.
    # Then check if y_y - z_y = t * v_y.
    # If v_x = (0,1), then y_x - z_x must be 0. And y_y - z_y is any (t is determined by y_y-z_y? No, v_x=0 means t*v_x=0, so y_x=z_x. t is free? No, v=(0,1) is a specific vector. t*(0,1) = (0,t). So y_y - z_y = t. This is always possible for any y_y.
    # Wait, if v=(0,1), the line is { (z_x, z_y + t) | t in F_q }.
    # So y is on the line iff y_x == z_x.
    
    def is_on_line(z, y, v):
        # z, y are points ((xa,xb),(ya,yb))
        # v is ((vx_a, vx_b), (vy_a, vy_b))
        vx, vy = v
        # Check if y - z is a multiple of v
        # diff = y - z
        dx = sub(y[0], z[0])
        dy = sub(y[1], z[1])
        
        if vx == (1, 0):
            # t = dx
            # check if dy == t * vy
            tv = mul(dx, vy)
            return dy == tv
        elif vx == (0, 1):
            # t = dy
            # check if dx == t * vx = 0
            return dx == (0, 0)
        else:
            # General case, but we only use (1,0) and (0,1) as x-components in our direction list?
            # No, v is a vector in F_q^2. My direction list has v_x=(1,0) or (0,1).
            # So this covers all.
            return False

    # Initialize C and V
    C = set()
    V = {} # z -> set of direction indices that are valid for z
    
    # Shuffle points for randomization
    indices = list(range(len(all_points)))
    random.shuffle(indices)
    
    for idx in indices:
        y = all_points[idx]
        
        # 1. Find valid directions for y
        valid_dirs_y = set()
        for vi, v in enumerate(directions):
            ok = True
            for z_idx in C:
                z = all_points[z_idx]
                if is_on_line(z, y, v):
                    ok = False
                    break
            if ok:
                valid_dirs_y.add(vi)
                
        if not valid_dirs_y:
            continue
            
        # 2. Check if adding y invalidates any z in C
        # For each z in C, check if y lies on any of z's currently valid lines
        new_V = {}
        valid_add = True
        for z_idx in C:
            z = all_points[z_idx]
            current_valids = V[z_idx]
            new_valids = set()
            for vi in current_valids:
                v = directions[vi]
                if is_on_line(z, y, v):
                    # This direction is now blocked by y
                    continue
                else:
                    new_valids.add(vi)
            if not new_valids:
                valid_add = False
                break
            new_V[z_idx] = new_valids
            
        if not valid_add:
            continue
            
        # 3. Add y to C
        C.add(idx)
        V[idx] = valid_dirs_y
        # Update V for existing points
        for z_idx in new_V:
            V[z_idx] = new_V[z_idx]
            
    # Construct N = F_q^2 \ C
    points_in_N = []
    for i, pt in enumerate(all_points):
        if i not in C:
            (xa, xb), (ya, yb) = pt
            points_in_N.append([[xa, xb], [ya, yb]])
            
    return np.array(points_in_N, dtype=np.int64)
