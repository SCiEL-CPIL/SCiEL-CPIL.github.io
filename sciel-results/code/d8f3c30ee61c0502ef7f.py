# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_q^2 with q = 49 (p=7).
    Uses a greedy "add then clean" strategy to maximize the complement C.
    """
    p = 7
    q = p * p
    deadline = time.time() + 580  # Safety margin
    
    k_const = 3
    
    # Point indexing
    # Point (xa, xb, ya, yb) -> idx = ((xa * p + xb) * p + ya) * p + yb
    def pt_to_idx(pt):
        xa, xb, ya, yb = pt
        return ((xa * p + xb) * p + ya) * p + yb
    
    def idx_to_pt(idx):
        yb = idx % p
        idx //= p
        ya = idx % p
        idx //= p
        xb = idx % p
        idx //= p
        xa = idx
        return (xa, xb, ya, yb)
    
    n_points = p**4
    all_pts = [idx_to_pt(i) for i in range(n_points)]
    
    # Precompute line indices
    # line_indices[x][v] is a numpy array of indices of other points on the line
    # Directions: 50
    # v=0..48: (1, m) where m = (v//p, v%p)
    # v=49: (0, 1)
    
    def get_dir(v_idx):
        if v_idx < 49:
            ma = v_idx // p
            mb = v_idx % p
            return (1, 0, ma, mb)
        else:
            return (0, 1, 0, 0)
    
    def mul_t_v(ta, tb, v):
        vxa, vxb, vya, vyb = v
        return ((ta*vxa + tb*vxb*k_const) % p,
                (ta*vxb + tb*vxa) % p,
                (ta*vya + tb*vyb*k_const) % p,
                (ta*vyb + tb*vya) % p)
    
    def add_pt_idx(idx, delta):
        # delta is (dxa, dxb, dya, dyb)
        pt = idx_to_pt(idx)
        xa, xb, ya, yb = pt
        dxa, dxb, dya, dyb = delta
        nxa = (xa + dxa) % p
        nxb = (xb + dxb) % p
        nya = (ya + dya) % p
        nyb = (yb + dyb) % p
        return pt_to_idx((nxa, nxb, nya, nyb))
    
    line_indices = []
    for x_idx in range(n_points):
        x_lines = []
        for v_idx in range(50):
            v = get_dir(v_idx)
            pts_on_line = []
            for ta in range(p):
                for tb in range(p):
                    if ta == 0 and tb == 0:
                        continue
                    delta = mul_t_v(ta, tb, v)
                    y_idx = add_pt_idx(x_idx, delta)
                    pts_on_line.append(y_idx)
            x_lines.append(np.array(pts_on_line, dtype=np.int32))
        line_indices.append(x_lines)
    
    # C is a boolean array of size n_points
    C = np.zeros(n_points, dtype=bool)
    
    def has_free_direction(x_idx):
        for v_idx in range(50):
            # Check if any point on the line is in C
            if not C[line_indices[x_idx][v_idx]].any():
                return True
        return False
    
    # Greedy loop
    while time.time() < deadline:
        # Find candidates
        candidates = []
        for x_idx in range(n_points):
            if not C[x_idx]:
                if has_free_direction(x_idx):
                    candidates.append(x_idx)
        
        if not candidates:
            break
        
        # Add a random subset of candidates to avoid immediate invalidation
        # Or add all? Adding all is faster but might invalidate many.
        # Let's add all, then clean.
        for x_idx in candidates:
            C[x_idx] = True
        
        # Clean: remove points that no longer have a free direction
        # Only points in C need to be checked
        # We can just iterate through all points in C
        changed = True
        while changed:
            changed = False
            for x_idx in range(n_points):
                if C[x_idx]:
                    if not has_free_direction(x_idx):
                        C[x_idx] = False
                        changed = True
            if time.time() > deadline:
                break
    
    # Build N = all_points - C
    # N indices are where C is False
    N_indices = np.where(~C)[0]
    
    # Convert to array of shape (k, 2, 2)
    points = np.zeros((len(N_indices), 2, 2), dtype=np.int64)
    for i, idx in enumerate(N_indices):
        xa, xb, ya, yb = idx_to_pt(idx)
        points[i, 0, 0] = xa
        points[i, 0, 1] = xb
        points[i, 1, 0] = ya
        points[i, 1, 1] = yb
    
    return points

# EVOLVE-BLOCK-END
