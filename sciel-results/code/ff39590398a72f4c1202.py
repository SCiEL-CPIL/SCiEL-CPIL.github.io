# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    total = q * q
    
    k_const = 3
    deadline = time.time() + 580

    def add_f(A, B):
        return (A + B) % p

    def mul_f(A, B):
        a1, b1 = A[..., 0], A[..., 1]
        a2, b2 = B[..., 0], B[..., 1]
        real = (a1 * a2 + b1 * b2 * k_const) % p
        imag = (a1 * b2 + b1 * a2) % p
        return np.stack([real, imag], axis=-1)

    def to_flat(pt):
        return (pt[..., 0] * 343 + pt[..., 1] * 49 + pt[..., 2] * 7 + pt[..., 3])

    def from_flat(idx):
        idx = idx.astype(np.int64)
        xa = (idx // 343) % p
        xb = (idx // 49) % p
        ya = (idx // 7) % p
        yb = idx % p
        return np.stack([xa, xb, ya, yb], axis=-1)

    all_flat = np.arange(total, dtype=np.int64)
    all_pts = from_flat(all_flat)

    # Directions
    dirs = []
    for ma in range(p):
        for mb in range(p):
            dirs.append((1, 0, ma, mb))
    dirs.append((0, 1, 0, 0))
    n_dirs = len(dirs)

    # Precompute affected[y] = list of (x, mask_v)
    # y is in M(x, v) iff x is in M(y, v)
    # We need to compute for each y, which (x, v) pairs have y in M(x, v).
    # This is equivalent to: for each y, for each v, find all x such that y is a mate of x in direction v.
    # x + t*v = y => x = y - t*v.
    # So for a fixed y and v, the set of x is {y - t*v : t in F_49, t != 0}.
    
    # t_vals: 48 non-zero elements
    t_vals = []
    for ta in range(p):
        for tb in range(p):
            if ta == 0 and tb == 0:
                continue
            t_vals.append((ta, tb))
    t_arr = np.array(t_vals, dtype=np.int64)

    affected = [[] for _ in range(total)]
    
    # For each y, for each v, compute the 48 x's.
    # We can vectorize this.
    # But let's just loop y and v. 2401 * 50 = 120k iterations.
    # Inside, we compute 48 x's.
    
    for y_idx in range(total):
        y_pt = all_pts[y_idx]
        for v_idx, v in enumerate(dirs):
            vxa, vxb, vya, vyb = v
            # Compute t*v for all t
            ta = t_arr[:, 0]
            tb = t_arr[:, 1]
            tv_xa = (ta * vxa + tb * vxb * k_const) % p
            tv_xb = (ta * vxb + tb * vxa) % p
            tv_ya = (ta * vya + tb * vyb * k_const) % p
            tv_yb = (ta * vyb + tb * vya) % p
            tv_pts = np.stack([tv_xa, tv_xb, tv_ya, tv_yb], axis=-1)
            
            # x = y - tv
            x_pts = (y_pt[None, :] - tv_pts) % p
            x_flats = to_flat(x_pts)
            
            # For each x, the mask is just bit v_idx
            mask_v = 1 << v_idx
            for x_flat in x_flats:
                affected[y_idx].append((int(x_flat), mask_v))
                
    # Optimization: Combine masks for same x in affected[y_idx]
    # affected[y_idx] is a list of (x, mask). Multiple entries for same x?
    # Yes, if y is a mate of x in multiple directions?
    # No, for a fixed y and v, x is unique for each t.
    # But different v's give different x's?
    # Wait, x is determined by y, v, t.
    # For a fixed y and v, the 48 x's are distinct.
    # For different v, the x's might overlap?
    # If x is on two lines through y, then y-x is parallel to two directions?
    # In a vector space, a non-zero vector has a unique direction (up to scalar).
    # But here directions are (1, m) and (0, 1).
    # The lines are distinct.
    # So for a fixed y, the sets of x for different v are disjoint?
    # x = y - t*v.
    # If x = y - t1*v1 = y - t2*v2, then t1*v1 = t2*v2.
    # Since v1, v2 are distinct directions, this implies t1=t2=0, which is excluded.
    # So the x's are distinct for different v.
    # So no merging needed.
    
    # Initialize Z
    Z = np.full(total, (1 << n_dirs) - 1, dtype=np.uint64)
    
    candidates = list(range(total))
    random.shuffle(candidates)
    
    S_mask = np.zeros(total, dtype=bool)
    
    while candidates and time.time() < deadline - 5:
        # Pick a random candidate
        y = candidates.pop()
        
        # Check if safe
        safe = True
        for x, mask_v in affected[y]:
            if Z[x] & ~mask_v == 0:
                safe = False
                break
        
        if safe:
            S_mask[y] = True
            # Update Z
            for x, mask_v in affected[y]:
                Z[x] &= ~mask_v
        # If not safe, y is discarded (not added to S, not kept in candidates)
        
    # Build N
    N_mask = ~S_mask
    N_pts = all_pts[N_mask]
    k = len(N_pts)
    
    if k == 0:
        points = np.zeros((1, 2, 2), dtype=np.int64)
    else:
        points = np.empty((k, 2, 2), dtype=np.int64)
        points[:, 0, 0] = N_pts[:, 0]
        points[:, 0, 1] = N_pts[:, 1]
        points[:, 1, 0] = N_pts[:, 2]
        points[:, 1, 1] = N_pts[:, 3]
        
    return points

# EVOLVE-BLOCK-END
