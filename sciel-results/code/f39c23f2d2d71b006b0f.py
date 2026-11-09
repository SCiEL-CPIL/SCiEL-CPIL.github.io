# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    
    Strategy:
    1. Precompute field arithmetic tables for F_49.
    2. Initialize N using the Hermitian unital construction.
    3. Precompute the 'affects' mapping.
    4. Use Randomized Greedy with multiple restarts to minimize |N|.
       - In each restart, shuffle the order of points in N.
       - Greedily remove points if valid.
       - Keep the best result (smallest |N|) across restarts.
    """
    deadline = time.time() + 620  # Safety margin before 630s limit

    p = 7
    q = p * p  # 49

    # Field F_q = F_p[alpha], alpha^2 = k
    # Find smallest quadratic non-residue mod p
    k_field = None
    for c in range(2, p):
        if pow(c, (p - 1) // 2, p) == p - 1:
            k_field = c
            break
    if k_field is None:
        k_field = 3  # fallback

    # Represent elements of F_q as integers 0..48: element (a, b) -> a + b*p
    def to_int(a, b):
        return a + b * p

    def to_pair(x):
        return (x % p, (x // p) % p)

    # Precompute field arithmetic tables
    # f_add[x, y] = x + y
    # f_mul[x, y] = x * y
    # f_neg[x] = -x
    f_add = np.zeros((q, q), dtype=np.int32)
    f_mul = np.zeros((q, q), dtype=np.int32)
    f_neg = np.zeros(q, dtype=np.int32)

    for x in range(q):
        xa, xb = to_pair(x)
        f_neg[x] = to_int((-xa) % p, (-xb) % p)
        for y in range(q):
            ya, yb = to_pair(y)
            # Addition
            f_add[x, y] = to_int((xa + ya) % p, (xb + yb) % p)
            # Multiplication: (xa + xb*alpha)(ya + yb*alpha)
            real = (xa * ya + xb * yb * k_field) % p
            imag = (xa * yb + xb * ya) % p
            f_mul[x, y] = to_int(real, imag)

    # Directions: v = (1, m) for m in F_q, or v = (0, 1)
    directions = []
    for m in range(q):
        directions.append((1, m))  # v = (1, m)
    directions.append((0, 1))  # vertical
    n_dir = len(directions)

    # All points in F_q^2. Index: x_idx * q + y_idx
    point_index = {}
    all_points = []
    for x_idx in range(q):
        for y_idx in range(q):
            idx = x_idx * q + y_idx
            point_index[(x_idx, y_idx)] = idx
            all_points.append((x_idx, y_idx))
    
    idx_to_point = [None] * (q * q)
    for pt, idx in point_index.items():
        idx_to_point[idx] = pt

    # Initialize N
    in_N = np.ones(q * q, dtype=bool)

    # Hermitian unital: U = {(x, y) : Trace(x) + Norm(y) = 0}
    # Trace((a,b)) = 2a, Norm((a,b)) = a^2 - b^2*k
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(to_int((u * u) % p, v))

    for x_idx in range(q):
        xa, xb = to_pair(x_idx)
        trace_x = (2 * xa) % p
        for y_idx in range(q):
            ya, yb = to_pair(y_idx)
            norm_y = (ya * ya - yb * yb * k_field) % p
            if (trace_x + norm_y) % p == 0:
                # In unital
                if y_idx not in y_special:
                    # Excluded
                    in_N[point_index[(x_idx, y_idx)]] = False

    # not_in_N_count[p_idx, d_idx] = number of points on line(p, v_d) \ {p} that are NOT in N
    not_in_N_count = np.zeros((q * q, n_dir), dtype=np.int32)

    # Compute initial counts
    for d_idx, (vx, vy) in enumerate(directions):
        # Precompute t*v for t in 1..q-1
        t_v = []
        for t in range(1, q):
            tx = f_mul[t, vx]
            ty = f_mul[t, vy]
            t_v.append((tx, ty))
        
        for p_idx in range(q * q):
            px, py = idx_to_point[p_idx]
            count = 0
            for tx, ty in t_v:
                nx = f_add[px, tx]
                ny = f_add[py, ty]
                n_idx = point_index[(nx, ny)]
                if not in_N[n_idx]:
                    count += 1
            not_in_N_count[p_idx, d_idx] = count

    # cover_count[p_idx] = number of directions d such that not_in_N_count[p_idx, d] == 0
    cover_count = np.sum(not_in_N_count == 0, axis=1).astype(np.int32)

    # Precompute affects
    # affects[r_idx] = list of (p_idx, d_idx) such that r is on line(p, d) \ {p}
    affects = [[] for _ in range(q * q)]
    for r_idx in range(q * q):
        rx, ry = idx_to_point[r_idx]
        for d_idx, (vx, vy) in enumerate(directions):
            for t in range(1, q):
                # p = r - t*v
                tv_x = f_mul[t, vx]
                tv_y = f_mul[t, vy]
                neg_tv_x = f_neg[tv_x]
                neg_tv_y = f_neg[tv_y]
                px = f_add[rx, neg_tv_x]
                py = f_add(ry, neg_tv_y)
                if (px, py) in point_index:
                    p_idx = point_index[(px, py)]
                    affects[r_idx].append((p_idx, d_idx))

    # Helper to run one greedy pass with a given order
    def run_greedy(order):
        # Deep copy state
        in_N_curr = in_N.copy()
        not_in_N_curr = not_in_N_count.copy()
        cover_count_curr = cover_count.copy()
        
        for r_idx in order:
            if not in_N_curr[r_idx]:
                continue
            # Check if removal is valid
            invalid = False
            affected = []
            
            for p_idx, d_idx in affects[r_idx]:
                old_count = not_in_N_curr[p_idx, d_idx]
                new_count = old_count + 1
                
                if old_count == 0:
                    if cover_count_curr[p_idx] == 1:
                        invalid = True
                        break
                    affected.append((p_idx, d_idx, old_count))
                else:
                    # old_count > 0, so cover_count doesn't change, but not_in_N_count does
                    affected.append((p_idx, d_idx, old_count))
            
            if invalid:
                continue
            
            # Commit changes
            for p_idx, d_idx, old_count in affected:
                not_in_N_curr[p_idx, d_idx] = old_count + 1
                if old_count == 0:
                    cover_count_curr[p_idx] -= 1
            
            in_N_curr[r_idx] = False
            
        return in_N_curr

    # Get initial list of points in N
    initial_N_indices = [i for i in range(q * q) if in_N[i]]
    
    best_in_N = in_N.copy()
    best_size = np.sum(in_N)
    
    # Randomized Restarts
    # Estimate time per restart
    # q*q = 2401 points.
    # Inner loop is small.
    # Let's try 100 restarts.
    
    start_time = time.time()
    num_restarts = 0
    max_restarts = 200 # Tune based on time
    
    # First, run deterministic greedy to get a baseline
    order_det = initial_N_indices
    best_in_N = run_greedy(order_det)
    best_size = np.sum(best_in_N)
    num_restarts += 1
    
    while time.time() < deadline - 10:
        random.shuffle(initial_N_indices)
        in_N_curr = run_greedy(initial_N_indices)
        curr_size = np.sum(in_N_curr)
        
        if curr_size < best_size:
            best_size = curr_size
            best_in_N = in_N_curr.copy()
            # print(f"Improved to {best_size}")
        
        num_restarts += 1
        if num_restarts % 10 == 0:
            # print(f"Restart {num_restarts}, Best: {best_size}")
            pass

    # Build result from best_in_N
    points = []
    for i in range(q * q):
        if best_in_N[i]:
            x, y = idx_to_point[i]
            xa, xb = to_pair(x)
            ya, yb = to_pair(y)
            points.append([[xa, xb], [ya, yb]])

    if not points:
        # Fallback
        points = []
        for x_idx in range(q):
            xa, xb = to_pair(x_idx)
            for y_idx in range(q):
                ya, yb = to_pair(y_idx)
                if in_N[point_index[(x_idx, y_idx)]]:
                    points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
