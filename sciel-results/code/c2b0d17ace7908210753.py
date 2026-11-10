# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using a greedy approach to maximize the
    complement size.
    """
    start_time = time.time()
    deadline = 590  # seconds, leave margin
    
    p = 7
    q = p * p  # 49
    total_points = q * q  # 2401
    
    # Field F_49 = F_7[alpha] where alpha^2 = 3
    k = 3
    
    def idx(xa, xb, ya, yb):
        return ((xa * p + xb) * p + ya) * p + yb
    
    def deidx(i):
        yb = i % p
        ya = (i // p) % p
        xb = (i // (p*p)) % p
        xa = (i // (p*p*p)) % p
        return xa, xb, ya, yb
    
    # Directions
    dirs = []
    for ma in range(p):
        for mb in range(p):
            dirs.append((1, 0, ma, mb))
    dirs.append((0, 0, 0, 1))
    num_dirs = len(dirs)  # 49
    
    # Precompute field multiplication and addition tables
    # Represent field element as index 0..48: idx = a*7 + b for a+b*alpha
    def felem_idx(a, b):
        return a * p + b
    
    def felem_deidx(i):
        return (i // p, i % p)
    
    # Multiplication table
    mul_table = np.zeros((q, q), dtype=np.int32)
    add_table = np.zeros((q, q), dtype=np.int32)
    
    for i in range(q):
        a, b = felem_deidx(i)
        for j in range(q):
            c, d = felem_deidx(j)
            # (a+b*alpha)(c+d*alpha) = ac + bd*alpha^2 + (ad+bc)*alpha
            ra = (a*c + b*d*k) % p
            rb = (a*d + b*c) % p
            mul_table[i, j] = felem_idx(ra, rb)
            # Addition
            sa = (a+c) % p
            sb = (b+d) % p
            add_table[i, j] = felem_idx(sa, sb)
    
    # Negation
    neg_table = np.zeros(q, dtype=np.int32)
    for i in range(q):
        a, b = felem_deidx(i)
        neg_table[i] = felem_idx((-a) % p, (-b) % p)
    
    # Build reverse_map using tables
    reverse_map = [[] for _ in range(total_points)]
    
    for p_idx in range(total_points):
        px, pb, py, pbb = deidx(p_idx)
        p_xi = felem_idx(px, pb)
        p_yi = felem_idx(py, pbb)
        
        for d_idx in range(num_dirs):
            vx_a, vx_b, vy_a, vy_b = dirs[d_idx]
            v_xi = felem_idx(vx_a, vx_b)
            v_yi = felem_idx(vy_a, vy_b)
            
            for t_i in range(q):
                if t_i == 0:
                    continue
                tv_xi = mul_table[t_i, v_xi]
                tv_yi = mul_table[t_i, v_yi]
                neg_tv_xi = neg_table[tv_xi]
                neg_tv_yi = neg_table[tv_yi]
                x_xi = add_table[p_xi, neg_tv_xi]
                x_yi = add_table[p_yi, neg_tv_yi]
                xa, xb = felem_deidx(x_xi)
                ya, yb = felem_deidx(x_yi)
                x_idx = idx(xa, xb, ya, yb)
                reverse_map[p_idx].append((x_idx, d_idx))
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 60:
        return _baseline()
    
    # Greedy with multiple random orders
    best_S_size = 0
    best_in_S = None
    
    num_trials = 20
    for trial in range(num_trials):
        if time.time() - start_time > deadline - 30:
            break
        
        np.random.seed(trial)
        order = np.random.permutation(total_points)
        
        s_count = np.zeros((total_points, num_dirs), dtype=np.int8)
        work_count = np.full(total_points, num_dirs, dtype=np.int8)
        in_S = np.zeros(total_points, dtype=np.int8)
        
        for p_idx in order:
            ok = True
            for (x, d) in reverse_map[p_idx]:
                if s_count[x, d] == 0 and work_count[x] == 1:
                    ok = False
                    break
            
            if ok:
                in_S[p_idx] = 1
                for (x, d) in reverse_map[p_idx]:
                    s_count[x, d] += 1
                    if s_count[x, d] == 1:
                        work_count[x] -= 1
        
        s_size = int(in_S.sum())
        if s_size > best_S_size:
            best_S_size = s_size
            best_in_S = in_S.copy()
    
    if best_in_S is None:
        return _baseline()
    
    points = []
    for i in range(total_points):
        if not best_in_S[i]:
            xa, xb, ya, yb = deidx(i)
            points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)


def _baseline() -> np.ndarray:
    """Fallback baseline construction."""
    p = 7
    k = 3
    
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(((u * u) % p, v))
    
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    in_unital = (2 * xa + ya * ya - yb * yb * k) % p == 0
                    in_y_special = (ya, yb) in y_special
                    if in_unital and not in_y_special:
                        continue
                    points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
