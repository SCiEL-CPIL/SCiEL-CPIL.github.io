import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    start_time = time.time()
    deadline = 580.0

    p = 7
    k = 3
    total_points = p * p * p * p
    num_dirs = p * p + 1

    def point_flat(xa, xb, ya, yb):
        return ((xa * p + xb) * p + ya) * p + yb

    def point_unflat(idx):
        yb = idx % p
        ya = (idx // p) % p
        xb = (idx // (p*p)) % p
        xa = (idx // (p*p*p)) % p
        return (xa, xb, ya, yb)

    line_other = []
    dir_0 = []
    for x in range(total_points):
        xa, xb, ya, yb = point_unflat(x)
        others = []
        for ta in range(p):
            for tb in range(p):
                if ta == 0 and tb == 0: continue
                nya = (ya + ta) % p
                nyb = (yb + tb) % p
                others.append(point_flat(xa, xb, nya, nyb))
        dir_0.append(others)
    line_other.append(dir_0)

    for ma in range(p):
        for mb in range(p):
            dir_m = []
            for x in range(total_points):
                xa, xb, ya, yb = point_unflat(x)
                others = []
                for ta in range(p):
                    for tb in range(p):
                        if ta == 0 and tb == 0: continue
                        nxa = (xa + ta) % p
                        nxb = (xb + tb) % p
                        tma = (ta * ma + tb * mb * k) % p
                        tmb = (ta * mb + tb * ma) % p
                        nya = (ya + tma) % p
                        nyb = (yb + tmb) % p
                        others.append(point_flat(nxa, nxb, nya, nyb))
                dir_m.append(others)
            line_other.append(dir_m)

    affected = [[] for _ in range(total_points)]
    for v in range(num_dirs):
        for p_idx in range(total_points):
            for x_idx in line_other[v][p_idx]:
                affected[p_idx].append((x_idx, v))

    best_S = np.zeros(total_points, dtype=bool)
    best_count = 0

    while time.time() - start_time < deadline:
        is_clean = np.ones((total_points, num_dirs), dtype=bool)
        clean_count = np.full(total_points, num_dirs, dtype=np.int32)
        unsafe_reasons = np.zeros(total_points, dtype=np.int32)
        in_S = np.zeros(total_points, dtype=bool)
        
        safe_queue = list(range(total_points))
        random.shuffle(safe_queue)
        
        added = 0
        while safe_queue:
            if time.time() - start_time > deadline:
                break
            p_idx = safe_queue.pop()
            if in_S[p_idx]:
                continue
            if unsafe_reasons[p_idx] != 0:
                continue
                
            in_S[p_idx] = True
            added += 1
            
            for x_idx, v_idx in affected[p_idx]:
                if is_clean[x_idx, v_idx]:
                    is_clean[x_idx, v_idx] = False
                    old_count = clean_count[x_idx]
                    clean_count[x_idx] -= 1
                    
                    if old_count == 1:
                        for q_idx in line_other[v_idx][x_idx]:
                            unsafe_reasons[q_idx] -= 1
                            if unsafe_reasons[q_idx] == 0:
                                safe_queue.append(q_idx)

        if added > best_count:
            best_count = added
            best_S = in_S.copy()

    points = []
    for x in range(total_points):
        if not best_S[x]:
            xa, xb, ya, yb = point_unflat(x)
            points.append([[xa, xb], [ya, yb]])
            
    return np.array(points, dtype=np.int64)
