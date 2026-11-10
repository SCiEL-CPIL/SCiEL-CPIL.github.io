import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    total_points = q * q  # 2401
    k = 3  # alpha^2 = 3 in F_7
    
    start_time = time.time()
    deadline = start_time + 580  # safety margin
    
    # Field elements as integers 0..48: (a, b) -> a + b*p
    def to_int(a, b):
        return a + b * p
    
    def to_pair(val):
        return (val % p, val // p)
    
    # Precompute addition and multiplication tables
    add_tbl = np.zeros((q, q), dtype=np.int16)
    mul_tbl = np.zeros((q, q), dtype=np.int16)
    
    for i in range(q):
        ai, bi = to_pair(i)
        for j in range(q):
            aj, bj = to_pair(j)
            # Addition
            add_tbl[i, j] = to_int((ai + aj) % p, (bi + bj) % p)
            # Multiplication: (ai + bi*alpha)(aj + bj*alpha) = (ai*aj + bi*bj*3) + (ai*bj + bi*aj)*alpha
            mul_tbl[i, j] = to_int((ai * aj + bi * bj * k) % p, (ai * bj + bi * aj) % p)
    
    # Build lines: lines[x_idx][d] = array of 48 point indices on the line through x with direction d
    # d = 0..48: direction (1, m) where m = d
    # d = 49: direction (0, 1)
    num_dirs = q + 1  # 50
    
    lines = np.zeros((total_points, num_dirs, q - 1), dtype=np.int32)
    
    for x_int in range(q):
        for y_int in range(q):
            x_idx = x_int * q + y_int
            
            # Direction (0, 1): vertical
            cnt = 0
            for t in range(1, q):
                y2 = add_tbl[y_int, t]
                lines[x_idx, q, cnt] = x_int * q + y2
                cnt += 1
            
            # Directions (1, m)
            for m in range(q):
                cnt = 0
                for t in range(1, q):
                    x2 = add_tbl[x_int, t]
                    tm = mul_tbl[t, m]
                    y2 = add_tbl[y_int, tm]
                    lines[x_idx, m, cnt] = x2 * q + y2
                    cnt += 1
    
    # For each point s, precompute affected[s] = list of (x, d) where s is in lines[x][d], x != s
    # Equivalently, x is in lines[s][d]
    affected = []
    for s in range(total_points):
        pairs = []
        for d in range(num_dirs):
            for x in lines[s, d]:
                pairs.append((x, d))
        affected.append(pairs)
    
    # Greedy construction
    random.seed(42)
    
    best_S_size = 0
    best_S = set()
    
    def try_greedy(order):
        nonlocal best_S_size, best_S
        if time.time() > deadline:
            return
        
        covered = np.ones((total_points, num_dirs), dtype=np.int8)
        num_cov = np.full(total_points, num_dirs, dtype=np.int8)
        S = set()
        in_S = np.zeros(total_points, dtype=bool)
        
        for s in order:
            if in_S[s]:
                continue
            # Check if s can be added
            ok = True
            for (x, d) in affected[s]:
                if num_cov[x] == 1 and covered[x, d] == 1:
                    ok = False
                    break
            if ok:
                for (x, d) in affected[s]:
                    if covered[x, d] == 1:
                        covered[x, d] = 0
                        num_cov[x] -= 1
                S.add(s)
                in_S[s] = True
        
        if len(S) > best_S_size:
            best_S_size = len(S)
            best_S = set(S)
    
    # Multiple random restarts
    for trial in range(50):
        if time.time() > deadline - 30:
            break
        order = list(range(total_points))
        random.shuffle(order)
        try_greedy(order)
    
    # Build N = complement of best_S
    points = []
    for x_int in range(q):
        for y_int in range(q):
            idx = x_int * q + y_int
            if idx not in best_S:
                xa, xb = to_pair(x_int)
                ya, yb = to_pair(y_int)
                points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)
