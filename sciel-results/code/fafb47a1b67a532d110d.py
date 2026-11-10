# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def nikodym_set() -> np.ndarray:
    p = 7
    k = 3
    
    n_total = p * p * p * p
    
    def to_idx(xa, xb, ya, yb):
        return ((xa * p + xb) * p + ya) * p + yb

    def to_coords(idx):
        yb = idx % p
        ya = (idx // p) % p
        xb = (idx // (p * p)) % p
        xa = (idx // (p * p * p)) % p
        return xa, xb, ya, yb

    def add_field(a1, b1, a2, b2):
        return ((a1 + a2) % p, (b1 + b2) % p)

    def mul_field(a1, b1, a2, b2):
        ra = (a1 * a2 + b1 * b2 * k) % p
        rb = (a1 * b2 + b1 * a2) % p
        return ra, rb

    # Precompute lines
    # Directions: 50
    directions = [((0, 0), (0, 1))]
    for ma in range(p):
        for mb in range(p):
            directions.append(((1, 0), (ma, mb)))
            
    lines = []
    lines_of_point = [[] for _ in range(n_total)]
    
    seen_lines = set()
    
    for i in range(n_total):
        xa, xb, ya, yb = to_coords(i)
        for d_idx in range(len(directions)):
            (dx_a, dx_b), (dy_a, dy_b) = directions[d_idx]
            # Generate all points on this line
            pts = set()
            for t_a in range(p):
                for t_b in range(p):
                    tx_a, tx_b = mul_field(t_a, t_b, dx_a, dx_b)
                    ty_a, ty_b = mul_field(t_a, t_b, dy_a, dy_b)
                    nxa = (xa + tx_a) % p
                    nx_b = (xb + tx_b) % p
                    nya = (ya + ty_a) % p
                    ny_b = (yb + ty_b) % p
                    j = to_idx(nxa, nx_b, nya, ny_b)
                    pts.add(j)
            # Canonicalize line to avoid duplicates
            # Sort points
            pts_list = sorted(pts)
            line_key = tuple(pts_list)
            if line_key not in seen_lines:
                seen_lines.add(line_key)
                line_id = len(lines)
                lines.append(pts_list)
                for pt in pts_list:
                    lines_of_point[pt].append(line_id)

    n_lines = len(lines)
    
    best_S = set()
    n_starts = 20
    rng = random.Random(42)
    
    start_time = time.time()
    
    for start in range(n_starts):
        if time.time() - start_time > 500:
            break
            
        S = set()
        in_S = [False] * n_total
        counts = [0] * n_lines
        only_point = [-1] * n_lines
        n_clean = [0] * n_total
        
        order = list(range(n_total))
        rng.shuffle(order)
        
        for x in order:
            valid = True
            x_has_clean = False
            to_decrement = []
            
            for L in lines_of_point[x]:
                c = counts[L]
                if c == 0:
                    x_has_clean = True
                elif c == 1:
                    y = only_point[L]
                    to_decrement.append(y)
                    if n_clean[y] == 1:
                        valid = False
                        break
                else:
                    pass
                    
            if valid and x_has_clean:
                S.add(x)
                in_S[x] = True
                n_clean[x] = 0
                for L in lines_of_point[x]:
                    c = counts[L]
                    counts[L] = c + 1
                    if c == 0:
                        only_point[L] = x
                        n_clean[x] += 1
                    elif c == 1:
                        y = only_point[L]
                        n_clean[y] -= 1
                        only_point[L] = -1
                    else:
                        pass

        if len(S) > len(best_S):
            best_S = S.copy()
            
    n_points = n_total - len(best_S)
    points = []
    for i in range(n_total):
        if i not in best_S:
            xa, xb, ya, yb = to_coords(i)
            points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
