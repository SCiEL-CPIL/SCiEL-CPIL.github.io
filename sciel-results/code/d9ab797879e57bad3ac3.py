# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    k = 3  # alpha^2 = 3

    def add_f49(u, v):
        return ((u[0] + v[0]) % p, (u[1] + v[1]) % p)

    def mul_f49(u, v):
        a, b = u
        c, d = v
        return ((a*c + b*d*k) % p, (a*d + b*c) % p)

    # Generate all directions
    directions = []
    for ma in range(p):
        for mb in range(p):
            if ma == 0 and mb == 0:
                continue
            directions.append(((1, 0), (ma, mb))) # (1, m)
    directions.append(((0, 1), (0, 0))) # (0, 1)
    
    # Map points to indices
    points = []
    point_to_idx = {}
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    pt = ((xa, xb), (ya, yb))
                    idx = len(points)
                    points.append(pt)
                    point_to_idx[pt] = idx

    # Precompute lines
    line_map = {}
    lines = []
    point_lines = np.zeros((len(points), 50), dtype=int)
    
    for x_idx, x in enumerate(points):
        for d_idx, (dx, dy) in enumerate(directions):
            line_pts = []
            for ta in range(p):
                for tb in range(p):
                    t = (ta, tb)
                    px = add_f49(x[0], mul_f49(t, dx))
                    py = add_f49(x[1], mul_f49(t, dy))
                    pt = (px, py)
                    line_pts.append(point_to_idx[pt])
            
            line_key = tuple(sorted(line_pts))
            if line_key not in line_map:
                line_map[line_key] = len(lines)
                lines.append(line_key)
            
            lid = line_map[line_key]
            point_lines[x_idx, d_idx] = lid

    num_lines = len(lines)
    
    # Identify points in U
    in_U = np.zeros(len(points), dtype=bool)
    for pt, idx in point_to_idx.items():
        xa, xb = pt[0]
        ya, yb = pt[1]
        if (2 * xa + ya * ya - yb * yb * k) % p == 0:
            in_U[idx] = True
            
    # Initialize S
    S_mask = np.zeros(len(points), dtype=bool)
    for pt, idx in point_to_idx.items():
        xa, xb = pt[0]
        ya, yb = pt[1]
        if in_U[idx] and ya in {0, 3, 5, 6}:
            S_mask[idx] = True
            
    # Calculate initial line counts
    if np.any(S_mask):
        line_counts = np.bincount(point_lines[S_mask].ravel(), minlength=num_lines)
    else:
        line_counts = np.zeros(num_lines, dtype=int)

    def is_valid(S_mask, line_counts):
        counts_for_points = line_counts[point_lines]
        
        if not np.all(np.any(counts_for_points[~S_mask] == 0, axis=1)):
            return False
        if not np.all(np.any(counts_for_points[S_mask] == 1, axis=1)):
            return False
            
        return True

    # Initial check
    if not is_valid(S_mask, line_counts):
        pass

    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s limit

    # Multi-pass greedy: add points to S, then remove points from S
    for iteration in range(20):
        if time.time() > deadline:
            break
        
        # Pass 1: Try to add candidates to S
        candidates = []
        for pt, idx in point_to_idx.items():
            if in_U[idx] and not S_mask[idx]:
                candidates.append(idx)
                
        def sort_key(idx):
            pt = points[idx]
            xa, xb = pt[0]
            ya, yb = pt[1]
            return (ya, yb, xa, xb)
            
        candidates.sort(key=sort_key)
        
        for idx in candidates:
            if time.time() > deadline:
                break
            
            S_mask[idx] = True
            line_counts[point_lines[idx]] += 1
            
            if is_valid(S_mask, line_counts):
                pass
            else:
                S_mask[idx] = False
                line_counts[point_lines[idx]] -= 1
        
        # Pass 2: Try to remove points from S (to shrink S and increase complement)
        s_points = [idx for idx in range(len(points)) if S_mask[idx]]
        s_points.sort(key=sort_key, reverse=True)
        
        for idx in s_points:
            if time.time() > deadline:
                break
            
            S_mask[idx] = False
            line_counts[point_lines[idx]] -= 1
            
            if is_valid(S_mask, line_counts):
                pass
            else:
                S_mask[idx] = True
                line_counts[point_lines[idx]] += 1

    # Construct N
    N_points = []
    for pt, idx in point_to_idx.items():
        if not S_mask[idx]:
            xa, xb = pt[0]
            ya, yb = pt[1]
            N_points.append([[xa, xb], [ya, yb]])
            
    return np.array(N_points, dtype=np.int64)

# EVOLVE-BLOCK-END
