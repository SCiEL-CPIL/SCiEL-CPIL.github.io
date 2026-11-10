# EVOLVE-BLOCK-START
import numpy as np
import random

def nikodym_set() -> np.ndarray:
    p = 7
    q = 49
    total_points = q * q
    num_directions = q + 1  # 50
    
    def point_to_idx(xa, xb, ya, yb):
        return ((xa * p + xb) * p + ya) * p + yb
    
    def idx_to_point(idx):
        yb = idx % p
        ya = (idx // p) % p
        xb = (idx // (p * p)) % p
        xa = (idx // (p * p * p)) % p
        return xa, xb, ya, yb

    # Precompute neighbors[x] = list of (y, d) such that x is on L(y, d) \ {y}
    # This is equivalent to: y is on L(x, d) \ {x} ?
    # No. L(y, d) is the line through y with direction d.
    # x is on L(y, d) \ {y} iff y is on L(x, d) \ {x}.
    # Because the line is the same.
    # So we can just iterate over lines.
    
    neighbors = [[] for _ in range(total_points)]
    
    # Generate all lines for each direction
    # Direction d is defined by vector v_d.
    # We need to identify unique lines.
    # A line is determined by a point and a direction.
    # To avoid duplicates, we can pick a canonical point for each line, e.g., the one with smallest index.
    # Or simply iterate over all points and directions, but only add if current point is the "first" on the line?
    # Easier: For each direction, iterate over all points, and for each point, find the "start" of the line.
    # Or just generate all pairs (y, x) on the same line with direction d.
    
    # Let's just iterate over all y and all d, and find the other points on the line.
    # For fixed y and d, the points are y + t*v_d for t in F_49.
    # We can compute these 49 points.
    # For each x in these points (x != y), add (y, d) to neighbors[x].
    
    # We need the vector v_d in F_49 coords.
    vectors = []
    for ma in range(p):
        for mb in range(p):
            # v = (1, m)
            # vx = (1, 0), vy = (ma, mb)
            vectors.append(((1, 0), (ma, mb)))
    # Vertical v = (0, 1)
    # vx = (0, 0), vy = (1, 0)
    vectors.append(((0, 0), (1, 0)))
    
    for d, (vx, vy) in enumerate(vectors):
        vx_a, vx_b = vx
        vy_a, vy_b = vy
        
        for y_idx in range(total_points):
            ya_a, ya_b, yb_a, yb_b = idx_to_point(y_idx)
            
            # Generate points on line
            # x = y + t*v
            # x_a = y_a + t*v_x
            # x_b = y_b + t*v_y
            # t is in F_49.
            
            for t_a in range(p):
                for t_b in range(p):
                    # t = (t_a, t_b)
                    # t * vx
                    # (t_a, t_b) * (vx_a, vx_b)
                    # = (t_a*vx_a + t_b*vx_b*3, t_a*vx_b + t_b*vx_a)
                    tx_a = (t_a * vx_a + t_b * vx_b * 3) % p
                    tx_b = (t_a * vx_b + t_b * vx_a) % p
                    
                    # t * vy
                    ty_a = (t_a * vy_a + t_b * vy_b * 3) % p
                    ty_b = (t_a * vy_b + t_b * vy_a) % p
                    
                    xa_a = (ya_a + tx_a) % p
                    xa_b = (ya_b + tx_b) % p
                    xb_a = (yb_a + ty_a) % p
                    xb_b = (yb_b + ty_b) % p
                    
                    x_idx = point_to_idx(xa_a, xa_b, xb_a, xb_b)
                    
                    if x_idx != y_idx:
                        neighbors[x_idx].append((y_idx, d))
    
    # Greedy removal
    best_N_size = total_points
    best_N_mask = np.ones(total_points, dtype=bool)
    
    for trial in range(10):
        N_mask = np.ones(total_points, dtype=bool)
        is_valid_dir = np.ones((total_points, num_directions), dtype=bool)
        valid_lines = np.full(total_points, num_directions, dtype=np.int32)
        
        points_to_check = list(range(total_points))
        random.shuffle(points_to_check)
        
        for x in points_to_check:
            if not N_mask[x]:
                continue
            
            safe = True
            for y, d in neighbors[x]:
                if is_valid_dir[y][d]:
                    if valid_lines[y] == 1:
                        safe = False
                        break
            
            if safe:
                N_mask[x] = False
                for y, d in neighbors[x]:
                    if is_valid_dir[y][d]:
                        is_valid_dir[y][d] = False
                        valid_lines[y] -= 1
        
        current_size = np.sum(N_mask)
        if current_size < best_N_size:
            best_N_size = current_size
            best_N_mask = N_mask.copy()
            
    result_indices = [i for i in range(total_points) if best_N_mask[i]]
    result = np.zeros((len(result_indices), 2, 2), dtype=np.int64)
    for i, idx in enumerate(result_indices):
        xa, xb, ya, yb = idx_to_point(idx)
        result[i] = [[xa, xb], [ya, yb]]
        
    return result

# EVOLVE-BLOCK-END
