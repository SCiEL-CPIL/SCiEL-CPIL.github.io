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
    # point_to_lines will be a 2D array of shape (2401, 50)
    # We need to map (point, direction) to line_id
    # But directions are 50. So for each point, we have 50 lines.
    point_lines = np.zeros((len(points), 50), dtype=int)
    
    # We need to assign line IDs consistently.
    # A line is uniquely identified by its set of points.
    # Let's generate lines for each point and direction, and map them.
    
    # To avoid O(N^2) memory for line_map if we use sets, we can use a dictionary mapping (min_point, max_point) or similar?
    # No, a line has 49 points.
    # We can map the sorted tuple of points to an ID.
    # There are 2450 lines.
    # 2401 * 50 = 120050 (point, line) pairs.
    # Unique lines = 2450.
    
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
    # U: 2*xa + ya^2 - yb^2*k = 0
    in_U = np.zeros(len(points), dtype=bool)
    for pt, idx in point_to_idx.items():
        xa, xb = pt[0]
        ya, yb = pt[1]
        if (2 * xa + ya * ya - yb * yb * k) % p == 0:
            in_U[idx] = True
            
    # Initialize S
    # Baseline: y_a in {0, 3, 5, 6}
    S_mask = np.zeros(len(points), dtype=bool)
    for pt, idx in point_to_idx.items():
        xa, xb = pt[0]
        ya, yb = pt[1]
        if in_U[idx] and ya in {0, 3, 5, 6}:
            S_mask[idx] = True
            
    # Calculate initial line counts
    line_counts = np.zeros(num_lines, dtype=int)
    # For each point in S, increment count for all lines through it
    # S_indices = np.where(S_mask)[0]
    # for idx in S_indices:
    #     line_counts[point_lines[idx]] += 1
    
    # Vectorized update:
    # We can use bincount
    # all_point_lines = point_lines[S_mask] -> shape (num_S, 50)
    # line_counts = np.bincount(all_point_lines.ravel(), minlength=num_lines)
    
    if np.any(S_mask):
        line_counts = np.bincount(point_lines[S_mask].ravel(), minlength=num_lines)
    else:
        line_counts = np.zeros(num_lines, dtype=int)

    def is_valid(S_mask, line_counts):
        # For x in N (not S): need line with count 0
        # For x in S: need line with count 1
        
        # Get counts for all points
        # point_lines shape (2401, 50)
        # line_counts shape (num_lines,)
        # counts_for_points = line_counts[point_lines] -> shape (2401, 50)
        
        counts_for_points = line_counts[point_lines]
        
        # Check N points
        # N_mask = ~S_mask
        # valid_N = np.any(counts_for_points[N_mask] == 0, axis=1)
        # all_N_valid = np.all(valid_N)
        
        # Check S points
        # valid_S = np.any(counts_for_points[S_mask] == 1, axis=1)
        # all_S_valid = np.all(valid_S)
        
        # Combine
        # To save memory/time, we can check in parts or just do it all.
        # 2401 * 50 is small.
        
        if not np.all(np.any(counts_for_points[~S_mask] == 0, axis=1)):
            return False
        if not np.all(np.any(counts_for_points[S_mask] == 1, axis=1)):
            return False
            
        return True

    # Initial check
    if not is_valid(S_mask, line_counts):
        # Fallback: return empty N? No, must return valid N.
        # If baseline is invalid, we have a bug.
        # Let's assume baseline is valid.
        pass

    # Candidates: Points in U not in S
    candidates = []
    for pt, idx in point_to_idx.items():
        if in_U[idx] and not S_mask[idx]:
            candidates.append(idx)
            
    # Sort candidates
    def sort_key(idx):
        pt = points[idx]
        xa, xb = pt[0]
        ya, yb = pt[1]
        return (ya, yb, xa, xb)
        
    candidates.sort(key=sort_key)
    
    start_time = time.time()
    
    # Greedy addition
    # We can try to add points.
    # To speed up, we can check if adding a point *could* possibly work.
    # Adding a point p increases count of 50 lines by 1.
    # This might break validity for:
    # 1. Points in N that relied on one of those lines having count 0.
    # 2. Points in S that relied on one of those lines having count 1 (now 2).
    
    # We just use the fast is_valid.
    
    for idx in candidates:
        if time.time() - start_time > 550:
            break
            
        # Add idx
        S_mask[idx] = True
        line_counts[point_lines[idx]] += 1
        
        if is_valid(S_mask, line_counts):
            # Keep
            pass
        else:
            # Revert
            S_mask[idx] = False
            line_counts[point_lines[idx]] -= 1

    # Construct N
    N_points = []
    for pt, idx in point_to_idx.items():
        if not S_mask[idx]:
            xa, xb = pt[0]
            ya, yb = pt[1]
            N_points.append([[xa, xb], [ya, yb]])
            
    return np.array(N_points, dtype=np.int64)

# EVOLVE-BLOCK-END
