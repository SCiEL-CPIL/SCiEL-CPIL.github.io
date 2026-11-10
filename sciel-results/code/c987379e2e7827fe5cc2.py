import numpy as np

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    k = 3 # alpha^2 = 3

    # Field ops
    def add_f(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
        
    def sub_f(a, b):
        return ((a[0] - b[0]) % p, (a[1] - b[1]) % p)
        
    def mul_f(a, b):
        return ((a[0]*b[0] + a[1]*b[1]*k) % p, (a[0]*b[1] + a[1]*b[0]) % p)
        
    def div_f(a, b):
        inv_denom = (b[0]*b[0] - b[1]*b[1]*k) % p
        if inv_denom == 0:
            return None
        inv_denom_inv = pow(inv_denom, p-2, p)
        inv_b = ((b[0] * inv_denom_inv) % p, (-b[1] * inv_denom_inv) % p)
        return mul_f(a, inv_b)

    # Generate points
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    points.append(((xa, xb), (ya, yb)))
                    
    n_points = len(points)
    point_to_idx = {pt: i for i, pt in enumerate(points)}
    
    # Directions
    # 0..48: slope m
    # 49: infinity (vertical)
    dirs = []
    for ma in range(p):
        for mb in range(p):
            dirs.append(((1, 0), (ma, mb)))
    dirs.append(((0, 1), (0, 0)))
    
    n_dirs = len(dirs)
    
    # Precompute lines: lines[x_idx][d_idx] = list of point indices
    lines = [[[] for _ in range(n_dirs)] for _ in range(n_points)]
    for x_idx, x in enumerate(points):
        for d_idx, dir_vec in enumerate(dirs):
            dx, dy = dir_vec
            line_pts = []
            for t_a in range(p):
                for t_b in range(p):
                    t = (t_a, t_b)
                    tdx = mul_f(t, dx)
                    tdy = mul_f(t, dy)
                    pt_x = add_f(x[0], tdx)
                    pt_y = add_f(x[1], tdy)
                    pt_full = (pt_x, pt_y)
                    line_pts.append(point_to_idx[pt_full])
            lines[x_idx][d_idx] = line_pts
            
    # Precompute direction between any two points
    # dir_between[x_idx][y_idx] = d_idx
    # If x == y, undefined (set to -1)
    dir_between = np.full((n_points, n_points), -1, dtype=np.int8)
    for x_idx in range(n_points):
        for d_idx in range(n_dirs):
            for y_idx in lines[x_idx][d_idx]:
                if y_idx != x_idx:
                    dir_between[x_idx][y_idx] = d_idx
                    dir_between[y_idx][x_idx] = d_idx # Direction is symmetric for line identification?
                    # Wait, direction (1, m) vs (1, m) is same line.
                    # The set of points is the same.
                    # So dir_between is symmetric.
                    
    # Greedy
    S = set()
    A = set(range(n_points))
    valid_dirs = [set(range(n_dirs)) for _ in range(n_points)]
    
    while A:
        best_x = -1
        best_d = -1
        best_cost = float('inf')
        
        for x in A:
            if not valid_dirs[x]:
                continue
            for d in valid_dirs[x]:
                # Cost: number of points in A that are on Line(x, d) and != x
                # These points will be removed from A
                cost = 0
                for y in lines[x][d]:
                    if y == x:
                        continue
                    if y in A:
                        cost += 1
                if cost < best_cost:
                    best_cost = cost
                    best_x = x
                    best_d = d
                    
        if best_x == -1:
            break
            
        S.add(best_x)
        
        # Remove best_x and all points on Line(best_x, best_d) from A
        line_pts = lines[best_x][best_d]
        for y in line_pts:
            A.discard(y)
            
        # Update valid_dirs for remaining points in A
        for y in A:
            d_xy = dir_between[best_x][y]
            if d_xy != -1:
                valid_dirs[y].discard(d_xy)
                if not valid_dirs[y]:
                    A.discard(y)
                    
    # N is complement of S
    N = [points[i] for i in range(n_points) if i not in S]
    arr = np.array(N, dtype=np.int64)
    return arr
