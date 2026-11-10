import numpy as np
import time

def nikodym_set() -> np.ndarray:
    p = 7
    k = 3  # Smallest quadratic non-residue mod 7 is 3 (1, 2, 4 are squares)
    
    # Field operations
    def add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
    
    def sub(a, b):
        return ((a[0] - b[0]) % p, (a[1] - b[1]) % p)
    
    def mul(a, b):
        return ((a[0]*b[0] + a[1]*b[1]*k) % p, (a[0]*b[1] + a[1]*b[0]) % p)
    
    def neg(a):
        return ((-a[0]) % p, (-a[1]) % p)
        
    # Generate all points
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    points.append(((xa, xb), (ya, yb)))
    
    q = p * p
    n_points = q * q
    
    # Map point to index
    def pt_to_idx(pt):
        x, y = pt
        xa, xb = x
        ya, yb = y
        return ((xa * p + xb) * q + (ya * p + yb))
        
    # Map index to point
    def idx_to_pt(idx):
        yb = idx % p
        ya = (idx // p) % p
        xb = (idx // (p*p)) % p
        xa = (idx // (p*p*p)) % p
        return ((xa, xb), (ya, yb))
        
    # Precompute lines
    # Directions: 0..48 are slopes m, 49 is vertical
    lines = [] # List of lists of point indices
    point_to_lines = [[] for _ in range(n_points)]
    
    # Vertical lines: x = c
    for c in range(q):
        line_pts = []
        for y in range(q):
            # x is c, y is y
            # Need to convert c, y to field elements
            # c = ca + cb*alpha, y = ya + yb*alpha
            # Actually, it's easier to iterate over all points and check
            pass
        
    # Better way to generate lines:
    # Iterate over all points and all directions, but avoid duplicates.
    # Or just generate all lines explicitly.
    
    # Slopes m in F_q
    slopes = [((a, b)) for a in range(p) for b in range(p)]
    
    # For each slope, there are q lines.
    # Line equation: y = m*x + c.
    # For a given m and c, the line is { (x, m*x+c) : x in F_q }.
    
    all_lines = []
    
    for m in slopes:
        for c in points: # c is the y-intercept
            line_pts = []
            for x in points:
                y = add(mul(m, x), c)
                pt = (x, y)
                line_pts.append(pt_to_idx(pt))
            all_lines.append(line_pts)
            
    # Vertical lines: x = c
    for c in points:
        line_pts = []
        for y in points:
            pt = (c, y)
            line_pts.append(pt_to_idx(pt))
        all_lines.append(line_pts)
        
    # Map point to lines
    for i, line in enumerate(all_lines):
        for pt_idx in line:
            point_to_lines[pt_idx].append(i)
            
    # Map (point, direction) to line index?
    # We need to find the line through x with direction m quickly.
    # Let's create a dict or array.
    # dir 0..49.
    # line_dir[line_idx] = dir
    line_dir = []
    for i in range(len(slopes)):
        line_dir.append(i)
    for i in range(len(points)):
        line_dir.append(49)
        
    # point_line_dir[x][m] = line_idx
    # x in 0..2400, m in 0..49
    point_line_dir = [[-1]*50 for _ in range(n_points)]
    for line_idx, line_pts in enumerate(all_lines):
        m = line_dir[line_idx]
        for pt_idx in line_pts:
            point_line_dir[pt_idx][m] = line_idx
            
    # Greedy Removal
    in_S = np.ones(n_points, dtype=bool)
    obs_count = np.full((n_points, 50), q-2, dtype=np.int32) # 48
    c_count = np.zeros(n_points, dtype=np.int32)
    
    # Initially, no one is isolated.
    
    def remove_point(y):
        in_S[y] = False
        for m in range(50):
            line_idx = point_line_dir[y][m]
            line_pts = all_lines[line_idx]
            for x in line_pts:
                if x == y:
                    continue
                obs_count[x][m] -= 1
                if obs_count[x][m] == 0:
                    c_count[x] += 1
                    
    def add_point(z):
        # Check if adding z keeps validity
        # z must be isolated on some line? No, z just needs to be in S.
        # But adding z might break others.
        # Also, z itself must be isolated on some line in the NEW S.
        # Before adding, z is not in S.
        # obs_count for z is not tracked (or is q-2? No, it's irrelevant).
        # When we add z, we need to know if z is isolated on some line.
        # Isolated on line L if no other point in S on L.
        # So we need to check if there exists m such that obs_count[z][m] == 0?
        # But obs_count[z][m] counts points in S.
        # If we haven't added z, obs_count[z][m] is the number of OTHER points in S on that line.
        # If obs_count[z][m] == 0, then z is isolated on that line.
        
        # First, check if z can be isolated.
        if np.min(obs_count[z]) > 0:
            return False
            
        # Check if adding z breaks any existing point.
        # For each neighbor x of z, obs_count[x][m] increases by 1.
        # If obs_count[x][m] was 0, it becomes 1, so c_count[x] decreases by 1.
        # If c_count[x] becomes 0, invalid.
        
        # Save state
        changes = []
        valid = True
        for m in range(50):
            line_idx = point_line_dir[z][m]
            line_pts = all_lines[line_idx]
            for x in line_pts:
                if x == z:
                    continue
                if in_S[x]:
                    if obs_count[x][m] == 0:
                        # This line was the only isolation for x
                        c_count[x] -= 1
                        changes.append((x, m, 1)) # 1 means c_count decreased
                        if c_count[x] < 0: # Should not happen if valid
                            valid = False
                        if c_count[x] == 0:
                            valid = False
                    else:
                        obs_count[x][m] += 1
                        changes.append((x, m, 0)) # 0 means obs_count increased
                    
        if not valid:
            # Revert
            for x, m, typ in changes:
                if typ == 0:
                    obs_count[x][m] -= 1
                else:
                    c_count[x] += 1
            return False
            
        # Commit
        in_S[z] = True
        # We already updated obs_count and c_count in the loop above?
        # Wait, in the check loop, I modified them.
        # If valid, the modifications are already there.
        # But I need to update obs_count for z itself?
        # obs_count[z] is not used for z's validity in the future?
        # It IS used if we remove z later? No, we don't remove in add phase.
        # But it is used if we try to add another point?
        # Yes, if we add w, and w is on a line with z, obs_count[w][m] should count z.
        # So we need obs_count[z][m] to be correct?
        # No, obs_count[x][m] counts points in S OTHER THAN x.
        # So for z, obs_count[z][m] should be the number of points in S on line m through z.
        # Currently, obs_count[z][m] is the number of points in S (before adding z) on that line.
        # So it is correct for z's perspective?
        # Wait, if I add z, does obs_count[z] change?
        # No, obs_count[z][m] counts OTHERS. So it stays the same.
        # But for others, it increases.
        # So the state is consistent.
        
        return True

    # Greedy Phase
    start_time = time.time()
    deadline = 600 - 30 # 30s safety margin
    
    while time.time() < deadline:
        # Find bad points
        bad = []
        for x in range(n_points):
            if in_S[x] and c_count[x] == 0:
                bad.append(x)
                
        if not bad:
            break
            
        # Score points
        scores = np.zeros(n_points, dtype=np.int32)
        for x in bad:
            min_obs = np.min(obs_count[x])
            for m in range(50):
                if obs_count[x][m] == min_obs:
                    line_idx = point_line_dir[x][m]
                    line_pts = all_lines[line_idx]
                    for y in line_pts:
                        if y != x and in_S[y]:
                            scores[y] += 1
                            
        # Pick max score
        max_score = np.max(scores)
        if max_score == 0:
            # Should not happen if bad is not empty
            # Pick random
            y = bad[0]
            # Find a neighbor
            m = np.argmin(obs_count[y])
            line_idx = point_line_dir[y][m]
            line_pts = all_lines[line_idx]
            candidates = [p for p in line_pts if p != y and in_S[p]]
            if candidates:
                y = candidates[0]
            else:
                break
        else:
            candidates = np.where(scores == max_score)[0]
            y = candidates[np.random.randint(len(candidates))]
            
        remove_point(y)
        
    # Local Search Phase (Add Back)
    # Try to add points from N to S
    # Iterate multiple times
    for _ in range(10):
        if time.time() > deadline:
            break
        improved = False
        # Shuffle order to break symmetry
        order = np.random.permutation(n_points)
        for z in order:
            if in_S[z]:
                continue
            if add_point(z):
                improved = True
        if not improved:
            break
            
    # Construct N
    N_indices = [i for i in range(n_points) if not in_S[i]]
    
    # Convert to array
    result = np.zeros((len(N_indices), 2, 2), dtype=np.int64)
    for i, idx in enumerate(N_indices):
        x, y = idx_to_pt(idx)
        result[i, 0, 0] = x[0]
        result[i, 0, 1] = x[1]
        result[i, 1, 0] = y[0]
        result[i, 1, 1] = y[1]
        
    return result

