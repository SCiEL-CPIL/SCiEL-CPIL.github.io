import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    start_time = time.time()
    deadline = 580.0
    
    p = 7
    k = 3  # alpha^2 = 3
    
    # Field operations
    def field_add(x, y):
        return ((x[0]+y[0])%p, (x[1]+y[1])%p)
    
    def field_mul(x, y):
        return ((x[0]*y[0] + k*x[1]*y[1])%p, (x[0]*y[1]+x[1]*y[0])%p)
    
    # Enumerate all points in F_49^2
    # Index: (xa, xb, ya, yb) -> xa*343 + xb*49 + ya*7 + yb
    n_points = p**4
    all_points = []
    point_to_idx = {}
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    idx = xa * (p**3) + xb * (p**2) + ya * p + yb
                    pt = ((xa, xb), (ya, yb))
                    all_points.append(pt)
                    point_to_idx[pt] = idx
    
    # Build lines
    # Lines are defined by direction v and offset c
    # v = (1, m) or v = (0, 1)
    # Point x = (x, y). Line: y = m*x + c (for v=(1,m)) or x = c (for v=(0,1))
    
    lines = [] # List of lists of point indices
    point_lines = [[] for _ in range(n_points)]
    
    # Vertical lines: direction (0, 1) => x = const
    for xa in range(p):
        for xb in range(p):
            line_pts = []
            for ya in range(p):
                for yb in range(p):
                    pt = ((xa,xb),(ya,yb))
                    idx = point_to_idx[pt]
                    line_pts.append(idx)
            lines.append(line_pts)
    
    # Slanted lines: direction (1, m) => y = m*x + c
    for ma in range(p):
        for mb in range(p):
            m = (ma, mb)
            for ca in range(p):
                for cb in range(p):
                    c = (ca, cb)
                    line_pts = []
                    for xa in range(p):
                        for xb in range(p):
                            x = (xa, xb)
                            mx = field_mul(m, x)
                            y = field_add(mx, c)
                            pt = (x, y)
                            idx = point_to_idx[pt]
                            line_pts.append(idx)
                    lines.append(line_pts)
    
    n_lines = len(lines)
    
    # Map point to lines
    for li, line in enumerate(lines):
        for pt_idx in line:
            point_lines[pt_idx].append(li)
    
    # State
    in_S = np.zeros(n_points, dtype=bool)
    count_S = np.zeros(n_lines, dtype=np.int32)
    good_lines = np.zeros(n_points, dtype=np.int32)
    unsatisfied = n_points # Initially S is empty, so all points are NOT in S.
                           # For x not in S, good line is L with count_S[L] == 0.
                           # Since S is empty, count_S[L] is 0 for all L.
                           # So all points have good_lines[x] = number of lines through x.
                           # Wait, if S is empty, is it valid?
                           # Condition for x not in S: exists L with count_S[L] == 0. Yes.
                           # So unsatisfied should be 0.
    
    # Initialize good_lines
    for x in range(n_points):
        # x is not in S
        for li in point_lines[x]:
            if count_S[li] == 0:
                good_lines[x] += 1
        if good_lines[x] == 0:
            unsatisfied += 1
            
    # Actually, since S is empty, count_S is 0 everywhere.
    # Every point is on 50 lines. So good_lines[x] = 50.
    # unsatisfied = 0.
    unsatisfied = 0
    best_S = in_S.copy()
    best_size = 0
    
    # Simulated Annealing
    # We want to maximize |S| subject to unsatisfied == 0.
    # Objective: Minimize (unsatisfied * 1000 - |S|)
    
    T = 50.0
    T_min = 0.1
    alpha = 0.9995
    
    current_size = 0
    current_cost = 0 # unsatisfied * 1000 - current_size
    
    # Precompute random order for points
    points_list = list(range(n_points))
    
    steps = 0
    max_steps = 500000 # Adjust based on speed
    
    while time.time() - start_time < deadline - 10:
        if steps % 1000 == 0:
            # Check time
            if time.time() - start_time > deadline - 5:
                break
            T *= alpha
            if T < T_min:
                T = T_min
                # Optional: restart with higher T if stuck?
        
        steps += 1
        
        x = random.choice(points_list)
        
        # Try to flip x
        old_in_S = in_S[x]
        
        # Calculate delta unsatisfied
        # We need to update count_S for lines through x
        # And then recalculate good_lines for points on those lines
        
        affected_points = set()
        for li in point_lines[x]:
            affected_points.update(lines[li])
            
        # Snapshot old good_lines for affected points
        old_good = {pt: good_lines[pt] for pt in affected_points}
        old_unsat_count = sum(1 for pt in affected_points if old_good[pt] == 0)
        
        # Perform flip
        if old_in_S:
            # Remove x from S
            in_S[x] = False
            for li in point_lines[x]:
                count_S[li] -= 1
        else:
            # Add x to S
            in_S[x] = True
            for li in point_lines[x]:
                count_S[li] += 1
        
        # Recalculate good_lines for affected points
        new_unsat_count = 0
        for pt in affected_points:
            new_good = 0
            for li in point_lines[pt]:
                c = count_S[li]
                if in_S[pt]:
                    if c == 1:
                        new_good += 1
                else:
                    if c == 0:
                        new_good += 1
            good_lines[pt] = new_good
            if new_good == 0:
                new_unsat_count += 1
        
        delta_unsat = new_unsat_count - old_unsat_count
        delta_size = (1 if not old_in_S else -1)
        
        delta_cost = delta_unsat * 1000 - delta_size
        
        # Acceptance
        accept = False
        if delta_cost < 0:
            accept = True
        else:
            # Probability of accepting worse solution
            # Use delta_cost as energy difference
            # If T is high, accept more
            if T > 0:
                r = random.random()
                if r < np.exp(-delta_cost / T):
                    accept = True
        
        if accept:
            current_size += delta_size
            current_cost += delta_cost
            if current_cost < best_cost if 'best_cost' in locals() else True:
                 # We want to minimize cost, so track best cost
                 pass
            # Track best valid solution (unsatisfied == 0)
            if current_cost == -current_size: # This implies unsatisfied == 0
                if current_size > best_size:
                    best_size = current_size
                    best_S = in_S.copy()
        else:
            # Revert
            in_S[x] = old_in_S
            for li in point_lines[x]:
                if old_in_S:
                    count_S[li] += 1
                else:
                    count_S[li] -= 1
            for pt in affected_points:
                good_lines[pt] = old_good[pt]
                
    # Restore best S
    in_S = best_S
    
    # Construct N (complement of S)
    N_points = []
    for i, pt in enumerate(all_points):
        if not in_S[i]:
            xa, xb = pt[0]
            ya, yb = pt[1]
            N_points.append([[xa, xb], [ya, yb]])
            
    return np.array(N_points, dtype=np.int64)
