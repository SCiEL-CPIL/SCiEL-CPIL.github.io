import numpy as np
from itertools import product

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    
    # Field F_49 = F_7[alpha] where alpha^2 = 3
    # Elements represented as (a, b) meaning a + b*alpha
    # Multiplication: (a,b)*(c,d) = (ac+3bd, ad+bc) mod 7
    
    def field_add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
    
    def field_mul(a, b):
        return ((a[0]*b[0] + 3*b[1]*a[1]) % p, 
                (a[0]*b[1] + a[1]*b[0]) % p)
    
    # Map field element tuples to indices 0..48
    field_elements = [(a, b) for a in range(p) for b in range(p)]
    elem_to_idx = {e: i for i, e in enumerate(field_elements)}
    
    # Precompute multiplication table for speed
    mul_table = np.zeros((q, q), dtype=np.int64)
    for i, a in enumerate(field_elements):
        for j, b in enumerate(field_elements):
            mul_table[i, j] = elem_to_idx[field_mul(a, b)]

    # Points in F_49^2
    # Point index: x_idx * q + y_idx
    # x_idx, y_idx in 0..48
    
    all_points = list(range(q * q))
    
    # Generate all lines
    # Line representation: (point_idx, direction_idx)
    # Directions: 49 slopes m (for v=(1,m)) + 1 vertical (v=(0,1))
    # Total 50 directions.
    
    lines = []
    point_to_lines = {pt: [] for pt in all_points}
    
    # Helper to get point index from coords
    def get_point_idx(x_idx, y_idx):
        return x_idx * q + y_idx

    # We need to map field element indices to coords for generation?
    # Or just work with indices and field ops.
    
    # Let's generate lines by iterating through points and directions?
    # No, that generates duplicates.
    # Better: iterate through all possible lines.
    # A line is determined by a point and a direction.
    # To avoid duplicates, we can just generate all unique sets of points.
    # Or, since we need `points_on_line` and `lines_through_point`,
    # we can generate each line once.
    
    # There are 2450 lines.
    # Vertical lines: x = const. 49 lines.
    # Non-vertical: y = m*x + c. 49 * 49 = 2401 lines.
    
    line_points_list = [] # list of lists of point indices
    
    # Vertical lines
    for x_idx in range(q):
        pts = []
        for y_idx in range(q):
            pts.append(get_point_idx(x_idx, y_idx))
        line_points_list.append(pts)
        
    # Non-vertical lines
    # y = m*x + c
    # m, c in F_49 (indices 0..48)
    for m_idx in range(q):
        for c_idx in range(q):
            pts = []
            for x_idx in range(q):
                # y = m*x + c
                # y_idx = mul_table[m_idx, x_idx] + c_idx mod 49?
                # No, field addition is component-wise mod 7.
                # We need to add field elements.
                # Let's precompute addition table too.
                pass
                
    # Let's precompute add table
    add_table = np.zeros((q, q), dtype=np.int64)
    for i, a in enumerate(field_elements):
        for j, b in enumerate(field_elements):
            add_table[i, j] = elem_to_idx[field_add(a, b)]

    for m_idx in range(q):
        for c_idx in range(q):
            pts = []
            for x_idx in range(q):
                # y = m*x + c
                # y_elem = m_elem * x_elem + c_elem
                # y_idx = add_table[mul_table[m_idx, x_idx], c_idx]
                y_idx = add_table[mul_table[m_idx, x_idx], c_idx]
                pts.append(get_point_idx(x_idx, y_idx))
            line_points_list.append(pts)
            
    # Total lines: 49 + 2401 = 2450
    # Build incidence
    lines_through_point = {pt: [] for pt in all_points}
    for l_idx, pts in enumerate(line_points_list):
        for pt in pts:
            lines_through_point[pt].append(l_idx)
            
    # Each point should have 50 lines
    # assert all(len(v) == 50 for v in lines_through_point.values())

    # Greedy algorithm
    best_C_size = 0
    best_C_set = set()
    
    import random
    
    # Number of restarts
    num_restarts = 10
    
    for restart in range(num_restarts):
        random.seed(restart)
        C_set = set()
        valid_lines_count = [50] * (q * q)
        C_on_line = [set() for _ in range(len(line_points_list))]
        
        candidates = list(all_points)
        random.shuffle(candidates)
        
        for p in candidates:
            # Check if adding p is valid
            # Calculate deltas
            deltas = {}
            is_valid = True
            
            for l_idx in lines_through_point[p]:
                k_L = len(C_on_line[l_idx])
                
                if k_L == 0:
                    # Line L becomes {p} in C.
                    # For all x in L, x != p: valid_lines_count[x] decreases by 1.
                    for x in line_points_list[l_idx]:
                        if x != p:
                            if x in deltas:
                                deltas[x] -= 1
                            else:
                                deltas[x] = -1
                elif k_L == 1:
                    # Line L has one point x_old in C.
                    # x_old loses this valid line.
                    x_old = next(iter(C_on_line[l_idx]))
                    if x_old in deltas:
                        deltas[x_old] -= 1
                    else:
                        deltas[x_old] = -1
                else:
                    # k_L > 1, no changes to validity counts
                    pass
            
            # Check if any count becomes 0
            for x, delta in deltas.items():
                if valid_lines_count[x] + delta == 0:
                    is_valid = False
                    break
            
            if is_valid:
                # Commit
                C_set.add(p)
                for l_idx in lines_through_point[p]:
                    C_on_line[l_idx].add(p)
                for x, delta in deltas.items():
                    valid_lines_count[x] += delta
                
        if len(C_set) > best_C_size:
            best_C_size = len(C_set)
            best_C_set = set(C_set)
            
    # Construct N
    N_set = set(all_points) - best_C_set
    
    # Convert to array
    points_list = []
    for pt_idx in sorted(N_set):
        x_idx = pt_idx // q
        y_idx = pt_idx % q
        xa, xb = field_elements[x_idx]
        ya, yb = field_elements[y_idx]
        points_list.append([[xa, xb], [ya, yb]])
        
    if not points_list:
        # Fallback
        for pt_idx in sorted(all_points):
            x_idx = pt_idx // q
            y_idx = pt_idx % q
            xa, xb = field_elements[x_idx]
            ya, yb = field_elements[y_idx]
            points_list.append([[xa, xb], [ya, yb]])
            
    return np.array(points_list, dtype=np.int64)
