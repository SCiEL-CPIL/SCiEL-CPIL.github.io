<<<<<<< SEARCH
def _two_row_construction():
    """
    Construct points on two rows using a Sidon-like approach.
    Place points on rows y=0 and y=h such that no three form isosceles triangle.
    """
    h = N // 2
    # Greedily add points to row 0
    row0 = []
    used_diffs_0 = set()
    for x in range(N):
        diffs = {abs(x - y) for y in row0}
        if not (diffs & used_diffs_0):
            used_diffs_0 |= diffs
            row0.append(x)
    
    # Greedily add points to row h
    row1 = []
    all_points = [(x, 0) for x in row0]
    used_diffs_1 = set()
    for x in range(N):
        # Check distances to all existing points
        valid = True
        dists_from_new = set()
        for px, py in all_points:
            d2 = (x - px)**2 + (h - py)**2
            if d2 in dists_from_new:
                valid = False
                break
            dists_from_new.add(d2)
        # Check if any existing point is apex
        if valid:
            for i, (px, py) in enumerate(all_points):
                d2 = (x - px)**2 + (h - py)**2
                # Check if this distance already exists from point i
                for j, (qx, qy) in enumerate(all_points):
                    if i == j:
                        continue
                    d2_ij = (px - qx)**2 + (py - qy)**2
                    if d2 == d2_ij:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            row1.append(x)
            all_points.append((x, h))
    
    return all_points
=======
def _two_row_construction():
    """
    Construct points on two rows using a Sidon-like approach with bitset tracking.
    """
    h = N // 2
    all_positions = [(x, y) for x in range(N) for y in range(N)]
    pos_to_idx = {p: i for i, p in enumerate(all_positions)}
    
    pos_arr = np.array(all_positions, dtype=np.int32)
    dx_all = pos_arr[:, 0:1] - pos_arr[0:1, 0]
    dy_all = pos_arr[:, 1:2] - pos_arr[0:1, 1]
    dist_matrix = (dx_all * dx_all + dy_all * dy_all).astype(np.int32)
    
    current_points = []
    current_indices = []
    current_dist_sets = []
    k = 0
    
    # Try all positions in row 0 first, then row h
    candidates = [(x, 0) for x in range(N)] + [(x, h) for x in range(N)]
    
    for pos in candidates:
        idx = pos_to_idx[pos]
        if k == 0:
            valid = True
            dists_list = []
        else:
            dists_list = dist_matrix[idx, current_indices].tolist()
            pos_bitset = 0
            for d2 in dists_list:
                pos_bitset |= (1 << d2)
            if bin(pos_bitset).count('1') != k:
                continue
            valid = True
            for i, d2 in enumerate(dists_list):
                if current_dist_sets[i] & (1 << d2):
                    valid = False
                    break
            if not valid:
                continue
        
        if valid:
            current_points.append(pos)
            current_indices.append(idx)
            new_bitset = 0
            for d2 in dists_list:
                new_bitset |= (1 << d2)
            for i, d2 in enumerate(dists_list):
                current_dist_sets[i] |= (1 << d2)
            current_dist_sets.append(new_bitset)
            k += 1
    
    return current_points
>>>>>>> REPLACE
