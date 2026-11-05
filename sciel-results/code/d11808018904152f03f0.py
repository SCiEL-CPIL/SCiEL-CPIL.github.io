import time

def construct():
    start_time = time.time()
    deadline = 140
    
    # Generate hexagon of radius 4
    R = 4
    points = []
    for a in range(-R, R + 1):
        for b in range(-R, R + 1):
            if abs(a + b) <= R:
                points.append((a, b))
                
    n_points = len(points) # 61
    
    # Map point to index
    point_to_idx = {p: i for i, p in enumerate(points)}
    
    # Build adjacency
    adj = [set() for _ in range(n_points)]
    for i in range(n_points):
        a1, b1 = points[i]
        for j in range(i + 1, n_points):
            a2, b2 = points[j]
            da = a1 - a2
            db = b1 - b2
            d2 = da*da + db*db + da*db
            if d2 == 1:
                adj[i].add(j)
                adj[j].add(i)
                
    active = [True] * n_points
    count = n_points
    
    # Greedy removal: remove point with lowest degree among active points
    # Tie-breaking: prefer removing points further from center to keep center dense?
    # Or just lowest degree.
    # Actually, to maximize edges, we want to keep high-degree nodes.
    # So we remove the one with the lowest degree.
    
    while count > 32:
        min_deg = float('inf')
        min_idx = -1
        for i in range(n_points):
            if not active[i]:
                continue
            deg = sum(1 for j in adj[i] if active[j])
            # Tie break: remove the one with largest distance from center?
            # This helps maintain a compact shape.
            dist = abs(points[i][0]) + abs(points[i][1]) + abs(points[i][0] + points[i][1])
            # We want to minimize (deg, -dist) ? No, minimize deg.
            # If deg is same, remove the one with LARGER dist (outermost).
            if deg < min_deg or (deg == min_deg and dist > abs(points[min_idx][0]) + abs(points[min_idx][1]) + abs(points[min_idx][0] + points[min_idx][1])):
                min_deg = deg
                min_idx = i
                
        if min_idx == -1:
            break
            
        active[min_idx] = False
        count -= 1
        
        if time.time() > deadline:
            break

    result = []
    for i in range(n_points):
        if active[i]:
            result.append(list(points[i]))
            
    # Verify count
    if len(result) != 32:
        # Fallback if something went wrong
        return [list(p) for p in points[:32]]
        
    # Count edges
    edge_count = 0
    for i in range(32):
        a1, b1 = result[i]
        for j in range(i + 1, 32):
            a2, b2 = result[j]
            da = a1 - a2
            db = b1 - b2
            d2 = da*da + db*db + da*db
            if d2 == 1:
                edge_count += 1
                
    # If we didn't get 76, try local search (swaps)
    if edge_count < 76:
        # Simple local search: try swapping each active point with each inactive point
        # if it increases edge count
        active_set = set(result)
        inactive_points = [points[i] for i in range(n_points) if not active[i]]
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(32):
                p_active = result[i]
                for p_inactive in inactive_points:
                    # Check distance constraint: p_inactive must not be distance < 1 from any other active point (except the one being replaced)
                    # Actually, we replace p_active with p_inactive.
                    # So p_inactive must be dist >= 1 from all result[k] where k != i.
                    valid = True
                    for k in range(32):
                        if k == i: continue
                        a3, b3 = result[k]
                        da = p_inactive[0] - a3
                        db = p_inactive[1] - b3
                        d2 = da*da + db*db + da*db
                        if d2 < 1:
                            valid = False
                            break
                    if not valid: continue
                    
                    # Calculate change in edges
                    # Edges lost: edges from p_active to result[k] (k != i)
                    edges_lost = 0
                    for k in range(32):
                        if k == i: continue
                        a3, b3 = result[k]
                        da = p_active[0] - a3
                        db = p_active[1] - b3
                        d2 = da*da + db*db + da*db
                        if d2 == 1:
                            edges_lost += 1
                            
                    # Edges gained: edges from p_inactive to result[k] (k != i)
                    edges_gained = 0
                    for k in range(32):
                        if k == i: continue
                        a3, b3 = result[k]
                        da = p_inactive[0] - a3
                        db = p_inactive[1] - b3
                        d2 = da*da + db*db + da*db
                        if d2 == 1:
                            edges_gained += 1
                            
                    if edges_gained > edges_lost:
                        # Perform swap
                        result[i] = list(p_inactive)
                        # Update inactive list (swap logic)
                        inactive_points.remove(p_inactive)
                        inactive_points.append(list(p_active))
                        improved = True
                        break
                if improved:
                    break
                    
    return result
