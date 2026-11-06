def construct():
    # Generate all points in Hex(2)
    points = set()
    for a in range(-2, 3):
        for b in range(-2, 3):
            if a*a + a*b + b*b <= 4:
                points.add((a, b))
    
    # Generate Ring 3 points
    ring3 = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if a*a + a*b + b*b == 9:
                ring3.append((a, b))
                
    # Sort ring3 to form a cycle
    # Angle sort
    import math
    ring3.sort(key=lambda p: math.atan2(p[1], p[0]))
    
    best_points = None
    best_edges = -1
    
    # Try all contiguous arcs of length 13 in Ring 3
    n = len(ring3)
    for i in range(n):
        arc = set()
        for j in range(13):
            arc.add(ring3[(i+j) % n])
        
        candidate = points.union(arc)
        
        # Count edges
        edges = 0
        cand_list = list(candidate)
        for k in range(len(cand_list)):
            a1, b1 = cand_list[k]
            for l in range(k+1, len(cand_list)):
                a2, b2 = cand_list[l]
                da = a1 - a2
                db = b1 - b2
                d2 = da*da + da*db + db*db
                if d2 == 1:
                    edges += 1
                    if edges > best_edges:
                        best_edges = edges
                        best_points = candidate
        # Early exit if we hit 76
        if best_edges == 76:
            break
            
    # Convert to list of lists
    return [[a, b] for a, b in best_points]
