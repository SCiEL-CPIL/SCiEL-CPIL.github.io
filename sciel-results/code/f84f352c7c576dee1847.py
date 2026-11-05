def construct():
    # 19-point hexagon of radius 2 centered at (0,0)
    hex_pts = []
    for a in range(-2, 3):
        for b in range(-2, 3):
            if abs(a) <= 2 and abs(b) <= 2 and abs(a - b) <= 2:
                hex_pts.append([a, b])
    
    # Add 13 more points to extend the shape
    # These points are chosen to maintain high connectivity
    extra_pts = [
        [3, 0], [3, 1], [3, 2], [3, 3],
        [4, 1], [4, 2], [4, 3], [4, 4],
        [5, 2], [5, 3], [5, 4], [5, 5],
        [6, 3]
    ]
    
    all_pts = hex_pts + extra_pts
    
    # Ensure distinctness and bounds (all are within |a|,|b| <= 64)
    # Check for duplicates (should be none)
    unique_pts = []
    seen = set()
    for p in all_pts:
        t = tuple(p)
        if t not in seen:
            seen.add(t)
            unique_pts.append(p)
            
    # We expect exactly 32 points
    # If not, trim or pad (but logic above should yield 32)
    return unique_pts[:32]
