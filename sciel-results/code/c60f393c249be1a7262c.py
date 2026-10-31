def point_in_hex(px, py, cx, cy, S, angle_deg):
    """Check if point (px,py) is inside hexagon with center (cx,cy), side S, rotation angle_deg."""
    a = np.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    inr = S * np.sqrt(3) / 2
    for k in range(6):
        theta = a + k * np.pi / 3
        nx = np.cos(theta)
        ny = np.sin(theta)
        if dx * nx + dy * ny > inr + 1e-9:
            return False
    return True

def hex_overlap_dist(c1, a1, c2, a2):
    """
    Returns negative value if overlapping, positive if separated.
    """
    x1, y1 = c1
    x2, y2 = c2
    
    # Bounding box check
    if abs(x1 - x2) > 2.0 or abs(y1 - y2) > 2.0:
        return 1.0 

    v1 = hex_vertices(x1, y1, 1.0, a1)
    v2 = hex_vertices(x2, y2, 1.0, a2)
    
    # Check if any vertex of v1 is inside hex2
    for vx, vy in v1:
        if point_in_hex(vx, vy, x2, y2, 1.0, a2):
            return -1.0 # Overlapping
            
    # Check if any vertex of v2 is inside hex1
    for vx, vy in v2:
        if point_in_hex(vx, vy, x1, y1, 1.0, a1):
            return -1.0 # Overlapping
            
    # If no vertices are inside, they might still overlap if edges cross?
    # For convex polygons, if no vertices of A are in B and no vertices of B are in A,
    # they are disjoint.
    
    min_dist = float('inf')
    
    # Distance from v1 to edges of v2
    for i in range(6):
        for j in range(6):
            p1 = v2[j]
            p2 = v2[(j+1) % 6]
            d = p2 - p1
            f = v1[i] - p1
            denom = np.dot(d, d)
            if denom < 1e-15:
                continue
            t = np.dot(f, d) / denom
            t = max(0, min(1, t))
            closest = p1 + t * d
            dist = np.linalg.norm(v1[i] - closest)
            if dist < min_dist:
                min_dist = dist
                
    # Distance from v2 to edges of v1
    for i in range(6):
        for j in range(6):
            p1 = v1[j]
            p2 = v1[(j+1) % 6]
            d = p2 - p1
            f = v2[i] - p1
            denom = np.dot(d, d)
            if denom < 1e-15:
                continue
            t = np.dot(f, d) / denom
            t = max(0, min(1, t))
            closest = p1 + t * d
            dist = np.linalg.norm(v2[i] - closest)
            if dist < min_dist:
                min_dist = dist

    return min_dist
