def _generate_spiral_sorted(n):
    # Generate points in a large box
    R = 10
    pts = []
    for a in range(-R, R+1):
        for b in range(-R, R+1):
            hd = (abs(a) + abs(b) + abs(a+b)) // 2
            # Calculate angle for sorting
            # x = a + 0.5*b
            # y = 0.866*b
            x = a + 0.5 * b
            y = 0.8660254037844386 * b
            ang = math.atan2(y, x)
            pts.append((hd, ang, a, b))
    
    # Sort by hex distance, then by angle
    pts.sort(key=lambda x: (x[0], x[1]))
    
    # Extract first n points
    return [(a, b) for hd, ang, a, b in pts[:n]]
