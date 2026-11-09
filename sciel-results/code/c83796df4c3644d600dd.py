import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin

    # Generate hexagon of radius 3 (37 vertices)
    # Axial coordinates (a, b)
    # Distance from origin: max(abs(a), abs(b), abs(-a-b))
    # Or simply iterate and check if in hexagon
    points = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            # Check if (a,b) is in hexagon of radius 3
            # Condition: max(abs(a), abs(b), abs(a+b)) <= 3 ?
            # No, hexagon in axial coords:
            # -r <= a <= r
            # -r <= b <= r
            # -r <= a+b <= r
            if -3 <= a <= 3 and -3 <= b <= 3 and -3 <= a+b <= 3:
                points.append((a, b))
    
    # points should have 37 elements
    # Identify boundary points (degree < 6 in the set)
    point_set = set(points)
    boundary_points = []
    for p in points:
        a, b = p
        neighbors = [(a+1, b), (a-1, b), (a, b+1), (a, b-1), (a+1, b-1), (a-1, b+1)]
        deg = 0
        for n in neighbors:
            if n in point_set:
                deg += 1
        if deg < 6:
            boundary_points.append(p)
            
    # We need to remove 5 points to get 32.
    # Try removing 5 from boundary_points.
    # To maximize edges, we should remove points with lowest degree.
    # All boundary points in r=3 hexagon have degree 4.
    # So any 5 should be similar.
    # Let's just pick the first 5.
    
    removed = boundary_points[:5]
    removed_set = set(removed)
    
    final_points = [p for p in points if p not in removed_set]
    
    # Convert to list of lists
    result = [[a, b] for a, b in final_points]
    
    return result
