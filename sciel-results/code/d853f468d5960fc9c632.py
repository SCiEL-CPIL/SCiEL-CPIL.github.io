import math

def construct():
    # Generate candidate points within a reasonable bounding box
    # The optimal shape for n=32 is contained within hex distance 3
    candidates = []
    for q in range(-10, 11):
        for r in range(-10, 11):
            # Calculate hex distance (cube distance)
            # x = q, z = r, y = -q-r
            # dist = max(|x|, |y|, |z|) = max(|q|, |r|, |q+r|)
            dist = max(abs(q), abs(r), abs(q + r))
            
            # Calculate angle for sorting to create a spiral
            # Convert axial to cartesian for angle calculation
            x = q + 0.5 * r
            y = math.sqrt(3) / 2 * r
            angle = math.atan2(y, x)
            
            candidates.append((dist, angle, q, r))
    
    # Sort by distance, then by angle
    # This creates a hexagonal spiral order
    candidates.sort(key=lambda x: (x[0], x[1]))
    
    # Take the first 32 points
    # These form the most compact cluster, maximizing nearest-neighbor contacts
    selected = [c[2:] for c in candidates[:32]]
    
    # Format as list of lists of integers
    result = [[q, r] for q, r in selected]
    
    return result
