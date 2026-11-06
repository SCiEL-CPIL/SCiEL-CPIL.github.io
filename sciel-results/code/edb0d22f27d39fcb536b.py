import math
import time

def construct():
    start_time = time.time()
    
    def hex_dist(a, b):
        return max(abs(a), abs(b), abs(a + b))
    
    # Generate all points with hex_dist <= 3 (37 points total)
    all_pts = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if hex_dist(a, b) <= 3:
                all_pts.append((a, b))
    
    # Sort by (hex_dist, angle) for spiral ordering
    def sort_key(p):
        a, b = p
        d = hex_dist(a, b)
        x = a + b * 0.5
        y = b * (math.sqrt(3) / 2)
        angle = math.atan2(y, x)
        return (d, angle)
    
    all_pts.sort(key=sort_key)
    
    # Take first 32 points
    result = all_pts[:32]
    
    return [[a, b] for a, b in result]
