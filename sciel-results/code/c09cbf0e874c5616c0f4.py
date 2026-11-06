import time
import math

def construct():
    # Hardcoded core: hexagon of radius 2 (19 points)
    core = [
        [0, 0],
        [1, 0], [0, 1], [-1, 1], [-1, 0], [0, -1], [1, -1],
        [2, 0], [1, 1], [0, 2], [-1, 2], [-2, 1], [-2, 0], [-1, -1], [0, -2], [1, -2], [2, -1]
    ]
    
    # Generate ring 3 points
    def hex_dist(a, b):
        return (abs(a) + abs(b) + abs(a+b)) // 2
        
    def dist_sq(p1, p2):
        da = p1[0] - p2[0]
        db = p1[1] - p2[1]
        return da*da + da*db + db*db
        
    def count_edges(points):
        count = 0
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                if dist_sq(points[i], points[j]) == 1:
                    count += 1
        return count

    ring3 = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if hex_dist(a, b) == 3:
                ring3.append([a, b])
                
    # Sort by angle
    def angle(p):
        a, b = p
        x = a + b/2.0
        y = b * math.sqrt(3) / 2.0
        return math.atan2(y, x)
        
    ring3.sort(key=angle)
    
    # Try all 18 contiguous blocks of 13
    best_score = -1
    best_block = None
    
    for i in range(18):
        subset = [ring3[(i + k) % 18] for k in range(13)]
        full_set = core + subset
        
        if len(set(map(tuple, full_set))) != 32:
            continue
            
        if any(abs(p[0]) > 64 or abs(p[1]) > 64 for p in full_set):
            continue
            
        edges = count_edges(full_set)
        
        if edges > best_score:
            best_score = edges
            best_block = [list(p) for p in full_set]
            
    return best_block
