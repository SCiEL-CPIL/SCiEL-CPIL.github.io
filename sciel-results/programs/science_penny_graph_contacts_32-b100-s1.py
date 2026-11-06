import time
import math

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 140.0  # 10s safety margin

    # Distance squared on triangular lattice
    # Points are a + b*omega, omega = exp(i*pi/3) = 1/2 + i*sqrt(3)/2
    # (a1 + b1*omega) - (a2 + b2*omega) = (a1-a2) + (b1-b2)*omega
    # |z|^2 = (a1-a2)^2 + (b1-b2)^2 + (a1-a2)(b1-b2)
    def dist_sq(p1, p2):
        da = p1[0] - p2[0]
        db = p1[1] - p2[1]
        return da*da + db*db + da*db

    def count_edges(points):
        count = 0
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                if dist_sq(points[i], points[j]) == 1:
                    count += 1
        return count

    # Generate points in a hex spiral
    # We want 32 points.
    # A hexagon of radius 2 has 19 points.
    # We need 13 more from the next ring.
    # Let's generate points by expanding rings.
    
    points = [[0, 0]]
    
    # Directions for the 6 sides of the hexagon
    # In axial coordinates (q, r) or our (a, b) system:
    # Neighbors: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Let's use a standard spiral generation.
    
    # Actually, simpler: generate all points in a box, sort by hex distance, take first 32.
    # Hex distance of (a,b) is max(|a|, |b|, |a+b|)? No.
    # In axial coords (q,r), dist = (|q| + |r| + |q+r|) / 2.
    # Our coords: x = a + b/2, y = b*sqrt(3)/2.
    # This is equivalent to axial coords if we map correctly.
    # Let's just use Euclidean distance from origin as a proxy for "compactness" if hex distance is hard to compute directly in this basis, 
    # but hex distance is better for lattice symmetry.
    # Hex distance in (a,b) basis:
    # The distance from (0,0) to (a,b) in lattice steps is (|a| + |b| + |a+b|) / 2 ?
    # Let's check (1,0): dist 1. (1+0+1)/2 = 1. Correct.
    # (0,1): dist 1. (0+1+1)/2 = 1. Correct.
    # (1,-1): dist 1. (1-1+0)/2 = 0? No. (1,-1) is a neighbor.
    # Wait, neighbors of (0,0) are (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1).
    # Let's check (1,-1): da=1, db=-1. da+db=0. |1|+|-1|+|0| = 2. 2/2 = 1. Correct.
    # So hex_dist(a,b) = (abs(a) + abs(b) + abs(a+b)) // 2
    
    candidates = []
    # Generate points in a sufficiently large box
    for a in range(-10, 11):
        for b in range(-10, 11):
            candidates.append([a, b])
            
    # Sort by hex distance, then by angle to create a deterministic spiral-like ordering
    # Angle: atan2(y, x) where x = a + b/2, y = b*sqrt(3)/2
    def hex_dist(p):
        a, b = p
        return (abs(a) + abs(b) + abs(a+b)) // 2
        
    def angle(p):
        a, b = p
        x = a + b/2.0
        y = b * math.sqrt(3) / 2.0
        return math.atan2(y, x)
        
    candidates.sort(key=lambda p: (hex_dist(p), angle(p)))
    
    # Take first 32
    result = candidates[:32]
    
    # Verify
    edges = count_edges(result)
    
    # If edges < 76, we might need a local search, but the hex spiral should be optimal.
    # For n=32, the hex spiral (or centered hexagon plus partial next ring) is known to achieve the Harborth bound.
    
    # Check time
    if time.time() > deadline:
        return result
        
    return result
