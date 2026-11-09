import math

def construct():
    # Explicit hexagonal ring construction
    # This is a structural change from angle-sorting spiral to
    # layer-by-layer addition with connectivity prioritization.
    
    points = [(0, 0)]
    
    # Add rings 1, 2, 3 fully
    for k in range(1, 4):
        # Generate points in ring k
        # Corners: (k, 0), (0, k), (-k, k), (-k, 0), (0, -k), (k, -k)
        # Sides: between corners
        ring_points = []
        
        # Side 1: (k, 0) to (0, k)
        # q from k to 0, r from 0 to k
        for i in range(k):
            ring_points.append((k - i, i))
        
        # Side 2: (0, k) to (-k, k)
        # q from 0 to -k, r from k to k
        for i in range(k):
            ring_points.append((-i, k))
            
        # Side 3: (-k, k) to (-k, 0)
        # q from -k to -k, r from k to 0
        for i in range(k):
            ring_points.append((-k, k - i))
            
        # Side 4: (-k, 0) to (0, -k)
        # q from -k to 0, r from 0 to -k
        for i in range(k):
            ring_points.append((-k + i, -i))
            
        # Side 5: (0, -k) to (k, -k)
        # q from 0 to k, r from -k to -k
        for i in range(k):
            ring_points.append((i, -k))
            
        # Side 6: (k, -k) to (k, 0)
        # q from k to k, r from -k to 0
        for i in range(k):
            ring_points.append((k, -k + i))
            
        points.extend(ring_points)
        
        if len(points) >= 32:
            break
            
    # If we need more points from ring 4 (since 1+6+12+18 = 37 > 32)
    # We have 19 points from rings 0-3.
    # We need 13 more from ring 4.
    if len(points) < 32:
        k = 4
        ring_points = []
        # Generate ring 4 points
        for i in range(k):
            ring_points.append((k - i, i))
        for i in range(k):
            ring_points.append((-i, k))
        for i in range(k):
            ring_points.append((-k, k - i))
        for i in range(k):
            ring_points.append((-k + i, -i))
        for i in range(k):
            ring_points.append((i, -k))
        for i in range(k):
            ring_points.append((k, -k + i))
            
        # We need 13 points.
        # To maximize edges, we should pick points with max connectivity to existing points.
        # As analyzed, side points have 2 connections, corners have 1.
        # The ring generation above starts at (k,0) which is a corner.
        # Order: C, S, S, S, C, S, S, S...
        # We want to pick 13 points.
        # If we pick the first 13, we get 4 Corners and 9 Sides?
        # Let's just take the first 13 generated.
        # This mimics the spiral's contiguous block behavior.
        points.extend(ring_points[:13])
        
    # Return exactly 32 points
    result = [[q, r] for q, r in points[:32]]
    return result
