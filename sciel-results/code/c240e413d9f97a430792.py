import math

def construct():
    n = 32
    
    # 1. Generate points in rings 0, 1, 2
    # These are 1 + 6 + 12 = 19 points.
    points = []
    
    # Helper to generate points at hex distance d
    def get_ring(d):
        if d == 0:
            return [(0, 0)]
        
        pts = []
        # The 6 corners of the hexagon at distance d
        # Directions: (d, 0), (0, d), (-d, d), (-d, 0), (0, -d), (d, -d)
        # Actually, let's just generate all points with hex_dist == d
        
        def hex_dist(q, r):
            return abs(q) + abs(r) + abs(q + r)
        
        # Range of q, r
        # Max q, r is d
        for q in range(-d, d + 1):
            for r in range(-d, d + 1):
                if hex_dist(q, r) == 2 * d: # Wait, hex_dist returns sum of abs, so distance is /2?
                    # My previous hex_dist definition: abs(q)+abs(r)+abs(q+r)
                    # For (1,0): 1+0+1=2. Dist 1.
                    # So we want hex_dist == 2*d.
                    pts.append((q, r))
        return pts

    # Add rings 0, 1, 2
    for d in range(3):
        points.extend(get_ring(d))
        
    # We have 19 points. We need 32. So we need 13 more from ring 3.
    ring3 = get_ring(3)
    
    # We want to select 13 points from ring3 that maximize edges within ring3.
    # The points in ring3 form a cycle.
    # We should select a contiguous block of 13.
    
    # To find a contiguous block, we need to order ring3 points cyclically.
    # We can sort them by angle or by following the cycle.
    
    # Let's create a set for fast lookup
    ring3_set = set(ring3)
    
    # Define neighbors in the lattice
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
    
    # Find a starting point in ring3
    start = ring3[0]
    
    # Traverse the cycle
    cycle = [start]
    current = start
    visited = {start}
    
    for _ in range(len(ring3) - 1):
        # Find the next neighbor in ring3
        next_pt = None
        for dq, dr in directions:
            nxt = (current[0] + dq, current[1] + dr)
            if nxt in ring3_set and nxt not in visited:
                next_pt = nxt
                break
        if next_pt is None:
            # Should not happen if it's a simple cycle
            break
        current = next_pt
        cycle.append(current)
        visited.add(current)
        
    # Now 'cycle' contains all 18 points in ring3 in order.
    # We want to pick 13 contiguous points.
    # Any block of 13 will have 12 internal edges.
    # We just pick the first 13.
    selected_ring3 = cycle[:13]
    
    points.extend(selected_ring3)
    
    # Verify count
    assert len(points) == 32, f"Expected 32 points, got {len(points)}"
    
    # Verify distinctness
    assert len(set(points)) == 32
    
    # Return as list of lists
    result = [list(p) for p in points]
    return result
