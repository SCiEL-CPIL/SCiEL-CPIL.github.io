def construct():
    # Generate points in a hexagonal spiral (triangular lattice spiral)
    # This pattern is known to maximize the number of edges for a given number of points n.
    
    points = [(0, 0)]
    
    # Directions in axial coordinates (a, b)
    # 0: (1, 0)
    # 1: (0, 1)
    # 2: (-1, 1)
    # 3: (-1, 0)
    # 4: (0, -1)
    # 5: (1, -1)
    dirs = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    
    current_a, current_b = 0, 0
    # The spiral expands in rings.
    # Ring 0: 1 point (0,0)
    # Ring 1: 6 points
    # Ring 2: 12 points
    # Ring k: 6*k points
    
    # We need 32 points.
    # Ring 0: 1 point (total 1)
    # Ring 1: 6 points (total 7)
    # Ring 2: 12 points (total 19)
    # Ring 3: 18 points (total 37) -> We stop in the middle of Ring 3.
    
    # Let's just walk the spiral until we have 32 points.
    
    # Start at (0,0)
    # Move 1 step in dir 0? No, standard spiral generation:
    # The distance from center increases.
    
    # A simpler way:
    # The points in order of distance from origin, breaking ties by angle.
    # But spiral is better for connectivity.
    
    # Let's generate the spiral sequence properly.
    # Step 1: (1, 0)
    # Step 2: (0, 1)
    # Step 3: (-1, 1)
    # Step 4: (-1, 0)
    # Step 5: (0, -1)
    # Step 6: (1, -1)
    # Then we are back at (0,0)? No, (1,-1) + (1,0) is not (0,0).
    # (1,-1) is distance 1.
    # The 6 neighbors of (0,0) are at distance 1.
    
    # Let's use the property that the spiral visits points in increasing "hex distance".
    # Hex distance for (a,b) is (|a| + |b| + |a+b|) / 2 ? No, that's for cube coords.
    # In axial, dist = (|a| + |b| + |a+b|) / 2 is incorrect.
    # Dist = max(|a|, |b|, |a+b|) ? No.
    # Let's just use the walking algorithm.
    
    # Walking algorithm for hex spiral:
    # 1. Start at (0,0)
    # 2. For step_length = 1, 2, 3, ...:
    #    For dir_index in 0, 1, 2, 3, 4, 5:
    #       Move step_length steps in direction dir_index
    #       (But for the first "ring", we only move 1 step in each direction to form the hexagon of radius 1?
    #       Actually, the spiral goes:
    #       (0,0)
    #       (1,0)
    #       (1,1) -- wait, (1,0)+(0,1) = (1,1). Dist of (1,1) is 1?
    #       a=1, b=1. a^2+ab+b^2 = 1+1+1=3. sqrt(3) is not 1.
    #       My neighbor definition:
    #       (1,0) dist 1.
    #       (0,1) dist 1.
    #       (1,1) dist sqrt(3) approx 1.732.
    #       
    #       The neighbors of (0,0) are:
    #       (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1).
    #       These are all distance 1.
    #       
    #       So the first 6 points after (0,0) are these 6 neighbors.
    #       
    #       Let's trace:
    #       Start (0,0).
    #       Dir 0: (1,0).
    #       Dir 1: (1,0)+(0,1) = (1,1). Is (1,1) a neighbor of (0,0)? No.
    #       So the simple "walk 1 step in each direction" doesn't visit the ring of neighbors directly in a single loop if we don't reset or if the geometry is different.
        
    #       Let's look at the neighbors again:
    #       (1,0)
    #       (0,1)
    #       (-1,1)
    #       (-1,0)
    #       (0,-1)
    #       (1,-1)
        
    #       If I start at (0,0) and go to (1,0).
    #       Then I want to go to (0,1). Vector is (-1, 1). That's Dir 2.
    #       So from (1,0), adding Dir 2 (-1,1) gives (0,1).
    #       From (0,1), adding Dir 3 (-1,0) gives (-1,1).
    #       From (-1,1), adding Dir 4 (0,-1) gives (-1,0).
    #       From (-1,0), adding Dir 5 (1,-1) gives (0,-1).
    #       From (0,-1), adding Dir 0 (1,0) gives (1,-1).
    #       From (1,-1), adding Dir 1 (0,1) gives (1,0). Back to start of ring.
        
    #       So the sequence of directions to trace the ring of radius 1 is:
    #       Start (0,0) -> Dir 0 -> (1,0)
    #       (1,0) -> Dir 2 -> (0,1)
    #       (0,1) -> Dir 3 -> (-1,1)
    #       (-1,1) -> Dir 4 -> (-1,0)
    #       (-1,0) -> Dir 5 -> (0,-1)
    #       (0,-1) -> Dir 0 -> (1,-1)  <-- Wait, (0,-1)+(1,0) = (1,-1).
    #       (1,-1) -> Dir 1 -> (1,0).
        
    #       This visits 6 points.
    #       The directions used were 0, 2, 3, 4, 5, 0, 1? No.
    #       Let's re-verify the step from (0,-1) to (1,-1).
    #       (0,-1) + (1,0) = (1,-1). Correct.
    #       So the loop is:
    #       (0,0)
    #       + (1,0) -> (1,0)
    #       + (-1,1) -> (0,1)
    #       + (-1,0) -> (-1,1)
    #       + (0,-1) -> (-1,0)
    #       + (1,-1) -> (0,-1)
    #       + (1,0) -> (1,-1)
    #       + (0,1) -> (1,0) (back to start)
        
    #       This works for Ring 1.
    #       For Ring 2, we need to step *twice* in each direction?
    #       Let's try.
    #       Current pos (1,0).
    #       We want to go to the next point in the spiral.
    #       The standard hex spiral algorithm:
    #       for k in range(1, max_ring+1):
    #           for i in range(6):
    #               for j in range(k):
    #                   move in direction i
    #                   add point
    #           
    #       Let's test this for k=1.
    #       i=0, j=0: move (1,0) from (0,0) -> (1,0). Add.
    #       i=1, j=0: move (0,1) from (1,0) -> (1,1). Add.
    #       i=2, j=0: move (-1,1) from (1,1) -> (0,2). Add.
    #       i=3, j=0: move (-1,0) from (0,2) -> (-1,2). Add.
    #       i=4, j=0: move (0,-1) from (-1,2) -> (-1,1). Add.
    #       i=5, j=0: move (1,-1) from (-1,1) -> (0,0). Add? No, (0,0) is start.
    #       
    #       This generates points at distance > 1.
    #       (1,1) dist sqrt(3).
    #       (0,2) dist 2.
    #       This is NOT the ring of nearest neighbors.
    #       
    #       The "spiral" that maximizes edges is the one that fills the hexagon by shells of increasing hex-distance.
    #       Shell 0: (0,0)
    #       Shell 1: 6 points at dist 1.
    #       Shell 2: 12 points at dist 2.
    #       
    #       How to generate Shell 1?
    #       We identified the path: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1).
    #       
    #       How to generate Shell 2?
    #       The points at dist 2 are:
    #       (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,1), (1,2)?
    #       Let's list dist 2 points:
    #       a^2+ab+b^2 = 4.
    #       (2,0), (0,2), (-2,2), (-2,0), (0,-2), (2,-2) -> 6 points.
    #       (1,1), (-1,2), (-2,1), (-1,-1), (1,-2), (2,-1) -> 6 points.
    #       Total 12.
    #       
    #       The spiral should visit them in an order that keeps the cluster connected.
    #       A standard way to generate the "hexagonal spiral" that visits all points in order of distance is:
        
    points = [(0,0)]
    # We will generate points in order of their "hex distance" (Manhattan-like on this lattice)
    # and for same distance, by angle.
    # But easier: just use the known optimal shape for n=32.
    # For n=32, the shape is a hexagon of radius 2 (which has 19 points) plus 13 points from the next shell (radius 3).
    # To maximize edges, we should add the points from radius 3 that have the most neighbors in the existing set.
    # This is essentially the greedy algorithm but starting from the center and expanding outwards.
    
    # Let's implement the "expand from center" greedy.
    # 1. Start with (0,0).
    # 2. Candidate pool: all points within radius 6 (safe bound).
    # 3. At each step, pick the candidate point that has the maximum number of neighbors in the selected set.
    #    Tie-breaker: prefer points with smaller distance to origin (to keep it compact).
    
    selected = {(0,0)}
    # Pre-compute distances to origin for tie-breaking
    def hex_dist(p):
        return (abs(p[0]) + abs(p[1]) + abs(p[0]+p[1])) // 2
        
    # This distance formula is for cube coordinates?
    # For axial (a,b), the distance to origin is (|a| + |b| + |a+b|) / 2 is WRONG.
    # Correct axial distance: max(|a|, |b|, |a+b|) is also wrong.
    # The Euclidean distance is sqrt(a^2+ab+b^2).
    # The "hex distance" (number of steps) is (|a| + |b| + |a+b|) / 2 IF we use cube coords (x,y,z) with x+y+z=0.
    # Axial to Cube: x=a, z=b, y=-a-b.
    # dist = (|x| + |y| + |z|) / 2 = (|a| + |-a-b| + |b|) / 2 = (|a| + |a+b| + |b|) / 2.
    # Yes, that is correct.
    
    def calc_dist(a, b):
        return (abs(a) + abs(b) + abs(a+b)) // 2

    # Generate candidates up to distance 6
    candidates = []
    for a in range(-6, 7):
        for b in range(-6, 7):
            if a == 0 and b == 0:
                continue
            d = calc_dist(a, b)
            if d <= 6:
                candidates.append((a, b))
                
    # We need 31 more points.
    for _ in range(31):
        best_point = None
        max_conn = -1
        min_dist = float('inf')
        
        for p in candidates:
            if p in selected:
                continue
            # Count neighbors in selected
            conns = 0
            for da, db in [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]:
                if (p[0]+da, p[1]+db) in selected:
                    conns += 1
            
            # We want max conns, then min dist to origin
            if conns > max_conn or (conns == max_conn and calc_dist(p[0], p[1]) < min_dist):
                max_conn = conns
                min_dist = calc_dist(p[0], p[1])
                best_point = p
                
        if best_point:
            selected.add(best_point)
            
    return [[a, b] for a, b in selected]
