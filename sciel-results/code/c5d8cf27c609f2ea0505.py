import random
import math

def construct():
    # Local Search approach
    # 1. Generate initial hexagonal spiral
    # 2. Optimize via random moves/swaps
    
    # Directions in axial coordinates
    dirs = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    
    def hex_dist(a, b):
        return (abs(a) + abs(b) + abs(a + b)) // 2

    def get_neighbors(a, b):
        return [(a + da, b + db) for da, db in dirs]

    def count_edges(points_set):
        count = 0
        for p in points_set:
            for n in get_neighbors(p[0], p[1]):
                if n in points_set:
                    count += 1
        return count // 2

    # Generate initial spiral
    # Spiral generation:
    # (0,0)
    # Then walk around shells.
    # Standard algorithm:
    # pos = (0,0)
    # points = [(0,0)]
    # for step in range(1, max_steps):
    #   for i in range(6):
    #     for j in range(step):
    #       pos = pos + dirs[i]
    #       points.append(pos)
    #       if len(points) == 32: return ...
    # 
    # Let's verify this standard spiral.
    # step=1:
    # i=0, j=0: pos=(1,0). points=[(0,0),(1,0)]
    # i=1, j=0: pos=(1,1). points=[...,(1,1)]
    # i=2, j=0: pos=(0,2). points=[...,(0,2)]
    # i=3, j=0: pos=(-1,2). points=[...,-1,2]
    # i=4, j=0: pos=(-1,1). points=[...,-1,1]
    # i=5, j=0: pos=(0,0). Wait, this returns to center?
    # Let's re-evaluate the standard hex spiral algorithm.
    # Usually:
    # x, y = 0, 0
    # for k in range(1, ...):
    #   for i in range(6):
    #     for j in range(k):
    #       x += dirs[i][0]
    #       y += dirs[i][1]
    #       yield (x, y)
    #
    # Trace:
    # k=1:
    # i=0, j=0: x=1, y=0. (1,0)
    # i=1, j=0: x=1, y=1. (1,1)
    # i=2, j=0: x=0, y=2. (0,2)
    # i=3, j=0: x=-1, y=2. (-1,2)
    # i=4, j=0: x=-1, y=1. (-1,1)
    # i=5, j=0: x=0, y=0. (0,0) -- This is a collision.
    #
    # The standard spiral algorithm yields the *ring* points, not including the center again.
    # The sequence of points yielded is the spiral.
    # Let's check distances for (1,1). dist^2 = 1+1+1=3. Not a neighbor of (0,0).
    # The neighbors of (0,0) are (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1).
    # The sequence above yields (1,0), (1,1), (0,2)...
    # This spiral is NOT the one that keeps the cluster connected by edges in the way we want?
    # Actually, the "spiral" that maximizes edges is the one that fills the hexagon.
    # The points in the hexagon of radius R are all points with hex_dist <= R.
    # The number of points is 1 + 3R(R+1).
    # R=0: 1
    # R=1: 7
    # R=2: 19
    # R=3: 37
    # We need 32. So we take all points with dist <= 2 (19 points) and 13 points from dist=3.
    # To maximize edges, the 13 points from dist=3 should be chosen to have max neighbors in the dist<=2 set.
    # This is exactly what the greedy algorithm in the current code tries to do.
    #
    # So, a "Local Search" starting from the greedy result or the full hexagon-minus-corners might be robust.
    #
    # Let's implement a Local Search that starts with a valid configuration (e.g., first 32 points of the "distance sorted" list, which is the hexagon with some corners missing) and optimizes.
    
    # Generate all points within dist 3
    all_points = []
    for a in range(-4, 5):
        for b in range(-4, 5):
            d = hex_dist(a, b)
            if d <= 3:
                all_points.append((a, b))
                
    # Sort by distance, then by angle (or just distance)
    # To get a good initial state, take the 32 points with smallest distance.
    # Ties at distance 3: there are 18 points. We need 13.
    # Any 13 will do for start, local search will fix it.
    all_points.sort(key=lambda p: hex_dist(p[0], p[1]))
    
    # We need exactly 32.
    # The first 19 have dist <= 2.
    # The next 13 have dist 3.
    initial_set = set(all_points[:32])
    
    current_edges = count_edges(initial_set)
    best_set = set(initial_set)
    best_edges = current_edges
    
    # Local Search
    random.seed(42)
    for _ in range(10000):
        # Pick a random point to move
        p_list = list(initial_set)
        p = random.choice(p_list)
        
        # Pick a random target position
        # Target should be close to current set to maintain compactness
        # Generate random candidate in a box around origin
        ta = random.randint(-4, 4)
        tb = random.randint(-4, 4)
        t = (ta, tb)
        
        if t in initial_set:
            continue
            
        # Check if moving p to t is valid (no collision)
        # t is not in set, so no collision with others.
        # p is being removed.
        
        new_set = set(initial_set)
        new_set.remove(p)
        new_set.add(t)
        
        new_edges = count_edges(new_set)
        
        if new_edges > current_edges:
            initial_set = new_set
            current_edges = new_edges
            if current_edges > best_edges:
                best_edges = current_edges
                best_set = set(initial_set)
        elif new_edges == current_edges:
            # Accept with some probability to escape local minima
            if random.random() < 0.5:
                initial_set = new_set
                current_edges = new_edges
        else:
            # Reject (or accept with very low prob, but let's keep it simple hill climbing)
            pass
            
    return [[a, b] for a, b in best_set]
