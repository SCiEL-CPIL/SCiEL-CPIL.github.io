import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 60.0 # seconds
    
    n = 16
    
    # Theoretical maximum for triangle-free graph on 16 vertices is floor(16^2/4) = 64.
    # K(8,8) is triangle-free, has diameter 2, and every edge is essential 
    # (removing any edge (u,v) makes dist(u,v)=3, increasing diameter to 3).
    
    # Partition vertices into two sets of 8
    # Set A: 0..7
    # Set B: 8..15
    
    edges = []
    
    # Generate edges for K(8,8)
    for u in range(8):
        for v in range(8, 16):
            # Ensure types are strictly Python int
            edges.append([int(u), int(v)])
            
    # Check deadline (though this is instantaneous)
    if time.time() > deadline:
        # In case of timeout (unlikely), return best found
        pass

    return edges
