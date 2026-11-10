import time

def construct():
    """
    Constructs a simple undirected graph on 16 vertices with diameter exactly 2.
    The graph is the complete bipartite graph K_8,8, which has 64 edges.
    It is triangle-free, ensuring that removing any edge increases the distance
    between its endpoints to at least 3, thus increasing the diameter to 3.
    This maximizes the edge count under the constraint that every edge is critical
    for maintaining diameter 2.
    """
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit
    
    n = 16
    # Partition vertices into two sets of 8
    # Set A: 0-7, Set B: 8-15
    edges = []
    
    # Generate edges between A and B
    for u in range(8):
        if time.time() > deadline:
            # If we somehow run out of time (unlikely for 64 edges), return what we have
            # But we need a valid graph. So we should finish quickly.
            # This is just a guard.
            break
        for v in range(8, 16):
            edges.append([u, v])
            
    # Ensure types are correct: list of lists of ints
    # The loop above produces exactly that.
    
    return edges
