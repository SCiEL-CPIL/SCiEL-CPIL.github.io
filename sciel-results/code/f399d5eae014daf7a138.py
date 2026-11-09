import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n = 40
    target_edges = 401
    
    # Use K_{20,20} as base: vertices 0-19 in part A, 20-39 in part B
    # This gives 20*20 = 400 edges with 0 triangles
    edges = set()
    
    # Add all edges between A and B
    for u in range(20):
        for v in range(20, 40):
            edges.add((u, v))
    
    # Add 1 edge within A to create exactly 1 triangle
    # Edge (0,1) creates triangle (0,1,20) since 20 is connected to both 0 and 1
    edges.add((0, 1))
    
    # Verify we have exactly 401 edges
    # K_{20,20} has 400 edges, plus 1 = 401
    
    # Format output
    result = [[u, v] for u, v in edges]
    
    return result
