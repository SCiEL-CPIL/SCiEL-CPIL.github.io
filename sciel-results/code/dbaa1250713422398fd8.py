import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Construction: 8 disjoint copies of K4 + 1 universal vertex
    # n = 33
    # Vertices 0..31 form 8 K4s.
    # Vertex 32 is connected to all 0..31.
    
    n = 33
    edges = []
    
    # 8 disjoint K4s
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
                
    # Universal vertex 32
    for i in range(32):
        edges.append([32, i])
        
    return {'n': n, 'edges': edges}
