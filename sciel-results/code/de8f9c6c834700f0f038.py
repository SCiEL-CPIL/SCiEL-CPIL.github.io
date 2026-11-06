import time

def construct():
    # Construct a graph on 35 vertices.
    # Base: 8 disjoint K4s.
    # Vertices 0-31 are the 8 K4s.
    # Block i has vertices 4*i, 4*i+1, 4*i+2, 4*i+3.
    # Add 3 new vertices: 32, 33, 34.
    # Connect 32 to all vertices in Block 0, and one vertex (first) in Blocks 1-7.
    # Connect 33 to all vertices in Block 1, and one vertex (first) in Blocks 0, 2-7.
    # Connect 34 to all vertices in Block 2, and one vertex (first) in Blocks 0, 1, 3-7.
    # Connect 32, 33, 34 to each other (Triangle).
    
    n = 35
    edges = []
    
    # 1. Generate edges for the 8 K4 blocks
    for b in range(8):
        start = b * 4
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([start + i, start + j])
                
    # 2. Add edges for new vertices 32, 33, 34
    # Vertex 32 connects to Block 0 (all 4) and first vertex of Blocks 1-7
    v32 = 32
    for v in range(0, 4): # Block 0
        edges.append([v32, v])
    for b in range(1, 8):
        edges.append([v32, b * 4]) # First vertex of block b
        
    # Vertex 33 connects to Block 1 (all 4) and first vertex of Blocks 0, 2-7
    v33 = 33
    for v in range(4, 8): # Block 1
        edges.append([v33, v])
    edges.append([v33, 0]) # First vertex of Block 0
    for b in range(2, 8):
        edges.append([v33, b * 4]) # First vertex of block b
        
    # Vertex 34 connects to Block 2 (all 4) and first vertex of Blocks 0, 1, 3-7
    v34 = 34
    for v in range(8, 12): # Block 2
        edges.append([v34, v])
    edges.append([v34, 0]) # First vertex of Block 0
    edges.append([v34, 4]) # First vertex of Block 1
    for b in range(3, 8):
        edges.append([v34, b * 4]) # First vertex of block b
        
    # Connect the new vertices to each other
    edges.append([v32, v33])
    edges.append([v32, v34])
    edges.append([v33, v34])
    
    return {'n': n, 'edges': edges}
