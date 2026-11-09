import time

def construct():
    # The Clebsch graph is the folded 5-dimensional hypercube.
    # It has 16 vertices, 40 edges, is triangle-free, and has chromatic number 4.
    # This is the maximum number of edges for a triangle-free graph on 16 vertices 
    # with chromatic number at least 4.
    
    edges = []
    edge_set = set()
    
    # Map a 5-bit integer to its canonical representative 
    # (identifying v and v XOR 11111)
    def canonical(v):
        # 11111 is 31
        return min(v, v ^ 31)
    
    # Generate the 16 vertices
    vertices = set()
    for v in range(32):
        vertices.add(canonical(v))
        
    vertices = list(vertices)
    # vertices will be 0..15 essentially, but let's map them to 0..15 indices for clean output
    # Actually, the canonical forms are just 0..15 because min(v, v^31) for v in 0..31
    # If v < 16, v^31 > 15 (since 31 is 11111, v^31 is bitwise NOT of v in 5 bits, which is 31-v).
    # So min(v, 31-v).
    # For v=0, min(0,31)=0.
    # For v=1, min(1,30)=1.
    # ...
    # For v=15, min(15,16)=15.
    # For v=16, min(16,15)=15.
    # So the set of canonical vertices is exactly {0, 1, ..., 15}.
    
    # Construct edges
    for u in vertices:
        for i in range(5):
            v = u ^ (1 << i)
            v_canon = canonical(v)
            if u < v_canon:
                edge_set.add((u, v_canon))
            elif u > v_canon:
                edge_set.add((v_canon, u))
            # if u == v_canon, it's a self-loop, which shouldn't happen in this construction
            # because u and u^(1<<i) are distinct and not antipodal (dist 1 != 5)
            
    # Convert to list of lists [u, v]
    for u, v in edge_set:
        edges.append([u, v])
        
    return edges
