import time

def construct():
    """
    Returns the edge list of the Clebsch graph, which is a triangle-free graph
    on 16 vertices with chromatic number 4 and 40 edges.
    This is the known maximum for this constraint.
    """
    N = 16
    # The Clebsch graph vertices can be represented as 5-bit integers with even parity.
    # There are C(5,0) + C(5,2) + C(5,4) = 1 + 10 + 5 = 16 such vertices.
    vertices = []
    for i in range(32):
        if bin(i).count('1') % 2 == 0:
            vertices.append(i)
    
    # Map original values to indices 0..15
    val_to_idx = {v: i for i, v in enumerate(vertices)}
    
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            # Two vertices in the Clebsch graph are adjacent if their Hamming distance is 4.
            # This is equivalent to their XOR having exactly 4 bits set.
            xor_val = vertices[i] ^ vertices[j]
            if bin(xor_val).count('1') == 4:
                edges.append([i, j])
                
    return edges
