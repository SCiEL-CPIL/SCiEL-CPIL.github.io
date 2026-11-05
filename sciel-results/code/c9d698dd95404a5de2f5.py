import time
import sys

def construct():
    n = 16
    # The optimal triangle-free graph on 16 vertices with chromatic number 4
    # is the Clebsch graph, which has 40 edges.
    # We construct it deterministically.
    
    # Vertices are 5-bit integers with even parity
    vertices = []
    for i in range(32):
        if bin(i).count('1') % 2 == 0:
            vertices.append(i)
            
    edges = []
    
    # Connect vertices if Hamming distance is 4
    for i in range(n):
        for j in range(i + 1, n):
            xor_val = vertices[i] ^ vertices[j]
            if bin(xor_val).count('1') == 4:
                edges.append([i, j])
                
    return edges
