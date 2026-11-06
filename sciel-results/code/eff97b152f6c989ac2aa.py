import time

def construct():
    # Vertices 0..15
    # U = 0..3, V = 4..15
    edges = []
    
    # Clique on U (0..3)
    for i in range(4):
        for j in range(i+1, 4):
            edges.append([i, j])
            
    # Clique on V (4..15)
    for i in range(4, 16):
        for j in range(i+1, 16):
            edges.append([i, j])
            
    # Bipartite edges B between U and V
    # M_u are disjoint sets of size 3
    # M_0 = {4,5,6}, M_1 = {7,8,9}, M_2 = {10,11,12}, M_3 = {13,14,15}
    # u_i is connected to v_j if v_j not in M_i
    
    M = [
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
        [13, 14, 15]
    ]
    
    for i in range(4):
        for j in range(4, 16):
            if j not in M[i]:
                edges.append([i, j])
                
    return edges
