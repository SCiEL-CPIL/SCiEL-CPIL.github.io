import time

def construct():
    # Construct two disjoint K3,3 components
    # Component 1: Partitions {0, 1, 2} and {3, 4, 5}
    # Component 2: Partitions {6, 7, 8} and {9, 10, 11}
    
    edges = []
    
    # K3,3 on 0-5
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            edges.append([u, v])
            
    # K3,3 on 6-11
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            edges.append([u, v])
            
    return edges
