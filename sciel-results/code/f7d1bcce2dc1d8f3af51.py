import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    # The graph is two disjoint K3,3 components.
    # Vertices 0-5 form one K3,3.
    # Vertices 6-11 form the other K3,3.
    
    def make_k33(offset):
        edges = []
        for i in range(3):
            for j in range(3):
                u = offset + i
                v = offset + 3 + j
                edges.append([u, v])
        return edges
    
    edges = make_k33(0) + make_k33(6)
    
    # Ensure types are correct
    result = [list(e) for e in sorted(set(tuple(sorted(e)) for e in edges))]
    
    return result
