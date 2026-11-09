import time

def construct():
    """
    Constructs a K4-free graph on 18 vertices that minimizes the size of the 
    largest triangle-free induced subgraph.
    
    The optimal graph is the Turan graph T(18, 3), which is the complete 
    3-partite graph K_{6,6,6}.
    
    Reasoning:
    1. The graph must be K4-free. The Turan graph T(n, r) is the densest 
       K_{r+1}-free graph. For n=18 and K4-free, r=3.
    2. Removing edges from a graph G to form H (subset of edges) implies 
       that any triangle-free induced subgraph in G is also triangle-free in H.
       Therefore, the maximum size of a triangle-free induced subgraph in H 
       is greater than or equal to that in G.
    3. Consequently, the densest K4-free graph (T(18,3)) minimizes the 
       maximum size of a triangle-free induced subgraph.
    4. In K_{6,6,6}, any triangle-free set must be contained within at most 
       two parts (since vertices from three different parts form a triangle). 
       The maximum size of a union of two parts is 6 + 6 = 12.
    
    Thus, the minimum possible score is 12.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Check time (though construction is instant)
    if time.time() > deadline:
        # Fallback: return empty graph (invalid for score, but avoids timeout)
        # Actually, returning empty graph gives score 18 (all vertices triangle-free).
        # But we should return a valid K4-free graph.
        # T(18,3) is the best.
        pass

    # Construct K_{6,6,6}
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    edges = []
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    edges.append([a, b])
    
    return edges
