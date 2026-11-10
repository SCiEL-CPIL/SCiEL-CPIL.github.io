"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Constructs a dense graph on 16 vertices with diameter 2 where every edge is critical.
    
    The optimal graph is the complete bipartite graph K(8,8).
    - Vertices 0..7 form partition A.
    - Vertices 8..15 form partition B.
    - Edges connect every vertex in A to every vertex in B.
    
    Properties:
    1. Triangle-free: No triangles exist in a bipartite graph.
    2. Diameter 2: Any two nodes in the same partition are distance 2 apart.
    3. Critical Edges: For any edge (u,v), u and v are in different partitions.
       They share no common neighbors. Removing (u,v) increases their distance to 3,
       thus increasing the graph diameter from 2 to 3.
    
    Max edges = 8 * 8 = 64.
    """
    n = 16
    half = n // 2
    
    # Partition A: 0 to half-1
    # Partition B: half to n-1
    partition_a = list(range(0, half))
    partition_b = list(range(half, n))
    
    edges = []
    for u in partition_a:
        for v in partition_b:
            # Ensure u < v for consistent representation if needed, 
            # but [u, v] is sufficient as undirected.
            edges.append([u, v])
            
    return edges

# EVOLVE-BLOCK-END
