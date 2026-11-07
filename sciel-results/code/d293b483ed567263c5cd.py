"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Constructs a maximal edge-critical diameter-2 graph on 16 vertices.
    
    The optimal structure is the complete bipartite graph K_{8,8}, which achieves
    the Mantel bound of 64 edges for triangle-free graphs on 16 vertices.
    
    Why this is optimal:
    1. Diameter 2 requirement: In K_{8,8}, any two vertices in the same partition
       are distance 2 (via any vertex in the other partition). Any two vertices
       in different partitions are distance 1. So diameter is exactly 2.
       
    2. Edge-criticality: Removing any edge (u, v) where u is in partition A and
       v is in partition B increases the distance between u and v to 3. This is
       because there are no triangles in K_{8,8} (it's bipartite), so there's no
       alternative path of length 2 between u and v. The shortest path becomes
       u -> w -> x -> v where w is in B and x is in A, giving length 3.
       
    3. Maximality: By Mantel's theorem, any graph with more than 64 edges on 16
       vertices must contain a triangle. If a graph contains a triangle (u, v, w),
       then edge (u, v) is not critical because removing it leaves path u-w-v of
       length 2, so the diameter doesn't increase. Thus, any edge-critical 
       diameter-2 graph must be triangle-free, limiting it to at most 64 edges.
       
    Therefore K_{8,8} with 64 edges is optimal.
    """
    deadline = time.monotonic() + 55.0  # safety margin under 60s evaluator limit
    
    n = 16
    half = n // 2
    
    # Partition vertices into two sets A = {0..7} and B = {8..15}
    edges = []
    
    # Generate all edges between A and B
    for u in range(half):
        for v in range(half, n):
            edges.append([u, v])
    
    # Ensure types are strictly Python int and list
    # The loop above already produces Python ints and lists
    return edges
# EVOLVE-BLOCK-END
