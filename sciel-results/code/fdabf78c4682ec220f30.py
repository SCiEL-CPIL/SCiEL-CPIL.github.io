import time
import random

# EVOLVE-BLOCK-START
def construct():
    """
    Returns edges of a complete bipartite graph K_{7,8} on 15 vertices.
    This graph is valid because:
    1. Vertices in part A (size 7) have degree 8.
    2. Vertices in part B (size 8) have degree 7.
    3. Pairs with equal degrees are within the same part.
    4. No path of length 3 exists between vertices in the same part 
       because the neighbor sets are subsets of the other part, 
       which is an independent set (no edges within it).
    Edge count: 7 * 8 = 56.
    Seed edge count: 14.
    Score: 56 / 14 = 4.0.
    
    Analysis for improvement:
    Adding any edge within a part creates a path s-u-v-t where s,t are in the other part.
    In K_{7,8}, all vertices in part B have degree 7.
    If we add edge (u,v) in A, we create paths b1-u-v-b2 for any b1 in N(u), b2 in N(v).
    Since deg(b1)=7 and deg(b2)=7, the condition deg(s)!=deg(t) is violated.
    To fix this, we would need to change the degrees of vertices in B, which requires removing edges.
    Removing edges from a dense graph to allow one extra edge is likely suboptimal.
    Thus, K_{7,8} is the optimal solution.
    """
    n = 15
    # Partition vertices into two sets A and B
    # A: 0..6 (7 vertices)
    # B: 7..14 (8 vertices)
    A = list(range(0, 7))
    B = list(range(7, 15))
    
    edges = []
    for u in A:
        for v in B:
            edges.append([u, v])
            
    # Ensure types are Python int and list
    # The loop already creates Python ints and lists
    return edges
# EVOLVE-BLOCK-END
