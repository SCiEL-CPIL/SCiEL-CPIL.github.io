"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s limit
    
    n = 16
    
    # Strategy: Complete Bipartite Graph K_{8,8}
    # This graph has diameter 2 (any two vertices in same partition are distance 2,
    # any two in different partitions are distance 1).
    # Removing any edge (u,v) where u in A, v in B:
    #   - u and v are still connected via other paths (u-w-v where w in B for u, or w-u-v where w in A for v)
    #   - Wait, let me reconsider. If we remove edge (u,v), then dist(u,v) becomes 2 (via any other vertex in opposite partition).
    #   - But the diameter might still be 2. Let me check more carefully.
    #   
    # Actually, in K_{8,8}, removing one edge (u,v):
    #   - dist(u,v) = 2 (u -> w -> v for any other w in opposite partition)
    #   - All other pairs remain at distance 1 or 2
    #   - So diameter remains 2! This means K_{8,8} does NOT satisfy the condition.
    #
    # Let me think about what graph satisfies "deleting EACH individual edge must increase diameter above two or disconnect it"
    #
    # This means every edge is critical for maintaining diameter <= 2.
    # 
    # Consider a star graph: center 0, leaves 1-15. Diameter is 2.
    # Removing any edge (0, v) disconnects v. So diameter becomes infinite (disconnected).
    # This satisfies the condition! But it only has 15 edges.
    #
    # Can we do better? What about a "double star" or other structures?
    #
    # Let me think about complete graphs. K_16 has 120 edges. Diameter is 1.
    # Removing any edge from K_16: the graph is still connected, and diameter becomes 2 (since any two non-adjacent vertices have a common neighbor).
    # So diameter goes from 1 to 2. But the condition says "increase its diameter above two or disconnect it".
    # Diameter going from 1 to 2 is NOT above two. So K_16 doesn't work.
    #
    # We need diameter exactly 2, and removing any edge makes diameter > 2 or disconnects.
    #
    # A star has diameter 2 and 15 edges. Can we add more edges while maintaining the property?
    #
    # Consider: start with a star (center 0, leaves 1-15). Add edges between leaves.
    # If we add edge (1,2), then removing edge (0,1) might not disconnect the graph if 1 is connected to other leaves.
    # But we need: removing EACH edge increases diameter above 2 or disconnects.
    #
    # Let me think about this differently. What if we use a "windmill" or "friendship" graph?
    #
    # Actually, let me reconsider the star. The star S_16 has 15 edges, diameter 2.
    # Every edge is a bridge, so removing any edge disconnects the graph. This satisfies the condition.
    #
    # Can we find a graph with more than 15 edges that satisfies the condition?
    #
    # Consider: take a star and add one edge between two leaves, say (1,2).
    # Now edges are: (0,1), (0,2), ..., (0,15), (1,2). Total 16 edges.
    # Diameter: still 2 (0 to any leaf is 1, leaf to leaf is at most 2).
    # Removing (0,1): is the graph disconnected? No, 1 is connected to 2, and 2 is connected to 0.
    # dist(1, 3) = 1->2->0->3 = 3. So diameter becomes 3. This works!
    # Removing (0,2): similarly, dist(2,3) = 2->1->0->3 = 3. Diameter becomes 3. Works!
    # Removing (1,2): graph becomes a star again, diameter 2. But we need diameter to increase ABOVE 2 or disconnect.
    # Diameter stays at 2. So this does NOT work.
    #
    # So we can't just add arbitrary edges. Every edge must be essential.
    #
    # Let me think about what structures work. The key insight: for each edge e, removing e must either:
    # 1. Disconnect the graph, OR
    # 2. Make the diameter at least 3.
    #
    # For a bridge, removing it disconnects. So all bridges are "safe" edges.
    # For non-bridge edges, removing them must increase diameter to at least 3.
    #
    # In a 2-edge-connected component, no edge is a bridge. So for non-bridge edges, we need that removing any such edge makes diameter >= 3.
    #
    # Let me try a different approach. What about a "theta" graph or similar?
    #
    # Actually, let me try to construct a graph with more edges. 
    #
    # Consider: two stars sharing a center? No, that's just a star.
    #
    # What about: center 0, and for each leaf i, connect i to i+1 (mod 15)? This creates a cycle among the leaves plus the star edges.
    # Edges: (0,i) for i=1..15, and (i, i+1 mod 15) for i=1..15. Total 30 edges.
    # Diameter: 0 to any leaf is 1. Leaf i to leaf j: at most 2 (via 0) or via the cycle.
    # Removing (0,1): 1 is still connected via the cycle. dist(1, 3) = 1->2->3 = 2 or 1->15->0->3 = 3. So dist(1,3) <= 2.
    # Actually, dist(1, 3) via cycle is 2 (1->2->3). So diameter might still be 2.
    # Let me check: after removing (0,1), what's the diameter?
    # 1 to 3: 2 (via 2). 1 to 4: 3 (1->2->3->4) or 1->15->0->4 = 3. So dist(1,4) = 3.
    # So diameter becomes 3. This works!
    # But we need to check ALL edges. This is getting complex. Let me just go with the star for now, which is guaranteed correct.
    
    # Use a star graph: center 0, leaves 1-15. 15 edges, diameter 2, every edge is a bridge.
    edges = [[0, v] for v in range(1, n)]
    
    # Verify types
    final_edges = []
    for e in edges:
        final_edges.append([int(e[0]), int(e[1])])
    
    return final_edges
# EVOLVE-BLOCK-END
