"""K3,3-free dense graph on 20 vertices.

Strategy:
1. Build a strong deterministic starting point using a bipartite construction
   (K_{5,15} has no K3,3 and gives 75 edges).
2. Improve via randomized local search:
   - Randomly try to add non-edges, checking K3,3-freedom via common-neighbor counts.
   - Randomly try to swap edges (remove one, add another) to escape local optima.
   - Track best edge count found.
3. Use a deadline guard to return the best valid graph before the time limit.

K3,3-free condition: no two vertices share 3 or more common neighbors.
For a bipartite graph, K3,3 is exactly a 3x3 complete bipartite subgraph,
which occurs iff some pair of vertices on one side share >=3 common neighbors on the other side.
For a general graph, K3,3 is a bipartite subgraph with parts of size 3 each.
We check: for every pair of vertices (u,v), count common neighbors. If >=3, we have a K3,3
(u, v, and 3 common neighbors form a K3,3 where {u,v,w1} and {w1,w2,w3} are the two parts... 
actually need to be careful).

Actually, K3,3 is a 6-vertex subgraph with bipartition (A,B), |A|=|B|=3, all edges between A and B present.
So we need to check: do there exist 3 vertices a1,a2,a3 and 3 vertices b1,b2,b3 such that
every ai is adjacent to every bj?

Efficient check when adding edge (u,v): the new edge could create a K3,3 only if u and v are in
the same part of the K3,3 (i.e., they share 3 common neighbors) OR u and v are in different parts.
If u and v are in different parts, then we need 2 more vertices in u's part and 2 more in v's part
that are fully connected. This is more complex.

Simpler approach: for the bipartite construction, we can maintain the bipartite structure and only
check the bipartite K3,3 condition (common neighbors >= 3). But to go beyond bipartite, we need
general checking.

Let me use a simpler but effective approach: start with K_{5,15} (bipartite, 75 edges), then
try to add edges within the same part or cross edges more densely, checking for K3,3 via
counting common neighbors for pairs.

Actually, for general graphs: a K3,3 exists iff there exist 3 vertices with at least 3 common
neighbors (where those 3 vertices form one part and 3 of the common neighbors form the other part,
and the edges between them exist). Wait, that's not quite right either.

Let me think again. A K3,3 subgraph has parts A={a1,a2,a3} and B={b1,b2,b3} such that all 9
edges between A and B exist. So if I pick any 3 vertices in A, they must each be connected to all 3
in B. 

For checking if adding edge (u,v) creates a K3,3:
- Case 1: u and v are in the same part (say both in A). Then we need 3 common neighbors of u and v
  that are all in B. If u and v have >=3 common neighbors, and those common neighbors are mutually... 
  no, they don't need to be connected to each other. They just need to be the B part. So if u and v
  share >=3 common neighbors, and we pick 3 of those as B, then {u, v, w} (where w is another vertex 
  in A connected to all of B) would form K3,3. Wait, we need 3 vertices in A, not just 2.

Let me use a different approach: just check the full K3,3 condition by iterating over potential 
triples. For 20 vertices, checking all C(20,3) * C(remaining,3) is too slow per edge addition.

Better: precompute for each pair (u,v) the number of common neighbors. A K3,3 exists if there's
a set of 3 vertices each sharing >=3 common neighbors with each other... no.

Actually, the standard efficient check: a graph contains K3,3 iff there exist 3 vertices a,b,c 
such that |N(a) ∩ N(b) ∩ N(c)| >= 3. Because then a,b,c form one part and any 3 common neighbors 
form the other part.

Wait, that's only true if the 3 common neighbors are in a part by themselves. Let me verify:
If N(a) ∩ N(b) ∩ N(c) contains vertices x,y,z, then a,b,c are all connected to x,y,z. So 
{a,b,c} × {x,y,z} forms a K3,3. Yes! So the condition is: there exist 3 vertices whose 
intersection of neighborhoods has size >= 3.

So for checking if the graph is K3,3-free: for every triple (a,b,c), check 
|N(a) ∩ N(b) ∩ N(c)| <= 2.

For efficiency during local search, when adding edge (u,v), we can check if this creates a 
triple with >=3 common neighbors. The new edge (u,v) means v ∈ N(u) and u ∈ N(v). This can 
increase intersections involving u or v.

Let me implement a practical approach with the triple-intersection check, optimized for 
incremental updates.

For 20 vertices, there are C(20,3) = 1140 triples. Checking each triple's common neighbor 
count by iterating over neighbors is fast. Let me just recompute the max triple intersection 
after each edge change. With 1140 triples and at most 20 neighbors each, that's ~22800 ops 
per check, which is very fast.
"""

import time
import random
import numpy as np
from itertools import combinations

def _check_k33_free(adj):
    """Check if graph with adjacency matrix adj is K3,3-free.
    K3,3 exists iff there are 3 vertices with >= 3 common neighbors."""
    n = adj.shape[0]
    # For each triple, count common neighbors
    for a, b, c in combinations(range(n), 3):
        common = adj[a] & adj[b] & adj[c]
        if np.sum(common) >= 3:
            return False
    return True

def _count_edges(adj):
    return int(np.sum(adj) // 2)

def construct():
    deadline = time.time() + 140  # 140s to be safe under 150s limit
    
    n = 20
    # Start with K_{5,15}: vertices 0-4 connected to vertices 5-19
    adj = np.zeros((n, n), dtype=bool)
    for a in range(5):
        for b in range(5, 20):
            adj[a, b] = True
            adj[b, a] = True
    
    best_adj = adj.copy()
    best_count = _count_edges(best_adj)
    
    # Verify starting graph is K3,3-free
    assert _check_k33_free(best_adj), "Initial graph should be K3,3-free"
    
    rng = random.Random(42)
    
    while time.time() < deadline:
        # Try random edge additions and swaps
        # First pass: try adding non-edges
        for _ in range(200):
            u, v = rng.sample(range(n), 2)
            if not adj[u, v]:
                # Tentatively add edge
                adj[u, v] = True
                adj[v, u] = True
                if _check_k33_free(adj):
                    count = _count_edges(adj)
                    if count > best_count:
                        best_count = count
                        best_adj = adj.copy()
                else:
                    adj[u, v] = False
                    adj[v, u] = False
        
        # Second pass: try edge swaps
        for _ in range(100):
            # Pick a random edge to remove
            edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i, j]]
            if not edges:
                break
            u, v = rng.choice(edges)
            # Pick a random non-edge to add
            non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not adj[i, j]]
            if not non_edges:
                break
            a, b = rng.choice(non_edges)
            
            # Perform swap
            adj[u, v] = False
            adj[v, u] = False
            adj[a, b] = True
            adj[b, a] = True
            
            if _check_k33_free(adj):
                count = _count_edges(adj)
                if count > best_count:
                    best_count = count
                    best_adj = adj.copy()
                # Keep the swap if it doesn't reduce edge count
                if count < _count_edges(best_adj):
                    pass  # keep it anyway for exploration
            else:
                # Revert
                adj[u, v] = True
                adj[v, u] = True
                adj[a, b] = False
                adj[b, a] = False
    
    # Convert best_adj to edge list
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i, j]:
                result.append([i, j])
    
    return result
