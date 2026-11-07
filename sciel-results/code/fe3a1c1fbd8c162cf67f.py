<<<<<<< SEARCH
# EVOLVE-BLOCK-START
P = {'n': 16}

def construct():
    return [[0,v] for v in range(1,P['n'])]
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
import time
import itertools
from collections import deque

P = {'n': 16}

def _bfs_diameter(n, adj):
    """Compute diameter using BFS from each vertex. Returns (diameter, is_connected)."""
    max_dist = 0
    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        # Check connectivity and max distance
        for i in range(n):
            if dist[i] == -1:
                return -1, False  # disconnected
            if dist[i] > max_dist:
                max_dist = dist[i]
    return max_dist, True

def _edge_essential(n, adj, u, v):
    """Check if edge (u,v) is essential: removing it makes dist(u,v) > 2 or disconnects."""
    # After removing (u,v), check if u and v still have a common neighbor
    # or if they are connected by a path of length <= 2
    # Actually, we need to check if dist(u,v) > 2 in G - {(u,v)}
    # This is true iff there is no w such that (u,w) and (w,v) are both in G - {(u,v)}
    # i.e., no w != u, w != v with w in adj[u] and w in adj[v]
    for w in adj[u]:
        if w == v:
            continue
        if w in adj[v]:
            return False  # found common neighbor, edge not essential
    return True

def _is_valid_graph(n, adj):
    """Check if graph has diameter exactly 2 and every edge is essential."""
    diam, connected = _bfs_diameter(n, adj)
    if not connected or diam != 2:
        return False
    for u in range(n):
        for v in adj[u]:
            if v > u:
                if not _edge_essential(n, adj, u, v):
                    return False
    return True

def _adj_to_edges(n, adj):
    edges = []
    for u in range(n):
        for v in adj[u]:
            if v > u:
                edges.append([u, v])
    return edges

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s limit
    n = P['n']
    
    # Strategy: try to build a graph with maximum edges where every edge is essential
    # and diameter is exactly 2.
    
    # Approach: Start with empty graph, add edges greedily.
    # An edge (u,v) can be added if:
    # 1. Adding it doesn't make any existing edge non-essential
    # 2. The new edge itself is essential
    # 3. The graph maintains diameter 2 (or will achieve it)
    
    # Actually, let's try a different approach:
    # Start with a complete graph and remove edges that are NOT essential.
    # But removing an edge might make other edges essential or non-essential.
    
    # Better: Use local search. Start with a valid graph, try to add edges.
    
    # Let me try a construction based on the idea that for edge (u,v) to be essential,
    # N(u) \ {v} and N(v) \ {u} must be disjoint.
    
    # Try: a graph where vertices are partitioned into 4 groups of 4,
    # and edges exist only between certain groups in a structured way.
    
    # Let's try: complete 4-partite graph with parts of size 4.
    # K_{4,4,4,4}: every vertex connects to all vertices not in its part.
    # n=16, 4 parts of 4.
    # Diameter: 2 (any two vertices in same part have common neighbors in other parts)
    # Edge essentiality: for edge (u,v) where u in part A, v in part B,
    #   N(u) \ {v} = all vertices not in part A except v
    #   N(v) \ {u} = all vertices not in part B except u
    #   Intersection: vertices not in part A and not in part B and not u and not v
    #   That's parts C and D (8 vertices). So they share common neighbors.
    #   So edges are NOT essential. This doesn't work.
    
    # Let me try a different structure. What if we use a graph where
    # adjacent vertices have completely disjoint neighborhoods?
    
    # Try: a graph based on a 1-factorization or matching structure.
    
    # Alternative: Let's just do a greedy search.
    # Start with empty adj, try to add all possible edges, checking validity.
    
    # For efficiency, let's precompute which edges can coexist.
    # Two edges (u,v) and (x,y) are "conflicting" if having both makes one non-essential.
    # Edge (u,v) is non-essential if there exists w with (u,w) and (w,v) both present.
    
    # So the constraint is: for every edge (u,v), there should be no w with (u,w) and (w,v) present.
    # This means: if (u,v) is an edge, then N(u) \ {v} and N(v) \ {u} must be disjoint.
    
    # This is a strong constraint. Let's think about it as a graph coloring problem.
    
    # Actually, let me just try a random/greedy approach with the deadline.
    
    # Greedy: try to add edges in some order, checking if the graph remains valid.
    # Order: try all pairs, for each, check if adding it keeps the graph valid.
    
    adj = [set() for _ in range(n)]
    
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    # Try adding edges one by one
    for u, v in all_edges:
        if time.time() > deadline:
            break
        # Try adding edge (u,v)
        adj[u].add(v)
        adj[v].add(u)
        
        # Check if graph is valid
        if _is_valid_graph(n, adj):
            # Keep it
            continue
        else:
            # Remove it
            adj[u].discard(v)
            adj[v].discard(u)
    
    edges = _adj_to_edges(n, adj)
    return edges

# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, this greedy approach will likely produce a very sparse graph because the constraint is so strong. Let me think more carefully.

The constraint "for every edge (u,v), N(u)\{v} and N(v)\{u} are disjoint" is very restrictive. Let me think about what graphs satisfy this.

If (u,v) is an edge, then no vertex w can be adjacent to both u and v (other than u and v themselves). This means the closed neighborhoods of adjacent vertices intersect only at the edge itself.

This is equivalent to saying the graph has no triangles! Because if there's a triangle u-v-w-u, then for edge (u,v), w is a common neighbor, so (u,v) is not essential.

Wait, that's not quite right. If there's a triangle, then edge (u,v) has common neighbor w, so it's not essential. But the converse: if there's no triangle, does that mean every edge is essential? No, because two vertices could share a common neighbor without forming a triangle... wait, if u-w and w-v are edges, and u-v is an edge, that IS a triangle. If u-w and w-v are edges but u-v is NOT an edge, then w is a common neighbor of u and v, but u and v aren't adjacent, so this doesn't affect edge essentiality.

So: edge (u,v) is essential iff there is no triangle containing edge (u,v). In other words, the graph must be triangle-free for all edges to be essential!

Wait, let me re-examine. Edge (u,v) is essential iff N(u)\{v} ∩ N(v)\{u} = ∅. This means no vertex w is adjacent to both u and v. If such w exists, then (u,w,v) is a path of length 2, and also (u,v) is an edge, forming triangle u-w-v-u.

So yes: **all edges are essential iff the graph is triangle-free**.

And we need diameter exactly 2. So we need a triangle-free graph on 16 vertices with diameter 2, maximizing edges.

This is a well-studied problem! The maximum number of edges in a triangle-free graph on n vertices is given by Mantel's theorem: floor(n²/4). For n=16, that's 64.

But we also need diameter 2. The complete bipartite graph K_{8,8} is triangle-free with 64 edges. What's its diameter? Any two vertices in the same part have distance 2 (via any vertex in the other part). Any two vertices in different parts have distance 1. So diameter is 2!

So K_{8,8} has 64 edges, diameter 2, and is triangle-free, so every edge is essential!

Let me verify: in K_{8,8}, for edge (u,v) where u is in part A and v is in part B:
- N(u) = all of part B (8 vertices)
- N(v) = all of part A (8 vertices)
- N(u)\{v} = part B minus v = 7 vertices
- N(v)\{u} = part A minus u = 7 vertices
- Intersection: empty (part A and part B are disjoint)

Yes! So K_{8,8} works with 64 edges.

Can we do better? Mantel's theorem says 64 is the max for triangle-free graphs on 16 vertices. So 64 is optimal.

Let me verify the diameter: K_{8,8} has diameter 2. ✓
Every edge is essential: since it's triangle-free, removing any edge (u,v) makes dist(u,v) = 2 still (via any other vertex in the opposite part, since each part has 8 vertices, there are 7 other common neighbors). Wait, that means removing edge (u,v) still leaves dist(u,v) = 2, so the diameter doesn't increase!

Oh no, I made an error. If removing edge (u,v) still leaves a path of length 2 between u and v, then the diameter remains 2, so the edge is NOT essential!

Let me re-read the problem: "Deleting EACH individual edge must increase its diameter above two or disconnect it."

So for K_{8,8}, removing edge (u,v) where u∈A, v∈B: u and v are still connected via any w∈A (w≠u): path u-w-v has length 2. So dist(u,v) is still 2, diameter is still 2. The edge is NOT essential!

I was confused. Let me re-derive. Edge (u,v) is essential iff removing it either disconnects u and v or makes dist(u,v) > 2. In K_{8,8}, removing (u,v) still leaves many paths of length 2, so dist(u,v) = 2 still. Not essential.

So my earlier analysis was wrong. The condition is NOT just triangle-free. Let me re-examine.

Edge (u,v) is essential iff in G\{(u,v)}, dist(u,v) > 2 or u,v disconnected. This means there is NO path of length ≤ 2 from u to v in G\{(u,v)}. A path of length 2 from u to v goes through some w: u-w-v. So we need: no w such that (u,w) and (w,v) are both edges in G\{(u,v)}. Since (u,v) is removed but (u,w) and (w,v) don't involve the edge (u,v) directly (unless w=u or w=v, which isn't allowed), this means: no w ≠ u,v with (u,w) ∈ E and (w,v) ∈ E.

So edge (u,v) is essential iff N(u) ∩ N(v) = ∅ (where N is the open neighborhood, and we exclude u and v themselves since they're not in their own neighborhoods... wait, u ∉ N(u) and v ∉ N(v) by definition of open neighborhood).

Hmm, let me be careful. N(u) = {w : (u,w) ∈ E}. u ∉ N(u). For edge (u,v), after removing (u,v), u and v are still adjacent if... no, they're not adjacent anymore. A path u-w-v exists iff w ∈ N(u) ∩ N(v) in the original graph (since w≠u,v, the edges (u,w) and (w,v) don't involve (u,v)).

So edge (u,v) is essential iff N(u) ∩ N(v) = ∅.

In K_{8,8}: for u∈A, v∈B, N(u) = B, N(v) = A. N(u) ∩ N(v) = B ∩ A = ∅. So edges ARE essential!

Wait, I think I confused myself. Let me re-examine. In K_{8,8}, u∈A, v∈B. N(u) = all vertices in B (since u connects to all in B). N(v) = all vertices in A. N(u) ∩ N(v) = B ∩ A = ∅. So there's no common neighbor w. So removing edge (u,v), there's no path of length 2 from u to v. What about longer paths? u can go to any w∈B (w≠v), then w connects to all in A, so w can go to any x∈A (x≠u), then x connects to all in B including v. So path u-w-x-v has length 3. So dist(u,v) = 3 after removing (u,v). Diameter becomes 3 > 2. Edge is essential! ✓

I was wrong earlier when I said "path u-w-v has length 2" - that requires w to be adjacent to both u and v, but in K_{8,8}, vertices in B are not adjacent to each other, and vertices in A are not adjacent to each other. So if u∈A, v∈B, a path u-w-v requires w adjacent to u (so w∈B) and w adjacent to v (so w∈A). But w can't be in both A and B. So no such path exists!

So K_{8,8} IS valid with 64 edges! And by Mantel's theorem, 64 is the maximum for triangle-free graphs, and we need the graph to be such that every edge has no common neighbor, which is stronger than triangle-free... actually no, it's exactly that adjacent vertices have no common neighbors, which means no triangles.

Wait, I need to double-check: does every triangle-free graph have the property that adjacent vertices have no common neighbors? No! A triangle-free graph can have two adjacent vertices with a common neighbor - that would form a triangle. So triangle-free ⟺ no three vertices form a triangle ⟺ for every edge (u,v), N(u) ∩ N(v) = ∅. Yes, these are equivalent.

So we need a triangle-free graph with diameter 2, maximizing edges. Mantel's theorem gives max 64 edges (achieved by K_{8,8}), and K_{8,8} has diameter 2. So 64 is optimal.

Let me code this up:

