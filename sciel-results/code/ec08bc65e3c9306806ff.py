import time
import random
import numpy as np

def construct():
    """
    Constructs a C6-free graph on 40 vertices maximizing edges.
    Uses a local search with a bipartite double-cover (Boolean matrix) C4-detection
    to quickly test whether adding an edge would create a C6 in the original graph.
    """
    N = 40
    deadline = time.time() + 115

    # Precompute all pairs
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]

    # ---------- C4-in-cover checker using Boolean adjacency matrix ----------
    # The cover has 80 vertices: u0 (u top) for u in 0..39, u1 (u bottom) for u in 40..79.
    # Edges in cover: (u0, v1) and (u1, v0) for each edge (u,v) in original graph.
    # A C4 in cover <-> a C6 in original graph.
    #
    # To test if adding edge (u,v) creates a C4 in cover:
    #   We need to check if there exists a path of length 3 in the current cover
    #   from u0 to v1, u1 to v0, u0 to v0, u1 to v1, etc.
    #
    # Actually, the C4s that the new edge would complete:
    #   The new cover edges are (u0,v1) and (u1,v0).
    #   A C4 involving (u0,v1) would be: u0 - a - b - v1 where a is in bottom layer, b in top layer.
    #   But wait: in the cover, edges always go between top and bottom.
    #   So a C4 is: t1 - b1 - t2 - b2 - t1 where t's are top, b's are bottom.
    #   The new edge (u0,v1): u0 is top, v1 is bottom.
    #   A C4 containing this edge: u0 - b_x - t_y - v1 - u0? No, that's not right.
    #   Let me reconsider.
    #
    # In the bipartite cover B of G:
    #   Vertices: {u_0, u_1 : u in V(G)}
    #   Edges: (u_0, v_1) and (u_1, v_0) for each edge (u,v) in G.
    #
    # A C4 in B: u_0 - v_1 - w_0 - x_1 - u_0
    #   This means: (u,v) in E, (v,w) in E, (w,x) in E, (x,u) in E.
    #   So it corresponds to a C4 in G, NOT a C6!
    #
    # Hmm, I need to reconsider the mapping. The standard double cover for C6 detection:
    #   Actually, a C6 in G corresponds to a C4 in the "incidence" bipartite graph?
    #   No. Let me think again.
    #
    # A C6 in G: v0-v1-v2-v3-v4-v5-v0
    # In the bipartite graph where we split each vertex into two:
    #   Top: v0_t, v1_t, v2_t, v3_t, v4_t, v5_t
    #   Bottom: v0_b, v1_b, v2_b, v3_b, v4_b, v5_b
    #   Edges: (v_i_t, v_j_b) and (v_j_t, v_i_b) for each edge (v_i, v_j) in G.
    #
    # A C6 in G becomes: v0_t - v1_b - v2_t - v3_b - v4_t - v5_b - v0_t
    #   That's a 6-cycle in the bipartite graph, not a 4-cycle!
    #
    # So the double cover maps C6 in G to C6 in the cover, not C4.
    #
    # Let me use a different approach: use the Boolean matrix to count common neighbors.
    #
    # Adding edge (u,v) creates a C6 iff there exist a,b,c,d distinct from u,v
    # forming a path u-a-b-c-d-v.
    #
    # Equivalent: there exists d (neighbor of v, d != u) such that u and d have
    # a common neighbor c, and c and d have a common neighbor b, and b is adjacent to u,
    # with all of u,a,b,c,d,v distinct.
    #
    # This is getting complex. Let me use a simpler but faster approach:
    # Use the adjacency matrix to check C6 via (A^4)[u,d] for d in N(v)\{u},
    # but we need to ensure the path is simple and avoids v.
    #
    # Actually, for 40 vertices, let's just use a faster DFS with numpy bit operations
    # or precompute distances.
    #
    # Let me use a different strategy: maintain the adjacency matrix as a numpy array
    # and use it for fast neighbor lookups.

    # Adjacency matrix
    A = np.zeros((N, N), dtype=np.uint8)
    adj_set = [set() for _ in range(N)]
    edges = []

    # Initial: 8 disjoint K5s
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a + 1, 5):
                u, v = base + a, base + b
                A[u, v] = 1
                A[v, u] = 1
                adj_set[u].add(v)
                adj_set[v].add(u)
                edges.append([u, v])

    best_edges = len(edges)
    best_A = A.copy()

    def can_add_fast(u, v):
        """Check if adding (u,v) creates a C6. Uses matrix-based approach."""
        # A C6 containing new edge (u,v): u - a - b - c - d - v
        # We need a path of length 4 from u to some neighbor d of v (d != u),
        # avoiding v.
        #
        # Let's do a BFS/DFS from u to depth 4, avoiding v.
        # Use a visited mask for the path.
        #
        # For efficiency, we limit the branching. With 40 vertices and degree ~5-10,
        # depth-4 DFS is manageable.
        
        # Iterative DFS with backtracking
        # Path: u, p1, p2, p3, p4 where p4 is a neighbor of v
        # We need p4 != v, p4 != u, and all p_i distinct and != v.
        
        # Use a recursive-like approach with explicit stack
        # Stack entries: (node, depth, path_mask)
        # path_mask: bitmask of vertices in the current path (including u)
        
        start_mask = 1 << u
        stack = [(u, 0, start_mask)]
        
        while stack:
            node, depth, mask = stack.pop()
            
            if depth == 4:
                # Check if v is a neighbor of node and v not in path
                if A[node, v] == 1 and not (mask & (1 << v)):
                    return False
                continue
            
            # Get neighbors of node, skip v and already-visited
            for nb in range(N):
                if A[node, nb] == 0:
                    continue
                if nb == v:
                    continue
                if mask & (1 << nb):
                    continue
                stack.append((nb, depth + 1, mask | (1 << nb)))
        
        return True

    random.seed(42)
    
    while time.time() < deadline:
        # Build list of missing edges
        missing = [(u, v) for (u, v) in all_pairs if A[u, v] == 0]
        
        if not missing:
            break
        
        random.shuffle(missing)
        
        added = 0
        for u, v in missing:
            if time.time() > deadline:
                break
            if A[u, v] == 1:
                continue
            if can_add_fast(u, v):
                A[u, v] = 1
                A[v, u] = 1
                adj_set[u].add(v)
                adj_set[v].add(u)
                edges.append([u, v])
                added += 1
                if len(edges) > best_edges:
                    best_edges = len(edges)
                    best_A = A.copy()
        
        if added == 0:
            # Kick: remove random edges
            if len(edges) > 0:
                kick = random.randint(1, 3)
                for _ in range(kick):
                    if not edges:
                        break
                    idx = random.randint(0, len(edges) - 1)
                    u, v = edges[idx]
                    A[u, v] = 0
                    A[v, u] = 0
                    adj_set[u].discard(v)
                    adj_set[v].discard(u)
                    edges.pop(idx)

    # Return best
    final = []
    for i in range(N):
        for j in range(i + 1, N):
            if best_A[i, j] == 1:
                final.append([i, j])
    return final

if __name__ == "__main__":
    e = construct()
    print(f"Edges: {len(e)}")
