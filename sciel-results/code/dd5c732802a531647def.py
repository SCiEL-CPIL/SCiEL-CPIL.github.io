import time
import random
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    n = 18
    
    # Helper to convert adjacency matrix to edge list
    def adj_to_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    edges.append([i, j])
        return edges

    # Helper to check if graph is K4-free
    # G is K4-free iff H (complement) has no independent set of size 4
    def is_k4_free_g(g_adj):
        # Check all subsets of 4 vertices in G
        # If any 4 vertices form a K4 in G, it's invalid
        # Equivalently, in H (complement), these 4 vertices must have at least one edge.
        # Wait, K4 in G means all 6 edges present in G.
        # So in H, all 6 edges are absent. i.e., Independent set of size 4 in H.
        
        h_adj = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if not g_adj[i][j]:
                    h_adj[i][j] = 1
                    h_adj[j][i] = 1
        
        # Check for independent set of size 4 in H
        for quad in itertools.combinations(range(n), 4):
            u, v, w, x = quad
            if (not h_adj[u][v] and not h_adj[u][w] and not h_adj[u][x] and
                not h_adj[v][w] and not h_adj[v][x] and
                not h_adj[w][x]):
                return False
        return True

    # Exact Branch and Bound to find max size of triangle-free induced subgraph in G
    # A subgraph S is triangle-free in G iff S contains no 3 vertices with all 3 edges in G.
    # Equivalently, in H (complement), S contains no 3 vertices with 0 edges (independent set of size 3 in H).
    # Wait. 
    # G[S] triangle-free <=> No {u,v,w} in S such that G[u,v], G[v,w], G[w,u] are all edges.
    # This is equivalent to: For every {u,v,w} in S, at least one edge is missing in G.
    # Missing in G means present in H.
    # So H[S] must not have an independent set of size 3.
    # i.e., alpha(H[S]) <= 2.
    
    # So the score is the maximum |S| such that alpha(H[S]) <= 2.
    # This is the same as finding the largest subset S where the induced subgraph H[S] has no independent set of size 3.
    # This is equivalent to finding the largest subset S where the clique number of G[S] is <= 2?
    # No. G[S] triangle-free means omega(G[S]) <= 2.
    
    # Let's compute the score by finding the maximum k such that there exists a subset S of size k with omega(G[S]) <= 2.
    
    def max_triangle_free_size(g_adj):
        # We want to maximize |S| such that G[S] is triangle-free.
        # Use branch and bound on vertices.
        # State: (current_size, current_vertices, candidates)
        # But we can just check all subsets? 2^18 = 262k. 
        # For each subset, check if it's triangle-free.
        # Checking triangle-free takes O(k^3) or faster with bitmasks.
        # 262k * small constant is very fast.
        
        # Let's use bitmasks for speed.
        # g_mask[i] is bitmask of neighbors in G.
        g_mask = [0]*n
        for i in range(n):
            for j in range(n):
                if g_adj[i][j]:
                    g_mask[i] |= (1 << j)
        
        best_size = 0
        all_mask = (1 << n) - 1
        
        # Iterate over all subsets? 
        # Actually, we can do a BFS/DFS on independent sets of the "triangle graph"?
        # No, simpler: just iterate subsets and check.
        # To speed up checking: 
        # A subset S is triangle-free in G iff for every u in S, (S \ {u}) is a subset of the non-neighbors of u in G?
        # No.
        
        # Let's just iterate subsets of size k from 18 down to 1.
        # If we find one, return k.
        # Number of subsets of size 18: 1
        # Size 17: 18
        # Size 16: 153
        # Size 15: 816
        # Size 14: 3060
        # Size 13: 8568
        # Size 12: 18564
        # Total is small enough.
        
        # Check function for a specific subset (list of vertices)
        def is_tri_free(subset):
            # subset is a tuple of vertices
            k = len(subset)
            if k < 3: return True
            # Check all triples
            for i in range(k):
                u = subset[i]
                # For each pair v, w in subset after i, check if u is connected to both and v-w connected
                # Actually, just check if any triangle exists.
                # Triangle exists if there is u,v,w in subset such that 
                # g_adj[u][v] and g_adj[v][w] and g_adj[w][u]
                # We can precompute for each u, the set of neighbors in G.
                # If v is in N_G(u) and w is in N_G(u) and g_adj[v][w], then triangle.
                for j in range(i+1, k):
                    v = subset[j]
                    if not g_adj[u][v]: continue
                    for l in range(j+1, k):
                        w = subset[l]
                        if g_adj[u][w] and g_adj[v][w]:
                            return False
            return True

        for k in range(n, 0, -1):
            for subset in itertools.combinations(range(n), k):
                if is_tri_free(subset):
                    return k
        return 0

    # Strategy:
    # 1. Try a few deterministic constructions known to be good for this problem.
    # 2. Use random search with bitmask verification to find better ones.
    
    best_score = n
    best_g_adj = None
    best_edges = []
    
    # Candidate 1: Complete Tripartite K_{6,6,6}
    # This graph is K4-free.
    # G = K_{6,6,6}. Edges between parts, no edges within parts.
    # Triangle-free induced subgraph?
    # If we pick vertices from 3 different parts, we get a triangle (if edges exist).
    # In K_{6,6,6}, any 3 vertices from 3 distinct parts form a triangle.
    # So to be triangle-free, we can pick vertices from at most 2 parts.
    # Max size = 6 + 6 = 12.
    # Score 12. (18/12 = 1.5)
    
    # Candidate 2: Clebsch Graph (16 vertices) + 2 vertices?
    # Clebsch graph is K4-free? No, Clebsch graph is strongly regular (16, 5, 0, 2).
    # Independence number 5.
    # Let's check K4-freeness.
    # Local structure: Each vertex has 5 neighbors.
    # If there is a K4, it must contain a vertex and its neighbors.
    # In Clebsch, neighbors of a vertex form a 5-cycle (C5). C5 is triangle-free.
    # So no K4 containing the center.
    # What about K4 not containing the center? 
    # Clebsch graph is K4-free. (It is the Schläfli graph? No, Schläfli is 27 vertices. Clebsch is 16).
    # Actually, Clebsch graph is K4-free.
    
    # Let's construct Clebsch graph.
    # Vertices are binary strings of length 4 with even weight (16 vertices).
    # Two vertices are adjacent if their Hamming distance is 2.
    
    def construct_clebsch():
        verts = [i for i in range(16) if bin(i).count('1') % 2 == 0]
        # Map 16 vertices to 0..15
        # Wait, the indices 0..15 are already the vertices.
        # Let's just use 0..15.
        adj = [[0]*16 for _ in range(16)]
        for u in range(16):
            for v in range(u+1, 16):
                # Hamming distance
                xor = u ^ v
                if bin(xor).count('1') == 2:
                    adj[u][v] = 1
                    adj[v][u] = 1
        return adj

    # We need 18 vertices.
    # Add 2 vertices.
    # How to add them to maintain K4-free and minimize triangle-free subgraph?
    # If we connect the new vertices to many existing ones, we might create triangles.
    # If we don't connect them, they form independent sets.
    
    # Let's try a random search approach instead, it's more robust.
    # Generate random G, check K4-free, calculate score.
    
    def random_g():
        g_adj = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    g_adj[i][j] = 1
                    g_adj[j][i] = 1
        return g_adj

    def repair_k4(g_adj):
        # Remove edges to break K4s
        changed = True
        while changed:
            changed = False
            # Find a K4
            for quad in itertools.combinations(range(n), 4):
                u, v, w, x = quad
                if (g_adj[u][v] and g_adj[u][w] and g_adj[u][x] and
                    g_adj[v][w] and g_adj[v][x] and
                    g_adj[w][x]):
                    # Break one edge
                    # Remove (u,v)
                    g_adj[u][v] = 0
                    g_adj[v][u] = 0
                    changed = True
                    break
        return g_adj

    # Optimize: Use bitmasks for K4 check and repair
    def check_k4_bitmask(g_mask):
        # g_mask[i] is neighbors of i
        for u in range(n):
            # For any pair v, w in N(u), if they are connected, and share a common neighbor x?
            # No, K4 is u, v, w, x where all connected.
            # Iterate u < v < w < x
            # Use bit operations.
            for v in range(u+1, n):
                if not (g_mask[u] & (1 << v)): continue
                # Common neighbors of u and v
                common = g_mask[u] & g_mask[v]
                # Remove u and v from common
                common &= ~(1 << u)
                common &= ~(1 << v)
                # Iterate w in common
                w_bit = common
                while w_bit:
                    w_idx = (w_bit & -w_bit).bit_length() - 1
                    w_bit &= w_bit - 1
                    # Common neighbors of u, v, w
                    common2 = common & g_mask[w_idx]
                    # Check if any x in common2
                    if common2:
                        return True
        return False

    # This is still O(n^4) effectively.
    # For n=18, it's fast enough.
    
    # Let's run a loop
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        # Try different densities
        p = 0.5 + 0.1 * (iterations % 5) # 0.5 to 0.9
        g_adj = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    g_adj[i][j] = 1
                    g_adj[j][i] = 1
        
        # Repair
        # Instead of full repair, just check if K4-free. If not, skip or repair.
        # Repairing is expensive. Let's just generate and check.
        # If K4-free, calculate score.
        
        # Fast K4 check
        is_k4 = False
        for u in range(n):
            for v in range(u+1, n):
                if not g_adj[u][v]: continue
                # Check if u, v have common neighbors w, x such that w-x edge exists
                # And u-w, u-x, v-w, v-x edges exist.
                # This is checking for triangles in the subgraph induced by N(u) intersect N(v)?
                # No.
                # Let's just use the simple loop for check, it's fast for n=18.
                pass
        
        # Use the simple is_k4_free_g which uses H complement?
        # No, let's just use the direct check.
        
        # Direct K4 check
        k4_found = False
        for u in range(n):
            for v in range(u+1, n):
                if not g_adj[u][v]: continue
                for w in range(v+1, n):
                    if not (g_adj[u][w] and g_adj[v][w]): continue
                    for x in range(w+1, n):
                        if g_adj[u][x] and g_adj[v][x] and g_adj[w][x]:
                            k4_found = True
                            break
                    if k4_found: break
                if k4_found: break
            if k4_found: break
        
        if k4_found:
            continue
            
        # Calculate score
        score = max_triangle_free_size(g_adj)
        
        if score < best_score:
            best_score = score
            best_g_adj = [row[:] for row in g_adj]
            best_edges = adj_to_edges(g_adj)
            if best_score <= 5: # Good enough
                break
                
        # If time is running out, ensure we have a valid result
        if time.time() > deadline - 5:
            break
            
    if not best_edges:
        # Fallback to K_{6,6,6}
        g_adj = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                pi = i // 6
                pj = j // 6
                if pi != pj:
                    g_adj[i][j] = 1
                    g_adj[j][i] = 1
        best_edges = adj_to_edges(g_adj)
        
    return best_edges

# Run construct to test? No, just define it.
# The evaluator calls construct().

if __name__ == "__main__":
    # Quick test
    edges = construct()
    print(f"Found {len(edges)} edges")
