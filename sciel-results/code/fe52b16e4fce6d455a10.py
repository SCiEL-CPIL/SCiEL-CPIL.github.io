import random
import time

def construct():
    deadline = time.time() + 145
    n = 40
    m = 401
    
    # 1. Construct a base graph using a Paley-type design on Z_41
    # Vertices 0..39. Edge (u, v) exists if (u - v) mod 41 is a quadratic residue mod 41.
    # Since 41 is prime, the quadratic residues form a difference set.
    # The graph on 40 vertices (excluding 40) is a circulant graph.
    # Number of edges in the full Paley graph on 41 vertices is 41 * 10 / 2 = 205? No.
    # For p=41, p=1 mod 4. The Paley graph is self-complementary.
    # Degree is (p-1)/2 = 20.
    # Total edges in G_41 is 41 * 20 / 2 = 410.
    # We remove vertex 40.
    # Edges incident to 40: (40, v) for v in 0..39. (40-v) mod 41 is a QR.
    # Since -1 is a QR mod 41 (41=1 mod 4), if (40-v) is QR, then (v-40) is QR.
    # So the graph is regular.
    # Let's just compute the edges among 0..39.
    
    # Compute quadratic residues mod 41
    QR = set()
    for x in range(1, 41):
        QR.add((x * x) % 41)
    
    base_edges = []
    adj = [set() for _ in range(n)]
    
    for u in range(n):
        for v in range(u + 1, n):
            diff = (u - v) % 41
            if diff in QR:
                base_edges.append((u, v))
                adj[u].add(v)
                adj[v].add(u)
                
    # Check number of edges
    # In a Paley graph P_41, every vertex has degree 20.
    # Vertex 40 is connected to 20 vertices in 0..39.
    # The subgraph induced by 0..39 has 40 vertices.
    # Sum of degrees in full graph = 41 * 20 = 820.
    # Sum of degrees in subgraph = 820 - 2 * (deg of 40) = 820 - 40 = 780.
    # Number of edges = 780 / 2 = 390.
    # So we have 390 edges. We need 401. We need to add 11 edges.
    
    # The base graph is a (40, 390) graph.
    # Max book size in Paley graphs is small.
    # For p=41, the max number of common neighbors for any edge is small.
    # In a strongly regular graph (which Paley graphs are), the number of common neighbors
    # for adjacent vertices is lambda.
    # For Paley p, lambda = (p - 5) / 4 = (41 - 5) / 4 = 9.
    # So the max book size of the base graph is 9.
    
    # We need to add 11 edges.
    # Adding an edge (u, v) increases the book size of (u, v) to common_neighbors(u, v).
    # It also increases the book size of edges (u, w) and (v, w) where w is a common neighbor?
    # No, book size of edge (a, b) is |N(a) intersection N(b)|.
    # If we add (u, v):
    # 1. The book size of (u, v) becomes |N(u) intersection N(v)|.
    # 2. For any edge (u, w) where w is in N(u), the book size increases by 1 if v is in N(w).
    #    i.e., if w is a common neighbor of u and v.
    # 3. Similarly for edges (v, w).
    # 4. For edges not incident to u or v, the book size does not change.
    
    # Current max book is 9.
    # If we add (u, v), the new book size for (u, v) is the number of common neighbors.
    # In the base graph, common neighbors of non-adjacent vertices can vary.
    # In a strongly regular graph (v, k, lambda, mu), the number of common neighbors
    # for adjacent vertices is lambda (9), and for non-adjacent vertices is mu.
    # mu = (p - 1) / 4 = 40 / 4 = 10.
    # So if we add an edge between non-adjacent vertices u, v, the book size of (u, v) becomes 10.
    # This is already higher than the current max of 9.
    # So the max book will be at least 10.
    
    # Can we do better?
    # The theorem says score >= 40/6 = 6.66, so 7 is the lower bound.
    # The base graph has max book 9.
    # Adding edges will likely increase it.
    
    # Strategy: Add 11 edges that minimize the maximum increase in book size.
    # We should add edges between vertices with few common neighbors?
    # But in the base graph, all non-adjacent pairs have exactly 10 common neighbors.
    # So any edge we add will have a book size of 10.
    # Thus, the max book size will be at least 10.
    
    # Let's verify this logic.
    # If we add 11 edges, we create 11 new edges with book size 10.
    # Do they affect other edges?
    # Adding (u, v) increases book size of (u, w) and (v, w) for w in N(u) cap N(v).
    # Since |N(u) cap N(v)| = 10, there are 10 such w's.
    # The book size of (u, w) was 9 (since u, w adjacent in base graph).
    # It becomes 9 + 1 = 10?
    # Wait, is v a common neighbor of u and w?
    # v is a neighbor of u (by construction).
    # v is a neighbor of w? Yes, w is in N(u) cap N(v), so w is neighbor of v.
    # So v is in N(u) and N(w).
    # So the book size of (u, w) increases by 1.
    # So edges incident to u or v that share a common neighbor with the new edge will have book size 10.
    
    # So it seems difficult to keep the max book below 10.
    # Let's try to construct the 11 edges and see if we can keep it at 10.
    
    # Candidate edges: all non-edges in the base graph.
    non_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if v not in adj[u]:
                non_edges.append((u, v))
                
    # We need to pick 11 edges from non_edges.
    # Let's try a greedy approach: pick the edge that minimizes the resulting max book.
    # Since all non-edges have 10 common neighbors, the new edge itself will have book 10.
    # The neighbors will go from 9 to 10.
    # So the max book will be 10.
    
    # Is it possible to get 9?
    # Only if we don't add any edges? But we need 401 edges.
    # Base has 390.
    # So we must add 11.
    # Any added edge has book size 10.
    # So the max book is at least 10.
    
    # Therefore, the optimal score for this construction is likely 10.
    # Let's just add 11 arbitrary non-edges.
    
    edges = list(base_edges)
    # Add 11 non-edges
    for i in range(11):
        edges.append(non_edges[i])
        
    # Update adjacency for the new edges
    for u, v in non_edges[:11]:
        adj[u].add(v)
        adj[v].add(u)
        
    # Compute max book
    max_book = 0
    for u, v in edges:
        common = len(adj[u] & adj[v])
        if common > max_book:
            max_book = common
            
    # It should be 10.
    # Let's try to optimize further?
    # Maybe a different algebraic construction?
    # Or maybe local search to reduce the max book?
    # If we swap an edge, can we reduce the max book?
    # The max book is 10.
    # If we remove an edge with book 10 and add a different one, will the max book drop?
    # Probably not, because the structure is highly symmetric.
    
    # Let's try a simple local search to see if we can get to 9 or lower.
    # But 9 might be impossible if the lower bound is higher.
    # Edwards and Khadzhiivanov-Nikiforov: score >= n/6.
    # For n=40, 40/6 = 6.66.
    # For n=40, m=401.
    # The theorem for m > n^2/4 (which is 400) gives a book of size at least 7.
    # So 7 is the theoretical lower bound.
    # 10 is not far from 7.
    
    # Let's try to improve the 10 by local search.
    # Randomly swap edges to see if max book decreases.
    
    current_edges = set(edges)
    current_max = max_book
    best_edges = edges
    best_max = current_max
    
    # List of all possible edges
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Time loop
    while time.time() < deadline:
        # Pick a random edge to remove
        u, v = random.choice(list(current_edges))
        
        # Pick a random non-edge to add
        # Generate random non-edge
        while True:
            a, b = random.choice(all_possible)
            if (a, b) not in current_edges:
                break
                
        # Check if swap improves
        # Remove (u, v), add (a, b)
        
        # We need to update adjacency and recompute max book?
        # That's expensive.
        
        # Heuristic:
        # Book size of (u, v) is common_uv.
        # Book size of (a, b) is common_ab.
        # If common_ab < current_max, it might be good.
        # But it also affects neighbors.
        
        # Let's just do a full recompute for a few random swaps.
        
        # Remove u-v
        adj[u].discard(v)
        adj[v].discard(u)
        
        # Add a-b
        adj[a].add(b)
        adj[b].add(a)
        
        # Compute max book
        new_max = 0
        # We can just check the edges affected by the change?
        # Edges incident to u, v, a, b.
        # And the new edge (a, b).
        
        # Actually, just computing all edges is 401 iterations, which is fast.
        for x, y in current_edges:
             # Note: current_edges still has (u, v) and not (a, b) in the set?
             # I modified adj but not the set.
             pass
             
        # Let's be careful.
        # Create a temporary set of edges
        temp_edges = set(current_edges)
        temp_edges.remove((u, v))
        temp_edges.add((a, b))
        
        # Compute max book for temp_edges
        # We can use the updated adj.
        for x, y in temp_edges:
            common = len(adj[x] & adj[y])
            if common > new_max:
                new_max = common
                if new_max >= best_max:
                    break
                    
        if new_max < best_max:
            best_max = new_max
            best_edges = list(temp_edges)
            current_edges = temp_edges
        else:
            # Revert
            adj[u].add(v)
            adj[v].add(u)
            adj[a].discard(b)
            adj[b].discard(a)
            # current_edges unchanged
            
    return [[u, v] for u, v in best_edges]

construct()
