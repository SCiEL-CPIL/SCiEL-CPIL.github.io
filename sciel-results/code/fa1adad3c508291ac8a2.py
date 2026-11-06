import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # Leave 5s margin for 120s limit

    def time_check():
        return time.time() - start_time

    def check_triangle_free(n, adj_masks):
        """
        Returns True if the graph is triangle-free.
        """
        for u in range(n):
            for v in range(u + 1, n):
                if (adj_masks[u] >> v) & 1:
                    if adj_masks[u] & adj_masks[v]:
                        return False
        return True

    def max_independent_set_size(n, adj_masks):
        """
        Returns the size of the maximum independent set.
        Uses bitsets and branch and bound.
        """
        best = [0]
        
        # Precompute degrees for better vertex selection? 
        # For now, stick to a simple heuristic: pick vertex with max degree in candidates
        # to prune faster.
        
        def dfs(candidates, size):
            if time_check() > DEADLINE - 2:
                return
            if not candidates:
                if size > best[0]:
                    best[0] = size
                return
            
            # Upper bound
            if size + candidates.bit_count() <= best[0]:
                return
            
            # Pick a vertex
            # Heuristic: Pick the vertex in candidates with the highest degree within the candidate set.
            # This tends to prune branches where we must exclude many neighbors.
            # To keep it fast, we might just pick the first one if n is small, 
            # but for n~40, a simple max-degree heuristic is better.
            
            # Optimization: If the number of candidates is small, just pick lsb.
            # Let's try a simple "first" strategy but ordered by degree if possible?
            # Actually, calculating max degree every step is expensive.
            # Let's stick to lsb but maybe randomize the bit order? 
            # No, deterministic lsb is fast. Let's rely on the bound.
            
            lsb = candidates & -candidates
            v = lsb.bit_length() - 1
            
            # Include v
            new_candidates = candidates & ~lsb & ~adj_masks[v]
            dfs(new_candidates, size + 1)
            
            # Exclude v
            dfs(candidates & ~lsb, size)

        dfs((1 << n) - 1, 0)
        return best[0]

    def is_valid(n, adj_masks):
        if not check_triangle_free(n, adj_masks):
            return False
        alpha = max_independent_set_size(n, adj_masks)
        return alpha < 10

    def andrasfai_adj(n):
        """
        Construct the adjacency masks for the Andrásfai graph And(n).
        Vertices are 0..n-1.
        Edge (u, v) exists if (v - u) % n is in the set of quadratic residues mod n?
        Actually, Andrásfai graphs are defined on $3k+1$ or $3k+2$ vertices usually.
        For general n, we can use a circulant graph definition.
        A common triangle-free graph with alpha ~ n/3 is the circulant graph
        where i is connected to i+1 and i-1? No, that's a cycle, alpha ~ n/2.
        
        The Andrásfai graph And(k) is defined on 3k+1 vertices.
        It is a triangle-free graph with independence number k+1.
        For n=39, 39 = 3*13. And(13) has 40 vertices.
        Let's construct And(13) on 40 vertices, then remove one vertex to get 39?
        Or construct a circulant graph on n vertices with specific connection set.
        
        Let's use the definition:
        Vertices $Z_n$. Edge $(u,v)$ if $v-u \pmod n$ is a quadratic residue?
        This works for prime n. 39 is not prime.
        
        Alternative: Use the known construction for R(3,10) lower bounds.
        The known graph for n=39 is often derived from the Andrasfai graph.
        Let's try to construct a circulant graph on n vertices.
        For n=39, let's try connecting i to i+1, i+2, i+13, i+14? 
        We need to ensure triangle-free.
        
        Let's use a simpler approach: Start with a random graph and use the Andrásfai structure as a heuristic for "good" density.
        Actually, let's just use the random search but with a better initialization:
        Start with a cycle (triangle-free, high alpha) and add edges greedily.
        """
        adj = [0] * n
        # Try to build a graph similar to Andrásfai.
        # For n=39, we can try to embed a structure.
        # Let's just return empty adj for now and let the search handle it,
        # BUT we will implement a specific "Andrásfai-like" initialization.
        
        # A known triangle-free graph with alpha < n/3 is hard to construct simply.
        # However, the Andrásfai graph And(k) on 3k+1 vertices has alpha k+1.
        # For n=39, we can take And(12) which has 37 vertices? No, And(k) has 3k+1.
        # And(12) -> 37 vertices. Alpha = 13.
        # We need alpha < 10. So we need to reduce alpha significantly.
        
        # Let's stick to random search but improve the "Phase 2" to be more aggressive.
        return adj

    def search_graph(n, seed):
        random.seed(seed)
        edges = []
        adj_masks = [0] * n
        
        # Initialize with a random triangle-free graph
        # Use a random permutation of vertices to break symmetry
        vertices = list(range(n))
        random.shuffle(vertices)
        
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(pairs)
        
        # Phase 1: Random initialization
        for u, v in pairs:
            if adj_masks[u] & (1 << v): continue
            # Check if adding (u,v) creates triangle
            if adj_masks[u] & adj_masks[v]:
                continue
            # Add edge
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
            edges.append([u, v])
            
        # Check current alpha
        alpha = max_independent_set_size(n, adj_masks)
        if alpha < 10:
            return n, edges, adj_masks

        # Phase 2: Greedy improvement (Local Search)
        # Try to add edges that don't create triangles.
        # Also, try to remove edges if adding others is blocked?
        # For now, just add all safe edges.
        
        # Sort pairs by some heuristic? 
        # Let's just iterate through all possible edges again.
        for u, v in pairs:
            if time_check() > DEADLINE - 15:
                break
            if adj_masks[u] & (1 << v): continue
            if adj_masks[u] & adj_masks[v]:
                continue
            
            # Add edge
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
            edges.append([u, v])
            
            # Check alpha periodically to save time
            # If we add many edges, alpha should drop.
            # We don't need to check after EVERY edge, but it's safer.
            # Let's check every 10 edges or if time permits.
            if len(edges) % 10 == 0:
                alpha = max_independent_set_size(n, adj_masks)
                if alpha < 10:
                    return n, edges, adj_masks

        # Final check
        alpha = max_independent_set_size(n, adj_masks)
        if alpha < 10:
            return n, edges, adj_masks
        
        return None, None, None

    def local_search(n, adj_masks, edges, seed):
        """
        Perform local search on the edge set.
        Try to swap edges or add/remove to reduce alpha while maintaining triangle-free.
        """
        random.seed(seed)
        best_alpha = max_independent_set_size(n, adj_masks)
        
        # If we are already good, return
        if best_alpha < 10:
            return n, edges, adj_masks, best_alpha

        # Try to add random edges
        possible_edges = [(u, v) for u in range(n) for v in range(u+1, n) 
                          if not (adj_masks[u] & (1 << v)) and not (adj_masks[u] & adj_masks[v])]
        
        random.shuffle(possible_edges)
        
        for u, v in possible_edges:
            if time_check() > DEADLINE - 10:
                break
            
            # Add edge
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
            edges.append([u, v])
            
            new_alpha = max_independent_set_size(n, adj_masks)
            
            if new_alpha < best_alpha:
                best_alpha = new_alpha
                if new_alpha < 10:
                    return n, edges, adj_masks, new_alpha
            else:
                # Revert
                adj_masks[u] ^= (1 << v)
                adj_masks[v] ^= (1 << u)
                edges.pop()
                
        return n, edges, adj_masks, best_alpha

    best_n = 0
    best_edges = []
    
    # Try n=39
    for seed in range(50):
        if time_check() > DEADLINE - 20:
            break
        n_found, edges_found, adj_found = search_graph(39, seed)
        if n_found:
            best_n = 39
            best_edges = edges_found
            break
        
        # If search didn't find a valid one, try local search on the best candidate from search?
        # The search_graph function returns None if not valid.
        # We should modify search_graph to return the best candidate found even if invalid?
        # For simplicity, let's just run more seeds.
        
    if best_n == 0:
        # Try n=38
        for seed in range(50):
            if time_check() > DEADLINE - 10:
                break
            n_found, edges_found, adj_found = search_graph(38, seed)
            if n_found:
                best_n = 38
                best_edges = edges_found
                break

    if best_n == 0:
        # Fallback to a known small valid graph.
        # We need a graph with n >= 26 and alpha < 10.
        # The Andrasfai graph And(9) has 28 vertices and alpha 10. (Too high, need < 10).
        # And(8) has 25 vertices, alpha 9. This is valid! n=25, alpha=9 < 10.
        # Let's construct And(8).
        n = 25
        adj = [0] * n
        # And(k) construction for 3k+1 vertices.
        # k=8, n=25.
        # Vertices 0..24.
        # Edge (u,v) if v-u mod 25 is a quadratic residue?
        # Quadratic residues mod 25: 1, 4, 6, 9, 11, 14, 16, 19, 21, 24.
        # Let's verify triangle-free.
        # If x+y+z = 0 mod p and x,y,z are QR, does it form a triangle?
        # In And(k), the graph is triangle-free.
        
        qrs = [1, 4, 6, 9, 11, 14, 16, 19, 21, 24]
        edges = []
        for u in range(n):
            for d in qrs:
                v = (u + d) % n
                if u < v:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append([u, v])
        
        # Check validity
        if is_valid(n, adj):
            best_n = n
            best_edges = edges
        else:
            # If And(8) construction is flawed, use a simple cycle-based graph
            # Cycle C_25 has alpha 12. Too high.
            # Let's just return n=1 if all else fails (though score will be low)
            best_n = 1
            best_edges = []

    return {'n': best_n, 'edges': best_edges}
