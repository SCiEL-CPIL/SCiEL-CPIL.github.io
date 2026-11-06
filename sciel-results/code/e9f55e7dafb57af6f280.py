"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time

# EVOLVE-BLOCK-START

N = 18
DEADLINE_MARGIN = 5.0  # safety margin in seconds

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN
    rng = random.Random(42)
    
    # We represent G (the K4-free graph) as a 18x18 boolean adjacency matrix.
    # G must be K4-free. We want to minimize the size of the largest triangle-free induced subgraph.
    # Equivalent: maximize the size of the largest K3 (triangle) in G? No.
    # A triangle-free induced subgraph is a set of vertices S such that G[S] has no triangles.
    # We want to minimize max{|S| : G[S] is triangle-free}.
    
    # Strategy: Start with a random K4-free graph, then use local search to reduce the max triangle-free induced subgraph size.
    # A K4-free graph can be constructed by ensuring no 4-cliques.
    
    # Let's use a different representation: the complement H = complement of G.
    # G is K4-free <=> H is triangle-free.
    # G[S] is triangle-free <=> H[S] is K3-free (triangle-free)? No.
    # G[S] triangle-free means no triangles in G[S].
    # Triangles in G correspond to independent sets of size 3 in H.
    # So G[S] is triangle-free <=> H[S] has no independent set of size 3 <=> H[S] has independence number <= 2.
    # We want to minimize the max size of S such that H[S] has alpha(H[S]) <= 2.
    # Wait, that's not right either.
    
    # Let's stick to G.
    # G is K4-free.
    # We want to minimize max{|S| : G[S] is triangle-free}.
    
    # A good initial structure: Complete tripartite graph K6,6,6.
    # Parts: A={0..5}, B={6..11}, C={12..17}.
    # Edges between different parts, no edges within parts.
    # This is K4-free (max clique size is 3).
    # Triangle-free induced subgraphs:
    # - Any subset of a single part is triangle-free (max size 6).
    # - Two vertices from different parts: triangle-free (size 2).
    # - One from each part: forms a triangle (not triangle-free).
    # - Two from A, one from B: triangle-free (size 3).
    # - So max triangle-free induced subgraph size is 6.
    
    # Can we do better than 6?
    # The seed program gives score 1.0, which corresponds to some value.
    # The task says "score is the largest number of vertices inducing a triangle-free subgraph".
    # For K6,6,6, this is 6.
    
    # Let's try to find a K4-free graph with smaller max triangle-free induced subgraph.
    # Random search + local improvement.
    
    def is_k4_free(adj):
        """Check if graph with adjacency matrix adj is K4-free."""
        for a in range(N):
            for b in range(a+1, N):
                if not adj[a][b]:
                    continue
                for c in range(b+1, N):
                    if not adj[b][c]:
                        continue
                    if not adj[a][c]:
                        continue
                    # a,b,c form a triangle. Check if any d connects to all three.
                    for d in range(c+1, N):
                        if adj[a][d] and adj[b][d] and adj[c][d]:
                            return False
        return True
    
    def max_triangle_free_size(adj):
        """Find the size of the largest triangle-free induced subgraph."""
        # Brute force for small n, but n=18 is too large for full brute force.
        # Use a heuristic or branch and bound.
        # For now, use a greedy approach or random sampling.
        # Actually, let's use a proper branch and bound for accuracy in the evaluator.
        # But for the search, we need speed.
        # Let's use a randomized local search to estimate, and verify at the end.
        
        # Simple upper bound: 18
        # Let's try to find a large triangle-free subset.
        # Greedy: start with all vertices, remove vertices involved in triangles? No, that's for independent sets.
        
        # Actually, let's just compute it exactly for the final answer, and use a proxy for the search.
        # Proxy: number of edges in the complement of the largest triangle-free subgraph?
        
        # Let's implement a fast exact solver for n=18.
        # We can use the fact that a triangle-free graph has at most floor(n^2/4) edges (Mantel's theorem).
        # But we need the max size of a triangle-free induced subgraph.
        
        # Let's use a recursive backtracking with pruning.
        # This might be slow, so let's cache or use a heuristic for the search loop.
        
        # For the search loop, let's use a heuristic:
        # Randomly sample subsets and check if they are triangle-free.
        # Keep track of the max size found.
        
        # Actually, let's use a different approach for the search:
        # We want to minimize the max triangle-free size.
        # Let's try to make the graph as "clique-like" as possible, but K4-free.
        # So we want many triangles, but no K4s.
        
        # Let's start with a random K4-free graph.
        
        return -1  # placeholder
    
    # Let's create a random K4-free graph.
    # Method: Start with empty graph, add edges randomly, ensuring no K4 is formed.
    
    def random_k4_free_graph():
        adj = [[False]*N for _ in range(N)]
        edges = []
        for u in range(N):
            for v in range(u+1, N):
                if rng.random() < 0.5:
                    # Check if adding (u,v) creates a K4.
                    # A K4 would require two other vertices w,x that are connected to u,v and to each other.
                    # Actually, it's enough to check if there's a triangle containing u,v that can be extended.
                    # Simpler: check if there exist w,x such that adj[u][w], adj[v][w], adj[u][x], adj[v][x], adj[w][x] are all true.
                    # This is O(N^2) per edge, so O(N^4) total, which is 18^4 = 104976, acceptable.
                    
                    creates_k4 = False
                    for w in range(N):
                        if w == u or w == v:
                            continue
                        if not adj[u][w] or not adj[v][w]:
                            continue
                        for x in range(w+1, N):
                            if x == u or x == v:
                                continue
                            if adj[u][x] and adj[v][x] and adj[w][x]:
                                creates_k4 = True
                                break
                        if creates_k4:
                            break
                    if not creates_k4:
                        adj[u][v] = True
                        adj[v][u] = True
                        edges.append([u, v])
        return adj, edges
    
    # Local search: try to flip edges (add or remove) to reduce the max triangle-free size.
    # Flipping an edge must maintain K4-free property.
    
    def max_triangle_free_size_exact(adj):
        """Exact computation of the largest triangle-free induced subgraph."""
        # Use branch and bound or DP.
        # For n=18, we can use inclusion-exclusion or backtracking.
        # Let's use a recursive approach with memoization on subsets? 2^18 = 262144, feasible.
        
        # Triangle-free means no three vertices form a triangle.
        # We want the max size of a subset S such that G[S] has no triangles.
        
        # Let's use a simple backtracking:
        # Sort vertices. Build the subset incrementally.
        # At each step, decide to include or exclude the next vertex.
        # Prune if the current size + remaining vertices <= best found.
        # Also prune if adding a vertex creates a triangle with two already included vertices.
        
        best = [0]
        
        def is_triangle_free(subset):
            # Check if the induced subgraph on subset is triangle-free.
            # For each triple in subset, check if it forms a triangle.
            for i in range(len(subset)):
                for j in range(i+1, len(subset)):
                    for k in range(j+1, len(subset)):
                        a, b, c = subset[i], subset[j], subset[k]
                        if adj[a][b] and adj[b][c] and adj[a][c]:
                            return False
            return True
        
        # This is O(n^3) per check, and there are 2^n subsets, so O(n^3 * 2^n) = 18^3 * 262144 ~ 1.5e9, too slow.
        
        # Better: build the subset incrementally.
        # When adding a vertex v to a current subset S, check if v forms a triangle with any pair in S.
        # A pair (u,w) in S forms a triangle with v if adj[u][v] and adj[w][v] and adj[u][w].
        
        # Let's do DFS:
        # State: (current_subset, next_vertex_to_consider)
        # To avoid duplicates, we only consider vertices in increasing order.
        
        max_size = 0
        
        def dfs(current_subset, start_idx):
            nonlocal max_size
            if len(current_subset) + (N - start_idx) <= max_size:
                return
            if start_idx == N:
                if len(current_subset) > max_size:
                    max_size = len(current_subset)
                return
            
            # Option 1: Don't include start_idx
            dfs(current_subset, start_idx + 1)
            
            # Option 2: Include start_idx, if it doesn't create a triangle
            v = start_idx
            can_add = True
            for u in current_subset:
                for w in current_subset:
                    if u == w:
                        continue
                    if adj[u][v] and adj[w][v] and adj[u][w]:
                        can_add = False
                        break
                if not can_add:
                    break
            
            if can_add:
                dfs(current_subset + [v], start_idx + 1)
        
        dfs([], 0)
        return max_size
    
    # This exact solver might be slow for the search loop.
    # Let's use a heuristic for the search:
    # Count the number of "bad" configurations, or use a sampling-based estimate.
    
    # For now, let's use the exact solver but with a timeout.
    
    def max_triangle_free_size_heuristic(adj):
        """Heuristic estimate of the max triangle-free induced subgraph size."""
        # Try multiple random starting points and grow the triangle-free set.
        best = 0
        for _ in range(100):
            # Start with a random vertex
            start = rng.randint(0, N-1)
            current = [start]
            for v in range(N):
                if v == start:
                    continue
                # Check if v can be added to current
                can_add = True
                for u in current:
                    for w in current:
                        if u == w:
                            continue
                        if adj[u][v] and adj[w][v] and adj[u][w]:
                            can_add = False
                            break
                    if not can_add:
                        break
                if can_add:
                    current.append(v)
            if len(current) > best:
                best = len(current)
        return best
    
    # Search loop
    best_adj = None
    best_edges = None
    best_score = float('inf')
    
    # Initialize with K6,6,6
    adj = [[False]*N for _ in range(N)]
    edges = []
    parts = [range(0, 6), range(6, 12), range(12, 18)]
    for i in range(3):
        for j in range(i+1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    adj[a][b] = True
                    adj[b][a] = True
                    edges.append([a, b])
    
    best_adj = [row[:] for row in adj]
    best_edges = [e[:] for e in edges]
    best_score = max_triangle_free_size_exact(best_adj)
    
    # Try random initial graphs
    for init_iter in range(50):
        if time.time() > deadline:
            break
        adj, edges = random_k4_free_graph()
        score = max_triangle_free_size_heuristic(adj)
        if score < best_score:
            exact_score = max_triangle_free_size_exact(adj)
            if exact_score < best_score:
                best_score = exact_score
                best_adj = [row[:] for row in adj]
                best_edges = [e[:] for e in edges]
    
    # Local search
    if time.time() < deadline:
        current_adj = [row[:] for row in best_adj]
        current_score = best_score
        
        for iter in range(500):
            if time.time() > deadline:
                break
            # Try flipping a random edge
            u = rng.randint(0, N-1)
            v = rng.randint(0, N-1)
            if u == v:
                continue
            
            new_adj = [row[:] for row in current_adj]
            new_adj[u][v] = not new_adj[u][v]
            new_adj[v][u] = new_adj[u][v]
            
            # Check if still K4-free
            if not is_k4_free(new_adj):
                continue
            
            new_score = max_triangle_free_size_heuristic(new_adj)
            if new_score < current_score:
                exact_score = max_triangle_free_size_exact(new_adj)
                if exact_score < current_score:
                    current_score = exact_score
                    current_adj = new_adj
                    if current_score < best_score:
                        best_score = current_score
                        best_adj = [row[:] for row in current_adj]
                        # Rebuild edges
                        best_edges = []
                        for a in range(N):
                            for b in range(a+1, N):
                                if current_adj[a][b]:
                                    best_edges.append([a, b])
    
    # Return the best graph found
    if best_edges is not None:
        return best_edges
    else:
        # Fallback to K6,6,6
        parts = [range(0, 6), range(6, 12), range(12, 18)]
        return [[a, b] for i in range(3) for j in range(i + 1, 3) for a in parts[i] for b in parts[j]]

# EVOLVE-BLOCK-END
