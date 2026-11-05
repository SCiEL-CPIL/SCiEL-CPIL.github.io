"""C6-free graph construction on 40 vertices via local search with kicks."""
import time
import random
from itertools import combinations

def _has_c6(adj, n):
    """Check if graph with adjacency matrix adj (n x n bool) contains a C6.
    Uses a fast check: for each pair (u,v), count common neighbors, etc.
    Actually, a more efficient approach: enumerate all paths of length 5 and check if endpoints are adjacent.
    But that's expensive. Let's use a smarter method.
    
    A C6 exists if there exist distinct vertices v0,v1,v2,v3,v4,v5 such that
    v0-v1-v2-v3-v4-v5-v0 are all edges.
    
    Efficient check: For each unordered pair {a,b} of non-adjacent vertices,
    check if there's a path of length 5 between them. That's still expensive.
    
    Alternative: For each triple of edges forming a path of length 2 (i.e., a 'V' shape),
    ... this is getting complex.
    
    Let's use the standard approach: for each pair of vertices (u,v) that are NOT adjacent,
    check if there exists a path u - a - b - c - d - v where all intermediate vertices are distinct
    and not equal to u or v.
    
    Actually, the most practical approach for n=40 is to use the following:
    For each ordered pair (u, v) with u < v and not adjacent, count the number of paths of length 5 from u to v.
    If any such count > 0, we have a C6.
    
    To count paths of length 5 from u to v:
    paths_5(u,v) = sum over a,b,c,d: adj[u][a] * adj[a][b] * adj[b][c] * adj[c][d] * adj[d][v]
    where a,b,c,d are all distinct and not equal to u or v.
    
    This is O(n^5) per pair, which is 40^5 = 102M per pair, and ~800 pairs, so ~80 billion operations. Too slow.
    
    Better approach: Precompute for each vertex u, the set of vertices at distance 2, 3, 4 from u.
    Then a C6 through u-v (non-edge) exists if there's a path of length 5 from u to v.
    
    Actually, let me use a different strategy. Since we're doing local search and need to check
    C6-freeness after each edge addition/removal, let's precompute a data structure.
    
    For adding edge (u,v): A C6 is created iff there exists a path of length 5 from u to v
    that doesn't use the edge (u,v) (which isn't there yet). So we need to check if
    dist(u,v) in the current graph (without edge uv) is exactly 5, OR if there's a path of length 5.
    
    Actually, we need: does there exist a simple path of length 5 from u to v in G - (u,v)?
    
    This can be checked by: 
    Let N(u) = neighbors of u, N(v) = neighbors of v.
    We need a path u-a-b-c-d-v.
    So a in N(u), d in N(v), and there's a path of length 3 from a to d in G - {u,v} - {a,d}... 
    
    This is still complex. Let me just do a direct BFS/DFS for paths of length exactly 5.
    
    For n=40, a simple approach:
    For each candidate edge (u,v) to add:
      Check if there's a C6 that would be formed. This happens iff there's a path of length 5
      from u to v in the current graph (which doesn't have edge uv).
      
    To check this efficiently:
      Step 1: Compute all vertices reachable from u in exactly k steps for k=1..5,
              keeping track of the path to avoid revisiting vertices.
      
    Actually for n=40, let's just do a simple check:
      For each (u,v) non-edge, check if adding it creates a C6 by checking all
      5-tuples (a,b,c,d,e) where u-a-b-c-d-e-v is a path.
      That's O(deg(u) * deg(a) * ...) which could be large.
    
    Let me use a cleaner method: 
    has_c6_after_adding(u, v, adj, n):
        # Check if there's a path of length 5 from u to v in current graph
        # Use DFS with depth limit 5
        def dfs(current, depth, visited):
            if depth == 5:
                return adj[current][v]  # need edge to v
            for nxt in range(n):
                if not visited[nxt] and adj[current][nxt]:
                    visited[nxt] = True
                    if dfs(nxt, depth + 1, visited):
                        visited[nxt] = False
                        return True
                    visited[nxt] = False
            return False
        
        # Start from neighbors of u
        for a in range(n):
            if a != u and a != v and adj[u][a]:
                visited = set([u, a])
                if dfs(a, 1, visited):
                    return True
        return False
    
    This is O(n^5) worst case but with pruning should be fast for sparse-ish graphs.
    """
    pass  # We'll implement this differently

def _count_c6s(adj, n):
    """Count number of C6s in the graph. Used to verify C6-freeness."""
    count = 0
    # For each 6-tuple of distinct vertices, check if they form a C6
    # This is O(n^6) which is 40^6 = 4 billion, too slow.
    # Better: for each pair of non-adjacent vertices (u,v), count paths of length 5.
    # A C6 corresponds to a pair (u,v) non-adjacent with a path of length 5 between them.
    # But each C6 is counted multiple times (12 times: 6 starting vertices * 2 directions).
    
    # Let's use: for each ordered pair (u,v) with u<v and not adj[u][v],
    # count simple paths of length 5 from u to v.
    # Sum all such counts. If total > 0, graph has C6.
    
    # For efficiency, let's do it smarter:
    # For each triple (a,b,c) where a-b-c is a path (a!=c, a-b and b-c are edges, a!=b, b!=c),
    # and for each triple (d,e,f) where d-e-f is a path,
    # check if we can connect them to form a 6-cycle.
    
    # Actually, let's just verify C6-freeness using the following approach:
    # A graph is C6-free iff for every pair of non-adjacent vertices (u,v),
    # there is no path of length 5 between them.
    
    # For n=40, let's use a matrix approach:
    # Let A be the adjacency matrix.
    # Number of walks of length 5 from u to v is (A^5)[u][v].
    # But walks can repeat vertices. We need simple paths.
    
    # For verification purposes (checking if final graph is C6-free), 
    # let's use a direct enumeration that's feasible for n=40.
    
    # Enumerate all 6-cycles: choose 6 vertices out of 40, and check if they form a C6.
    # C(40,6) = 3,838,380. For each set of 6 vertices, check if they induce a C6.
    # A set of 6 vertices induces a C6 if the induced subgraph is exactly a 6-cycle.
    # That means each vertex has degree 2 in the induced subgraph, and it's connected.
    
    for verts in combinations(range(n), 6):
        # Check if induced subgraph on verts is a C6
        # Build adjacency within these 6 vertices
        deg = [0] * 6
        edges_in = 0
        for i in range(6):
            for j in range(i+1, 6):
                if adj[verts[i]][verts[j]]:
                    deg[i] += 1
                    deg[j] += 1
                    edges_in += 1
        # A C6 has exactly 6 edges and all degrees are 2
        if edges_in == 6 and all(d == 2 for d in deg):
            count += 1
    return count

def construct():
    n = 40
    deadline = time.time() + 140  # safety margin
    
    # Start with 8 disjoint K5s
    adj = [[False]*n for _ in range(n)]
    for k in range(8):
        for a in range(5):
            for b in range(a+1, 5):
                u = 5*k + a
                v = 5*k + b
                adj[u][v] = adj[v][u] = True
    
    best_edges = 0
    best_adj = [row[:] for row in adj]
    
    def get_edge_count(adj):
        return sum(sum(row) for row in adj) // 2
    
    def is_c6_free(adj, n):
        """Check if graph is C6-free by enumerating all 6-subsets."""
        for verts in combinations(range(n), 6):
            deg = [0] * 6
            edges_in = 0
            for i in range(6):
                for j in range(i+1, 6):
                    if adj[verts[i]][verts[j]]:
                        deg[i] += 1
                        deg[j] += 1
                        edges_in += 1
            if edges_in == 6 and all(d == 2 for d in deg):
                return False
        return True
    
    def would_create_c6(adj, n, u, v):
        """Check if adding edge (u,v) creates a C6.
        This happens iff there's a path of length 5 from u to v in current graph."""
        # DFS from neighbors of u, looking for path of length 5 to v
        # Path: u - a - b - c - d - e - v where a,b,c,d,e are distinct and not u,v
        def dfs(current, depth, visited):
            if depth == 5:
                return adj[current][v]
            for nxt in range(n):
                if nxt != u and nxt != v and not visited[nxt] and adj[current][nxt]:
                    visited[nxt] = True
                    if dfs(nxt, depth + 1, visited):
                        visited[nxt] = False
                        return True
                    visited[nxt] = False
            return False
        
        for a in range(n):
            if a != u and a != v and adj[u][a]:
                visited = [False]*n
                visited[u] = True
                visited[v] = True
                visited[a] = True
                if dfs(a, 1, visited):
                    return True
        return False
    
    def try_add_edge(adj, n, u, v):
        """Try to add edge (u,v). Returns True if successful (graph remains C6-free)."""
        if adj[u][v]:
            return False
        if would_create_c6(adj, n, u, v):
            return False
        adj[u][v] = adj[v][u] = True
        return True
    
    def remove_edge(adj, n, u, v):
        adj[u][v] = adj[v][u] = False
    
    # Local search with kicks
    random.seed(42)
    
    current_adj = [row[:] for row in adj]
    current_edges = get_edge_count(current_adj)
    
    # First, greedily add edges
    for u in range(n):
        if time.time() > deadline - 30:
            break
        for v in range(u+1, n):
            if not current_adj[u][v]:
                if try_add_edge(current_adj, n, u, v):
                    current_edges += 1
    
    if current_edges > best_edges:
        best_edges = current_edges
        best_adj = [row[:] for row in current_adj]
    
    # Now do local search: try to add random edges, and if it creates C6, 
    # remove some edges to fix it, then try again
    stuck_count = 0
    max_stuck = 50
    
    while time.time() < deadline - 10:
        # Try to add a random non-edge
        improved = False
        attempts = 0
        while attempts < 200 and time.time() < deadline - 10:
            attempts += 1
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v or current_adj[u][v]:
                continue
            if try_add_edge(current_adj, n, u, v):
                current_edges += 1
                improved = True
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = [row[:] for row in current_adj]
        
        if improved:
            stuck_count = 0
        else:
            stuck_count += 1
        
        if stuck_count >= max_stuck:
            # Kick: remove a random edge and restart local search
            # Find all edges
            edges_list = [(i,j) for i in range(n) for j in range(i+1,n) if current_adj[i][j]]
            if edges_list:
                u, v = random.choice(edges_list)
                remove_edge(current_adj, n, u, v)
                current_edges -= 1
            stuck_count = 0
    
    # Final verification
    if not is_c6_free(best_adj, n):
        # Fallback to initial construction
        adj_init = [[False]*n for _ in range(n)]
        for k in range(8):
            for a in range(5):
                for b in range(a+1, 5):
                    u = 5*k + a
                    v = 5*k + b
                    adj_init[u][v] = adj_init[v][u] = True
        best_adj = adj_init
        best_edges = get_edge_count(best_adj)
    
    # Convert to edge list
    result = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u][v]:
                result.append([u, v])
    
    return result

construct()
