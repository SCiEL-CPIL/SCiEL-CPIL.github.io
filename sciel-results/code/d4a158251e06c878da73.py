"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def construct():
    n = 15
    deadline = time.time() + 85  # safety margin before 90s kill
    
    # Precompute all possible 3-edge paths
    # A 3-edge path is s-u-v-t where (s,u),(u,v),(v,t) are edges, s!=t
    # For a given edge set, we need: for all such paths, deg(s) != deg(t)
    
    # Strategy: Start with complete graph, remove edges to satisfy constraint
    # Or: start with bipartite and try to add edges
    
    # Better: use the structure that vertices of same degree must not be at distance 3
    # Key insight: if we can partition vertices into groups where each group has
    # a unique degree, and vertices in the same group are never at distance 3, we're good.
    
    # Let's try: make a graph where degrees are as "spread out" as possible
    # and use a greedy + local search approach
    
    # First, let's try a construction: 
    # Use a "blow-up" of a small valid graph
    
    # Actually, let's just do a smart search:
    # 1. Start with K_{7,8} (56 edges)
    # 2. Try adding intra-partition edges one at a time, checking validity
    # 3. Also try removing some inter-partition edges to allow more intra-partition edges
    
    # Efficient validity check:
    # For a given adjacency matrix, compute degrees, then for each pair (s,t) with deg(s)==deg(t),
    # check if there's a 3-path between them.
    
    def check_valid(adj):
        """adj is n x n boolean matrix. Return True if valid."""
        degs = [sum(adj[i]) for i in range(n)]
        # For each pair with same degree, check if distance 3 exists
        for s in range(n):
            for t in range(s+1, n):
                if degs[s] == degs[t] and s != t:
                    # Check if there's a 3-edge path s-u-v-t
                    # u must be neighbor of s, v must be neighbor of t, u-v must be edge, u!=t, v!=s
                    found = False
                    for u in range(n):
                        if adj[s][u] and u != t:
                            for v in range(n):
                                if adj[u][v] and adj[v][t] and v != s and u != t:
                                    found = True
                                    break
                            if found:
                                break
                    if found:
                        return False
        return True
    
    # Start with K_{7,8}
    adj = [[False]*n for _ in range(n)]
    for i in range(7):
        for j in range(7, 15):
            adj[i][j] = True
            adj[j][i] = True
    
    best_edges = sum(sum(row) for row in adj) // 2
    
    # Try to add intra-partition edges
    # First, try adding all possible intra-A edges, then remove conflicts
    # Greedy: try adding each possible edge, keep if valid
    
    candidate_edges = []
    for i in range(7):
        for j in range(i+1, 7):
            candidate_edges.append((i, j))
    for i in range(7, 15):
        for j in range(i+1, 15):
            candidate_edges.append((i, j))
    
    # Shuffle for variety
    import random
    random.seed(42)
    random.shuffle(candidate_edges)
    
    for (i, j) in candidate_edges:
        if time.time() > deadline:
            break
        if not adj[i][j]:
            # Try adding
            adj[i][j] = True
            adj[j][i] = True
            if check_valid(adj):
                best_edges += 1
            else:
                adj[i][j] = False
                adj[j][i] = False
    
    # Local search: try swapping - remove one edge, add another
    all_edges = []
    for i in range(n):
        for j in range(i+1, n):
            all_edges.append((i, j))
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing an edge and adding a different one
        for (i, j) in all_edges:
            if time.time() > deadline:
                break
            if adj[i][j]:
                # Try removing (i,j)
                adj[i][j] = False
                adj[j][i] = False
                for (k, l) in all_edges:
                    if time.time() > deadline:
                        break
                    if not adj[k][l] and (k,l) != (i,j) and (k,l) != (j,i):
                        adj[k][l] = True
                        adj[l][k] = True
                        if check_valid(adj):
                            best_edges += 1
                            improved = True
                            break
                        else:
                            adj[k][l] = False
                            adj[l][k] = False
                # Restore (i,j) if we didn't improve
                if not improved or not adj[k][l]:
                    adj[i][j] = True
                    adj[j][i] = True
    
    # Build edge list
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                edges.append([i, j])
    
    return edges
# EVOLVE-BLOCK-END
