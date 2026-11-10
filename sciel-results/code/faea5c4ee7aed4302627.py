"""Program for dense odd-order graphs avoiding equal-degree path endpoints."""

# EVOLVE-BLOCK-START
import random

P = {'n': 15}

def construct():
    n = P['n']
    
    def get_degrees(adj):
        return [sum(adj[i]) for i in range(n)]
    
    def check_valid(adj, n):
        """Check if any simple 3-edge path s-u-v-t has deg(s)==deg(t)."""
        deg = get_degrees(adj)
        # For each edge (u,v), check all s in N(u)\{v} and t in N(v)\{u}
        for u in range(n):
            for v in range(u+1, n):
                if not adj[u][v]:
                    continue
                for s in range(n):
                    if s == u or s == v:
                        continue
                    if not adj[s][u]:
                        continue
                    for t in range(n):
                        if t == s or t == u or t == v:
                            continue
                        if not adj[v][t]:
                            continue
                        if deg[s] == deg[t]:
                            return False
        return True

    def count_edges(adj):
        return sum(sum(row) for row in adj) // 2

    best_edges = 0
    best_adj = None
    
    # Try multiple random initializations
    for _ in range(20):
        adj = [[False] * n for _ in range(n)]
        
        # Random initial graph
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i][j] = adj[j][i] = True
        
        # Greedily add edges
        added = True
        while added:
            added = False
            # Shuffle order to break symmetry
            edges_to_try = [(i, j) for i in range(n) for j in range(i+1, n) if not adj[i][j]]
            random.shuffle(edges_to_try)
            
            for i, j in edges_to_try:
                if adj[i][j]:
                    continue
                adj[i][j] = adj[j][i] = True
                if check_valid(adj, n):
                    added = True
                    break
                else:
                    adj[i][j] = adj[j][i] = False
        
        # Try to add more edges with different order?
        # The loop above adds one edge at a time.
        
        curr_edges = count_edges(adj)
        if curr_edges > best_edges:
            best_edges = curr_edges
            best_adj = [row[:] for row in adj]

    # Build edge list from best_adj
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i][j]:
                edges.append([i, j])
    
    # Final validation
    assert check_valid(best_adj, n), "Graph is invalid!"
    
    # Convert to proper Python ints
    edges = [[int(u), int(v)] for u, v in edges]
    return edges
# EVOLVE-BLOCK-END
