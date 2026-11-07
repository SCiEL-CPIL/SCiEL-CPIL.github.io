import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5 second safety margin
    
    n = 15
    
    # We'll use a greedy approach with local search
    # Start with a star or some initial graph, then try to add edges
    
    # Key constraint: For every path s-u-v-t (simple, length 3), deg(s) != deg(t)
    # This means: for edge (u,v), no degree value can appear in both N(u)\{v} and N(v)\{u}
    
    def is_valid(adj, deg):
        """Check if the graph satisfies the constraint."""
        for u in range(n):
            for v in adj[u]:
                if v <= u:
                    continue
                # For edge (u,v), check that no degree appears in both N(u)\{v} and N(v)\{u}
                degs_u = set()
                for s in adj[u]:
                    if s != v:
                        degs_u.add(deg[s])
                for t in adj[v]:
                    if t != u:
                        if deg[t] in degs_u:
                            # Check if there exist s != t with deg(s) = deg(t)
                            # We need to verify s != t
                            found = False
                            for s in adj[u]:
                                if s != v and deg[s] == deg[t] and s != t:
                                    found = True
                                    break
                            if found:
                                return False
        return True
    
    def is_valid_fast(adj, deg):
        """Faster validity check."""
        for u in range(n):
            for v in adj[u]:
                if v <= u:
                    continue
                # Build set of (degree, vertex) for N(u)\{v}
                u_side = {}
                for s in adj[u]:
                    if s != v:
                        d = deg[s]
                        if d not in u_side:
                            u_side[d] = []
                        u_side[d].append(s)
                for t in adj[v]:
                    if t != u:
                        d = deg[t]
                        if d in u_side:
                            # Check if there's s in u_side[d] with s != t
                            for s in u_side[d]:
                                if s != t:
                                    return False
        return True
    
    # Greedy construction: try to add edges in a smart order
    # Start with empty graph
    adj = [[] for _ in range(n)]
    deg = [0] * n
    edge_set = set()
    
    # Try all possible edges, add if valid
    # Order: try to maximize edges, so try all pairs
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    # Shuffle for diversity, but also try multiple orders
    random.seed(42)
    
    best_edges = 0
    best_adj = None
    best_deg = None
    
    for attempt in range(200):
        if time.time() > deadline - 10:
            break
            
        adj = [[] for _ in range(n)]
        deg = [0] * n
        edge_set = set()
        
        # Try edges in random order
        edges_to_try = all_edges[:]
        random.shuffle(edges_to_try)
        
        for u, v in edges_to_try:
            if time.time() > deadline - 10:
                break
            # Try adding edge (u, v)
            adj[u].append(v)
            adj[v].append(u)
            deg[u] += 1
            deg[v] += 1
            edge_set.add((u, v))
            
            if is_valid_fast(adj, deg):
                pass  # Keep the edge
            else:
                # Remove the edge
                adj[u].remove(v)
                adj[v].remove(u)
                deg[u] -= 1
                deg[v] -= 1
                edge_set.discard((u, v))
        
        if len(edge_set) > best_edges:
            best_edges = len(edge_set)
            best_adj = [list(x) for x in adj]
            best_deg = list(deg)
    
    # Try local search: remove some edges and re-add
    if best_adj is None:
        # Fallback: return a star
        return [[0, v] for v in range(1, n)]
    
    # Convert to list of edges
    edges = []
    for u in range(n):
        for v in best_adj[u]:
            if u < v:
                edges.append([u, v])
    
    # Verify
    adj_final = [[] for _ in range(n)]
    deg_final = [0] * n
    for u, v in edges:
        adj_final[u].append(v)
        adj_final[v].append(u)
        deg_final[u] += 1
        deg_final[v] += 1
    
    if not is_valid_fast(adj_final, deg_final):
        # Fallback to star
        return [[0, v] for v in range(1, n)]
    
    return edges
