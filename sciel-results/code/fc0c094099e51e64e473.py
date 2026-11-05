import time
import random
import math

def construct():
    n = 20
    deadline = time.time() + 118.0  # Safety margin before 120s limit

    # Precompute all possible edges
    all_edges = []
    edge_to_idx = {}
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            all_edges.append((u, v))
            edge_to_idx[(u, v)] = idx
            idx += 1
    m_total = len(all_edges)  # 190

    # Precompute conflict structure for K2,3
    # Two edges (u,v) and (x,y) conflict if they form a K2,3 with some third edge
    # Actually, the condition is: no two vertices have 3 common neighbors.
    # So we need to check: for any two vertices a,b, |N(a) ∩ N(b)| <= 2.
    # Equivalently, we cannot have three edges (a,w1), (a,w2), (a,w3) and (b,w1), (b,w2), (b,w3)
    # where w1,w2,w3 are distinct.
    # 
    # A simpler way: when adding edge (u,v), check that for every w, 
    # |N(u) ∩ N(w)| <= 2 and |N(v) ∩ N(w)| <= 2 (where N includes the new edge).
    #
    # Let's use a bitmask representation for adjacency.

    def init_adj():
        """Initialize with a good starting graph: 5 disjoint K4s gives 5*6=30 edges."""
        adj = [0] * n
        for k in range(5):
            base = 4 * k
            for a in range(4):
                for b in range(a + 1, 4):
                    u = base + a
                    v = base + b
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def edge_count(adj):
        count = 0
        for u in range(n):
            count += bin(adj[u]).count('1')
        return count // 2

    def is_valid_add(adj, u, v):
        """Check if adding edge (u,v) keeps graph K2,3-free."""
        # After adding (u,v), for every vertex w != u,v:
        #   |N(u) ∩ N(w)| <= 2 and |N(v) ∩ N(w)| <= 2
        # Note: N(u) now includes v, N(v) now includes u.
        
        # Check for u: for each w in N(u) (excluding v), |N(u) ∩ N(w)| must be <= 2
        # But N(u) changes: it gains v.
        # So we need: for each w != u, |N(u) ∩ N(w)| <= 2 where N(u) includes v.
        
        # More carefully:
        # For w in current N(u) (not including v since edge not added yet):
        #   new common neighbors of u and w = old common + possibly v (if v in N(w))
        # For w = v: common neighbors of u and v = N(u) ∩ N(v) (old, since v not in old N(u))
        #   Actually N(u) ∩ N(v) in old graph, plus u and v don't count as neighbors of themselves.
        #   Wait, N(u) ∩ N(v) in the new graph: N(u) = old N(u) ∪ {v}, N(v) = old N(v) ∪ {u}
        #   Common neighbors = (old N(u) ∪ {v}) ∩ (old N(v) ∪ {u}) 
        #   = (old N(u) ∩ old N(v)) ∪ (old N(u) ∩ {u}) ∪ ({v} ∩ old N(v)) ∪ ({v} ∩ {u})
        #   = (old N(u) ∩ old N(v)) since u not in old N(u), v not in old N(v), u≠v
        #   So |N(u) ∩ N(v)| = |old N(u) ∩ old N(v)|
        
        # For w in N(v) (not u):
        #   Similarly |N(v) ∩ N(w)| = |old N(v) ∩ old N(w)|
        
        # For w not in {u,v} and w not in N(u) and w not in N(v):
        #   N(u) ∩ N(w) = old N(u) ∩ N(w) (since v not in N(w) unless v-w edge exists)
        #   Actually if (v,w) is an edge, then v ∈ N(w), so v might be in N(u)∩N(w)
        
        # Let me just check all w != u,v:
        #   |N_new(u) ∩ N_new(w)| <= 2
        #   |N_new(v) ∩ N_new(w)| <= 2
        
        # N_new(u) = adj[u] | (1 << v)
        # N_new(v) = adj[v] | (1 << u)
        
        nu_new = adj[u] | (1 << v)
        nv_new = adj[v] | (1 << u)
        
        for w in range(n):
            if w == u or w == v:
                continue
            nw = adj[w]
            # |N_new(u) ∩ N_new(w)|
            common_uw = nu_new & nw
            if bin(common_uw).count('1') > 2:
                return False
            # |N_new(v) ∩ N_new(w)|
            common_vw = nv_new & nw
            if bin(common_vw).count('1') > 2:
                return False
        
        # Also check |N_new(u) ∩ N_new(v)| <= 2
        common_uv = nu_new & nv_new
        if bin(common_uv).count('1') > 2:
            return False
        
        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    # Start with 5 disjoint K4s
    current_adj = init_adj()
    current_edges = edge_count(current_adj)
    best_adj = current_adj[:]
    best_edges = current_edges

    # Local search: try to add edges greedily, then use swaps to escape local optima
    def try_add_all(adj, edges_count):
        """Greedily add all possible edges."""
        made_progress = True
        while made_progress:
            made_progress = False
            for u, v in all_edges:
                if not ((adj[u] >> v) & 1):
                    if is_valid_add(adj, u, v):
                        add_edge(adj, u, v)
                        edges_count += 1
                        made_progress = True
        return adj, edges_count

    # Initial greedy pass
    current_adj, current_edges = try_add_all(current_adj, current_edges)
    best_adj = current_adj[:]
    best_edges = current_edges

    # Now use simulated annealing with swaps
    T = 2.0
    cooling_rate = 0.99995

    # Precompute list of edges for random selection
    # We'll randomly pick an edge to remove and an edge to add
    
    while time.time() < deadline:
        move_type = random.random()
        
        if move_type < 0.7:
            # Try to add a random non-existent edge
            u, v = random.choice(all_edges)
            if not ((current_adj[u] >> v) & 1):
                if is_valid_add(current_adj, u, v):
                    add_edge(current_adj, u, v)
                    current_edges += 1
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = current_adj[:]
        else:
            # Try a swap: remove an existing edge, add a different one
            # Pick a random existing edge to remove
            # First collect existing edges
            existing = []
            for u, v in all_edges:
                if (current_adj[u] >> v) & 1:
                    existing.append((u, v))
            
            if not existing:
                continue
                
            u, v = random.choice(existing)
            # Pick a random non-existent edge to add
            x, y = random.choice(all_edges)
            if (current_adj[x] >> y) & 1 or (x == u and y == v) or (x == v and y == u):
                continue
            
            # Try the swap
            new_adj = current_adj[:]
            remove_edge(new_adj, u, v)
            if is_valid_add(new_adj, x, y):
                add_edge(new_adj, x, y)
                # Edge count is the same, so we accept with some probability
                # to explore the space
                if random.random() < 0.3:  # Accept swaps to explore
                    current_adj = new_adj
                    # current_edges unchanged
        
        T *= cooling_rate
        if T < 0.01:
            T = 0.01

    # Final greedy pass on best_adj
    final_adj, final_edges = try_add_all(best_adj[:], best_edges)
    if final_edges > best_edges:
        best_adj = final_adj
        best_edges = final_edges

    # Convert to list of edges
    result = []
    for u, v in all_edges:
        if (best_adj[u] >> v) & 1:
            result.append([u, v])
    
    return result
