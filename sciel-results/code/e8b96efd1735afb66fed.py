"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time

# EVOLVE-BLOCK-START
def construct():
    n = 40
    target_edges = 401
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # Represent graph as adjacency bitmask: adj[i] is an integer where bit j is set if edge (i,j) exists
    # Edge (u,v) with u<v: represented by index u*n + v or we use a 2D structure
    # For speed, use list of ints as bitmasks
    
    def init_adj():
        adj = [0] * n
        return adj
    
    def add_edge(adj, u, v):
        if u > v:
            u, v = v, u
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    def remove_edge(adj, u, v):
        if u > v:
            u, v = v, u
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    
    def has_edge(adj, u, v):
        if u > v:
            u, v = v, u
        return (adj[u] >> v) & 1
    
    def count_edges(adj):
        total = 0
        for i in range(n):
            total += bin(adj[i]).count('1')
        return total // 2
    
    def max_book(adj):
        """Compute the maximum book size using bitset operations."""
        max_bk = 0
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    # Common neighbors = adj[u] & adj[v], count bits
                    common = adj[u] & adj[v]
                    bk = bin(common).count('1')
                    if bk > max_bk:
                        max_bk = bk
        return max_bk
    
    def get_edge_list(adj):
        """Return list of edges as (u,v) tuples with u<v."""
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append((u, v))
        return edges
    
    # Initialize: K_{20,20} + one intra-partition edge
    adj = init_adj()
    for a in range(20):
        for b in range(20, 40):
            add_edge(adj, a, b)
    add_edge(adj, 0, 1)
    
    current_score = max_book(adj)
    best_adj = adj[:]
    best_score = current_score
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    cooling = 0.9995
    
    while time.time() < deadline and T > T_min:
        # Generate a random move: add one edge and remove one edge
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
        
        # Check if this is a non-edge to add
        if has_edge(adj, u, v):
            continue
        
        # Find a random edge to remove
        # Sample from existing edges
        while True:
            ru = random.randint(0, n - 1)
            rv = random.randint(0, n - 1)
            if ru == rv:
                continue
            if ru > rv:
                ru, rv = rv, ru
            if has_edge(adj, ru, rv):
                break
        
        # Apply move
        add_edge(adj, u, v)
        remove_edge(adj, ru, rv)
        
        new_score = max_book(adj)
        delta = new_score - current_score
        
        if delta <= 0 or random.random() < pow(2.718281828, -delta / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj = adj[:]
        else:
            # Revert
            remove_edge(adj, u, v)
            add_edge(adj, ru, rv)
        
        T *= cooling
    
    # Convert best_adj to edge list
    result = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                result.append([u, v])
    
    return result
# EVOLVE-BLOCK-END
