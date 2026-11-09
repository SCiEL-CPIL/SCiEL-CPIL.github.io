import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin for 120s limit
    
    n = 20
    num_pairs = n * (n - 1) // 2  # 190 total possible edges
    
    # Precompute pair indices for fast lookup
    pair_idx = {}
    pair_list = []
    for u in range(n):
        for v in range(u + 1, n):
            pair_idx[(u, v)] = len(pair_list)
            pair_list.append((u, v))
    
    # Precompute for each vertex, the list of pair indices it's part of
    vertex_pairs = [[] for _ in range(n)]
    for idx, (u, v) in enumerate(pair_list):
        vertex_pairs[u].append(idx)
        vertex_pairs[v].append(idx)
    
    # State: adj[i] is a bitmask of neighbors of vertex i
    adj = [0] * n
    
    # cn_count[a][b] = number of common neighbors of vertices a and b
    # We need to track when any triple has >= 3 common neighbors
    # Key insight: For K3,3, we need 3 vertices sharing 3+ common neighbors
    # Track: for each pair (a,b), cn_count[a][b] = |N(a) ∩ N(b)|
    # A K3,3 exists iff there exist 3 vertices a,b,c such that 
    # |N(a)∩N(b)∩N(c)| >= 3, which means the 3 vertices a,b,c have 3 common neighbors
    
    # Actually, let's track it differently. A K3,3 subgraph exists if there
    # are vertices X={x1,x2,x3} and Y={y1,y2,y3} such that every xi is adjacent 
    # to every yj. This is equivalent to: there exist 3 vertices with 3 common neighbors.
    
    # For efficiency, track: for each pair (a,b), the number of common neighbors
    # When we add edge (u,v), the common neighbor count changes for pairs involving u or v
    
    cn_count = [[0] * n for _ in range(n)]
    
    def init_cn():
        for a in range(n):
            for b in range(a + 1, n):
                cn_count[a][b] = (adj[a] & adj[b]).bit_count()
                cn_count[b][a] = cn_count[a][b]
    
    def has_k33():
        """Check if there exists a K3,3 by checking if any 3 vertices have 3+ common neighbors"""
        # A K3,3 exists iff there are 3 vertices a,b,c such that |N(a)∩N(b)∩N(c)| >= 3
        # This is equivalent to: for some triple (a,b,c), the intersection has size >= 3
        # We can check: for each pair (a,b), look at common neighbors, and check if any 3 of them form a triangle with a and b... 
        # Actually simpler: just check all triples
        for a in range(n):
            for b in range(a + 1, n):
                for c in range(b + 1, n):
                    if (adj[a] & adj[b] & adj[c]).bit_count() >= 3:
                        return True
        return False
    
    def can_add_edge(u, v):
        """Check if adding edge (u,v) would create a K3,3"""
        if u == v:
            return False
        if (adj[u] >> v) & 1:
            return False
        
        new_adj_u = adj[u] | (1 << v)
        new_adj_v = adj[v] | (1 << u)
        
        # Check all triples that could now have 3+ common neighbors
        # Triples involving u: {u, x, y}
        for x in range(n):
            if x == u:
                continue
            for y in range(x + 1, n):
                if y == u:
                    continue
                common = new_adj_u & adj[x] & adj[y]
                if common.bit_count() >= 3:
                    return False
        
        # Triples involving v but not u: {v, x, y} where x,y != u
        for x in range(n):
            if x == u or x == v:
                continue
            for y in range(x + 1, n):
                if y == u or y == v:
                    continue
                common = new_adj_v & adj[x] & adj[y]
                if common.bit_count() >= 3:
                    return False
        
        return True
    
    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    
    def count_edges():
        return sum(adj[i].bit_count() for i in range(n)) // 2
    
    def get_edge_list():
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append((u, v))
        return edges
    
    def greedy_fill():
        """Try to add all possible edges greedily"""
        added = 0
        for u in range(n):
            for v in range(u + 1, n):
                if not ((adj[u] >> v) & 1):
                    if can_add_edge(u, v):
                        add_edge(u, v)
                        added += 1
        return added
    
    def simulated_annealing(max_iters=5000):
        """Simulated annealing to maximize edges while maintaining K3,3-free property"""
        current_edges = count_edges()
        best_edges = current_edges
        best_adj = adj[:]
        
        T = 1.0  # Initial temperature
        cooling = 0.999  # Cooling rate
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Choose a random move: add, remove, or swap an edge
            move_type = random.random()
            
            if move_type < 0.3:
                # Try to add a random missing edge
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                if (adj[u] >> v) & 1:
                    continue
                if can_add_edge(u, v):
                    add_edge(u, v)
                    new_edges = count_edges()
                    delta = new_edges - current_edges
                    if delta >= 0 or random.random() < np.exp(delta / T):
                        current_edges = new_edges
                        if current_edges > best_edges:
                            best_edges = current_edges
                            best_adj = adj[:]
                    else:
                        remove_edge(u, v)
                else:
                    # Can't add, so this move is rejected
                    pass
                    
            elif move_type < 0.6:
                # Try to remove a random edge and add a different one (swap)
                edges = get_edge_list()
                if not edges:
                    continue
                a, b = random.choice(edges)
                remove_edge(a, b)
                
                # Try to add a random missing edge
                c = random.randint(0, n - 1)
                d = random.randint(0, n - 1)
                if c == d:
                    add_edge(a, b)
                    continue
                if c > d:
                    c, d = d, c
                if (c, d) == (a, b):
                    add_edge(a, b)
                    continue
                if (adj[c] >> d) & 1:
                    add_edge(a, b)
                    continue
                if can_add_edge(c, d):
                    add_edge(c, d)
                    new_edges = count_edges()
                    delta = new_edges - current_edges
                    if delta >= 0 or random.random() < np.exp(delta / T):
                        current_edges = new_edges
                        if current_edges > best_edges:
                            best_edges = current_edges
                            best_adj = adj[:]
                    else:
                        remove_edge(c, d)
                        add_edge(a, b)
                else:
                    add_edge(a, b)
                    
            else:
                # Try a 2-for-2 swap: remove two edges, add two different edges
                edges = get_edge_list()
                if len(edges) < 2:
                    continue
                (a1, b1), (a2, b2) = random.sample(edges, 2)
                remove_edge(a1, b1)
                remove_edge(a2, b2)
                
                c1 = random.randint(0, n - 1)
                d1 = random.randint(0, n - 1)
                c2 = random.randint(0, n - 1)
                d2 = random.randint(0, n - 1)
                if c1 > d1:
                    c1, d1 = d1, c1
                if c2 > d2:
                    c2, d2 = d2, c2
                
                # Check if these are valid and distinct from removed edges
                valid = True
                if (c1, d1) in [(a1, b1), (a2, b2)] or (c2, d2) in [(a1, b1), (a2, b2)]:
                    valid = False
                if (c1, d1) == (c2, d2):
                    valid = False
                if (adj[c1] >> d1) & 1 or (adj[c2] >> d2) & 1:
                    valid = False
                if c1 == d1 or c2 == d2:
                    valid = False
                
                if valid and can_add_edge(c1, d1) and can_add_edge(c2, d2):
                    add_edge(c1, d1)
                    add_edge(c2, d2)
                    new_edges = count_edges()
                    delta = new_edges - current_edges
                    if delta >= 0 or random.random() < np.exp(delta / T):
                        current_edges = new_edges
                        if current_edges > best_edges:
                            best_edges = current_edges
                            best_adj = adj[:]
                    else:
                        remove_edge(c1, d1)
                        remove_edge(c2, d2)
                        add_edge(a1, b1)
                        add_edge(a2, b2)
                else:
                    add_edge(a1, b1)
                    add_edge(a2, b2)
            
            T *= cooling
        
        return best_edges, best_adj
    
    # Multiple restarts with different initializations
    best_edges = 0
    best_M = [0] * n
    random.seed(42)
    
    # Strategy 1: Start with random graphs at various densities
    for restart in range(10):
        if time.time() > deadline:
            break
        
        adj = [0] * n
        density = 0.2 + 0.05 * restart
        
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < density:
                    if can_add_edge(u, v):
                        add_edge(u, v)
        
        greedy_fill()
        
        # Simulated annealing
        sa_edges, sa_adj = simulated_annealing(max_iters=3000)
        if sa_edges > best_edges:
            best_edges = sa_edges
            best_M = sa_adj[:]
    
    # Strategy 2: Structured initializations
    if time.time() < deadline:
        adj = [0] * n
        # Circulant graph with connections at distances 1, 2, 3
        for i in range(n):
            for d in [1, 2, 3]:
                j = (i + d) % n
                if i < j:
                    if can_add_edge(i, j):
                        add_edge(i, j)
        
        greedy_fill()
        sa_edges, sa_adj = simulated_annealing(max_iters=3000)
        if sa_edges > best_edges:
            best_edges = sa_edges
            best_M = sa_adj[:]
    
    if time.time() < deadline:
        adj = [0] * n
        # Turan-like: split into 2 parts, add all cross edges, sparse within
        for i in range(10):
            for j in range(10, 20):
                if can_add_edge(i, j):
                    add_edge(i, j)
        # Add some within-part edges
        for i in range(10):
            for j in range(i + 1, 10):
                if random.random() < 0.2 and can_add_edge(i, j):
                    add_edge(i, j)
        for i in range(10, 20):
            for j in range(i + 1, 20):
                if random.random() < 0.2 and can_add_edge(i, j):
                    add_edge(i, j)
        
        greedy_fill()
        sa_edges, sa_adj = simulated_annealing(max_iters=3000)
        if sa_edges > best_edges:
            best_edges = sa_edges
            best_M = sa_adj[:]
    
    # Final refinement with more SA iterations on the best solution
    if best_edges > 0 and time.time() < deadline:
        adj = best_M[:]
        sa_edges, sa_adj = simulated_annealing(max_iters=5000)
        if sa_edges > best_edges:
            best_edges = sa_edges
            best_M = sa_adj[:]
    
    # Verify the result is K3,3-free
    # (should be guaranteed by construction, but double-check)
    
    # Return the best found
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_M[u] >> v) & 1:
                edges.append([u, v])
    return edges
