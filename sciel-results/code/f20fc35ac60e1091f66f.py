import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill
    
    def make_k4_graph(num_copies):
        n = 4 * num_copies
        edges = []
        for b in range(num_copies):
            base = 4 * b
            for i in range(4):
                for j in range(i + 1, 4):
                    edges.append([base + i, base + j])
        return {'n': n, 'edges': edges}
    
    def build_adj(n, edges):
        adj = [[False]*n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
        return adj
    
    def build_adj_list(adj, n):
        adj_list = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    adj_list[i].append(j)
                    adj_list[j].append(i)
        return adj_list
    
    def has_c5(adj, n):
        """Check if graph contains a 5-cycle using adjacency lists."""
        if n < 5:
            return False
        adj_list = build_adj_list(adj, n)
        
        # For each edge (a,b), check if there's a path b->c->d->e->a of length 4
        # i.e., c in N(b)\{a}, d in N(c)\{b}, e in N(d)\{c}, a in N(e)\{d}
        # We use a smarter approach: for each pair (a,b) that is an edge,
        # check if there exist c,d,e such that b-c-d-e-a forms a path.
        
        for a in range(n):
            for b in adj_list[a]:
                if b <= a:
                    continue
                # Find common neighbors of a and b? No, we need a path of length 4 from b to a
                # b -> c -> d -> e -> a where c,d,e are distinct and not a,b
                # c must be in N(b) \ {a}
                for c in adj_list[b]:
                    if c == a:
                        continue
                    # d must be in N(c) \ {b, a}
                    for d in adj_list[c]:
                        if d == b or d == a:
                            continue
                        # e must be in N(d) \ {c, a} and a must be in N(e) \ {d}
                        for e in adj_list[d]:
                            if e == c or e == a:
                                continue
                            if adj[e][a] and e != d:
                                return True
        return False
    
    def max_independent_set_size(adj, n):
        """Compute maximum independent set size using branch and bound with Tomita's algorithm."""
        if n == 0:
            return 0
        
        adj_list = build_adj_list(adj, n)
        
        # Tomita's algorithm for maximum clique on complement graph
        # Maximum independent set in G = Maximum clique in complement of G
        # Complement adjacency
        comp_adj_list = [[] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j and not adj[i][j]:
                    comp_adj_list[i].append(j)
        
        best = [0]
        
        # Order vertices by degree (greedy coloring for upper bound)
        # Use a simple branch and bound
        def color_sort(P, order):
            """Color the vertices in P using greedy coloring, return color classes."""
            # Simple approach: just return len(P) as upper bound
            return len(P)
        
        def expand(R, P):
            """Branch and bound to find max clique in complement graph."""
            if time.time() > deadline:
                return
            if not P:
                if len(R) > best[0]:
                    best[0] = len(R)
                return
            
            # Upper bound: len(R) + len(P)
            if len(R) + len(P) <= best[0]:
                return
            
            # Greedy coloring for tighter upper bound
            # Color vertices in P
            uncolored = list(P)
            colors = {}
            color_num = 0
            while uncolored:
                color_num += 1
                new_uncolored = []
                for v in uncolored:
                    can_color = True
                    for u in uncolored:
                        if u == v:
                            continue
                        if colors.get(u) == color_num and v in comp_adj_list[u]:
                            can_color = False
                            break
                    if can_color:
                        colors[v] = color_num
                    else:
                        new_uncolored.append(v)
                uncolored = new_uncolored
            
            # Process vertices in reverse color order
            # Group by color
            max_color = max(colors.values()) if colors else 0
            for c in range(max_color, 0, -1):
                # Vertices with color c
                verts_c = [v for v in P if colors.get(v) == c]
                for v in verts_c:
                    if len(R) + c <= best[0]:
                        break  # No need to check remaining vertices with this or lower color
                    # Include v
                    new_R = R + [v]
                    new_P = [u for u in P if u > v and u in comp_adj_list[v]]
                    expand(new_R, new_P)
        
        expand([], list(range(n)))
        return best[0]
    
    def check_valid(adj, n):
        """Check C5-free and alpha < 9."""
        if has_c5(adj, n):
            return False
        alpha = max_independent_set_size(adj, n)
        return alpha < 9
    
    # Start with 8xK4
    result = make_k4_graph(8)
    n = result['n']
    edges = result['edges']
    adj = build_adj(n, edges)
    
    # Verify base case
    if not check_valid(adj, n):
        return make_k4_graph(7)
    
    best_n = n
    best_edges = edges[:]
    
    # Try to extend by adding vertices one at a time
    current_n = n
    current_edges = edges[:]
    current_adj = adj
    
    for target_n in range(current_n + 1, min(current_n + 4, 36)):
        if time.time() > deadline:
            break
        
        found = False
        random.seed(42 + target_n)
        
        # Try various connection patterns for the new vertex
        for density in [0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8, 0.1, 0.9]:
            if time.time() > deadline:
                break
            if found:
                break
            
            for trial in range(100):
                if time.time() > deadline:
                    break
                
                # Randomly choose connections for new vertex
                new_conn = []
                for v in range(current_n):
                    if random.random() < density:
                        new_conn.append(v)
                
                # Build new adjacency
                new_adj = [[False]*target_n for _ in range(target_n)]
                for i in range(current_n):
                    for j in range(current_n):
                        new_adj[i][j] = current_adj[i][j]
                
                new_edges = current_edges[:]
                for v in new_conn:
                    new_adj[current_n][v] = True
                    new_adj[v][current_n] = True
                    new_edges.append([current_n, v])
                
                if check_valid(new_adj, target_n):
                    found = True
                    best_n = target_n
                    best_edges = new_edges[:]
                    current_n = target_n
                    current_edges = new_edges[:]
                    current_adj = new_adj
                    break
        
        if not found:
            break
    
    return {'n': best_n, 'edges': best_edges}
