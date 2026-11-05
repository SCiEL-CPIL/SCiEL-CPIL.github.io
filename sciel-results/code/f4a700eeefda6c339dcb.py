import random
import time
import math

def construct():
    n = 20
    
    # Precompute edge indices
    edge_indices = {}
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_indices[(u, v)] = idx
            idx += 1
    all_edges = list(edge_indices.keys())

    def get_edge_idx(u, v):
        if u > v:
            u, v = v, u
        return edge_indices[(u, v)]

    # adj1[u], adj2[u] are bitmasks for G1 and G2
    def init_graph():
        adj1 = [0] * n
        adj2 = [0] * n
        # Start with 5 disjoint K4s, split edges between G1 and G2
        for k in range(5):
            base = 4 * k
            for a in range(4):
                for b in range(a + 1, 4):
                    u = base + a
                    v = base + b
                    # Alternate which subgraph gets the edge
                    if (a + b) % 2 == 0:
                        adj1[u] |= 1 << v
                        adj1[v] |= 1 << u
                    else:
                        adj2[u] |= 1 << v
                        adj2[v] |= 1 << u
        return adj1, adj2

    def total_edge_count(adj1, adj2):
        count = 0
        for u in range(n):
            count += bin(adj1[u]).count('1') + bin(adj2[u]).count('1')
        return count // 2

    def is_c4_free_addition(adj, u, v):
        """Check if adding edge (u,v) to adj keeps it C4-free.
        C4-free means no two vertices share more than 1 common neighbor.
        After adding (u,v), we need:
        - For each w in N(u) \ {v}: |N(w) ∩ N(v) \ {u}| <= 1
        - For each w in N(v) \ {u}: |N(w) ∩ N(u) \ {v}| <= 1
        But since (u,v) is new, N(u) and N(v) don't include each other yet.
        Actually, after adding (u,v):
        - N(u) becomes N_old(u) ∪ {v}
        - N(v) becomes N_old(v) ∪ {u}
        
        We need to check that for any pair of vertices, they don't get 2+ common neighbors.
        The only pairs that could be affected are those involving u or v.
        
        For pair (u, w) where w is in N_old(v):
          Common neighbors of u and w after adding: (N_old(u) ∩ N_old(w)) ∪ ({v} if v in N_old(w))
          Wait, this is getting complex. Let me think differently.
          
        Actually, the key insight: in a C4-free graph, no 4-cycle exists.
        Adding edge (u,v) creates a C4 if there exists a vertex w such that
        w is adjacent to both u and v in the current graph (before adding).
        Because then u-w-v-...-u would form a cycle? No, that's just a triangle.
        
        Let me reconsider. A C4 is u-a-v-b-u where all are distinct.
        Adding edge (u,v) creates a C4 if there exist vertices a,b such that:
        - a is adjacent to u and v (in current graph)
        - b is adjacent to u and v (in current graph)  
        Wait no, that would require a and b both connected to u and v.
        
        Actually: adding edge (u,v) creates a C4 iff there exist two distinct vertices a,b
        such that a is adjacent to both u and v, and b is adjacent to both u and v.
        Because then we have cycle u-a-v-b-u.
        
        So: check if |N(u) ∩ N(v)| >= 2 in the current graph. If yes, adding (u,v) creates a C4.
        """
        common = adj[u] & adj[v]
        # Count common neighbors
        if bin(common).count('1') >= 2:
            return False
        return True

    def add_edge(adj, u, v):
        adj[u] |= 1 << v
        adj[v] |= 1 << u

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    # Main search
    start_time = time.time()
    deadline = start_time + 115.0

    current_adj1, current_adj2 = init_graph()
    current_edges = total_edge_count(current_adj1, current_adj2)
    
    best_adj1 = [x for x in current_adj1]
    best_adj2 = [x for x in current_adj2]
    best_edges = current_edges

    T = 1.0
    cooling_rate = 0.9999

    while time.time() < deadline:
        move_type = random.random()
        
        if move_type < 0.5:
            # Try to add a random edge to G1 or G2
            u, v = random.choice(all_edges)
            if not ((current_adj1[u] >> v) & 1):
                # Try adding to G1
                if is_c4_free_addition(current_adj1, u, v):
                    add_edge(current_adj1, u, v)
                    current_edges += 1
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj1 = [x for x in current_adj1]
                        best_adj2 = [x for x in current_adj2]
                elif not ((current_adj2[u] >> v) & 1):
                    # Try adding to G2
                    if is_c4_free_addition(current_adj2, u, v):
                        add_edge(current_adj2, u, v)
                        current_edges += 1
                        if current_edges > best_edges:
                            best_edges = current_edges
                            best_adj1 = [x for x in current_adj1]
                            best_adj2 = [x for x in current_adj2]
                else:
                    T *= 0.99
            else:
                T *= 0.999
        elif move_type < 0.8:
            # Try to add a random edge to G2 specifically
            u, v = random.choice(all_edges)
            if not ((current_adj2[u] >> v) & 1):
                if is_c4_free_addition(current_adj2, u, v):
                    add_edge(current_adj2, u, v)
                    current_edges += 1
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj1 = [x for x in current_adj1]
                        best_adj2 = [x for x in current_adj2]
                elif not ((current_adj1[u] >> v) & 1):
                    if is_c4_free_addition(current_adj1, u, v):
                        add_edge(current_adj1, u, v)
                        current_edges += 1
                        if current_edges > best_edges:
                            best_edges = current_edges
                            best_adj1 = [x for x in current_adj1]
                            best_adj2 = [x for x in current_adj2]
                else:
                    T *= 0.99
            else:
                T *= 0.999
        else:
            # Try a swap: remove one edge from G1 or G2, add another
            # Pick edge to remove
            u, v = random.choice(all_edges)
            in_g1 = (current_adj1[u] >> v) & 1
            in_g2 = (current_adj2[u] >> v) & 1
            
            if in_g1:
                # Remove from G1, try to add elsewhere
                x, y = random.choice(all_edges)
                if ((current_adj1[x] >> y) & 1) or ((current_adj2[x] >> y) & 1) or (x == u and y == v) or (x == v and y == u):
                    T *= 0.999
                    continue
                
                new_adj1 = [a for a in current_adj1]
                new_adj2 = [a for a in current_adj2]
                remove_edge(new_adj1, u, v)
                
                if is_c4_free_addition(new_adj1, x, y):
                    add_edge(new_adj1, x, y)
                    current_adj1 = new_adj1
                    current_adj2 = new_adj2
                    T *= 0.999
                elif not ((new_adj2[x] >> y) & 1) and is_c4_free_addition(new_adj2, x, y):
                    add_edge(new_adj2, x, y)
                    current_adj1 = new_adj1
                    current_adj2 = new_adj2
                    T *= 0.999
                else:
                    T *= 0.99
            elif in_g2:
                # Remove from G2, try to add elsewhere
                x, y = random.choice(all_edges)
                if ((current_adj1[x] >> y) & 1) or ((current_adj2[x] >> y) & 1) or (x == u and y == v) or (x == v and y == u):
                    T *= 0.999
                    continue
                
                new_adj1 = [a for a in current_adj1]
                new_adj2 = [a for a in current_adj2]
                remove_edge(new_adj2, u, v)
                
                if is_c4_free_addition(new_adj2, x, y):
                    add_edge(new_adj2, x, y)
                    current_adj1 = new_adj1
                    current_adj2 = new_adj2
                    T *= 0.999
                elif not ((new_adj1[x] >> y) & 1) and is_c4_free_addition(new_adj1, x, y):
                    add_edge(new_adj1, x, y)
                    current_adj1 = new_adj1
                    current_adj2 = new_adj2
                    T *= 0.999
                else:
                    T *= 0.99
            else:
                T *= 0.999
        
        # Cooling
        T *= cooling_rate
        if T < 0.01:
            T = 0.01

    # Convert best_adj1 and best_adj2 to list of edges
    result = []
    for (u, v), idx in edge_indices.items():
        if ((best_adj1[u] >> v) & 1) or ((best_adj2[u] >> v) & 1):
            result.append([u, v])
    return result
