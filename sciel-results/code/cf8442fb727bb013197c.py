import random
import time

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Build edge list
    edge_list = []
    edge_indices = {}
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_indices[(u, v)] = idx
            edge_list.append((u, v))
            idx += 1
    m = idx  # 190 total possible edges

    def edge_count(adj):
        count = 0
        for u in range(n):
            count += bin(adj[u]).count('1')
        return count // 2

    def is_valid_addition(adj, u, v):
        # Check if adding (u, v) keeps graph K2,3-free
        # After adding (u,v), for any w != u,v: |N(u) ∩ N(w)| <= 2 and |N(v) ∩ N(w)| <= 2
        neighbors_v = adj[v]
        temp_v = neighbors_v
        while temp_v:
            lsb = temp_v & -temp_v
            w = lsb.bit_length() - 1
            temp_v ^= lsb
            if w == u:
                continue
            common = adj[u] & adj[w]
            if bin(common).count('1') > 2:
                return False
        
        neighbors_u = adj[u]
        temp_u = neighbors_u
        while temp_u:
            lsb = temp_u & -temp_u
            w = lsb.bit_length() - 1
            temp_u ^= lsb
            if w == v:
                continue
            common = adj[v] & adj[w]
            if bin(common).count('1') > 2:
                return False
        
        return True

    def add_edge(adj, u, v):
        adj[u] |= 1 << v
        adj[v] |= 1 << u

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    # Greedy initialization: build a good starting graph
    def greedy_build():
        adj = [0] * n
        # Start with 5 disjoint K4s
        for k in range(5):
            base = 4 * k
            for a in range(4):
                for b in range(a + 1, 4):
                    u = base + a
                    v = base + b
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
        return adj

    current_adj = greedy_build()
    current_edges = edge_count(current_adj)
    
    best_adj = [x for x in current_adj]
    best_edges = current_edges

    # Local search to improve initial solution
    all_edges = list(edge_list)
    improved = True
    while improved:
        improved = False
        for u, v in all_edges:
            if not ((current_adj[u] >> v) & 1):
                if is_valid_addition(current_adj, u, v):
                    add_edge(current_adj, u, v)
                    current_edges += 1
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = [x for x in current_adj]
                    improved = True

    best_solution = [x for x in best_adj]
    best_solution_edges = best_edges

    # Multiple restarts with simulated annealing
    random.seed(42)
    for restart in range(200):
        if time.time() > deadline:
            break
        
        # Random restart: start from a random subset of edges
        random_adj = [0] * n
        for u, v in all_edges:
            if random.random() < 0.3:  # 30% chance to include
                if is_valid_addition(random_adj, u, v):
                    add_edge(random_adj, u, v)
        
        random_edges = edge_count(random_adj)
        
        # Simulated annealing from this start
        T = 1.0
        cooling_rate = 0.9999
        current = [x for x in random_adj]
        curr_edges = random_edges
        
        for step in range(8000):
            if time.time() > deadline:
                break
            
            move_type = random.random()
            
            if move_type < 0.7:
                # Try to add a random edge
                u, v = random.choice(all_edges)
                if not ((current[u] >> v) & 1):
                    if is_valid_addition(current, u, v):
                        add_edge(current, u, v)
                        curr_edges += 1
                        if curr_edges > best_solution_edges:
                            best_solution_edges = curr_edges
                            best_solution = [x for x in current]
                    else:
                        T *= 0.99
                else:
                    T *= 0.999
            else:
                # Try a swap
                u, v = random.choice(all_edges)
                if ((current[u] >> v) & 1):
                    x, y = random.choice(all_edges)
                    if ((current[x] >> y) & 1) or (x == u and y == v) or (x == v and y == u):
                        T *= 0.999
                        continue
                    
                    new_adj = [a for a in current]
                    remove_edge(new_adj, u, v)
                    
                    if is_valid_addition(new_adj, x, y):
                        add_edge(new_adj, x, y)
                        # Same edge count, accept to explore
                        current = new_adj
                        T *= 0.999
                    else:
                        T *= 0.99
                else:
                    T *= 0.999
            
            T *= cooling_rate
            if T < 0.01:
                T = 0.01

    # Convert best solution to list of edges
    result = []
    for (u, v), idx in edge_indices.items():
        if (best_solution[u] >> v) & 1:
            result.append([u, v])
    return result
