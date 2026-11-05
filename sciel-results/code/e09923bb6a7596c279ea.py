import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n = 16
    
    # Clebsch graph construction
    def get_clebsch():
        vectors = [i for i in range(32) if bin(i).count('1') % 2 == 0]
        adj = [0] * n
        for u_idx in range(n):
            for v_idx in range(u_idx + 1, n):
                xor = vectors[u_idx] ^ vectors[v_idx]
                if bin(xor).count('1') == 4:
                    adj[u_idx] |= (1 << v_idx)
                    adj[v_idx] |= (1 << u_idx)
        return adj

    def count_edges(adj):
        count = 0
        for i in range(n):
            count += bin(adj[i]).count('1')
        return count // 2

    def is_triangle_free(adj):
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    if (adj[i] & adj[j]) != 0:
                        return False
        return True

    def is_3_colorable(adj):
        """
        Exact backtracking 3-coloring with MRV heuristic.
        Returns True if graph is 3-colorable.
        """
        colors = [-1] * n
        
        # Precompute adjacency lists for speed
        adj_list = []
        for i in range(n):
            neighbors = []
            for j in range(n):
                if i != j and (adj[i] & (1 << j)):
                    neighbors.append(j)
            adj_list.append(neighbors)
        
        def backtrack():
            # Find uncolored vertex with MRV (fewest available colors)
            best_v = -1
            best_available = 4
            for v in range(n):
                if colors[v] == -1:
                    used_colors = set()
                    for u in adj_list[v]:
                        if colors[u] != -1:
                            used_colors.add(colors[u])
                    available = 3 - len(used_colors)
                    if available < best_available:
                        best_available = available
                        best_v = v
                        if available <= 0:
                            return False
            
            if best_v == -1:
                return True  # All vertices colored
            
            # Try each color
            used_colors = set()
            for u in adj_list[best_v]:
                if colors[u] != -1:
                    used_colors.add(colors[u])
            
            for c in range(3):
                if c not in used_colors:
                    colors[best_v] = c
                    if backtrack():
                        return True
                    colors[best_v] = -1
            
            return False
        
        return backtrack()

    def can_add_edge(adj, u, v):
        """Check if adding edge (u,v) preserves triangle-freeness."""
        if adj[u] & (1 << v):
            return False  # Edge already exists
        # Check if u and v have a common neighbor
        if (adj[u] & adj[v]) != 0:
            return False
        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    # Start with Clebsch graph (40 edges)
    best_adj = get_clebsch()
    best_count = count_edges(best_adj)
    best_is_4chromatic = not is_3_colorable(best_adj)
    
    # Simulated annealing to find denser 4-chromatic triangle-free graphs
    # State: adj (list of bitmasks)
    # Objective: maximize edges among 4-chromatic triangle-free graphs
    
    current_adj = [x for x in best_adj]
    current_count = best_count
    current_is_4chromatic = best_is_4chromatic
    
    # Temperature schedule
    T0 = 10.0
    T_end = 0.01
    cooling_rate = 0.999
    
    random.seed(42)
    
    T = T0
    step = 0
    
    while time.time() < deadline and T > T_end:
        step += 1
        
        # Generate a random move
        # Options: add edge, remove edge, or swap
        move_type = random.random()
        
        if move_type < 0.4:
            # Try to add a random edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v or adj[u] & (1 << v):
                continue
            if not can_add_edge(current_adj, u, v):
                continue
            
            new_adj = [x for x in current_adj]
            add_edge(new_adj, u, v)
            new_count = current_count + 1
            new_is_4chromatic = not is_3_colorable(new_adj)
            
            # Acceptance: prefer 4-chromatic with more edges
            if new_is_4chromatic and new_count > current_count:
                # Always accept improvement in edge count for 4-chromatic
                current_adj = new_adj
                current_count = new_count
                current_is_4chromatic = True
                if current_count > best_count:
                    best_adj = [x for x in current_adj]
                    best_count = current_count
                    best_is_4chromatic = True
            elif new_is_4chromatic and new_count == current_count:
                # Same edge count, still 4-chromatic - accept with some probability
                if random.random() < 0.5:
                    current_adj = new_adj
                    current_count = new_count
                    current_is_4chromatic = True
            elif not new_is_4chromatic:
                # Became 3-colorable - accept with probability based on temperature
                # This allows escaping local optima
                prob = min(1.0, new_count / max(1, current_count))
                if random.random() < prob:
                    current_adj = new_adj
                    current_count = new_count
                    current_is_4chromatic = False
            # If new_is_4chromatic but new_count < current_count, reject
            
        elif move_type < 0.7:
            # Try to remove a random edge
            if current_count == 0:
                continue
            # Pick a random existing edge
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if current_adj[u] & (1 << v):
                        edges.append((u, v))
            if not edges:
                continue
            u, v = random.choice(edges)
            
            new_adj = [x for x in current_adj]
            remove_edge(new_adj, u, v)
            new_count = current_count - 1
            new_is_4chromatic = not is_3_colorable(new_adj)
            
            # Removing edges is rarely beneficial unless it helps us add more later
            # Accept with low probability
            if random.random() < 0.1:
                current_adj = new_adj
                current_count = new_count
                current_is_4chromatic = new_is_4chromatic
        
        else:
            # Edge swap: remove one edge, add another
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if current_adj[u] & (1 << v):
                        edges.append((u, v))
            if not edges:
                continue
            u1, v1 = random.choice(edges)
            
            # Try to add a random edge after removing (u1,v1)
            new_adj = [x for x in current_adj]
            remove_edge(new_adj, u1, v1)
            
            # Try to add a different edge
            u2 = random.randint(0, n - 1)
            v2 = random.randint(0, n - 1)
            if u2 == v2:
                T *= cooling_rate
                continue
            if new_adj[u2] & (1 << v2):
                T *= cooling_rate
                continue
            if not can_add_edge(new_adj, u2, v2):
                T *= cooling_rate
                continue
            
            add_edge(new_adj, u2, v2)
            new_count = current_count  # Removed one, added one
            new_is_4chromatic = not is_3_colorable(new_adj)
            
            if new_is_4chromatic and current_is_4chromatic:
                # Both 4-chromatic, same edge count - explore
                if random.random() < 0.5:
                    current_adj = new_adj
                    current_count = new_count
                    current_is_4chromatic = True
            elif new_is_4chromatic and not current_is_4chromatic:
                # Became 4-chromatic - great!
                current_adj = new_adj
                current_count = new_count
                current_is_4chromatic = True
                if current_count > best_count:
                    best_adj = [x for x in current_adj]
                    best_count = current_count
                    best_is_4chromatic = True
            elif not new_is_4chromatic and current_is_4chromatic:
                # Lost 4-chromaticity - accept with temperature-based probability
                prob = min(1.0, 0.5 * T / T0)
                if random.random() < prob:
                    current_adj = new_adj
                    current_count = new_count
                    current_is_4chromatic = False
        
        # Cool down
        T *= cooling_rate
        
        # Periodically try to add all possible edges greedily
        if step % 100 == 0 and time.time() < deadline:
            greedy_adj = [x for x in current_adj]
            improved = True
            while improved:
                improved = False
                for u in range(n):
                    for v in range(u + 1, n):
                        if not (greedy_adj[u] & (1 << v)) and can_add_edge(greedy_adj, u, v):
                            # Check if adding keeps it 4-chromatic or at least doesn't make it worse
                            test_adj = [x for x in greedy_adj]
                            add_edge(test_adj, u, v)
                            test_4chromatic = not is_3_colorable(test_adj)
                            if test_4chromatic:
                                greedy_adj = test_adj
                                improved = True
                                if count_edges(greedy_adj) > best_count:
                                    best_adj = [x for x in greedy_adj]
                                    best_count = count_edges(greedy_adj)
                                    best_is_4chromatic = True
                                break
                    if improved:
                        break
            
            if count_edges(greedy_adj) > current_count:
                current_adj = greedy_adj
                current_count = count_edges(greedy_adj)
                current_is_4chromatic = not is_3_colorable(current_adj)
    
    # Final greedy completion on best_adj
    final_adj = [x for x in best_adj]
    improved = True
    while improved:
        improved = False
        for u in range(n):
            for v in range(u + 1, n):
                if not (final_adj[u] & (1 << v)) and can_add_edge(final_adj, u, v):
                    test_adj = [x for x in final_adj]
                    add_edge(test_adj, u, v)
                    if not is_3_colorable(test_adj):
                        final_adj = test_adj
                        improved = True
                        break
            if improved:
                break
    
    best_adj = final_adj
    best_count = count_edges(best_adj)
    
    return [[u, v] for u in range(n) for v in range(u + 1, n) if best_adj[u] & (1 << v)]
