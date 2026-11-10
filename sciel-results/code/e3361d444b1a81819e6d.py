import time
import random
import itertools

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 110  # 110s to be safe
    
    best_edges = []
    best_count = 0
    
    # Generate all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def is_valid_edge(adj, common, u, v):
        if common[u][v] > 2:
            return False
        for w in range(n):
            if w == u or w == v:
                continue
            if adj[u][w] and common[v][w] >= 2:
                return False
            if adj[v][w] and common[u][w] >= 2:
                return False
        return True

    def add_edge(adj, common, active_edges, u, v):
        adj[u][v] = 1
        adj[v][u] = 1
        active_edges.add((u, v))
        for w in range(n):
            if w == u or w == v:
                continue
            if adj[v][w]:
                common[u][w] += 1
                common[w][u] += 1
            if adj[u][w]:
                common[v][w] += 1
                common[w][v] += 1

    def remove_edge(adj, common, active_edges, u, v):
        adj[u][v] = 0
        adj[v][u] = 0
        active_edges.discard((u, v))
        for w in range(n):
            if w == u or w == v:
                continue
            if adj[v][w]:
                common[u][w] -= 1
                common[w][u] -= 1
            if adj[u][w]:
                common[v][w] -= 1
                common[w][v] -= 1

    def greedy_construct():
        adj = [[0] * n for _ in range(n)]
        common = [[0] * n for _ in range(n)]
        active_edges = set()
        
        shuffled_edges = all_edges[:]
        random.shuffle(shuffled_edges)
        
        for u, v in shuffled_edges:
            if is_valid_edge(adj, common, u, v):
                add_edge(adj, common, active_edges, u, v)
        
        return adj, common, active_edges

    def simulate_annealing(adj, common, active_edges, temp, steps):
        current_adj = [row[:] for row in adj]
        current_common = [row[:] for row in common]
        current_edges = set(active_edges)
        current_count = len(current_edges)
        best_count_sa = current_count
        best_edges_sa = set(current_edges)
        
        for i in range(steps):
            if time.time() > deadline:
                break
            
            # Choose a move: Add or Remove
            if random.random() < 0.6: # Bias towards adding
                # Pick a random edge from all_edges
                u, v = random.choice(all_edges)
                if (u, v) in current_edges:
                    continue
                if is_valid_edge(current_adj, current_common, u, v):
                    add_edge(current_adj, current_common, current_edges, u, v)
                    current_count += 1
                    if current_count > best_count_sa:
                        best_count_sa = current_count
                        best_edges_sa = set(current_edges)
                    temp = max(temp * 0.995, 0.1)
                else:
                    # Failed to add, cool down slightly
                    temp = max(temp * 0.99, 0.1)
            else:
                # Remove a random edge
                if not current_edges:
                    continue
                edge_list = list(current_edges)
                u, v = random.choice(edge_list)
                remove_edge(current_adj, current_common, current_edges, u, v)
                current_count -= 1
                # Cooling
                temp = max(temp * 0.98, 0.1)
                
        return best_count_sa, best_edges_sa

    # Run multiple SA instances starting from greedy
    while time.time() < deadline:
        adj, common, active_edges = greedy_construct()
        count = len(active_edges)
        
        if count > best_count:
            best_count = count
            best_edges = list(active_edges)
            
        # Run SA for a fixed number of steps
        sa_count, sa_edges = simulate_annealing(adj, common, active_edges, 10.0, 1000)
        
        if sa_count > best_count:
            best_count = sa_count
            best_edges = list(sa_edges)

    return [[u, v] for u, v in best_edges]
