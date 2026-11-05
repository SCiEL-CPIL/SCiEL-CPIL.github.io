import random
import time
import math

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    n = 31
    target_edges = 91
    
    def count_c4(adj_sets):
        """Count 4-cycles using adjacency sets."""
        total = 0
        for u in range(n):
            for v in range(u+1, n):
                if v not in adj_sets[u]:
                    common = len(adj_sets[u] & adj_sets[v])
                    if common >= 2:
                        total += common * (common - 1) // 2
        return total // 2
    
    def edges_to_adj_sets(edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def adj_sets_to_edges(adj_sets):
        edges = []
        for u in range(n):
            for v in adj_sets[u]:
                if u < v:
                    edges.append([u, v])
        return edges

    def generate_initial_graph():
        # Generate a random graph with 91 edges
        # Or a greedy one
        random.seed(42)
        edges = []
        edge_set = set()
        # Try to make it somewhat random but not too dense in clusters
        # Just random shuffle of all possible edges
        all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_possible)
        edges = [list(e) for e in all_possible[:target_edges]]
        return edges

    # Simulated Annealing
    def simulated_annealing(edges, max_time):
        adj = edges_to_adj_sets(edges)
        current_c4 = count_c4(adj)
        best_edges = list(edges)
        best_c4 = current_c4
        best_adj = [set(s) for s in adj]
        
        T = 10.0
        cooling_rate = 0.999
        min_T = 0.01
        
        all_possible_edges = set((i, j) for i in range(n) for j in range(i+1, n))
        current_edge_set = set(tuple(e) for e in edges)
        
        while time.time() < max_time:
            if T < min_T:
                break
                
            # Generate a random swap
            # Pick random edge to remove
            remove_edge = random.choice(edges)
            ru, rv = remove_edge[0], remove_edge[1]
            
            # Pick random non-edge to add
            while True:
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u < v and (u, v) not in current_edge_set:
                    add_edge = (u, v)
                    break
            
            # Apply swap to adj sets
            adj[ru].discard(rv)
            adj[rv].discard(ru)
            current_edge_set.discard((ru, rv))
            
            adj[u].add(v)
            adj[v].add(u)
            current_edge_set.add((u, v))
            
            new_c4 = count_c4(adj)
            delta = new_c4 - current_c4
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                # Accept
                current_c4 = new_c4
                edges = adj_sets_to_edges(adj)
                
                if current_c4 < best_c4:
                    best_c4 = current_c4
                    best_edges = list(edges)
                    best_adj = [set(s) for s in adj]
            else:
                # Reject, revert
                adj[ru].add(rv)
                adj[rv].add(ru)
                current_edge_set.add((ru, rv))
                
                adj[u].discard(v)
                adj[v].discard(u)
                current_edge_set.discard((u, v))
            
            T *= cooling_rate
            
        return best_edges, best_c4

    # Run multiple restarts if time permits
    edges = generate_initial_graph()
    best_overall_edges = edges
    best_overall_c4 = count_c4(edges_to_adj_sets(edges))
    
    # Use 90% of time for SA
    sa_time = deadline - time.time() - 5
    if sa_time > 10:
        edges, c4 = simulated_annealing(edges, time.time() + sa_time)
        if c4 < best_overall_c4:
            best_overall_c4 = c4
            best_overall_edges = edges
            
    return best_overall_edges
