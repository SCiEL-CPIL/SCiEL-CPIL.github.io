import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    n = 31
    target_edges = 91
    
    def count_c4(adj_sets):
        """Count 4-cycles using adjacency sets.
        For each non-adjacent pair {u,v}, count C(|N(u)∩N(v)|, 2).
        Each 4-cycle is counted twice (once per pair of opposite vertices).
        """
        total = 0
        for u in range(n):
            for v in range(u+1, n):
                if v not in adj_sets[u]:  # non-adjacent
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
    
    # Generate initial graph: try to build a C4-free graph greedily, then add one edge
    def generate_initial_graph():
        random.seed(42)
        # Try multiple random orderings to build a C4-free graph
        best_edges = None
        best_c4 = float('inf')
        
        for attempt in range(20):
            adj = [set() for _ in range(n)]
            vertex_order = list(range(n))
            random.shuffle(vertex_order)
            
            for u in vertex_order:
                # Try to connect u to as many vertices as possible without creating C4
                candidates = [v for v in range(n) if v != u and v not in adj[u]]
                random.shuffle(candidates)
                
                for v in candidates:
                    if len(adj[u]) >= 3:  # degree cap to help avoid C4
                        break
                    # Check if adding edge (u,v) creates a C4
                    common = len(adj[u] & adj[v])
                    if common < 2:  # Would create C4 if common >= 2
                        adj[u].add(v)
                        adj[v].add(u)
            
            edges = adj_sets_to_edges(adj)
            if len(edges) >= target_edges:
                # Trim to target_edges
                random.shuffle(edges)
                edges = edges[:target_edges]
                c4 = count_c4(edges_to_adj_sets(edges))
                if c4 < best_c4:
                    best_c4 = c4
                    best_edges = edges
            
            if len(edges) < target_edges:
                # Add random edges to reach target
                all_possible = [[i,j] for i in range(n) for j in range(i+1, n)]
                existing = set(tuple(e) for e in edges)
                for e in all_possible:
                    if len(edges) >= target_edges:
                        break
                    if tuple(e) not in existing:
                        edges.append(e)
                        existing.add(tuple(e))
                c4 = count_c4(edges_to_adj_sets(edges))
                if c4 < best_c4:
                    best_c4 = c4
                    best_edges = edges
        
        if best_edges is None:
            # Fallback: random graph
            all_possible = [[i,j] for i in range(n) for j in range(i+1, n)]
            random.shuffle(all_possible)
            best_edges = all_possible[:target_edges]
        
        return best_edges
    
    # Local search: try edge swaps to reduce C4
    def local_search(edges, max_time):
        adj = edges_to_adj_sets(edges)
        current_c4 = count_c4(adj)
        best_edges = list(edges)
        best_c4 = current_c4
        best_adj = [set(s) for s in adj]
        
        # Precompute all possible edges
        all_edges = set((i,j) for i in range(n) for j in range(i+1, n))
        current_edge_set = set(tuple(e) for e in edges)
        
        while time.time() < max_time:
            improved = False
            # Try all possible edge swaps: remove one edge, add another
            # To make this efficient, we'll sample random swaps
            for _ in range(500):
                if time.time() >= max_time:
                    break
                
                # Pick a random edge to remove
                remove_edge = random.choice(edges)
                ru, rv = remove_edge[0], remove_edge[1]
                
                # Pick a random non-edge to add
                while True:
                    u = random.randint(0, n-1)
                    v = random.randint(0, n-1)
                    if u < v and (u,v) not in current_edge_set:
                        add_edge = [u, v]
                        break
                
                # Check if this swap keeps exactly target_edges
                # It does: remove one, add one
                
                # Compute new C4 count efficiently
                # Remove edge (ru,rv)
                best_adj[ru].discard(rv)
                best_adj[rv].discard(ru)
                current_edge_set.discard((ru,rv))
                
                # Add edge (u,v)
                best_adj[u].add(v)
                best_adj[v].add(u)
                current_edge_set.add((u,v))
                
                new_c4 = count_c4(best_adj)
                
                if new_c4 < best_c4:
                    best_c4 = new_c4
                    best_edges = adj_sets_to_edges(best_adj)
                    improved = True
                else:
                    # Revert
                    best_adj[ru].add(rv)
                    best_adj[rv].add(ru)
                    current_edge_set.add((ru,rv))
                    best_adj[u].discard(v)
                    best_adj[v].discard(u)
                    current_edge_set.discard((u,v))
            
            if not improved and time.time() > max_time - 5:
                break
        
        return best_edges, best_c4
    
    # Generate initial graph
    edges = generate_initial_graph()
    adj = edges_to_adj_sets(edges)
    current_c4 = count_c4(adj)
    
    # Run local search with time budget
    remaining = deadline - time.time()
    if remaining > 5:
        edges, current_c4 = local_search(edges, time.time() + remaining - 3)
    
    # Final check
    if len(edges) != target_edges:
        # Fix if needed
        edge_set = set(tuple(e) for e in edges)
        if len(edges) < target_edges:
            all_possible = [(i,j) for i in range(n) for j in range(i+1, n)]
            for e in all_possible:
                if len(edges) >= target_edges:
                    break
                if e not in edge_set:
                    edges.append(list(e))
                    edge_set.add(e)
        elif len(edges) > target_edges:
            edges = edges[:target_edges]
    
    return edges
