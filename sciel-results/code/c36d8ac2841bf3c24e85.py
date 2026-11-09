"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n = 16
    target_edges = 65
    
    # Start with K_{8,8} + one edge
    def initial_edges():
        return [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
    
    def edge_set(edges):
        return set((min(u,v), max(u,v)) for u,v in edges)
    
    def compute_degrees(adj, n):
        return [sum(adj[i]) for i in range(n)]
    
    def find_triangles(adj, n):
        """Find all triangles in the graph."""
        triangles = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    common = []
                    for k in range(j+1, n):
                        if adj[i][k] and adj[j][k]:
                            common.append(k)
                    for k in common:
                        triangles.append((i, j, k))
        return triangles
    
    def score_graph(adj, n, degrees):
        """Compute the score: max degree sum over all triangles."""
        triangles = find_triangles(adj, n)
        if not triangles:
            return 0
        max_sum = 0
        for (i, j, k) in triangles:
            s = degrees[i] + degrees[j] + degrees[k]
            if s > max_sum:
                max_sum = s
        return max_sum
    
    def make_adj(edges, n):
        adj = [[0]*n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = 1
            adj[v][u] = 1
        return adj
    
    def all_pairs(n):
        return [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Initial graph
    edges = initial_edges()
    adj = make_adj(edges, n)
    degrees = compute_degrees(adj, n)
    best_score = score_graph(adj, n, degrees)
    best_adj = [row[:] for row in adj]
    
    # Local search: try edge swaps (remove one edge, add another)
    all_pairs_list = all_pairs(n)
    
    def try_swap(adj, degrees, n, remove_pair, add_pair, best_score):
        """Try swapping: remove remove_pair, add add_pair. Return new adj, degrees, score if improved."""
        u, v = remove_pair
        a, b = add_pair
        
        # Check validity
        if adj[u][v] == 0:
            return None  # edge not present
        if adj[a][b] == 1:
            return None  # edge already present
        if u == a or u == b or v == a or v == b:
            # This would create a multi-edge or self-loop issue; actually just check
            pass
        
        # Create modified adj
        new_adj = [row[:] for row in adj]
        new_adj[u][v] = 0
        new_adj[v][u] = 0
        new_adj[a][b] = 1
        new_adj[b][a] = 1
        
        new_degrees = degrees[:]
        new_degrees[u] -= 1
        new_degrees[v] -= 1
        new_degrees[a] += 1
        new_degrees[b] += 1
        
        new_score = score_graph(new_adj, n, new_degrees)
        
        return (new_adj, new_degrees, new_score)
    
    # Simulated annealing / hill climbing
    current_adj = [row[:] for row in adj]
    current_degrees = degrees[:]
    current_score = best_score
    
    random.seed(42)
    
    for iteration in range(50000):
        if time.time() > deadline:
            break
        
        # Pick a random edge to remove and a random non-edge to add
        present_edges = [(i, j) for i in range(n) for j in range(i+1, n) if current_adj[i][j]]
        absent_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not current_adj[i][j]]
        
        if not present_edges or not absent_edges:
            continue
        
        remove_pair = random.choice(present_edges)
        add_pair = random.choice(absent_edges)
        
        result = try_swap(current_adj, current_degrees, n, remove_pair, add_pair, current_score)
        if result is None:
            continue
        
        new_adj, new_degrees, new_score = result
        
        if new_score < current_score:
            current_adj = new_adj
            current_degrees = new_degrees
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_adj = [row[:] for row in current_adj]
        
        # Occasional restart to escape local minima
        elif random.random() < 0.01 and current_score > best_score:
            # Partial restart: keep best
            pass
    
    # Final polish: try all single-edge swaps from best
    for _ in range(5):
        if time.time() > deadline:
            break
        
        improved = False
        for remove_pair in all_pairs_list:
            if time.time() > deadline:
                break
            if not best_adj[remove_pair[0]][remove_pair[1]]:
                continue
            for add_pair in all_pairs_list:
                if best_adj[add_pair[0]][add_pair[1]]:
                    continue
                result = try_swap(best_adj, compute_degrees(best_adj, n), n, remove_pair, add_pair, best_score)
                if result is None:
                    continue
                new_adj, new_degrees, new_score = result
                if new_score < best_score:
                    best_adj = new_adj
                    best_score = new_score
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    # Convert best_adj back to edge list
    best_edges = [[i, j] for i in range(n) for j in range(i+1, n) if best_adj[i][j]]
    
    return {'n': n, 'edges': best_edges}
# EVOLVE-BLOCK-END
