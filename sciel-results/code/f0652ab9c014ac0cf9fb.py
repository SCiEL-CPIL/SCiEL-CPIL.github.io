import random
import time
import math

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 140  # 140s to be safe
    
    def is_valid_add(adj, C, u, v):
        if adj[u][v]:
            return False
        # Check constraints
        for w in adj[u]:
            if C[w][v] > 1:
                return False
        for w in adj[v]:
            if C[u][w] > 1:
                return False
        return True

    def add_edge(adj, C, u, v):
        adj[u][v] = True
        adj[v][u] = True
        for w in adj[u]:
            if w != v:
                C[w][v] += 1
                C[v][w] += 1
        for w in adj[v]:
            if w != u:
                C[u][w] += 1
                C[w][u] += 1

    def remove_edge(adj, C, u, v):
        adj[u][v] = False
        adj[v][u] = False
        for w in adj[u]:
            if w != v:
                C[w][v] -= 1
                C[v][w] -= 1
        for w in adj[v]:
            if w != u:
                C[u][w] -= 1
                C[w][u] -= 1

    best_edges = []
    best_count = 0
    
    def get_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    edges.append([i, j])
        return edges

    def greedy_init():
        adj = [[False] * n for _ in range(n)]
        C = [[0] * n for _ in range(n)]
        edges_to_try = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(edges_to_try)
        for u, v in edges_to_try:
            if is_valid_add(adj, C, u, v):
                add_edge(adj, C, u, v)
        return adj, C, get_edges(adj)

    # Run multiple restarts
    num_restarts = 50
    
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        adj, C, edges = greedy_init()
        current_count = len(edges)
        best_in_run = edges[:]
        best_count_in_run = current_count
        
        # Simulated Annealing
        T = 1.0
        T_min = 0.01
        cooling = 0.995
        
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        while T > T_min and time.time() < deadline - 5:
            # Pick a move
            move_type = random.random()
            
            if move_type < 0.5:
                # Add edge
                u, v = random.choice(all_pairs)
                if not adj[u][v] and is_valid_add(adj, C, u, v):
                    add_edge(adj, C, u, v)
                    delta = 1
                    # Accept if better (always) or with prob
                    if delta > 0 or random.random() < math.exp(delta / T):
                        # Always accept if better
                        if current_count + delta > best_count_in_run:
                            best_count_in_run = current_count + delta
                            best_in_run = get_edges(adj)
                        current_count += delta
                    else:
                        remove_edge(adj, C, u, v)
                else:
                    # Move invalid, do nothing
                    pass
                    
            elif move_type < 0.7:
                # Remove edge
                if current_count > 0:
                    # Pick a random edge to remove
                    # We need a list of current edges
                    # For efficiency, we can just pick a random pair and check if it's an edge
                    # But picking a random edge from a list is better
                    # Let's maintain a list of edges for removal
                    # Actually, picking random pair and checking adj is O(1)
                    while True:
                        u, v = random.choice(all_pairs)
                        if adj[u][v]:
                            break
                    remove_edge(adj, C, u, v)
                    delta = -1
                    if delta > 0 or random.random() < math.exp(delta / T):
                        # Accept
                        if current_count + delta > best_count_in_run:
                            best_count_in_run = current_count + delta
                            best_in_run = get_edges(adj)
                        current_count += delta
                    else:
                        add_edge(adj, C, u, v)
                        
            else:
                # Swap
                if current_count > 0:
                    # Pick edge to remove
                    while True:
                        u1, v1 = random.choice(all_pairs)
                        if adj[u1][v1]:
                            break
                    
                    # Pick edge to add
                    while True:
                        u2, v2 = random.choice(all_pairs)
                        if not adj[u2][v2]:
                            break
                    
                    # Try swap
                    remove_edge(adj, C, u1, v1)
                    if is_valid_add(adj, C, u2, v2):
                        add_edge(adj, C, u2, v2)
                        delta = 0 # Net change 0? 
                        # Wait, we removed 1, added 1. Count is same.
                        # But structure is different.
                        # We accept with prob 1 if same count? 
                        # Or use T to explore.
                        # Since count is same, delta = 0.
                        # Accept if random() < exp(0/T) = 1. Always accept.
                        # This allows random walk in the space of same-size graphs.
                        # This is good for escaping local minima.
                        
                        # Update best? No, count is same.
                        # But maybe this leads to a better state later.
                        
                        # We don't update best_count_in_run here as it didn't increase.
                    else:
                        # Revert
                        add_edge(adj, C, u1, v1)
                        
            T *= cooling
            
        if best_count_in_run > best_count:
            best_count = best_count_in_run
            best_edges = best_in_run[:]
            
    return best_edges

if __name__ == "__main__":
    # Test
    edges = construct()
    # Validate
    n = 20
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    valid = True
    for i in range(n):
        for j in range(i + 1, n):
            if len(adj[i].intersection(adj[j])) > 2:
                valid = False
                break
        if not valid:
            break
            
    print(f"Edges: {len(edges)}, Valid: {valid}")
