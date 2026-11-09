import random
import time
import itertools

def construct():
    n = 16
    target_edges = 65
    
    def get_edges(adj_matrix):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj_matrix[i][j]:
                    edges.append([i, j])
        return edges

    def calculate_score(adj_matrix):
        # Calculate degrees
        deg = [sum(row) for row in adj_matrix]
        
        max_sum = 0
        # Iterate over all triples
        for u, v, w in itertools.combinations(range(n), 3):
            if adj_matrix[u][v] and adj_matrix[v][w] and adj_matrix[w][u]:
                s = deg[u] + deg[v] + deg[w]
                if s > max_sum:
                    max_sum = s
        return max_sum

    # Initialize
    # Start with K_8,8 + 1 edge as a good baseline, but randomize slightly
    # Or start random?
    # Let's try a few random starts and the baseline.
    
    best_graph = None
    best_score = float('inf')
    
    start_time = time.time()
    timeout = 140.0 # Safety margin
    
    # Helper to create K_8,8 + 1 edge
    def make_baseline():
        adj = [[0]*n for _ in range(n)]
        # Part A: 0-7, Part B: 8-15
        for u in range(8):
            for v in range(8, 16):
                adj[u][v] = 1
                adj[v][u] = 1
        # Add one internal edge
        # Randomly pick two in A
        u, v = random.sample(range(8), 2)
        adj[u][v] = 1
        adj[v][u] = 1
        return adj

    # Helper to create random graph with 65 edges
    def make_random():
        adj = [[0]*n for _ in range(n)]
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        for u, v in all_edges[:65]:
            adj[u][v] = 1
            adj[v][u] = 1
        return adj

    # Candidates
    candidates = [make_baseline()]
    # Add a few random ones
    for _ in range(5):
        candidates.append(make_random())
    
    # Local Search
    for start_adj in candidates:
        if time.time() - start_time > timeout:
            break
            
        current_adj = [row[:] for row in start_adj]
        current_score = calculate_score(current_adj)
        
        if current_score < best_score:
            best_score = current_score
            best_graph = [row[:] for row in current_adj]
            
        if best_score <= 23: # Can't do much better than 23 likely? 23.4 is asymptotic.
            break

        # Local Search Loop
        no_improve_count = 0
        last_improve_time = time.time()
        
        while time.time() - start_time < timeout:
            # Pick random edge to remove
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if current_adj[i][j]:
                        edges.append((i, j))
            
            if not edges:
                break
                
            u, v = random.choice(edges)
            
            # Pick random non-edge to add
            non_edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if not current_adj[i][j]:
                        non_edges.append((i, j))
                        
            if not non_edges:
                break
                
            x, y = random.choice(non_edges)
            
            # Apply swap
            current_adj[u][v] = 0
            current_adj[v][u] = 0
            current_adj[x][y] = 1
            current_adj[y][x] = 1
            
            new_score = calculate_score(current_adj)
            
            if new_score < current_score:
                current_score = new_score
                no_improve_count = 0
                last_improve_time = time.time()
                
                if current_score < best_score:
                    best_score = current_score
                    best_graph = [row[:] for row in current_adj]
                    if best_score <= 23:
                        break
            else:
                # Revert
                current_adj[u][v] = 1
                current_adj[v][u] = 1
                current_adj[x][y] = 0
                current_adj[y][x] = 0
                no_improve_count += 1
                
                # If stuck for too long, restart or accept equal?
                # Let's accept equal moves to explore
                if new_score == current_score and random.random() < 0.1:
                    current_adj[u][v] = 0
                    current_adj[v][u] = 0
                    current_adj[x][y] = 1
                    current_adj[y][x] = 1
                    # Keep it
                else:
                    no_improve_count += 1

            # If no improvement for a while, break to try another start?
            if no_improve_count > 500:
                break

    # Convert best_graph to edge list
    if best_graph is None:
        # Fallback to baseline
        best_graph = make_baseline()
        
    edges = get_edges(best_graph)
    return {'n': 16, 'edges': edges}
