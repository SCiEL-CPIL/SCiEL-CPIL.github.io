import time
import random
import copy

def construct():
    n = 16
    # Initial baseline: K_{8,8} + (0,1)
    # This structure is known to be a strong starting point.
    adj = [[False]*n for _ in range(n)]
    for u in range(8):
        for v in range(8, 16):
            adj[u][v] = True
            adj[v][u] = True
    adj[0][1] = True
    adj[1][0] = True
    
    def get_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    edges.append([i, j])
        return edges

    def calc_score(adj):
        degrees = [0]*n
        for i in range(n):
            # Sum of adjacency row
            # Using sum for clarity, could optimize with numpy if needed
            deg_i = 0
            row = adj[i]
            for j in range(n):
                if row[j]:
                    deg_i += 1
            degrees[i] = deg_i
        
        max_sum = 0
        # Iterate over all triangles
        for i in range(n):
            row_i = adj[i]
            deg_i = degrees[i]
            for j in range(i+1, n):
                if row_i[j]:
                    row_j = adj[j]
                    deg_j = degrees[j]
                    for k in range(j+1, n):
                        if row_i[k] and row_j[k]:
                            s = deg_i + deg_j + degrees[k]
                            if s > max_score:
                                max_score = s
        return max_score

    current_adj = copy.deepcopy(adj)
    current_score = calc_score(current_adj)
    best_adj = copy.deepcopy(current_adj)
    best_score = current_score
    
    start_time = time.time()
    timeout = 140.0  # Safety margin
    
    T = 10.0
    cooling = 0.99995
    min_T = 0.01
    
    # Precompute a list of all possible edges to speed up random selection
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    while time.time() - start_time < timeout:
        if T < min_T:
            break
            
        # Pick a random edge to remove
        # We need an edge that is currently present
        # And a non-edge that is currently absent
        # To avoid bias, we can pick random indices and check
        
        # Optimization: maintain list of current edges?
        # For n=16, checking is fast.
        
        while True:
            u, v = random.choice(all_pairs)
            if current_adj[u][v]:
                break
        
        while True:
            x, y = random.choice(all_pairs)
            if not current_adj[x][y]:
                break
        
        # Apply swap
        current_adj[u][v] = False
        current_adj[v][u] = False
        current_adj[x][y] = True
        current_adj[y][x] = True
        
        new_score = calc_score(current_adj)
        
        # Acceptance criterion
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj = copy.deepcopy(current_adj)
        else:
            # Simulated annealing acceptance
            # delta = current_score - new_score (negative)
            # prob = exp(delta / T)
            if random.random() < pow(2.718281828, (current_score - new_score) / T):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_adj = copy.deepcopy(current_adj)
            else:
                # Revert
                current_adj[u][v] = True
                current_adj[v][u] = True
                current_adj[x][y] = False
                current_adj[y][x] = False
                
        T *= cooling

    return {'n': n, 'edges': get_edges(best_adj)}
