import time
import random
import itertools

def construct():
    start_time = time.time()
    time_limit = 290  # Keep 10s buffer for safety
    
    # 1. Generate the seed: Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    n_seed = q
    edges_seed = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    
    # Represent graph as adjacency matrix for fast checking
    # adj[i][j] = 1 if edge exists
    adj = [[0] * n_seed for _ in range(n_seed)]
    for i, j in edges_seed:
        adj[i][j] = 1
        adj[j][i] = 1
        
    # Helper to check if a set of vertices contains a K5
    def has_k5(vertices, adj_mat):
        # vertices is a list of indices in the CURRENT graph
        # We only care about K5s that include the NEW vertex? 
        # Actually, we need to check if the WHOLE graph has K5.
        # Since the seed has no K5, we only need to check K5s involving at least one new vertex.
        pass

    # Better approach: Check if adding vertex v with neighbor set N_v creates a K5 or barK5.
    # A K5 involving v consists of v and 4 neighbors of v that form a K4 in the subgraph induced by N_v.
    # A barK5 involving v consists of v and 4 non-neighbors of v that form a barK4 (independent set of size 4) in the subgraph induced by non-N_v.
    
    def creates_k5(v, neighbors, adj_mat):
        # Check if neighbors contain a K4
        # neighbors is a set/list of indices
        if len(neighbors) < 4:
            return False
        # Iterate over all 4-subsets of neighbors
        for combo in itertools.combinations(neighbors, 4):
            # Check if combo forms a K4
            is_k4 = True
            for i in range(4):
                for j in range(i+1, 4):
                    if adj_mat[combo[i]][combo[j]] == 0:
                        is_k4 = False
                        break
                if not is_k4:
                    break
            if is_k4:
                return True
        return False

    def creates_bark5(v, non_neighbors, adj_mat):
        # Check if non_neighbors contain an independent set of size 4
        if len(non_neighbors) < 4:
            return False
        for combo in itertools.combinations(non_neighbors, 4):
            is_bark4 = True
            for i in range(4):
                for j in range(i+1, 4):
                    if adj_mat[combo[i]][combo[j]] == 1:
                        is_bark4 = False
                        break
                if not is_bark4:
                    break
            if is_bark4:
                return True
        return False

    # We want to extend to n = 42
    target_n = 42
    current_n = n_seed
    current_adj = [row[:] for row in adj]
    
    # List of edges for final output
    current_edges = set(edges_seed)
    
    # Attempt to add vertices one by one
    success = True
    while current_n < target_n:
        if time.time() - start_time > time_limit - 10:
            break # Out of time, keep current best
        
        v_new = current_n
        # Try to find a valid neighbor set for v_new
        # Heuristic: Start with a random subset of existing vertices
        # To speed up, we can use a greedy approach:
        # 1. Start with no neighbors.
        # 2. Try to add edges to vertices that are "safe".
        
        # Simpler: Random restarts for this vertex
        found = False
        attempts = 0
        while not found and attempts < 500:
            if time.time() - start_time > time_limit - 5:
                break
            attempts += 1
            
            # Random neighbor set
            # Bias towards half density?
            # Let's try a random permutation and take a random prefix
            perm = list(range(current_n))
            random.shuffle(perm)
            k = random.randint(0, current_n)
            neighbors = set(perm[:k])
            non_neighbors = set(range(current_n)) - neighbors
            
            if creates_k5(v_new, neighbors, current_adj):
                continue
            if creates_bark5(v_new, non_neighbors, current_adj):
                continue
            
            # If valid, add the vertex
            found = True
            
        if not found:
            # Try a different random seed or strategy?
            # If we fail to find a neighbor set, we might need to backtrack or give up on this n
            # For simplicity, if we fail, we stop extending
            break
        
        # Add v_new to graph
        current_adj.append([0] * (current_n + 1))
        for i in range(current_n):
            current_adj[i].append(0)
        
        for u in neighbors:
            current_adj[v_new][u] = 1
            current_adj[u][v_new] = 1
            current_edges.add((min(u, v_new), max(u, v_new)))
            
        current_n += 1

    # If we reached target_n, we have a 42-vertex graph
    # Verify final graph just in case (optional, but good for safety)
    # The incremental checks should have ensured validity.
    
    if current_n > n_seed:
        return current_n, list(current_edges)
    else:
        return n_seed, edges_seed
