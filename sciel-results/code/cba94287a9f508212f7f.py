"""Combined program: Deterministic seed + Local Search Extension."""
import time
import random
import copy

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 115.0  # 115 seconds to leave margin for overhead/return

    # 1. Seed: Greenwood-Gleason colouring of K16 (Clebsch) doubled to n=32
    # GF(16) multiplication
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    # Precompute log table for GF(16)
    log = {}
    x = 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
    
    # Initial n=32 construction
    def get_seed_graph(n):
        graph = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if i // 16 == j // 16:
                    c = log[(i % 16) ^ (j % 16)] % 3
                else:
                    c = 3
                graph[i][j] = c
                graph[j][i] = c
        return graph

    current_n = 32
    graph = get_seed_graph(current_n)
    best_n = current_n
    best_graph = copy.deepcopy(graph)

    def is_valid_graph(g, n):
        """Check if the graph has no monochromatic triangles."""
        for i in range(n):
            for j in range(i + 1, n):
                c_ij = g[i][j]
                for k in range(j + 1, n):
                    if g[i][k] == c_ij and g[j][k] == c_ij:
                        return False
        return True

    def add_vertex_backtrack(g, n):
        """
        Attempts to add a new vertex at index n to graph g (size n).
        Returns True if successful (g is updated to size n+1), False otherwise.
        Uses backtracking with pruning.
        """
        # We will assign colors to edges (i, n) for i in 0..n-1
        # assignments[i] will be the color of edge (i, n)
        assignments = [-1] * n
        
        # Precompute neighbors for pruning? 
        # For efficiency in the recursive step, we just check against previously assigned.
        
        def backtrack(i):
            if i == n:
                return True
            
            # Try each color for edge (i, n)
            # Heuristic: try colors that appear less often in the neighborhood of i? 
            # Or just 0,1,2,3. Randomizing might help avoid symmetry traps.
            colors_to_try = [0, 1, 2, 3]
            random.shuffle(colors_to_try)
            
            for c in colors_to_try:
                assignments[i] = c
                # Pruning: Check triangles (k, i, n) for all k < i
                # Triangle is (k, i, n). Edges: (k,i), (i,n), (k,n).
                # Colors: g[k][i], assignments[i], assignments[k]
                # We need g[k][i] != assignments[i] OR g[k][i] != assignments[k] OR assignments[i] != assignments[k]
                # Since assignments[i] is fixed to c, we check:
                # For any k < i, if g[k][i] == c, then assignments[k] must NOT be c.
                # i.e., if g[k][i] == c, we require assignments[k] != c.
                
                valid = True
                for k in range(i):
                    if g[k][i] == c:
                        if assignments[k] == c:
                            valid = False
                            break
                
                if valid:
                    if backtrack(i + 1):
                        return True
                
            assignments[i] = -1
            return False

        if backtrack(0):
            # Success: Update graph
            # Extend graph to n+1
            for i in range(n):
                row = g[i]
                row.append(assignments[i])
            g.append([0] * (n + 1))
            last_row = g[n]
            for i in range(n):
                last_row[i] = assignments[i]
            last_row[n] = 0 # self loop, irrelevant
            return True
        return False

    attempts = 0
    max_attempts = 20000
    consecutive_failures = 0
    
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        
        # Try to add a vertex
        # We work on a copy of the current graph to allow rollback if we want to try kicks
        # But add_vertex_backtrack modifies in place. 
        # To be safe, we can snapshot if we want to retry with different kicks.
        # However, if add_vertex fails, the graph is unchanged (backtrack returns False without modification if we structure it carefully).
        # In the implementation above, modifications happen ONLY if backtrack(0) returns True.
        # So if it returns False, graph is unchanged.
        
        if add_vertex_backtrack(graph, current_n):
            current_n += 1
            best_n = current_n
            best_graph = copy.deepcopy(graph)
            consecutive_failures = 0
            # If we reached a very high n, we might want to stop?
            # But let's keep going to see if we can go higher.
            # Also, adding a vertex makes the graph denser, harder to extend.
            
            # Optional: If we just extended, maybe do a quick check if we can extend again immediately?
            # The loop will handle that.
            
        else:
            consecutive_failures += 1
            
            # If we fail to add a vertex, we need to modify the existing graph to create "room".
            # Strategy: Randomly flip some edges (Kick).
            # We must ensure the graph remains valid after the kick.
            
            # Probability of kicking: Higher if we've failed many times in a row.
            kick_prob = min(0.5, 0.1 * consecutive_failures)
            
            if random.random() < kick_prob:
                # Save state
                old_graph = copy.deepcopy(graph)
                old_n = current_n
                
                # Perform Kick: Flip K edges
                # K should be small, e.g., 5-10% of edges
                num_edges = current_n * (current_n - 1) // 2
                num_flips = max(1, num_edges // 10)
                
                for _ in range(num_flips):
                    i = random.randint(0, current_n - 1)
                    j = random.randint(0, current_n - 1)
                    if i != j:
                        graph[i][j] = random.randint(0, 3)
                        graph[j][i] = graph[i][j]
                
                # Validate
                if is_valid_graph(graph, current_n):
                    # Kick was successful in maintaining validity.
                    # We can now try to add a vertex again in the next iteration.
                    # It's possible this kick opens up a solution for adding a vertex.
                    pass
                else:
                    # Kick broke validity. Rollback.
                    graph = old_graph
                    current_n = old_n
                    # Increment failures to encourage stronger kicks or different strategies?
                    # Or just accept that this kick failed.
            
            # If we have failed many times and kicking isn't helping, we might be stuck.
            # But we keep trying until deadline.

    # Finalize
    # Ensure we return the best valid state
    # best_graph is guaranteed to be valid because we only update it on successful add_vertex_backtrack
    # which implies the new graph is valid.
    # Initial state is valid.
    
    n = best_n
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append(best_graph[i][j])
            
    return {'n': n, 'colors': colors}
# EVOLVE-BLOCK-END
