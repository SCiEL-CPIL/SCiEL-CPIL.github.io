import time
import random

def construct():
    start_time = time.time()
    time_limit = 115  # Safety margin
    
    def get_seed_graph(n):
        # Return a valid graph of size n, or None if not found quickly
        # We will try to repair a random graph
        
        # Helper: Check K4
        def find_k4(adj, n):
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u] & (1 << v):
                        common = adj[u] & adj[v]
                        c = common
                        while c:
                            lsb = c & -c
                            w = lsb.bit_length() - 1
                            c &= c - 1
                            if w == u or w == v: continue
                            # u, v, w form a triangle?
                            # u-v is edge.
                            # w is in N(u) and N(v). So u-w and v-w are edges.
                            # Yes, triangle.
                            # Check for x in N(u) & N(v) & N(w)
                            common3 = adj[u] & adj[v] & adj[w]
                            if common3:
                                x_lsb = common3 & -common3
                                x = x_lsb.bit_length() - 1
                                return (u, v, w, x)
            return None

        # Helper: Find Independent Set of size 6
        def find_i6(adj, n):
            # Backtracking search for I6
            # We search in the complement graph for K6
            # To speed up, we use a heuristic: sort vertices by degree in complement (descending)
            # Degree in complement = n - 1 - degree in original
            
            # Precompute degrees
            degs = [bin(adj[i]).count('1') for i in range(n)]
            comp_degs = [n - 1 - d for d in degs]
            
            # Order vertices by comp_degree descending
            order = sorted(range(n), key=lambda x: comp_degs[x], reverse=True)
            
            # Map original vertex to index in order? No, just use the list of candidates.
            
            def backtrack(candidates, current_set):
                if len(current_set) == 6:
                    return current_set
                if len(candidates) < 6 - len(current_set):
                    return None
                
                # Try each candidate
                for i in range(len(candidates)):
                    v = candidates[i]
                    # New candidates must be non-neighbors of v in original graph
                    # i.e., neighbors in complement
                    # And must be from candidates[i+1:]
                    new_cands = []
                    for j in range(i + 1, len(candidates)):
                        u = candidates[j]
                        if not (adj[v] & (1 << u)):
                            new_cands.append(u)
                    
                    result = backtrack(new_cands, current_set + [v])
                    if result:
                        return result
                return None

            return backtrack(order, [])

        # Repair Loop
        adj = [0] * n
        # Initialize with random edges
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        steps = 0
        max_steps = 20000
        while time.time() - start_time < time_limit and steps < max_steps:
            k4 = find_k4(adj, n)
            if k4:
                u, v, w, x = k4
                # Remove a random edge from the K4
                edges = [(u,v), (u,w), (u,x), (v,w), (v,x), (w,x)]
                ru, rv = random.choice(edges)
                adj[ru] &= ~(1 << rv)
                adj[rv] &= ~(1 << ru)
            else:
                i6 = find_i6(adj, n)
                if i6:
                    # Add a random edge within the independent set
                    # Pick two distinct vertices from i6
                    u, v = random.sample(i6, 2)
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                else:
                    # Valid graph found
                    return adj
            steps += 1
            
        return None

    # Try n=35
    best_n = 17
    best_adj = None
    
    # Try to find a valid graph for n=35
    for target_n in [35, 36]:
        if time.time() - start_time > time_limit - 5:
            break
            
        found_adj = None
        for restart in range(5):
            if time.time() - start_time > time_limit - 5:
                break
            found_adj = get_seed_graph(target_n)
            if found_adj:
                best_n = target_n
                best_adj = found_adj
                break
        
        if best_adj:
            break

    # If we found a valid graph, return it
    if best_adj:
        edges = []
        for i in range(best_n):
            for j in range(i + 1, best_n):
                if best_adj[i] & (1 << j):
                    edges.append([i, j])
        return {'n': best_n, 'edges': edges}
    
    # Fallback to seed
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}
