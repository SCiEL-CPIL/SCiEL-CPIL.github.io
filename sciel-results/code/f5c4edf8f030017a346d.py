import time
import random
import sys

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Total possible edges in K20 is 190
    total_edges = n * (n - 1) // 2
    
    # Map edge indices to (u, v)
    edge_list = []
    idx_to_edge = {}
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            idx_to_edge[idx] = (u, v)
            edge_list.append((u, v))
            idx += 1
            
    # Precompute for each edge index, the list of other edge indices that share a vertex with it?
    # Actually, for K2,3 constraint:
    # A set of edges is K2,3-free if for any pair of vertices (u, v), 
    # the number of common neighbors is <= 2.
    # Common neighbors of u and v are vertices w such that (u,w) and (v,w) are in the graph.
    # So, if we have edges (u,w) and (v,w), they contribute to the common neighbor count of u and v.
    # Let's define a conflict:
    # Two edges e1=(u1, v1) and e2=(u2, v2) don't directly conflict.
    # A set of edges conflicts if there exist u, v such that |N(u) ∩ N(v)| >= 3.
    # This means there are 3 vertices w1, w2, w3 such that (u,wi) and (v,wi) are all in the graph.
    
    # Efficient check for validity of a bitmask:
    # Iterate over all pairs (u, v).
    # Count common neighbors.
    
    def is_valid(mask):
        # For each pair u < v
        for u in range(n):
            # Neighbors of u
            nu = 0
            for v in range(u + 1, n):
                if mask & (1 << idx_to_edge[(u, v)][0] if False else 0): # dummy
                    pass
            # Let's use adjacency masks for speed
            adj = [0] * n
            for i in range(total_edges):
                if mask & (1 << i):
                    u, v = edge_list[i]
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            for u in range(n):
                for v in range(u + 1, n):
                    common = (adj[u] & adj[v]).bit_count()
                    if common > 2:
                        return False
        return True

    def count_edges(mask):
        return mask.bit_count()

    def repair(mask):
        """Remove edges to make the graph K2,3-free. 
        Strategy: While invalid, find a violating pair (u,v) with >=3 common neighbors,
        and remove one of the edges incident to one of the common neighbors."""
        adj = [0] * n
        for i in range(total_edges):
            if mask & (1 << i):
                u, v = edge_list[i]
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        changed = True
        while changed:
            changed = False
            # Find a violation
            viol_u, viol_v, common_mask = -1, -1, 0
            for u in range(n):
                for v in range(u + 1, n):
                    common = adj[u] & adj[v]
                    if common.bit_count() > 2:
                        viol_u, viol_v = u, v
                        common_mask = common
                        break
                if viol_u != -1:
                    break
            
            if viol_u == -1:
                break
            
            # We have a violation. Pick a random common neighbor w.
            # Remove one of the edges (u, w) or (v, w).
            # Removing (u, w) reduces common neighbors of u with all its neighbors.
            # Let's just pick a random edge from the triangle (u, v, w) structure?
            # Actually, removing (u, w) or (v, w) fixes the count for (u, v).
            # But might break other constraints? No, removing edges only decreases counts.
            # So we can safely remove one.
            
            # Get list of common neighbors
            w_list = []
            cm = common_mask
            while cm:
                lsb = cm & -cm
                w = lsb.bit_length() - 1
                cm -= lsb
                w_list.append(w)
            
            w = random.choice(w_list)
            
            # Decide which edge to remove: (u, w) or (v, w)
            # Remove the one that seems "less useful" or just random.
            if random.random() < 0.5:
                u_rem, w_rem = u, w
            else:
                u_rem, w_rem = v, w
            
            # Find edge index
            e_idx = idx_to_edge[(min(u_rem, w_rem), max(u_rem, w_rem))]
            
            # Remove edge
            mask &= ~(1 << e_idx)
            
            # Update adjacency
            adj[u_rem] &= ~(1 << w_rem)
            adj[w_rem] &= ~(1 << u_rem)
            
            changed = True
            
        return mask

    def crossover(parent1, parent2):
        """Uniform crossover with repair."""
        child = 0
        for i in range(total_edges):
            if random.random() < 0.5:
                child |= (parent1 >> i) & 1
            else:
                child |= (parent2 >> i) & 1
            child <<= 0 # dummy to keep python happy if I was doing shifts, but I'm doing bitwise
        # Wait, the loop above is wrong for building the mask.
        child = 0
        for i in range(total_edges):
            bit = 0
            if random.random() < 0.5:
                bit = (parent1 >> i) & 1
            else:
                bit = (parent2 >> i) & 1
            if bit:
                child |= (1 << i)
        
        return repair(child)

    def mutation(mask, rate=0.05):
        """Add or remove random edges, then repair if necessary."""
        # To maintain validity, we can only add edges if we repair afterwards.
        # Or we can try to add a valid edge.
        # For GA, let's do a "random perturbation" followed by repair.
        new_mask = mask
        for i in range(total_edges):
            if random.random() < rate:
                if new_mask & (1 << i):
                    new_mask &= ~(1 << i)
                else:
                    new_mask |= (1 << i)
        
        return repair(new_mask)

    def greedy_initial():
        """Generate a valid graph using greedy heuristic."""
        mask = 0
        # Sort edges randomly
        edges = list(range(total_edges))
        random.shuffle(edges)
        for i in edges:
            # Try adding edge i
            new_mask = mask | (1 << i)
            if is_valid(new_mask):
                mask = new_mask
        return mask

    # Population initialization
    pop_size = 40
    population = []
    
    # Seed with a few good greedy solutions
    for _ in range(pop_size):
        if time.time() > deadline:
            break
        m = greedy_initial()
        population.append(m)
        
    # If we didn't fill population due to time, fill with empty or random valid
    while len(population) < pop_size:
        population.append(0)
        
    best_mask = max(population, key=count_edges)
    best_count = count_edges(best_mask)
    
    # GA Loop
    elite_size = 5
    max_generations = 1000 # Limit, but time is the real limit
    
    for gen in range(max_generations):
        if time.time() > deadline:
            break
            
        # Sort population by fitness (edge count)
        population.sort(key=count_edges, reverse=True)
        
        # Update best
        if count_edges(population[0]) > best_count:
            best_count = count_edges(population[0])
            best_mask = population[0]
            
        # Create new population
        new_population = []
        
        # Elitism
        for i in range(elite_size):
            new_population.append(population[i])
            
        # Generate offspring to fill the rest
        while len(new_population) < pop_size:
            if time.time() > deadline:
                break
                
            # Tournament selection
            def select():
                candidates = random.sample(population, 3)
                return max(candidates, key=count_edges)
            
            p1 = select()
            p2 = select()
            
            # Crossover
            if random.random() < 0.9:
                child = crossover(p1, p2)
            else:
                child = p1
                
            # Mutation
            if random.random() < 0.5:
                child = mutation(child)
                
            new_population.append(child)
            
        population = new_population

    # Final polish: Local search on the best found
    # Try to add any valid edge to best_mask
    final_mask = best_mask
    adj = [0] * n
    for i in range(total_edges):
        if final_mask & (1 << i):
            u, v = edge_list[i]
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
    # Try adding missing edges
    for i in range(total_edges):
        if not (final_mask & (1 << i)):
            u, v = edge_list[i]
            # Check if adding (u,v) is valid
            # Common neighbors of u and v in current graph must be <= 2
            # Adding (u,v) doesn't change common neighbors of u and v.
            # It changes common neighbors of (u, w) for w in N(v) and (v, w) for w in N(u).
            # Specifically, for any w in N(v), common(u, w) increases by 1 (v becomes common neighbor).
            # So we need common(u, w) <= 1 for all w in N(v).
            # And common(v, w) <= 1 for all w in N(u).
            
            valid_add = True
            # Check w in N(v)
            nv = adj[v]
            w = nv
            while w:
                lsb = w & -w
                w_idx = lsb.bit_length() - 1
                w -= lsb
                if (adj[u] & adj[w_idx]).bit_count() >= 2:
                    valid_add = False
                    break
            if valid_add:
                nu = adj[u]
                w = nu
                while w:
                    lsb = w & -w
                    w_idx = lsb.bit_length() - 1
                    w -= lsb
                    if (adj[v] & adj[w_idx]).bit_count() >= 2:
                        valid_add = False
                        break
            
            if valid_add:
                final_mask |= (1 << i)
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
    # Reconstruct edges from final_mask
    edges = []
    for i in range(total_edges):
        if final_mask & (1 << i):
            u, v = edge_list[i]
            edges.append([u, v])
            
    return edges

if __name__ == "__main__":
    start = time.time()
    edges = construct()
    end = time.time()
    print(f"Time: {end-start:.2f}s, Edges: {len(edges)}")
    # Verify validity
    n = 20
    adj = [0]*n
    for u, v in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    valid = True
    for u in range(n):
        for v in range(u+1, n):
            common = (adj[u] & adj[v]).bit_count()
            if common >= 3:
                valid = False
                print(f"Invalid: {u}, {v} have {common} common neighbors")
    print(f"Valid: {valid}")

