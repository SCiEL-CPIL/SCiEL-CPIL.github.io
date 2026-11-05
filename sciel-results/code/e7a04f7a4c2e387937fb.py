import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s margin for safety

    def bitmask_to_edges(n, adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def fast_k4_count(n, adj):
        """Count K4 cliques using bit operations. Returns 0 if none."""
        count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    # Count edges in the common neighborhood
                    c = common
                    while c:
                        lsb = c & -c
                        w = lsb.bit_length() - 1
                        c ^= lsb
                        count += (adj[w] & c).bit_count()
        return count // 6

    def fast_has_ind6(n, adj):
        """Check if there's an independent set of size 6. Uses complement graph clique search."""
        mask = (1 << n) - 1
        # Build complement adjacency
        comp = [0] * n
        for i in range(n):
            comp[i] = mask & ~(adj[i] | (1 << i))
        
        # Quick necessary condition: each vertex in comp must have degree >= 5
        for i in range(n):
            if comp[i].bit_count() < 5:
                return False
        
        # Bron-Kerbosch style with pivot for finding a 6-clique in comp
        # Use iterative deepening or just recursive search with pruning
        def has_clique_size(cand_mask, size):
            if size == 0:
                return True
            if cand_mask.bit_count() < size:
                return False
            
            # Get candidate list
            c = cand_mask
            if c == 0:
                return False
            
            # Pick a vertex with max degree in cand_mask for better pruning
            best_v = -1
            best_deg = -1
            temp_c = c
            while temp_c:
                lsb = temp_c & -temp_c
                v = lsb.bit_length() - 1
                temp_c ^= lsb
                deg = (comp[v] & cand_mask).bit_count()
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
            
            v = best_v
            new_cand = cand_mask & comp[v] & ~(1 << v)
            
            # Recursive: include v
            if has_clique_size(new_cand, size - 1):
                return True
            
            # Exclude v
            return has_clique_size(cand_mask & ~(1 << v), size)
        
        return has_clique_size(mask, 6)

    def penalty(n, adj):
        """Compute penalty: 0 if valid, else positive value."""
        k4 = fast_k4_count(n, adj)
        if k4 > 0:
            return k4 * 100
        if fast_has_ind6(n, adj):
            return 1000
        return 0

    def local_search_sa(n, adj, time_limit):
        """Simulated annealing with edge flips."""
        current = adj[:]
        current_pen = penalty(n, current)
        
        if current_pen == 0:
            return current

        temp = 50.0
        cooling = 0.9995
        no_improve = 0
        best_pen = current_pen
        best_adj = current[:]

        while time.time() < time_limit:
            # Pick a random edge to flip
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            
            # Flip the edge
            had_edge = (current[u] >> v) & 1
            if had_edge:
                current[u] &= ~(1 << v)
                current[v] &= ~(1 << u)
            else:
                current[u] |= (1 << v)
                current[v] |= (1 << u)
            
            new_pen = penalty(n, current)
            
            if new_pen < current_pen:
                current_pen = new_pen
                no_improve = 0
                if new_pen < best_pen:
                    best_pen = new_pen
                    best_adj = current[:]
                if new_pen == 0:
                    return current
            else:
                delta = new_pen - current_pen
                if delta > 0 and random.random() < pow(2.71828, -delta / max(temp, 0.01)):
                    current_pen = new_pen
                    no_improve += 1
                else:
                    # Revert
                    if had_edge:
                        current[u] |= (1 << v)
                        current[v] |= (1 << u)
                    else:
                        current[u] &= ~(1 << v)
                        current[v] &= ~(1 << u)
                    no_improve += 1
            
            temp *= cooling
            
            # Restart if stuck
            if no_improve > 5000:
                no_improve = 0
                temp = max(temp, 10.0)
        
        return best_adj

    def local_search_hill(n, adj, time_limit):
        """Hill climbing with edge flips - only accept improvements."""
        current = adj[:]
        current_pen = penalty(n, current)
        
        if current_pen == 0:
            return current

        best_pen = current_pen
        best_adj = current[:]
        stuck_count = 0

        while time.time() < time_limit:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            
            had_edge = (current[u] >> v) & 1
            if had_edge:
                current[u] &= ~(1 << v)
                current[v] &= ~(1 << u)
            else:
                current[u] |= (1 << v)
                current[v] |= (1 << u)
            
            new_pen = penalty(n, current)
            
            if new_pen < current_pen:
                current_pen = new_pen
                stuck_count = 0
                if new_pen < best_pen:
                    best_pen = new_pen
                    best_adj = current[:]
                if new_pen == 0:
                    return current
            else:
                # Revert
                if had_edge:
                    current[u] |= (1 << v)
                    current[v] |= (1 << u)
                else:
                    current[u] &= ~(1 << v)
                    current[v] &= ~(1 << u)
                stuck_count += 1
            
            # If stuck too long, do a random restart
            if stuck_count > 2000:
                break
        
        return best_adj

    best_n = 17
    best_adj = None
    
    # Seed: Paley graph on 17
    squares = {x * x % 17 for x in range(1, 17)}
    adj17 = [0] * 17
    for i in range(17):
        for j in range(i + 1, 17):
            if (j - i) % 17 in squares:
                adj17[i] |= (1 << j)
                adj17[j] |= (1 << i)
    best_adj = adj17

    # Try to find larger graphs
    # Allocate time more aggressively to higher n
    targets = [35, 34, 33, 32, 31, 30, 29, 28, 26, 24, 22, 20]
    
    remaining_time = deadline - start_time
    # Give proportional time: higher n gets more attempts
    total_weight = sum(t for t in targets)
    
    for target_n in targets:
        if time.time() > deadline - 5:
            break
        
        # Time slice proportional to target_n
        time_slice = remaining_time * (target_n / total_weight)
        time_slice = max(time_slice, 2.0)  # Minimum 2s per target
        time_slice = min(time_slice, deadline - time.time() - 3)  # Don't exceed deadline
        
        if time_slice < 1.0:
            break
        
        # Multiple random restarts within the time slice
        num_restarts = max(1, int(time_slice / 3.0))
        found_valid = False
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random graph with appropriate edge density
            # For K4-free graphs, we want moderate density
            # For no ind-6, we want sufficient density
            adj = [0] * target_n
            # Use a structured initialization: start with random, then try to fix
            edge_prob = 0.4  # Moderate density
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if random.random() < edge_prob:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Run local search
            search_time = (deadline - time.time() - 3) / max(num_restarts - restart, 1)
            search_time = max(search_time, 1.0)
            
            if random.random() < 0.5:
                result_adj = local_search_sa(target_n, adj, time.time() + search_time)
            else:
                result_adj = local_search_hill(target_n, adj, time.time() + search_time)
            
            if penalty(target_n, result_adj) == 0:
                best_n = target_n
                best_adj = result_adj
                found_valid = True
                break
        
        if found_valid:
            # We found a valid graph for this n, try higher n? No, we go in descending order
            # So this is the best we can do. Stop.
            break

    # Fallback to seed if no improvement
    if best_n == 17:
        return {'n': 17, 'edges': bitmask_to_edges(17, best_adj)}
    
    return {'n': best_n, 'edges': bitmask_to_edges(best_n, best_adj)}
