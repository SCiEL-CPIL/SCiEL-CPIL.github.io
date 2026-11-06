import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 120s/150s kill

    n = 40

    # Precompute all possible edges
    all_possible_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]

    def would_create_c6(adj, u, v):
        """
        Check if adding edge (u, v) to the graph represented by adjacency bitmasks `adj`
        creates a C6.
        """
        Nu = adj[u]
        Nv = adj[v]
        
        a_mask = Nu & ~(1 << v)
        while a_mask:
            lsb_a = a_mask & -a_mask
            a = lsb_a.bit_length() - 1
            a_mask ^= lsb_a
            
            d_mask = Nv & ~(1 << u) & ~(1 << a)
            while d_mask:
                lsb_d = d_mask & -d_mask
                d = lsb_d.bit_length() - 1
                d_mask ^= lsb_d
                
                Na = adj[a]
                Nd = adj[d]
                
                b_mask = Na & ~(1 << u) & ~(1 << v) & ~(1 << d)
                c_mask = Nd & ~(1 << u) & ~(1 << v) & ~(1 << a)
                
                b_temp = b_mask
                while b_temp:
                    lsb_b = b_temp & -b_temp
                    b = lsb_b.bit_length() - 1
                    b_temp ^= lsb_b
                    
                    if adj[b] & c_mask:
                        return True
        
        return False

    def build_adj(edges_set):
        adj = [0] * n
        for u, v in edges_set:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def run_greedy(start_edges, seed_val):
        random.seed(seed_val)
        
        adj = build_adj(start_edges)
        edges = set(start_edges)
        
        possible_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    possible_edges.append((i, j))
        
        random.shuffle(possible_edges)
        
        for u, v in possible_edges:
            if time.time() > deadline:
                return edges
            if not would_create_c6(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.add((u, v))
        
        return edges

    def local_search(edges_set, seed_val):
        """
        Improve the edge set by adding edges and swapping.
        """
        random.seed(seed_val)
        
        adj = build_adj(edges_set)
        current_edges = set(edges_set)
        
        # Pass 1: Add any possible non-edge
        # We do this in a loop because adding one edge might make another non-edge addable?
        # No, adding an edge only creates more C6 constraints, so it can only make it *harder* to add other edges.
        # So a single pass is sufficient for "add if valid".
        non_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    non_edges.append((i, j))
        
        random.shuffle(non_edges)
        
        for u, v in non_edges:
            if time.time() > deadline - 2:
                break
            if not would_create_c6(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges.add((u, v))
        
        # Pass 2: Try swaps (remove 1, add 2)
        # We iterate as long as we have time and we keep finding improvements.
        # To avoid infinite loops on bad seeds, we limit the number of *consecutive* failures?
        # Or just run until time is up.
        
        removable = list(current_edges)
        random.shuffle(removable)
        
        # Re-generate non_edges list as it changed
        non_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    non_edges.append((i, j))
        random.shuffle(non_edges)
        
        consecutive_failures = 0
        max_failures = 50
        
        while time.time() < deadline - 2:
            if consecutive_failures > max_failures:
                break
                
            if not removable:
                break
            
            e_rem = removable[random.randint(0, len(removable) - 1)]
            u_rem, v_rem = e_rem
            
            # Save state
            old_adj = adj[:]
            old_edges = set(current_edges)
            
            # Remove edge
            adj[u_rem] &= ~(1 << v_rem)
            adj[v_rem] &= ~(1 << u_rem)
            current_edges.remove(e_rem)
            
            # Try to add 2 non-edges
            added = 0
            local_non_edges = non_edges[:]
            random.shuffle(local_non_edges)
            
            for u, v in local_non_edges:
                if added >= 2:
                    break
                if time.time() > deadline - 2:
                    break
                if (u, v) in current_edges:
                    continue
                if not would_create_c6(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    current_edges.add((u, v))
                    added += 1
            
            if added >= 2:
                # Success: Edge count increased by (added - 1)
                consecutive_failures = 0
                # Update removable and non_edges for next iteration
                removable = list(current_edges)
                random.shuffle(removable)
                
                non_edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if not (adj[i] & (1 << j)):
                            non_edges.append((i, j))
                random.shuffle(non_edges)
                
                # Continue searching for more improvements
            else:
                # Revert
                adj = old_adj
                current_edges = old_edges
                consecutive_failures += 1

        return current_edges

    best_edges = set()
    best_count = 0
    
    seed_counter = 42
    restart_count = 0
    
    # Initial phase: Standard restarts
    while time.time() < deadline - 10:
        
        # Alternate between K5 seed and empty seed
        if restart_count % 2 == 0:
            start_edges = set()
            # Add K5s
            for k in range(8):
                for a in range(5):
                    for b in range(a + 1, 5):
                        u, v = 5 * k + a, 5 * k + b
                        start_edges.add((u, v))
        else:
            start_edges = set()
        
        # Run a greedy construction
        edges = run_greedy(start_edges, seed_counter)
        
        # Run local search on the result
        if time.time() < deadline - 5:
            edges = local_search(edges, seed_counter + 1000)
            
        count = len(edges)
        if count > best_count:
            best_count = count
            best_edges = set(edges)
            
        seed_counter += 1
        restart_count += 1

    # Second phase: Perturbation of best graph
    # If we have a good graph, shake it and try to improve.
    while time.time() < deadline - 2:
        if not best_edges:
            break
            
        # Create a perturbed version
        perturbed_edges = set(best_edges)
        num_to_remove = random.randint(5, 10)
        removable_list = list(perturbed_edges)
        random.shuffle(removable_list)
        for i in range(min(num_to_remove, len(removable_list))):
            perturbed_edges.remove(removable_list[i])
        
        # Run greedy to fill in
        edges = run_greedy(perturbed_edges, seed_counter)
        
        # Run local search
        if time.time() < deadline - 1:
            edges = local_search(edges, seed_counter + 2000)
            
        count = len(edges)
        if count > best_count:
            best_count = count
            best_edges = set(edges)
            
        seed_counter += 1
        restart_count += 1

    # Convert back to list of lists
    result = [[u, v] for u, v in best_edges]
    return result
