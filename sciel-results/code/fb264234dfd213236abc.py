import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 150s kill

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

    def spectral_seed():
        """
        Create a high-density initial graph using a spectral/structured approach.
        We use a "shift" construction on a cyclic group Z_40.
        This creates a circulant graph. We pick a set of shifts that avoids C6.
        A circulant graph C_n(S) has C6 if there is a solution to:
        s1 + s2 + s3 + s4 + s5 + s6 = 0 (mod n) with si in S (allowing signs? no, just steps).
        Actually, a cycle of length 6 in a circulant graph corresponds to a sequence of 6 steps from S
        that sums to 0 mod n.
        
        Alternatively, we can just use a known good dense graph structure.
        The Paley graph of order 41 is C6-free? No, Paley(41) has many cycles.
        
        Let's try a different spectral seed:
        Start with a random graph with p=0.5, then greedily remove edges that form C6?
        Too slow.
        
        Let's try a structured seed:
        Divide vertices into 5 groups of 8.
        Connect all vertices in group i to group (i+1) mod 5?
        This creates a 5-partite graph.
        Bipartite graphs are C6-free? No, C6 is bipartite.
        
        Let's stick to the K5 seed but make it more structured or use a different high-density seed.
        The current K5 seed gives ~80 edges. Greedy adds ~20.
        
        Let's try a seed based on a strongly regular graph.
        The Hoffman-Singleton graph is (50, 22, 0, 1). Not 40.
        
        Let's just use a random seed with high density and then "repair" it.
        Or, use the K5 seed but add a few more edges to the seed itself to jumpstart the greedy.
        
        Actually, the inspiration suggests "spectral construction".
        Let's try initializing with a random graph with p=0.3, then running a local search
        that starts with "remove edges that create C6" (repair) and then "add edges".
        
        However, the most effective part of the current program is the local search.
        Let's improve the seed diversity.
        """
        edges = set()
        # K5 seed
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    u, v = 5 * k + a, 5 * k + b
                    edges.add((u, v))
        return edges

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

    def perturb_and_refine(edges_set, seed_val, num_remove):
        """
        Remove some edges randomly, then run greedy + local search to fill back in.
        """
        random.seed(seed_val)
        perturbed = set(edges_set)
        removable_list = list(perturbed)
        random.shuffle(removable_list)
        
        for i in range(min(num_remove, len(removable_list))):
            perturbed.remove(removable_list[i])
            
        edges = run_greedy(perturbed, seed_val + 1000)
        
        if time.time() < deadline - 5:
            edges = local_search(edges, seed_val + 2000)
            
        return edges

    best_edges = set()
    best_count = 0
    
    seed_counter = 42
    restart_count = 0
    
    # Initial phase: Standard restarts with varied seeds
    while time.time() < deadline - 15:
        
        # Alternate between different seed strategies
        strategy = restart_count % 3
        
        if strategy == 0:
            # K5 seed
            start_edges = set()
            for k in range(8):
                for a in range(5):
                    for b in range(a + 1, 5):
                        u, v = 5 * k + a, 5 * k + b
                        start_edges.add((u, v))
        elif strategy == 1:
            # Empty seed
            start_edges = set()
        else:
            # Random sparse seed
            start_edges = set()
            random.seed(seed_counter)
            for _ in range(30):
                u, v = random.sample(range(n), 2)
                if u > v: u, v = v, u
                start_edges.add((u, v))
        
        # Run a greedy construction
        edges = run_greedy(start_edges, seed_counter)
        
        # Run local search on the result
        if time.time() < deadline - 10:
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
        num_to_remove = random.randint(3, 8)
        edges = perturb_and_refine(best_edges, seed_counter, num_to_remove)
        
        count = len(edges)
        if count > best_count:
            best_count = count
            best_edges = set(edges)
            
        seed_counter += 1
        restart_count += 1

    # Convert back to list of lists
    result = [[u, v] for u, v in best_edges]
    return result
