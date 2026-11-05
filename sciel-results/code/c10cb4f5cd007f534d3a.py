import random
import time
import math

def construct():
    n = 20
    start_time = time.time()
    max_time = 115.0  # Safety margin
    
    best_edges = []
    best_count = 0
    
    # Precompute all pairs
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    pair_index = {(u, v): i for i, (u, v) in enumerate(all_pairs)}
    pair_index.update({(v, u): i for i, (u, v) in enumerate(all_pairs)})
    
    def is_valid_add(u, v, adj_mask, common):
        """Check if edge (u,v) can be added without violating K2,3-free constraint."""
        if (adj_mask[u] >> v) & 1:
            return False
        if common[u][v] > 2:
            return False
        
        # Adding (u,v) increases common[u][w] by 1 for all w in N(v), w != u
        # Need common[u][w] < 2 for all w in N(v), w != u
        mask_v = adj_mask[v] & ~(1 << u)
        if mask_v:
            w = 0
            m = mask_v
            while m:
                lsb = m & -m
                w = lsb.bit_length() - 1
                if common[u][w] >= 2:
                    return False
                m -= lsb
        
        # Similarly for common[v][w] < 2 for all w in N(u), w != v
        mask_u = adj_mask[u] & ~(1 << v)
        if mask_u:
            w = 0
            m = mask_u
            while m:
                lsb = m & -m
                w = lsb.bit_length() - 1
                if common[v][w] >= 2:
                    return False
                m -= lsb
        
        return True

    def add_edge(u, v, adj_mask, common, edges_set):
        """Add edge (u,v) to the graph state."""
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        edges_set.add((u, v))
        
        # Update common neighbor counts
        mask_v = adj_mask[v] & ~(1 << u)
        m = mask_v
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            common[u][w] += 1
            common[w][u] += 1
            m -= lsb
        
        mask_u = adj_mask[u] & ~(1 << v)
        m = mask_u
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            common[v][w] += 1
            common[w][v] += 1
            m -= lsb

    def remove_edge(u, v, adj_mask, common, edges_set):
        """Remove edge (u,v) from the graph state."""
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
        edges_set.discard((u, v))
        
        # For each w in N(v)\{u}, common[u][w] -= 1
        mask_v = adj_mask[v]
        m = mask_v
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            common[u][w] -= 1
            common[w][u] -= 1
            m -= lsb
        
        # For each w in N(u)\{v}, common[v][w] -= 1
        mask_u = adj_mask[u]
        m = mask_u
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            common[v][w] -= 1
            common[w][v] -= 1
            m -= lsb

    def greedy_fill(adj_mask, common, edges_set):
        """Greedy fill: add edges in random order if valid."""
        pairs = all_pairs[:]
        random.shuffle(pairs)
        for u, v in pairs:
            if is_valid_add(u, v, adj_mask, common):
                add_edge(u, v, adj_mask, common, edges_set)

    def simulated_annealing(initial_edges_set, deadline):
        """Simulated Annealing to maximize edges."""
        adj_mask = [0] * n
        common = [[0] * n for _ in range(n)]
        edges_set = set(initial_edges_set)
        
        for u, v in initial_edges_set:
            add_edge(u, v, adj_mask, common, edges_set)
            
        best_local_edges = set(edges_set)
        best_local_count = len(edges_set)
        
        # SA Parameters
        T = 1.0
        cooling_rate = 0.999
        min_T = 0.01
        
        while time.time() < deadline - 5:
            if T < min_T:
                T = 1.0 # Restart temperature if stuck too long or just reset for exploration
                # Or break if we've cooled down significantly and haven't improved
                # Let's just keep cooling, but if we hit min_T, we might restart search from best
                pass
            
            # Pick a random move: 
            # 1. Try to add a random missing edge
            # 2. Try to remove a random present edge (to escape local optimum)
            
            # Bias towards adding edges (90% add, 10% remove)
            if random.random() < 0.9:
                # Pick a random missing edge
                # We can pick a random pair and check if it's missing
                u, v = random.choice(all_pairs)
                if not ((adj_mask[u] >> v) & 1):
                    if is_valid_add(u, v, adj_mask, common):
                        add_edge(u, v, adj_mask, common, edges_set)
                        if len(edges_set) > best_local_count:
                            best_local_count = len(edges_set)
                            best_local_edges = set(edges_set)
            else:
                # Pick a random present edge to remove
                if edges_set:
                    u, v = random.choice(list(edges_set))
                    remove_edge(u, v, adj_mask, common, edges_set)
                    # After removal, we might be able to add more.
                    # We don't accept/reject here immediately, we just do the move.
                    # But we should track if this leads to a better state later.
                    # For simplicity, we just do the move. The "acceptance" is implicit 
                    # in the fact that we are exploring. 
                    # However, standard SA accepts worse states with probability p.
                    # Here, removing an edge always decreases count by 1.
                    # We should only do this if we are "stuck" or with prob p.
                    # Let's refine: 
                    # If we are removing, we do it with probability exp(-1/T).
                    # But we already decided to do it. 
                    # Let's restructure:
                    pass # This block is a bit messy. Let's use a cleaner SA loop.

        # The above SA loop is a bit pseudo. Let's implement a cleaner one.
        
        # Restart SA implementation properly
        adj_mask = [0] * n
        common = [[0] * n for _ in range(n)]
        edges_set = set(initial_edges_set)
        for u, v in initial_edges_set:
            add_edge(u, v, adj_mask, common, edges_set)
            
        best_local_edges = set(edges_set)
        best_local_count = len(edges_set)
        
        T = 1.0
        cooling_rate = 0.9995
        min_T = 0.05
        
        while time.time() < deadline - 5:
            if T < min_T:
                # Reset temperature to explore again
                T = 1.0
                # Optionally restart from best_local_edges to avoid getting lost
                # adj_mask = [0]*n
                # common = [[0]*n for _ in range(n)]
                # edges_set = set(best_local_edges)
                # for u,v in best_local_edges: add_edge(...)
                # This is expensive. Just reset T.
                
            # Generate a move
            # Try to add a random missing edge
            u, v = random.choice(all_pairs)
            if not ((adj_mask[u] >> v) & 1):
                if is_valid_add(u, v, adj_mask, common):
                    add_edge(u, v, adj_mask, common, edges_set)
                    if len(edges_set) > best_local_count:
                        best_local_count = len(edges_set)
                        best_local_edges = set(edges_set)
                else:
                    # Cannot add. Maybe remove a random edge to allow it?
                    # With probability p, remove a random edge from the current graph
                    # that is "blocking" or just random.
                    if random.random() < 0.1: # 10% chance to kick
                        if edges_set:
                            ru, rv = random.choice(list(edges_set))
                            remove_edge(ru, rv, adj_mask, common, edges_set)
            else:
                # Edge exists. Maybe remove it?
                # Only if we are stuck or with low probability
                if random.random() < 0.05:
                    remove_edge(u, v, adj_mask, common, edges_set)
                    
            T *= cooling_rate
            
        return best_local_edges

    def verify_edges(edge_list):
        """Verify that the edge list forms a K2,3-free graph."""
        adj = [0] * n
        for u, v in edge_list:
            if u == v:
                return False
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[u] & adj[v]
                if bin(common).count('1') > 2:
                    return False
        return True

    # Main search loop
    deadline = start_time + max_time
    while time.time() < deadline:
        # Greedy initialization
        adj_mask = [0] * n
        common = [[0] * n for _ in range(n)]
        edges_set = set()
        greedy_fill(adj_mask, common, edges_set)
        
        # Simulated Annealing
        result_edges = simulated_annealing(edges_set, deadline)
        
        # Verify and update best
        if verify_edges(result_edges) and len(result_edges) > best_count:
            best_count = len(result_edges)
            best_edges = [[u, v] for u, v in result_edges]
        
        if time.time() > deadline - 5:
            break

    if not best_edges:
        best_edges = []
        for i in range(5):
            base = i * 4
            for u in range(4):
                for v in range(u + 1, 4):
                    best_edges.append([base + u, base + v])
    
    return best_edges

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
