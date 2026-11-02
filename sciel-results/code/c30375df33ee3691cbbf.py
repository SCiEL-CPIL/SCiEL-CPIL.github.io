import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = 145  # Safety margin
    
    # Use a base graph with better girth to start?
    # The prompt says "KEEP this design... K5 base graph".
    # I will stick to K5 but use the cycle-breaking search.
    
    M = 64
    n_vertices = 5 * M  # 320
    
    base_edges = [
        (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 2), (1, 3), (1, 4),
        (2, 3), (2, 4),
        (3, 4)
    ]
    num_edges = len(base_edges)
    
    def build_adj_matrix(voltages):
        adj_mat = np.zeros((n_vertices, n_vertices), dtype=bool)
        for i, (u, v) in enumerate(base_edges):
            val = voltages[i]
            offsets = np.arange(M)
            
            src_u = u * M + offsets
            dst_v = v * M + (offsets + val) % M
            adj_mat[src_u, dst_v] = True
            
            src_v = v * M + offsets
            dst_u = u * M + (offsets + M - val) % M
            adj_mat[src_v, dst_u] = True
            
        return adj_mat

    def find_short_cycles(adj_mat, max_len):
        """
        Find all cycles of length < max_len.
        Returns a list of lists of vertices representing the cycles.
        """
        n = n_vertices
        cycles = []
        # To avoid duplicates, we can just find one per source if possible,
        # or collect all. For local search, just finding some is enough.
        
        for start in range(n):
            dist = np.full(n, -1, dtype=int)
            parent = np.full(n, -1, dtype=int)
            dist[start] = 0
            queue = [start]
            head = 0
            
            while head < len(queue):
                u = queue[head]
                head += 1
                du = dist[u]
                
                if du >= max_len - 1:
                    continue
                    
                neighbors = np.where(adj_mat[u])[0]
                for v in neighbors:
                    if dist[v] == -1:
                        dist[v] = du + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        if parent[u] != v:
                            cycle_len = du + dist[v] + 1
                            if cycle_len < max_len:
                                # Reconstruct cycle
                                # Path start -> ... -> u
                                # Path start -> ... -> v
                                # Edge (u, v)
                                # This logic is tricky to reconstruct quickly.
                                # For the purpose of perturbation, we just need to know 
                                # which *base edges* are involved? 
                                # Or just perturb random edges in the lifted graph?
                                # We can't perturb lifted edges directly, we perturb base voltages.
                                # A lifted edge (a, b) corresponds to a base edge (u, v) 
                                # where a = u*M + offset_a, b = v*M + offset_b.
                                # We need to map lifted edge to base edge index.
                                
                                # Let's just store the lifted edge (u, v) for now.
                                # And later map it to a base edge to perturb.
                                cycles.append((u, v, cycle_len))
                                
        return cycles

    def get_base_edge_index(u_lift, v_lift):
        """
        Map a lifted edge (u_lift, v_lift) to the base edge index.
        u_lift = u_base * M + offset_u
        v_lift = v_base * M + offset_v
        We need to find i such that base_edges[i] is (u_base, v_base) or (v_base, u_base).
        """
        u_base = u_lift // M
        v_base = v_lift // M
        
        # Check all base edges
        for i, (bu, bv) in enumerate(base_edges):
            if (bu == u_base and bv == v_base) or (bu == v_base and bv == u_base):
                return i
        return -1

    # Main Loop
    voltages = [random.randint(0, M - 1) for _ in range(num_edges)]
    adj_mat = build_adj_matrix(voltages)
    
    while time.time() < deadline:
        short_cycles = find_short_cycles(adj_mat, g)
        
        if not short_cycles:
            # Valid graph found
            edges = []
            for u in range(n_vertices):
                vs = np.where(adj_mat[u, u+1:])[0]
                for v in vs:
                    edges.append((u, u + 1 + v))
            return n_vertices, edges
            
        # Pick a random short cycle
        # Actually, just pick the first one found to save time
        u_lift, v_lift, _ = short_cycles[0]
        
        # Find the base edge index
        base_idx = get_base_edge_index(u_lift, v_lift)
        
        if base_idx != -1:
            # Perturb this voltage
            voltages[base_idx] = random.randint(0, M - 1)
            # Rebuild adjacency
            adj_mat = build_adj_matrix(voltages)
        else:
            # Should not happen
            break
            
    # Fallback
    edges = []
    for u in range(n_vertices):
        vs = np.where(adj_mat[u, u+1:])[0]
        for v in vs:
            edges.append((u, u + 1 + v))
    return n_vertices, edges
