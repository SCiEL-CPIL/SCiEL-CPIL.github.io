import random
import itertools

def has_independent_set_size_k(n, adj_masks, k):
    # adj_masks is a list of ints, where adj_masks[i] is the bitmask of neighbors of i
    
    def dfs(available, depth):
        if depth == k:
            return True
        if available == 0:
            return False
        if available.bit_count() < k - depth:
            return False
        
        # Find vertex with max degree in induced subgraph
        max_deg = -1
        v_idx = -1
        avail_bits = available
        while avail_bits:
            lsb = avail_bits & -avail_bits
            u = lsb.bit_length() - 1
            deg = (adj_masks[u] & available).bit_count()
            if deg > max_deg:
                max_deg = deg
                v_idx = u
            avail_bits ^= lsb
        
        if v_idx == -1:
            return False
            
        v_bit = 1 << v_idx
        
        # Branch 1: Include v
        # Remove v and its neighbors from available
        new_avail_include = available & ~(adj_masks[v_idx] | v_bit)
        if dfs(new_avail_include, depth + 1):
            return True
        
        # Branch 2: Exclude v
        new_avail_exclude = available & ~v_bit
        if dfs(new_avail_exclude, depth):
            return True
        
        return False

    return dfs((1 << n) - 1, 0)

def construct():
    # We want to maximize n.
    # Try n from 35 down to 28.
    for n in range(35, 27, -1):
        # Try multiple random trials
        for _ in range(500):
            # Initialize empty graph
            adj = {i: set() for i in range(n)}
            edges = []
            
            # List of all possible edges
            all_edges = list(itertools.combinations(range(n), 2))
            random.shuffle(all_edges)
            
            # common[a][c] is True if a and c have a common neighbor
            common = [[False]*n for _ in range(n)]
            
            for u, v in all_edges:
                # Check if adding (u, v) creates a C5
                # Path: u-a-b-c-v
                # a in N(u), c in N(v), b in N(a) intersect N(c)
                # common[a][c] is True if N(a) intersect N(c) is not empty
                
                creates_c5 = False
                for a in adj[u]:
                    for c in adj[v]:
                        if common[a][c]:
                            creates_c5 = True
                            break
                    if creates_c5:
                        break
                
                if creates_c5:
                    continue
                
                # Add edge
                adj[u].add(v)
                adj[v].add(u)
                edges.append([u, v])
                
                # Update common
                # v is now a neighbor of all a in N(u)
                # So for any a in N(u) and c in N(v), v is a common neighbor of a and c
                for a in adj[u]:
                    for c in adj[v]:
                        common[a][c] = True
                        common[c][a] = True
                for a in adj[v]:
                    for c in adj[u]:
                        common[a][c] = True
                        common[c][a] = True
            
            # Check alpha < 9
            # Convert adj sets to bitmasks
            adj_masks = [0] * n
            for i in range(n):
                mask = 0
                for j in adj[i]:
                    mask |= (1 << j)
                adj_masks[i] = mask
            
            if not has_independent_set_size_k(n, adj_masks, 9):
                return {'n': n, 'edges': edges}
        
        # If no valid graph found for this n, try smaller n
    # Fallback to seed
    return {'n': 28, 'edges': [[4 * b + i, 4 * b + j] for b in range(7) for i in range(4) for j in range(i + 1, 4)]}
