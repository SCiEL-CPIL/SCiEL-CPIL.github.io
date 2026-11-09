import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    def get_max_independent_set_size(adj, n, limit=10):
        """
        Returns the size of the largest independent set in G, capped at 'limit'.
        An independent set in G is a clique in the complement of G.
        We build the complement adjacency and find max clique.
        """
        # Complement adjacency: ~adj[i] & mask
        mask = (1 << n) - 1
        comp_adj = [0] * n
        for i in range(n):
            comp_adj[i] = mask ^ adj[i]
            comp_adj[i] &= ~(1 << i) # no self loops
        
        # Branch and bound for max clique in comp_adj
        best = 0
        # Order vertices by degree in comp_adj (descending) to prune better
        # Actually, standard Bron-Kerbosch with pivot or simple B&B
        # For small n (40) and limit 10, simple B&B is fine.
        
        def expand(clique, candidates):
            nonlocal best
            if len(clique) + len(candidates) <= best:
                return
            if len(clique) >= limit:
                best = limit
                return
            if not candidates:
                if len(clique) > best:
                    best = len(clique)
                return
            
            # Pick a pivot to partition candidates
            # Simple version: just iterate
            for i in range(len(candidates)):
                v = candidates[i]
                new_clique = clique | (1 << v)
                # New candidates are neighbors of v in comp_adj that are in candidates
                new_candidates_mask = candidates & comp_adj[v]
                new_candidates = []
                # Extract bits
                temp = new_candidates_mask
                while temp:
                    lsb = temp & -temp
                    idx = (lsb.bit_length() - 1)
                    new_candidates.append(idx)
                    temp ^= lsb
                
                expand(new_clique, new_candidates)
                
                # Remove v from candidates for subsequent iterations
                candidates_mask = candidates_mask ^ (1 << v) if False else None 
                # The loop above modifies candidates list implicitly by slicing? 
                # Better to pass masks.
            
        # Let's rewrite expand with masks for speed
        best_size = 0
        def expand_mask(clique_mask, cand_mask):
            nonlocal best_size
            if bin(clique_mask).count('1') + bin(cand_mask).count('1') <= best_size:
                return
            if bin(clique_mask).count('1') >= limit:
                best_size = limit
                return
            if cand_mask == 0:
                if bin(clique_mask).count('1') > best_size:
                    best_size = bin(clique_mask).count('1')
                return
            
            # Pivot selection: pick v in cand_mask with max degree in cand_mask
            # For simplicity, just pick first or random? 
            # Let's pick one with high degree to reduce branching
            best_v = -1
            best_deg = -1
            temp = cand_mask
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                deg = bin(cand_mask & comp_adj[v]).count('1')
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                temp ^= lsb
            
            if best_v == -1: return
            
            # Branch 1: Include best_v
            expand_mask(clique_mask | (1 << best_v), cand_mask & comp_adj[best_v])
            
            # Branch 2: Exclude best_v
            expand_mask(clique_mask, cand_mask ^ (1 << best_v))
            
        expand_mask(0, mask)
        return best_size

    def has_triangle(adj, n):
        for i in range(n):
            neighbors = adj[i]
            # Check if any two neighbors are connected
            # Iterate through neighbors
            temp = neighbors
            while temp:
                lsb1 = temp & -temp
                v1 = lsb1.bit_length() - 1
                temp ^= lsb1
                
                # Check edges from v1 to other neighbors of i
                # (adj[v1] & neighbors)
                if adj[v1] & neighbors:
                    return True
        return False

    def find_triangle(adj, n):
        for i in range(n):
            neighbors = adj[i]
            temp = neighbors
            while temp:
                lsb1 = temp & -temp
                v1 = lsb1.bit_length() - 1
                temp ^= lsb1
                common = adj[v1] & neighbors
                if common:
                    lsb2 = common & -common
                    v2 = lsb2.bit_length() - 1
                    return (i, v1, v2)
        return None

    def find_independent_set(adj, n, limit=10):
        # Returns a set of 'limit' vertices that are independent in G
        # i.e. clique in complement
        mask = (1 << n) - 1
        comp_adj = [0] * n
        for i in range(n):
            comp_adj[i] = mask ^ adj[i]
            comp_adj[i] &= ~(1 << i)
            
        found = [False]
        result_set = []
        
        def expand(clique_mask, cand_mask):
            if found[0]: return
            if bin(clique_mask).count('1') >= limit:
                # Extract vertices
                temp = clique_mask
                while temp:
                    lsb = temp & -temp
                    v = lsb.bit_length() - 1
                    result_set.append(v)
                    temp ^= lsb
                found[0] = True
                return
            if cand_mask == 0: return
            
            # Pivot
            best_v = -1
            best_deg = -1
            temp = cand_mask
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                deg = bin(cand_mask & comp_adj[v]).count('1')
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                temp ^= lsb
            
            if best_v == -1: return
            
            expand(clique_mask | (1 << best_v), cand_mask & comp_adj[best_v])
            if found[0]: return
            expand(clique_mask, cand_mask ^ (1 << best_v))
            
        expand(0, mask)
        if found[0]:
            return result_set
        return None

    def try_n(n):
        if time.time() > deadline:
            return None
        
        best_valid = None
        # Try multiple random restarts
        for restart in range(1000):
            if time.time() > deadline:
                break
            
            # Init random graph
            adj = [0] * n
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Phase 1: Remove triangles
            # Loop until no triangles or timeout
            for step in range(10000):
                if time.time() > deadline:
                    break
                tri = find_triangle(adj, n)
                if tri is None:
                    break
                u, v, w = tri
                # Remove random edge from triangle
                choice = random.randint(0, 2)
                if choice == 0:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                elif choice == 1:
                    adj[v] &= ~(1 << w)
                    adj[w] &= ~(1 << v)
                else:
                    adj[w] &= ~(1 << u)
                    adj[u] &= ~(1 << w)
            
            # Check if valid
            if find_triangle(adj, n) is None:
                indep_set = find_independent_set(adj, n, 10)
                if indep_set is None:
                    # Valid!
                    edges = []
                    for i in range(n):
                        for j in range(i+1, n):
                            if adj[i] & (1 << j):
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}
            
            # Phase 2: Try to fix independent set by adding edges
            # Only if no triangles
            if find_triangle(adj, n) is None:
                for step in range(10000):
                    if time.time() > deadline:
                        break
                    indep_set = find_independent_set(adj, n, 10)
                    if indep_set is None:
                        # Valid
                        edges = []
                        for i in range(n):
                            for j in range(i+1, n):
                                if adj[i] & (1 << j):
                                    edges.append([i, j])
                        return {'n': n, 'edges': edges}
                    
                    # Pick two random vertices from indep_set
                    u, v = random.sample(indep_set, 2)
                    # Add edge (u, v)
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    
                    # This might create triangles
                    # Find and break them
                    for sub_step in range(100):
                        tri = find_triangle(adj, n)
                        if tri is None:
                            break
                        a, b, c = tri
                        choice = random.randint(0, 2)
                        if choice == 0:
                            adj[a] &= ~(1 << b)
                            adj[b] &= ~(1 << a)
                        elif choice == 1:
                            adj[b] &= ~(1 << c)
                            adj[c] &= ~(1 << b)
                        else:
                            adj[c] &= ~(1 << a)
                            adj[a] &= ~(1 << c)
        
        return None

    # Try n=40 first
    res = try_n(40)
    if res:
        return res
        
    # Try n=39
    if time.time() < deadline - 10:
        res = try_n(39)
        if res:
            return res

    # Fallback: Andrasfai graph on 26
    n = 26
    return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}
