import time
import random

def construct():
    start_time = time.time()
    DEADLINE = start_time + 145.0  # Safety margin before 150s kill

    def gen_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [0] * n
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append((u, v))
        return adj, edges

    def bronk_max_clique(P, X, adj, deadline):
        """Find a maximum clique in the graph restricted to vertices in P."""
        best_clique = 0
        best_size = 0
        
        def bk(P, X, R):
            nonlocal best_clique, best_size
            if time.time() > deadline:
                return
            if not P and not X:
                if R.bit_count() > best_size:
                    best_size = R.bit_count()
                    best_clique = R
                return
            union = P | X
            if not union:
                return
            pivot = 0
            max_common = -1
            temp = union
            while temp:
                u = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                common = P & adj[u]
                count = common.bit_count()
                if count > max_common:
                    max_common = count
                    pivot = u
            candidates = P & ~adj[pivot]
            temp_cand = candidates
            while temp_cand:
                if time.time() > deadline:
                    return
                v = (temp_cand & -temp_cand).bit_length() - 1
                temp_cand &= temp_cand - 1
                bk(P & adj[v], X, R | (1 << v))
                P &= ~(1 << v)
                X |= (1 << v)
        
        bk(P, X, 0)
        return best_clique

    def greedy_min_common(n, edges, adj, rng, deadline):
        """
        Greedy: repeatedly pick edge with minimum common neighbors,
        expand to max clique containing that edge.
        """
        rem_adj = [0] * n
        for u, v in edges:
            rem_adj[u] |= (1 << v)
            rem_adj[v] |= (1 << u)
        
        remaining_edges = set(edges)
        if not remaining_edges:
            return []
        
        cliques = []
        
        while remaining_edges:
            if time.time() > deadline:
                return None
            
            # Find edge with minimum common neighbors among remaining
            # Sample if too many edges
            edge_list = list(remaining_edges)
            if len(edge_list) > 200:
                sample = rng.sample(edge_list, 200)
            else:
                sample = edge_list
            
            best_u = -1
            best_v = -1
            min_common = float('inf')
            
            for u, v in sample:
                if not (rem_adj[u] & (1 << v)):
                    continue
                common = rem_adj[u] & rem_adj[v]
                count = common.bit_count()
                if count < min_common:
                    min_common = count
                    best_u = u
                    best_v = v
                    if count == 0:
                        break
            
            if best_u == -1:
                # Fallback: pick any remaining edge
                u, v = next(iter(remaining_edges))
                best_u, best_v = u, v
                min_common = (rem_adj[u] & rem_adj[v]).bit_count()
            
            # Build max clique containing best_u and best_v
            # Start with P = common neighbors, X = 0
            P = rem_adj[best_u] & rem_adj[best_v]
            X = 0
            
            # Use Bron-Kerbosch to find max clique in P
            K = bronk_max_clique(P, X, rem_adj, deadline)
            clique_mask = K | (1 << best_u) | (1 << best_v)
            
            # Extract clique vertices
            clique_vertices = []
            cm = clique_mask
            while cm:
                v = (cm & -cm).bit_length() - 1
                cm &= cm - 1
                clique_vertices.append(v)
            
            if len(clique_vertices) < 2:
                clique_vertices = [best_u, best_v]
                clique_mask = (1 << best_u) | (1 << best_v)
            
            cliques.append(clique_vertices)
            
            # Remove all edges within clique from remaining
            clique_set = set(clique_vertices)
            new_remaining = set()
            for u, v in remaining_edges:
                if u in clique_set and v in clique_set:
                    rem_adj[u] &= ~(1 << v)
                    rem_adj[v] &= ~(1 << u)
                else:
                    new_remaining.add((u, v))
            
            remaining_edges = new_remaining
        
        return cliques

    def greedy_max_degree(n, edges, adj, rng, deadline):
        """
        Greedy: repeatedly pick vertex with max degree, find max clique around it.
        """
        rem_adj = [0] * n
        for u, v in edges:
            rem_adj[u] |= (1 << v)
            rem_adj[v] |= (1 << u)
        
        remaining_edges = set(edges)
        if not remaining_edges:
            return []
        
        cliques = []
        
        while remaining_edges:
            if time.time() > deadline:
                return None
            
            # Find vertex with max degree in remaining graph
            max_deg = -1
            best_v = -1
            for v in range(n):
                d = rem_adj[v].bit_count()
                if d > max_deg:
                    max_deg = d
                    best_v = v
            
            if best_v == -1:
                break
            
            P = rem_adj[best_v]
            X = 0
            K = bronk_max_clique(P, X, rem_adj, deadline)
            clique_mask = K | (1 << best_v)
            
            clique_vertices = []
            cm = clique_mask
            while cm:
                v = (cm & -cm).bit_length() - 1
                cm &= cm - 1
                clique_vertices.append(v)
            
            if len(clique_vertices) < 2:
                # Find any neighbor
                nb = rem_adj[best_v]
                if nb:
                    v = (nb & -nb).bit_length() - 1
                    clique_vertices = [best_v, v]
                    clique_mask = (1 << best_v) | (1 << v)
                else:
                    break
            
            cliques.append(clique_vertices)
            
            clique_set = set(clique_vertices)
            new_remaining = set()
            for u, v in remaining_edges:
                if u in clique_set and v in clique_set:
                    rem_adj[u] &= ~(1 << v)
                    rem_adj[v] &= ~(1 << u)
                else:
                    new_remaining.add((u, v))
            
            remaining_edges = new_remaining
        
        return cliques

    def solve_instance(n, p, seed, deadline):
        adj, edges = gen_graph(n, p, seed)
        if not edges:
            return []
        
        best_cliques = [[u, v] for u, v in edges]
        best_count = len(best_cliques)
        
        rng_master = random.Random(12345)
        
        elapsed = time.time() - start_time
        remaining_time = deadline - start_time - elapsed
        
        if n <= 32:
            num_restarts = min(300, max(30, int(remaining_time / 0.05)))
        else:
            num_restarts = min(150, max(15, int(remaining_time / 0.15)))
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            rng = random.Random(rng_master.randint(0, 2**31))
            
            # Alternate between strategies based on restart
            if restart % 2 == 0:
                result = greedy_min_common(n, edges, adj, rng, deadline)
            else:
                result = greedy_max_degree(n, edges, adj, rng, deadline)
            
            if result is not None and len(result) < best_count:
                best_count = len(result)
                best_cliques = result
        
        return best_cliques

    # Solve both instances with time allocation
    deadline1 = start_time + 55.0
    deadline2 = DEADLINE
    
    result1 = solve_instance(32, 0.6, 601, deadline1)
    
    if time.time() > DEADLINE:
        adj2, edges2 = gen_graph(64, 0.55, 602)
        result2 = [[u, v] for u, v in edges2]
    else:
        result2 = solve_instance(64, 0.55, 602, DEADLINE)
    
    return [result1, result2]
