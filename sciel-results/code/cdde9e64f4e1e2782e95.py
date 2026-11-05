import time
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill

    def time_up():
        return (time.time() - start_time) > DEADLINE

    def generate_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def get_edges(adj, n):
        edges = []
        for u in range(n):
            mask = adj[u]
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                if u < v:
                    edges.append((u, v))
                mask ^= lsb
        return edges

    def compute_centrality(adj, n, iterations=50):
        """
        Compute eigenvector centrality using power iteration.
        """
        # Create adjacency matrix
        A = np.zeros((n, n), dtype=np.float64)
        for u in range(n):
            mask = adj[u]
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                A[u, v] = 1.0
                mask ^= lsb
        
        # Initialize vector
        x = np.ones(n, dtype=np.float64)
        x /= np.linalg.norm(x)
        
        for _ in range(iterations):
            x_new = A @ x
            norm = np.linalg.norm(x_new)
            if norm == 0:
                break
            x_new /= norm
            if np.linalg.norm(x_new - x) < 1e-10:
                x = x_new
                break
            x = x_new
            
        return x

    def solve_instance(n, p, seed):
        adj = generate_graph(n, p, seed)
        edges = get_edges(adj, n)
        total_edges = len(edges)
        
        # Compute centrality
        centrality = compute_centrality(adj, n)
        
        # Copy adjacency for remaining edges
        rem = adj[:]
        
        cliques = []
        active = (1 << n) - 1
        
        while active and not time_up():
            # Find vertex with highest centrality among active vertices with remaining edges
            best_v = -1
            best_c = -1.0
            m = active
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                if rem[v]: # Has remaining edges
                    if centrality[v] > best_c:
                        best_c = centrality[v]
                        best_v = v
                m ^= lsb
            
            if best_v == -1:
                # Fallback: find any vertex with edges
                m = active
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    if rem[v]:
                        best_v = v
                        break
                    m ^= lsb
                if best_v == -1:
                    break

            # Expand best_v into a maximal clique in the remaining graph
            # We prioritize neighbors with high centrality
            clique = [best_v]
            # Candidates are neighbors of best_v in the remaining graph
            candidates = rem[best_v] & active
            candidates &= ~(1 << best_v)
            
            # Sort candidates by centrality descending
            cand_list = []
            m = candidates
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                cand_list.append((centrality[v], v))
                m ^= lsb
            cand_list.sort(reverse=True)
            
            # Greedy expansion
            current_clique_mask = 1 << best_v
            for c_val, v in cand_list:
                # Check if v is connected to all vertices in current clique
                # v must be in candidates (neighbor of best_v)
                # and neighbor of all other vertices in clique
                is_connected = True
                cm = current_clique_mask
                while cm:
                    lsb = cm & -cm
                    u = lsb.bit_length() - 1
                    if u != v and not (rem[v] & (1 << u)):
                        is_connected = False
                        break
                    cm ^= lsb
                
                if is_connected:
                    clique.append(v)
                    current_clique_mask |= (1 << v)
                
                # If time is tight, break early? No, just keep it fast.
                # The loop is small (n <= 64).
            
            # Add clique to result
            if len(clique) >= 2:
                cliques.append(clique)
                
                # Remove edges covered by this clique from rem
                for u in clique:
                    for v in clique:
                        if u != v:
                            rem[u] &= ~(1 << v)
            
            # Update active vertices
            active = 0
            for i in range(n):
                if rem[i]:
                    active |= (1 << i)
            
            # Check time again
            if time_up():
                break

        # Cover remaining edges with pairs if any left (should be few if greedy worked well)
        if active:
            for u in range(n):
                nb = rem[u]
                while nb:
                    lsb = nb & -nb
                    v = lsb.bit_length() - 1
                    if u < v:
                        cliques.append([u, v])
                        rem[u] &= ~(1 << v)
                        rem[v] &= ~(1 << u)
                    nb ^= lsb

        return cliques

    c1 = solve_instance(32, 0.6, 601)
    c2 = solve_instance(64, 0.55, 602)

    return [c1, c2]
