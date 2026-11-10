import time
import random
import sys

def construct():
    start_time = time.time()
    DEADLINE = start_time + 115.0

    def gen_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def build_adj(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def popcount(x):
        return x.bit_count()

    def max_clique_branch_bound(n, adj, mask, deadline):
        """
        Find a maximum clique in G[mask] using branch and bound with pivot.
        adj is the adjacency bitmasks for the full graph (n vertices).
        mask restricts to a subset of vertices.
        """
        # Order vertices by degree in the induced subgraph for better pruning
        vertices = []
        m = mask
        while m:
            bit = m & (-m)
            v = bit.bit_length() - 1
            vertices.append(v)
            m ^= bit
        
        # Sort by degree in induced subgraph (descending)
        vertices.sort(key=lambda v: popcount(adj[v] & mask), reverse=True)
        
        # Map vertex to index in sorted order
        vert_to_idx = {v: i for i, v in enumerate(vertices)}
        k = len(vertices)
        
        # Build adjacency in the induced subgraph as bitmasks over indices
        adj_idx = [0] * k
        for i in range(k):
            vi = vertices[i]
            for j in range(i + 1, k):
                vj = vertices[j]
                if adj[vi] & (1 << vj):
                    adj_idx[i] |= (1 << j)
                    adj_idx[j] |= (1 << i)
        
        best_clique = []
        
        def expand(R, P):
            nonlocal best_clique
            if not P:
                if len(R) > len(best_clique):
                    best_clique = R[:]
                return
            
            pc = popcount(P)
            if len(R) + pc <= len(best_clique):
                return
            
            # Pivot: choose vertex in P with max degree in P
            max_deg = -1
            pivot = -1
            P_temp = P
            while P_temp:
                bit = P_temp & (-P_temp)
                v = bit.bit_length() - 1
                P_temp ^= bit
                deg = popcount(P & adj_idx[v])
                if deg > max_deg:
                    max_deg = deg
                    pivot = v
            
            if pivot == -1:
                return
            
            # Only consider non-neighbors of pivot
            Q = P & ~adj_idx[pivot]
            
            while Q:
                if time.time() > deadline:
                    return
                
                # Choose vertex with max degree in P
                best_v = -1
                best_deg = -1
                Q_temp = Q
                while Q_temp:
                    bit = Q_temp & (-Q_temp)
                    v = bit.bit_length() - 1
                    Q_temp ^= bit
                    deg = popcount(P & adj_idx[v])
                    if deg > best_deg:
                        best_deg = deg
                        best_v = v
                
                if best_v == -1:
                    break
                
                v = best_v
                new_R = R + [v]
                new_P = P & adj_idx[v]
                expand(new_R, new_P)
                
                P &= ~(1 << v)
                Q &= P
        
        expand([], mask)
        
        # Convert back to vertex indices
        result = [vertices[i] for i in best_clique]
        return result

    def greedy_clique_partition(n, adj, edges, deadline):
        """
        Greedy clique partition:
        1. Find max clique in remaining graph
        2. Remove its edges
        3. Repeat
        """
        rem_adj = adj[:]
        cliques = []
        
        while True:
            if time.time() > deadline:
                break
            
            # Check if any edges remain
            has_edges = False
            for i in range(n):
                if rem_adj[i]:
                    has_edges = True
                    break
            if not has_edges:
                break
            
            # Find max clique in remaining graph
            mask = (1 << n) - 1
            # Only include vertices that have at least one remaining edge
            active_mask = 0
            for i in range(n):
                if rem_adj[i]:
                    active_mask |= (1 << i)
            
            if not active_mask:
                break
            
            clique = max_clique_branch_bound(n, rem_adj, active_mask, deadline)
            
            if len(clique) < 2:
                # No clique of size >= 2 found, take a single edge
                for u in range(n):
                    if rem_adj[u]:
                        v_mask = rem_adj[u] & (~((1 << (u + 1)) - 1))
                        if v_mask:
                            bit = v_mask & (-v_mask)
                            v = bit.bit_length() - 1
                            clique = [u, v]
                            break
                if len(clique) < 2:
                    break
            
            # Remove edges of this clique
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    u, v = clique[i], clique[j]
                    rem_adj[u] &= ~(1 << v)
                    rem_adj[v] &= ~(1 << u)
            
            cliques.append(clique)
        
        return cliques

    def greedy_clique_partition_heuristic(n, adj, edges, deadline):
        """
        Greedy clique partition with heuristic:
        Prefer edges with high common neighbor count, find max clique containing that edge.
        """
        rem_adj = adj[:]
        cliques = []
        
        while True:
            if time.time() > deadline:
                break
            
            has_edges = False
            for i in range(n):
                if rem_adj[i]:
                    has_edges = True
                    break
            if not has_edges:
                break
            
            # Find edge with maximum common neighbors
            best_u, best_v = -1, -1
            best_common = -1
            
            for u in range(n):
                if not rem_adj[u]:
                    continue
                v_mask = rem_adj[u] & (~((1 << (u + 1)) - 1))
                while v_mask:
                    bit = v_mask & (-v_mask)
                    v = bit.bit_length() - 1
                    v_mask ^= bit
                    common = rem_adj[u] & rem_adj[v]
                    cnt = popcount(common)
                    if cnt > best_common:
                        best_common = cnt
                        best_u, best_v = u, v
            
            if best_u == -1:
                break
            
            # Find max clique containing edge (best_u, best_v)
            # Common neighbors of best_u and best_v
            common_mask = rem_adj[best_u] & rem_adj[best_v]
            
            # Find max clique in G[common_mask]
            clique_core = max_clique_branch_bound(n, rem_adj, common_mask, deadline)
            clique = clique_core + [best_u, best_v]
            
            # Remove edges of this clique
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    u, v = clique[i], clique[j]
                    rem_adj[u] &= ~(1 << v)
                    rem_adj[v] &= ~(1 << u)
            
            cliques.append(clique)
        
        return cliques

    def validate_partition(n, edges, cliques):
        """Validate that cliques form a valid partition."""
        edge_set = set()
        for u, v in edges:
            edge_set.add((min(u, v), max(u, v)))
        
        covered = set()
        for clique in cliques:
            if len(clique) < 2:
                return False
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    u, v = clique[i], clique[j]
                    e = (min(u, v), max(u, v))
                    if e not in edge_set:
                        return False
                    if e in covered:
                        return False
                    covered.add(e)
        
        return covered == edge_set

    def solve_instance(n, p, seed, time_limit):
        edges = gen_graph(n, p, seed)
        adj = build_adj(n, edges)
        
        # Baseline: each edge as its own 2-clique
        baseline = [[u, v] for u, v in edges]
        best_cliques = baseline
        best_count = len(baseline)
        
        start = time.time()
        attempt = 0
        while time.time() < start + time_limit:
            if time.time() > DEADLINE:
                break
            
            rng = random.Random(seed * 1000 + attempt)
            
            if attempt % 2 == 0:
                result = greedy_clique_partition_heuristic(n, adj, edges, DEADLINE)
            else:
                result = greedy_clique_partition(n, adj, edges, DEADLINE)
            
            if result is not None and len(result) > 0:
                if validate_partition(n, edges, result):
                    count = len(result)
                    if count < best_count:
                        best_count = count
                        best_cliques = result
            attempt += 1
        
        return best_cliques

    # Time budget allocation
    t1 = 20.0
    t2 = 80.0

    current = time.time()
    remaining = DEADLINE - current
    if t1 + t2 > remaining:
        scale = remaining / (t1 + t2)
        t1 *= scale
        t2 *= scale

    result1 = solve_instance(32, 0.6, 601, t1)

    if time.time() > DEADLINE:
        edges2 = gen_graph(64, 0.55, 602)
        result2 = [[u, v] for u, v in edges2]
    else:
        result2 = solve_instance(64, 0.55, 602, t2)

    return [result1, result2]
