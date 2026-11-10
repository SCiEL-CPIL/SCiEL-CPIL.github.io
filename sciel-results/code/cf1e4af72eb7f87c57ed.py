import time
import random
import heapq

def construct():
    start_time = time.time()
    DEADLINE = start_time + 110.0

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

    def edge_clique_assignment(n, adj, rng, deadline):
        """
        Global edge-assignment framework:
        - Maintain a priority queue of edges keyed by "constraint" (how few cliques can cover them)
        - For each edge, find the best clique to assign it to (maximizes shared coverage)
        - This avoids local myopia by considering global edge constraints
        """
        rem_adj = adj[:]
        # Track which edges are still uncovered
        edge_set = set()
        for u in range(n):
            mask = rem_adj[u]
            while mask:
                v_bit = mask & (-mask)
                v = v_bit.bit_length() - 1
                if u < v:
                    edge_set.add((u, v))
                mask ^= v_bit
        
        cliques = []
        # For each vertex, track which cliques contain it and their edge sets
        vertex_cliques = {v: [] for v in range(n)}
        
        # Priority queue: (constraint_score, edge)
        # constraint_score = number of vertices that can be in a clique with this edge
        # Lower = more constrained = process first
        pq = []
        for (u, v) in edge_set:
            common = rem_adj[u] & rem_adj[v]
            score = popcount(common)
            heapq.heappush(pq, (score, u, v))
        
        while pq:
            if time.time() > deadline:
                return None
            
            score, u, v = heapq.heappop(pq)
            
            # Check if edge is still uncovered
            if (u, v) not in edge_set:
                continue
            
            # Find best clique to extend or create for this edge
            # Look at all vertices that form a triangle with (u,v)
            common_mask = rem_adj[u] & rem_adj[v]
            
            # Try to find a large clique containing u and v
            best_clique = [u, v]
            best_size = 2
            
            # Greedily add vertices to maximize clique size
            candidate_mask = common_mask
            while candidate_mask:
                # Pick vertex with max degree in remaining candidates
                max_deg = -1
                best_v = -1
                cm = candidate_mask
                while cm:
                    v_bit = cm & (-cm)
                    w = v_bit.bit_length() - 1
                    cm ^= v_bit
                    deg = popcount(candidate_mask & rem_adj[w])
                    if deg > max_deg:
                        max_deg = deg
                        best_v = w
                
                if best_v == -1:
                    break
                
                # Check if adding best_v maintains clique property
                if (rem_adj[u] >> best_v) & 1 and (rem_adj[v] >> best_v) & 1:
                    # Check against all existing clique members
                    valid = True
                    for w in best_clique:
                        if w != best_v and not ((rem_adj[w] >> best_v) & 1):
                            valid = False
                            break
                    if valid:
                        best_clique.append(best_v)
                        best_size += 1
                        candidate_mask &= rem_adj[best_v]
                else:
                    candidate_mask &= ~(1 << best_v)
            
            # If we found a clique with more than 2 vertices, use it
            if best_size > 2:
                clique = best_clique
                # Remove all edges in this clique from edge_set
                for i in range(len(clique)):
                    for j in range(i + 1, len(clique)):
                        a, b = clique[i], clique[j]
                        if a > b:
                            a, b = b, a
                        edge_set.discard((a, b))
                        rem_adj[a] &= ~(1 << b)
                        rem_adj[b] &= ~(1 << a)
                
                # Update vertex_cliques
                for w in clique:
                    vertex_cliques[w].append(len(cliques))
                
                cliques.append(clique)
                
                # Update priority queue for affected edges
                for w in clique:
                    mask = rem_adj[w]
                    while mask:
                        x_bit = mask & (-mask)
                        x = x_bit.bit_length() - 1
                        mask ^= x_bit
                        if w < x:
                            edge = (w, x)
                        else:
                            edge = (x, w)
                        if edge in edge_set:
                            common = rem_adj[edge[0]] & rem_adj[edge[1]]
                            new_score = popcount(common)
                            heapq.heappush(pq, (new_score, edge[0], edge[1]))
            
            else:
                # Just cover this edge as a 2-clique
                edge_set.discard((u, v))
                rem_adj[u] &= ~(1 << v)
                rem_adj[v] &= ~(1 << u)
                cliques.append([u, v])
        
        return cliques if cliques else None

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
            rng = random.Random(seed * 1000 + attempt)
            result = edge_clique_assignment(n, adj, rng, DEADLINE)
            if result is not None:
                count = len(result)
                if count < best_count:
                    best_count = count
                    best_cliques = result
            attempt += 1

        return best_cliques

    # Time budget allocation
    t1 = 15.0
    t2 = 90.0

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
