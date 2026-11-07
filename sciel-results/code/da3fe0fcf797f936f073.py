"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random
from collections import defaultdict
import sys

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def max_bipartite_matching(adj, n_left, n_right):
        pair_u = [-1] * n_left
        pair_v = [-1] * n_right
        dist = [0] * n_left
        
        def bfs():
            from collections import deque
            queue = deque()
            for u in range(n_left):
                if pair_u[u] == -1:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = -1
            found = False
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    pu = pair_v[v]
                    if pu == -1:
                        found = True
                    elif dist[pu] == -1:
                        dist[pu] = dist[u] + 1
                        queue.append(pu)
            return found

        def dfs(u):
            for v in adj[u]:
                pu = pair_v[v]
                if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                    pair_u[u] = v
                    pair_v[v] = u
                    return True
            dist[u] = -1
            return False

        matching = 0
        while bfs():
            for u in range(n_left):
                if pair_u[u] == -1:
                    if dfs(u):
                        matching += 1
        return pair_u

    def greedy_clique_cover(edges, n_vertices, seed=0):
        """
        Greedy algorithm to cover edges with cliques.
        Returns the list of cliques.
        """
        if not edges:
            return []
        
        rng = Random(seed)
        rem_edges_set = set(edges)
        cliques = []
        
        while rem_edges_set:
            if time.time() > deadline - 10:
                # Fallback
                for e in rem_edges_set:
                    cliques.append([e[0], e[1]])
                return cliques
            
            # Calculate degrees
            degrees = defaultdict(int)
            for u, v in rem_edges_set:
                degrees[u] += 1
                degrees[v] += 1
            
            if not degrees:
                break
                
            # Pick vertex with max degree (with random tie-breaking for restarts)
            max_deg = max(degrees.values())
            candidates = [v for v, d in degrees.items() if d == max_deg]
            max_v = rng.choice(candidates)
            
            # Find neighbors
            neighbors = set()
            for u, v in rem_edges_set:
                if u == max_v:
                    neighbors.add(v)
                elif v == max_v:
                    neighbors.add(u)
            
            # Greedy clique expansion
            clique_vertices = [max_v]
            # Sort candidates by degree
            candidate_neighbors = list(neighbors)
            candidate_neighbors.sort(key=lambda x: degrees.get(x, 0), reverse=True)
            
            for cand in candidate_neighbors:
                is_connected = True
                for cv in clique_vertices:
                    if cv == cand:
                        continue
                    if (min(cv, cand), max(cv, cand)) not in rem_edges_set:
                        is_connected = False
                        break
                if is_connected:
                    clique_vertices.append(cand)
            
            cliques.append(sorted(clique_vertices))
            
            # Remove covered edges
            for i in range(len(clique_vertices)):
                for j in range(i + 1, len(clique_vertices)):
                    u, w = clique_vertices[i], clique_vertices[j]
                    edge = (min(u, w), max(u, w))
                    if edge in rem_edges_set:
                        rem_edges_set.remove(edge)
                        
        return cliques

    def clique_partition(edges, n, k):
        # 1. Identify cross edges and neighbors
        neighbors_in_K = defaultdict(set)
        cross_edges = []
        k_edges_set = set()
        
        for u, v in edges:
            if u < k and v >= k:
                neighbors_in_K[v].add(u)
                cross_edges.append((u, v))
            elif u < k and v < k:
                k_edges_set.add((u, v))
        
        # 2. Build matching graph
        k_edges = []
        edge_index = {}
        idx = 0
        for u in range(k):
            for w in range(u + 1, k):
                if (u, w) in k_edges_set:
                    k_edges.append((u, w))
                    edge_index[(u, w)] = idx
                    idx += 1
        
        n_left = len(k_edges)
        candidate_i_vertices = []
        for v in range(k, n):
            if len(neighbors_in_K[v]) >= 2:
                candidate_i_vertices.append(v)
        
        n_right = len(candidate_i_vertices)
        
        adj = [[] for _ in range(n_left)]
        for j, v in enumerate(candidate_i_vertices):
            nv = neighbors_in_K[v]
            for i, (u, w) in enumerate(k_edges):
                if u in nv and w in nv:
                    adj[i].append(j)
        
        match = max_bipartite_matching(adj, n_left, n_right)
        
        # 3. Initialize state
        # covered_k_edges: set of K-edges covered by triangles
        # triangles: list of (k_edge_idx, i_vertex_idx)
        covered_k_edges = set()
        triangles = []
        for i, j in enumerate(match):
            if j != -1:
                covered_k_edges.add(k_edges[i])
                triangles.append((i, j))
        
        # Remaining K-edges
        rem_k_edges = [e for e in k_edges if e not in covered_k_edges]
        
        # 4. Local Search: Try to remove triangles to improve K-cover
        # Cost function: N_cross_uncovered_by_triangles + len(K_cover)
        # N_cross_uncovered = Total_Cross - 2 * len(triangles)
        # So we minimize: Total_Cross - 2*T + K_cover(T)
        # Equivalent to minimizing: K_cover(T) - 2*T
        
        def get_cost(tri_list):
            t_count = len(tri_list)
            cov_k = set()
            for i, j in tri_list:
                cov_k.add(k_edges[i])
            rem_k = [e for e in k_edges if e not in cov_k]
            
            # Estimate K cover cost using greedy (multiple restarts for better estimate)
            best_cover_len = float('inf')
            # Use a few restarts to get a good lower bound on cover size
            for seed in range(5):
                cover = greedy_clique_cover(rem_k, k, seed)
                if len(cover) < best_cover_len:
                    best_cover_len = len(cover)
            
            return len(cross_edges) - 2 * t_count + best_cover_len

        current_cost = get_cost(triangles)
        
        # Try removing each triangle
        improved = True
        while improved:
            improved = False
            for idx_t in range(len(triangles)):
                if time.time() > deadline - 20:
                    break
                
                # Try removing triangle at idx_t
                new_triangles = triangles[:idx_t] + triangles[idx_t+1:]
                new_cost = get_cost(new_triangles)
                
                if new_cost < current_cost:
                    current_cost = new_cost
                    triangles = new_triangles
                    improved = True
                    # Restart loop to check other triangles
                    break
        
        # 5. Construct final cliques
        cliques = []
        
        # Add triangles
        covered_i_vertices = set()
        v_to_triangle = {}
        for i, j in triangles:
            u, w = k_edges[i]
            v = candidate_i_vertices[j]
            cliques.append([u, w, v])
            covered_i_vertices.add(v)
            v_to_triangle[v] = (u, w)
        
        # Cover remaining K-edges
        covered_k_edges = set(k_edges[i] for i, j in triangles)
        rem_k_edges = [e for e in k_edges if e not in covered_k_edges]
        
        # Use the best greedy cover found (or just run it again with best seed)
        # For consistency, we run greedy_clique_cover once more
        k_cliques = greedy_clique_cover(rem_k_edges, k, seed=42)
        cliques.extend(k_cliques)
        
        # Cover remaining cross edges
        for u, v in cross_edges:
            if v in covered_i_vertices:
                tu, tw = v_to_triangle[v]
                if u == tu or u == tw:
                    continue # Covered by triangle
            cliques.append([u, v])
            
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        cliques = clique_partition(edges, n, k)
        results.append(cliques)
        
    return results
# EVOLVE-BLOCK-END
