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
        """
        Hopcroft-Karp algorithm for maximum matching in bipartite graph.
        adj: list of lists, adj[u] contains v for u in [0, n_left), v in [0, n_right)
        Returns list of matches for left nodes: match[u] = v or -1
        """
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

    def clique_partition(edges, n, k):
        # 1. Identify cross edges and build adjacency for I vertices to K vertices
        # I vertices are k..n-1. K vertices are 0..k-1.
        # For each v in I, let N(v) be the set of neighbors in K.
        neighbors_in_K = defaultdict(set)
        cross_edges = []
        
        for u, v in edges:
            if u < k and v >= k:
                # u is in K, v is in I
                neighbors_in_K[v].add(u)
                cross_edges.append((u, v))
            
        # 2. We want to find a maximum matching between:
        # Left nodes: Pairs of vertices in K (edges within K)
        # Right nodes: Vertices in I
        # Edge exists if the I-vertex is connected to both vertices of the K-pair.
        
        # To reduce the size of the bipartite graph, we only consider "dense" I vertices.
        # An I vertex v can only form a triangle if |N(v)| >= 2.
        # Let's index the K-edges.
        # There are k*(k-1)/2 K-edges.
        # Let's map each K-edge (u, w) to an index.
        # Since k is small (10, 20, 30), the number of K-edges is manageable (45, 190, 435).
        
        k_edges = []
        edge_index = {}
        idx = 0
        for u in range(k):
            for w in range(u + 1, k):
                k_edges.append((u, w))
                edge_index[(u, w)] = idx
                idx += 1
        
        n_left = len(k_edges)
        
        # Filter I vertices that can potentially match
        candidate_i_vertices = []
        for v in range(k, n):
            if len(neighbors_in_K[v]) >= 2:
                candidate_i_vertices.append(v)
        
        n_right = len(candidate_i_vertices)
        
        # Build adjacency list for matching
        # adj[i] is list of j such that candidate_i_vertices[j] is adjacent to both endpoints of k_edges[i]
        adj = [[] for _ in range(n_left)]
        
        for j, v in enumerate(candidate_i_vertices):
            nv = neighbors_in_K[v]
            for i, (u, w) in enumerate(k_edges):
                if u in nv and w in nv:
                    adj[i].append(j)
        
        # Run Hopcroft-Karp
        # We want to maximize the number of matched pairs.
        # match[i] = j means k_edge i is matched to I-vertex j.
        match = max_bipartite_matching(adj, n_left, n_right)
        
        # 3. Construct the cliques
        cliques = []
        
        # Set of covered K-edges
        covered_k_edges = set()
        # Set of covered I-vertices (those that formed a triangle)
        covered_i_vertices = set()
        
        # Type B cliques: Triangles {u, w, v}
        for i, j in enumerate(match):
            if j != -1:
                u, w = k_edges[i]
                v = candidate_i_vertices[j]
                cliques.append([u, w, v])
                covered_k_edges.add((u, w))
                covered_i_vertices.add(v)
        
        # Remaining K-edges need to be covered.
        # We can cover them in pairs (4-cycles in the line graph sense, or just partition the remaining edges of the clique K into cliques of size 2? No, {u,w} is a clique. But we want to minimize count.
        # The remaining subgraph of K is a graph on k vertices. We need to partition its edges into cliques.
        # Since K is a clique, any subset of vertices forms a clique.
        # So we just need to partition the set of remaining edges into the minimum number of cliques.
        # This is equivalent to edge coloring the complement? No.
        # It's the clique cover number of the graph formed by remaining edges.
        # For a graph G, the minimum number of cliques to cover edges is the chromatic number of the complement graph?
        # Actually, for small k (<=30), we can use a greedy approach or a simple ILP/Backtracking if needed, but greedy is likely good enough given the "O(n)" slack.
        
        # Let's find the remaining edges in K
        remaining_k_edges = []
        for u in range(k):
            for w in range(u + 1, k):
                if (u, w) not in covered_k_edges:
                    remaining_k_edges.append((u, w))
        
        # We need to cover remaining_k_edges with cliques.
        # A clique in K is just a set of vertices.
        # We want to minimize the number of sets.
        # This is the same as finding a coloring of the line graph? No.
        # It is the clique cover of the graph H=(K, remaining_k_edges).
        # Since H is a subgraph of a complete graph, we can use a greedy algorithm:
        # Pick a vertex with highest degree, form a clique with its neighbors, remove covered edges, repeat.
        # Or simply: Since k is small, we can try to pack as many edges as possible into each clique.
        # Max clique size in a graph is hard, but here the graph is dense (subgraph of K_k).
        # Let's use a simple greedy:
        # While edges remain:
        #   Find a maximal clique in the remaining graph.
        #   Add it to cliques.
        #   Remove its edges from remaining graph.
        
        # To find a maximal clique in a small graph, we can use Bron-Kerbosch or just greedy.
        # Given the time limit, a simple greedy is safer.
        
        # Let's represent remaining edges as a set of tuples
        rem_edges_set = set(remaining_k_edges)
        
        while rem_edges_set:
            # Find a vertex with max degree in rem_edges_set
            degrees = defaultdict(int)
            for u, v in rem_edges_set:
                degrees[u] += 1
                degrees[v] += 1
            
            if not degrees:
                break
                
            # Pick vertex with max degree
            max_v = max(degrees, key=degrees.get)
            
            # Find neighbors of max_v in rem_edges_set
            neighbors = set()
            for u, v in rem_edges_set:
                if u == max_v:
                    neighbors.add(v)
                elif v == max_v:
                    neighbors.add(u)
            
            # We want to form a clique containing max_v.
            # The clique can only contain vertices from neighbors that are mutually connected in rem_edges_set.
            # Let's find a maximal clique in the subgraph induced by neighbors.
            # For small k, we can just pick a greedy maximal clique.
            clique_vertices = [max_v]
            candidate_neighbors = list(neighbors)
            candidate_neighbors.sort(key=lambda x: degrees.get(x, 0), reverse=True)
            
            for cand in candidate_neighbors:
                # Check if cand is connected to all existing clique_vertices
                is_connected = True
                for cv in clique_vertices:
                    if cv == cand:
                        continue
                    if (min(cv, cand), max(cv, cand)) not in rem_edges_set:
                        is_connected = False
                        break
                if is_connected:
                    clique_vertices.append(cand)
            
            # Add this clique
            cliques.append(sorted(clique_vertices))
            
            # Remove all edges within this clique from rem_edges_set
            for i in range(len(clique_vertices)):
                for j in range(i + 1, len(clique_vertices)):
                    u, w = clique_vertices[i], clique_vertices[j]
                    edge = (min(u, w), max(u, w))
                    if edge in rem_edges_set:
                        rem_edges_set.remove(edge)
            
            # Check deadline
            if time.time() > deadline - 5:
                # Fallback: cover each remaining edge individually
                for e in rem_edges_set:
                    cliques.append([e[0], e[1]])
                rem_edges_set.clear()
                break

        # 4. Cover remaining cross edges
        # Cross edges (u, v) where v is in I.
        # If v was covered in a triangle, the edges (u, v) for u in the triangle are covered.
        # Specifically, if triangle is {a, b, v}, then edges (a,v) and (b,v) are covered.
        # Any other edge (u, v) with u not in {a,b} must be covered separately.
        # Also, if v was NOT covered in a triangle, ALL its edges (u, v) must be covered separately.
        
        # Let's track which cross edges are covered.
        # A cross edge (u, v) is covered if:
        # 1. v is in covered_i_vertices AND u is in the triangle that covered v.
        
        # We need to map v to the triangle vertices.
        v_to_triangle = {}
        for i, j in enumerate(match):
            if j != -1:
                u, w = k_edges[i]
                v = candidate_i_vertices[j]
                v_to_triangle[v] = (u, w)
        
        for u, v in cross_edges:
            if v in covered_i_vertices:
                tu, tw = v_to_triangle[v]
                if u == tu or u == tw:
                    continue # Covered by triangle
            # If not covered by triangle, add as 2-clique
            cliques.append([u, v])
            
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        cliques = clique_partition(edges, n, k)
        
        # Validation check (optional but good for debugging, removed for speed in final if slow)
        # The task says invalid result scores 0. We must be careful.
        # Let's do a quick sanity check on the first instance to ensure logic holds.
        
        results.append(cliques)
        
    return results
# EVOLVE-BLOCK-END
