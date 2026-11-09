import time
import random
from collections import deque
from itertools import combinations
import networkx as nx
import numpy as np

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 75.0

    n = P['n']
    edges = P['edges']

    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    degrees = [len(adj[i]) for i in range(n)]

    # ---------- Bron-Kerbosch with pivot to find max clique ----------
    best_clique = []

    def bron_kerbosch(R, Pset, X):
        nonlocal best_clique
        if time.time() > deadline:
            return
        if not Pset and not X:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
        # Pivot
        candidates_for_pivot = list(Pset) + list(X)
        u = None
        max_neighbors = -1
        for v in candidates_for_pivot:
            count = len(adj[v] & Pset)
            if count > max_neighbors:
                max_neighbors = count
                u = v
        if u is None:
            candidates = list(Pset)
        else:
            candidates = [v for v in Pset if v not in adj[u]]
        candidates.sort(key=lambda x: degrees[x], reverse=True)
        for v in candidates:
            if time.time() > deadline:
                return
            bron_kerbosch(R | {v}, Pset & adj[v], X & adj[v])
            Pset = Pset - {v}
            X = X | {v}

    bron_kerbosch(frozenset(), frozenset(range(n)), frozenset())

    # Greedy + random restarts to find good cliques
    def greedy_clique(start_order):
        remaining = set(start_order)
        clique = set()
        while remaining:
            best_v = max(remaining, key=lambda v: len(adj[v] & remaining))
            clique.add(best_v)
            remaining = remaining & adj[best_v]
        return clique

    for trial in range(100):
        if time.time() > deadline - 30:
            break
        shuffled = list(range(n))
        random.shuffle(shuffled)
        c = greedy_clique(shuffled)
        if len(c) > len(best_clique):
            best_clique = list(c)

    if not best_clique:
        best_clique = [0, 1]

    # ---------- Max-flow based path verification ----------
    # For a set of branch vertices B, we need k(k-1)/2 internally disjoint paths.
    # Model: split each non-branch vertex v into v_in -> v_out with capacity 1.
    # Branch vertices have infinite capacity (they can be endpoints of multiple paths).
    # For each pair (s,t) in B, add a source s' connected to s_in with capacity 1,
    # and t_out connected to sink t' with capacity 1. But this doesn't work directly
    # because we need ALL pairs simultaneously.
    # 
    # Better approach: Use a super-source and super-sink.
    # For each pair (s,t), create a flow unit. But standard max-flow gives max flow
    # from one source to one sink.
    #
    # Alternative: For each pair (s,t), verify that there's a path avoiding all
    # previously used interior vertices. Use iterative assignment with backtracking.
    #
    # Most efficient: For a candidate set B of size k, check if we can route
    # k(k-1)/2 units of flow where each unit goes from one branch to another,
    # and each non-branch vertex has capacity 1.
    #
    # This is a multi-commodity flow problem which is NP-hard in general, but
    # for small k we can use the following trick:
    # Create a network where each non-branch vertex v is split into v_in -> v_out
    # with capacity 1. Branch vertices are not split (or have infinite capacity).
    # Add a super-source S connected to each branch b with capacity (k-1) [since b
    # is endpoint of k-1 paths]. Add a super-sink T connected from each branch b
    # with capacity (k-1). For each edge (u,v) in the graph, add directed edges
    # u_out -> v_in and v_out -> u_in with capacity infinity (or large).
    # Then max flow from S to T should be k(k-1)/2 * 2 = k(k-1) (counting both
    # directions), but actually each path uses one unit from S to b_in and one unit
    # from b_out to T.
    #
    # Actually, let me think more carefully. We need k(k-1)/2 paths, each connecting
    # a distinct pair. Each path uses its two endpoints. So total "endpoint usage" is
    # k(k-1). Each branch b is endpoint of (k-1) paths, so it needs capacity (k-1)
    # on both input and output sides.
    #
    # Network construction:
    # - For each non-branch vertex v: v_in -> v_out, capacity 1
    # - For each branch vertex b: b_in -> b_out, capacity (k-1) [or infinity]
    # - For each edge (u,v): u_out -> v_in capacity INF, v_out -> u_in capacity INF
    # - Super-source S -> b_in for each branch b, capacity (k-1)
    # - b_out -> super-sink T for each branch b, capacity (k-1)
    # - Max flow should be k(k-1)
    #
    # If max flow == k(k-1), then we can extract the paths.

    def verify_with_maxflow(branches_list):
        """Verify if branches_list admits a valid subdivision using max-flow.
        Returns (True, paths_dict) or (False, None)."""
        k = len(branches_list)
        if k < 2:
            return False, None
        branches_set = set(branches_list)
        
        # Build flow network
        # Node naming: v_in = 2*v, v_out = 2*v+1
        # Super source = 2*n, super sink = 2*n+1
        num_nodes = 2 * n + 2
        S = 2 * n
        T = 2 * n + 1
        
        # Use NetworkX for max flow
        G = nx.DiGraph()
        INF = float('inf')
        
        for v in range(n):
            vin = 2 * v
            vout = 2 * v + 1
            if v in branches_set:
                cap = k - 1
            else:
                cap = 1
            G.add_edge(vin, vout, capacity=cap)
        
        for u, v in edges:
            uout = 2 * u + 1
            vin = 2 * v
            vout = 2 * v + 1
            uin = 2 * u
            G.add_edge(uout, vin, capacity=INF)
            G.add_edge(vout, uin, capacity=INF)
        
        for b in branches_list:
            bin_ = 2 * b
            bout = 2 * b + 1
            G.add_edge(S, bin_, capacity=k - 1)
            G.add_edge(bout, T, capacity=k - 1)
        
        try:
            flow_value, flow_dict = nx.maximum_flow(G, S, T)
        except Exception:
            return False, None
        
        if flow_value < k * (k - 1):
            return False, None
        
        # Extract paths from flow
        # For each pair (i,j) in branches, trace the flow from branches[i] to branches[j]
        # The flow from S to branches[i]_in is (k-1) units, distributed among outgoing edges.
        # We need to decompose the flow into individual paths.
        
        # Collect all flow-carrying edges
        # For each branch b, the flow entering b_in from S is (k-1).
        # The flow leaving b_out to T is (k-1).
        # We need to pair up: for each unit of flow entering b_in from some neighbor,
        # trace through the graph to find which branch it exits to.
        
        # Alternative: For each ordered pair (s, t) with s != t, find a path from s to t
        # in the flow-residual graph that carries 1 unit of flow.
        
        # Simpler: Just do BFS/DFS on the flow-carrying subgraph to extract paths.
        # Build a directed graph of flow-carrying edges (excluding S and T connections).
        flow_graph = nx.DiGraph()
        for u in range(n):
            for v in range(n):
                if u == v:
                    continue
                # Check if there's flow from u_out to v_in
                uout = 2 * u + 1
                vin = 2 * v
                if uout in flow_dict and vin in flow_dict[uout]:
                    f = flow_dict[uout][vin]
                    if f > 0:
                        flow_graph.add_edge(u, v, capacity=f)
                # Check vertex capacity edges
                vin_u = 2 * u
                vout_u = 2 * u + 1
                if vin_u in flow_dict and vout_u in flow_dict[vin_u]:
                    f = flow_dict[vin_u][vout_u]
                    if f > 0:
                        # This is internal flow through vertex u
                        pass
        
        # Extract paths: for each pair (s,t), find a simple path in flow_graph
        # This is complex. Let's use a different approach.
        
        # Use the fact that we already verified with BFS in the old code.
        # Instead, let's just use the iterative BFS approach but with a better ordering.
        return True, None  # Placeholder - we'll use BFS for actual path extraction

    # ---------- Path finding (shortest path avoiding forbidden vertices) ----------
    def shortest_path(s, t, forbidden):
        """BFS for shortest path from s to t avoiding forbidden set (interior only)."""
        if s == t:
            return [s]
        visited = {s}
        parent = {s: None}
        queue = deque([s])
        while queue:
            cur = queue.popleft()
            for nb in adj[cur]:
                if nb == t:
                    path = [t]
                    node = cur
                    while node is not None:
                        path.append(node)
                        node = parent[node]
                    path.reverse()
                    return path
                if nb in forbidden:
                    continue
                if nb not in visited:
                    visited.add(nb)
                    parent[nb] = cur
                    queue.append(nb)
        return None

    # ---------- Verify a clique subdivision ----------
    def verify_clique_subdivision(clique):
        """Try to find internally disjoint paths for all pairs in clique.
        Returns (True, paths_dict) or (False, None)."""
        k = len(clique)
        if k < 2:
            return False, None
        branches_set = set(clique)
        pairs = list(combinations(range(k), 2))
        
        # First check with max-flow if it's even possible
        # (skip for now, use direct BFS approach)
        
        # Strategy: for each pair, find a path that avoids all previously used interior vertices
        # Order pairs by "difficulty" (shorter available paths first)
        # Compute initial shortest paths
        pair_paths = {}
        for i, j in pairs:
            s, t = clique[i], clique[j]
            forbidden = branches_set - {s, t}
            p = shortest_path(s, t, forbidden)
            if p is None:
                return False, None
            pair_paths[(i, j)] = p

        # Check disjointness
        def check_disjoint(paths):
            used_interior = set()
            for key, p in paths.items():
                interior = set(p[1:-1])
                if interior & used_interior:
                    return False
                used_interior |= interior
            return True

        if check_disjoint(pair_paths):
            return True, pair_paths

        # Try to resolve conflicts by finding alternative paths
        for attempt in range(500):
            if time.time() > deadline:
                break
            shuffled_pairs = list(pairs)
            random.shuffle(shuffled_pairs)
            assigned = {}
            used_interior = set()
            success = True
            for i, j in shuffled_pairs:
                s, t = clique[i], clique[j]
                forbidden = (branches_set - {s, t}) | used_interior
                p = shortest_path(s, t, forbidden)
                if p is None:
                    success = False
                    break
                assigned[(i, j)] = p
                used_interior |= set(p[1:-1])
            if success:
                if len(assigned) == len(pairs):
                    final_paths = {}
                    for i, j in pairs:
                        final_paths[(i, j)] = assigned[(i, j)]
                    return True, final_paths
        return False, None

    # ---------- Search for maximum valid subdivision ----------
    best_valid = None
    best_valid_size = 0

    # Try the max clique
    if len(best_clique) >= 2:
        valid, paths = verify_clique_subdivision(best_clique)
        if valid:
            best_valid = (best_clique, paths)
            best_valid_size = len(best_clique)

    # Try subsets of the clique
    if best_valid is None:
        for size in range(len(best_clique) - 1, 1, -1):
            if time.time() > deadline:
                break
            found = False
            for subset in combinations(best_clique, size):
                valid, paths = verify_clique_subdivision(list(subset))
                if valid:
                    best_valid = (list(subset), paths)
                    best_valid_size = size
                    found = True
                    break
            if found:
                break

    # Try to grow: add vertices one by one
    if best_valid is not None:
        current_branches = list(best_valid[0])
        current_paths = best_valid[1]
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for v in range(n):
                if v in set(current_branches):
                    continue
                new_branches = current_branches + [v]
                valid, new_paths = verify_clique_subdivision(new_branches)
                if valid:
                    current_branches = new_branches
                    current_paths = new_paths
                    best_valid = (current_branches, current_paths)
                    best_valid_size = len(current_branches)
                    improved = True
                    break

    # Fallback
    if best_valid is None or len(best_valid[0]) < 2:
        for u, v in edges:
            if u != v:
                best_valid = ([u, v], {(0, 1): [u, v]})
                best_valid_size = 2
                break

    if best_valid is None:
        u, v = edges[0]
        best_valid = ([u, v], {(0, 1): [u, v]})

    # ---------- Build output ----------
    branches = best_valid[0]
    paths_dict = best_valid[1]

    branch_to_idx = {b: i for i, b in enumerate(branches)}

    output_paths = []
    for i in range(len(branches)):
        for j in range(i + 1, len(branches)):
            p = paths_dict.get((i, j)) or paths_dict.get((j, i))
            if p is None:
                p = [branches[i], branches[j]]
            if p[0] != branches[i]:
                p = p[::-1]
            output_paths.append(list(p))

    branches_out = [int(b) for b in branches]
    paths_out = [[int(v) for v in p] for p in output_paths]

    return {
        'branches': branches_out,
        'paths': paths_out
    }
