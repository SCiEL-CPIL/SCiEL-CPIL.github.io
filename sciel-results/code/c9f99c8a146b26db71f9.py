"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    start_time = time.time()
    deadline = start_time + 86.0  # 90s wall-clock, leave 4s margin

    # --- Max-flow based vertex-disjoint path oracle ---
    def max_flow_vertex_disjoint(branches, deadline):
        """
        Given a set of branch vertices, check if internally vertex-disjoint paths
        exist for all pairs, and extract them.
        
        We model each vertex v (except branches) as a node split into v_in -> v_out
        with capacity 1. Branch vertices have infinite capacity. Edges go from
        u_out to v_in with capacity 1.
        
        We create a super-source connected to each branch and a super-sink.
        Actually, for the complete graph subdivision, we need paths between ALL pairs.
        We handle this by adding demand for each pair via a multi-commodity approach,
        but that's hard. Instead, we use a sequential approach with flow verification.
        
        Better approach: For k branches, we need C(k,2) internally disjoint paths.
        We can model this as: create a flow network where each branch gets supply/demand.
        Actually, the standard trick: for a clique subdivision with k terminals,
        we need k(k-1)/2 vertex-disjoint paths. 
        
        Simpler: use the fact that we can check feasibility via max-flow on an
        expanded graph. For each pair (i,j) of branches, we need a path. 
        The vertex-disjointness constraint means each non-branch vertex can be used
        by at most one path.
        
        I'll use a different strategy: sequential path finding with flow-based
        feasibility check. First try to find paths greedily, then verify all
        are disjoint. If not, retry with different ordering.
        
        For the oracle: given a set of branches, build a flow network where
        we send 1 unit of flow from a super-source to each branch, and from each
        branch to a super-sink, requiring total flow = k(k-1)/2 through the
        non-branch vertices with capacity 1 each.
        
        Actually the cleanest: model as edge-disjoint in the line graph, but let's
        use vertex splitting.
        """
        branch_set = set(branches)
        k = len(branches)
        num_pairs = k * (k - 1) // 2
        
        # Vertex splitting: for each non-branch vertex, split into in/out with cap 1
        # For branch vertices, they have unlimited capacity (they're endpoints)
        # 
        # Network: 
        #   For each vertex v: v_in -> v_out (cap 1 if non-branch, inf if branch)
        #   For each edge (u,v): u_out -> v_in (cap 1) and v_out -> u_in (cap 1)
        #   Super source S -> each branch's in-node (cap 1)  [actually need multiple]
        #   
        # This is getting complex. Let me use a simpler sequential approach
        # with backtracking and flow-based pruning.
        
        # Sequential greedy with multiple orderings
        best_paths = None
        num_attempts = 30
        
        for attempt in range(num_attempts):
            if time.time() > deadline:
                break
            
            # Choose ordering of pairs
            pairs = list(itertools.combinations(branches, 2))
            if attempt == 0:
                # Direct edges first, then by degree
                pair_order = sorted(pairs, key=lambda uv: (0 if uv[1] in adj[uv[0]] else 1, len(adj[uv[0]]) + len(adj[uv[1]])))
            elif attempt == 1:
                pair_order = sorted(pairs, key=lambda uv: len(adj[uv[0]]) + len(adj[uv[1]]), reverse=True)
            elif attempt == 2:
                pair_order = list(pairs)
                random.shuffle(pair_order)
            else:
                pair_order = list(pairs)
                random.shuffle(pair_order)
            
            used_internal = set()
            paths_found = {}
            success = True
            
            for u, v in pair_order:
                if time.time() > deadline:
                    success = False
                    break
                forbidden = branch_set - {u, v} | used_internal
                
                # BFS for shortest path
                path = _bfs_path(u, v, forbidden, adj)
                
                if path is None:
                    # Try longer paths with limited DFS
                    path = _dfs_path(u, v, forbidden, adj, max_depth=6)
                
                if path is None:
                    success = False
                    break
                
                # Verify path
                if path[0] != u or path[-1] != v:
                    success = False
                    break
                for i in range(len(path) - 1):
                    if path[i+1] not in adj[path[i]]:
                        success = False
                        break
                if not success:
                    break
                for x in path[1:-1]:
                    if x in branch_set or x in used_internal:
                        success = False
                        break
                if not success:
                    break
                
                paths_found[(u, v)] = path
                for x in path[1:-1]:
                    used_internal.add(x)
            
            if success:
                best_paths = [paths_found[(u, v)] for (u, v) in pairs]
                break
        
        return best_paths
    
    def _bfs_path(src, dst, forbidden, adj):
        """Find shortest path from src to dst avoiding forbidden vertices."""
        if src == dst:
            return [src]
        if dst in forbidden:
            return None
        visited = {src}
        parent = {src: None}
        queue = deque([src])
        while queue:
            node = queue.popleft()
            for nb in sorted(adj[node]):
                if nb == dst:
                    path = [dst]
                    cur = node
                    while cur is not None:
                        path.append(cur)
                        cur = parent[cur]
                    path.reverse()
                    return path
                if nb not in forbidden and nb not in visited:
                    visited.add(nb)
                    parent[nb] = node
                    queue.append(nb)
        return None
    
    def _dfs_path(src, dst, forbidden, adj, max_depth=6):
        """Find a simple path from src to dst avoiding forbidden, with DFS backtracking."""
        if src == dst:
            return [src]
        
        def dfs(current, path, visited, depth):
            if depth > max_depth:
                return None
            for nb in sorted(adj[current], key=lambda x: (-len(adj[x]), x)):
                if nb == dst and len(path) >= 2:
                    return path + [dst]
                if nb not in forbidden and nb not in visited and nb != src and nb != dst:
                    visited.add(nb)
                    result = dfs(nb, path + [nb], visited, depth + 1)
                    if result is not None:
                        return result
                    visited.remove(nb)
            return None
        
        return dfs(src, [src], {src}, 0)
    
    # --- Flow-based oracle: check if k branches can support a clique subdivision ---
    def flow_feasible(branches, deadline):
        """
        Use max-flow to check if internally vertex-disjoint paths exist for all
        pairs of branch vertices.
        
        Construction:
        - Split each non-branch vertex v into v_in -> v_out with capacity 1.
        - Branch vertices are not split (or split with infinite capacity).
        - For each edge (u,v) in original graph: add edge u_out -> v_in with cap 1.
        - Add super source S and super sink T.
        - For each branch b: add S -> b_in with capacity (k-1), and b_out -> T with capacity (k-1).
        - Total required flow: k*(k-1)/2... 
        
        Actually this doesn't directly work because we need paths between specific pairs.
        
        Better approach: The problem is equivalent to checking if the graph contains
        a topological minor of K_k. By Menger's theorem generalization, this is
        related to connectivity.
        
        For practical purposes, I'll use a simpler check: for each branch b,
        verify that there are k-1 vertex-disjoint paths from b to the other k-1
        branches. This is a necessary condition. We can check this with max-flow
        where b is the source and we connect all other branches to a sink.
        """
        branch_list = list(branches)
        k = len(branch_list)
        if k < 2:
            return True
        branch_set = set(branch_list)
        
        # For each branch, check k-1 disjoint paths to other branches
        for b in branch_list:
            if time.time() > deadline:
                return False
            
            others = [x for x in branch_list if x != b]
            
            # Build flow network
            # Node indexing: for vertex v, in-node = 2*v, out-node = 2*v+1
            # Plus super source and sink
            num_nodes = 2 * n + 2
            SRC = 2 * n
            SNK = 2 * n + 1
            
            # Build adjacency for flow (use simple BFS-based max flow since graph is small)
            # Capacity matrix
            cap = [[0] * num_nodes for _ in range(num_nodes)]
            
            for v in range(n):
                if v in branch_set:
                    # Branch: infinite capacity through the vertex
                    cap[2*v][2*v+1] = 1000
                else:
                    cap[2*v][2*v+1] = 1
            
            for u, v in P['edges']:
                # u_out -> v_in
                cap[2*u+1][2*v] = 1
                # v_out -> u_in
                cap[2*v+1][2*u] = 1
            
            # Super source to b_in
            cap[SRC][2*b] = k - 1
            # Each other branch's out to sink
            for o in others:
                cap[2*o+1][SNK] = 1
            
            # Run max flow
            flow = _max_flow(cap, num_nodes, SRC, SNK)
            
            if flow < k - 1:
                return False
        
        return True
    
    def _max_flow(cap, num_nodes, src, sink):
        """Ford-Fulkerson with BFS (Edmonds-Karp)."""
        flow = 0
        while True:
            # BFS to find augmenting path
            parent = [-1] * num_nodes
            parent[src] = src
            queue = deque([src])
            found = False
            while queue and not found:
                u = queue.popleft()
                for v in range(num_nodes):
                    if parent[v] == -1 and cap[u][v] > 0:
                        parent[v] = u
                        if v == sink:
                            found = True
                            break
                        queue.append(v)
            
            if not found:
                break
            
            # Find bottleneck
            bottleneck = float('inf')
            v = sink
            while v != src:
                u = parent[v]
                bottleneck = min(bottleneck, cap[u][v])
                v = u
            
            # Update residual
            v = sink
            while v != src:
                u = parent[v]
                cap[u][v] -= bottleneck
                cap[v][u] += bottleneck
                v = u
            
            flow += bottleneck
        
        return flow
    
    # --- Bron-Kerbosch for max cliques ---
    def find_max_cliques(deadline):
        cliques = []
        
        def bronk(R, P, X):
            if time.time() > deadline:
                return
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(list(R))
                return
            if P or X:
                pivot = max(P | X, key=lambda v: len(adj[v] & P))
            else:
                return
            candidates = [v for v in P if v not in adj[pivot]]
            for v in candidates:
                bronk(R | {v}, P & adj[v], X & adj[v])
                P = P - {v}
                X = X | {v}
        
        bronk(frozenset(), frozenset(range(n)), frozenset())
        cliques.sort(key=len, reverse=True)
        return cliques
    
    # --- Main search ---
    best_result = None
    best_size = 0
    
    # Phase 1: Find largest clique
    cliques = find_max_cliques(deadline)
    
    for clique in cliques:
        if time.time() > deadline:
            break
        if len(clique) < 2:
            break
        if len(clique) <= best_size:
            continue
        
        # Quick flow feasibility check
        if not flow_feasible(clique, deadline):
            continue
        
        result = max_flow_vertex_disjoint(clique, deadline)
        if result is not None:
            best_result = {'branches': list(clique), 'paths': result}
            best_size = len(clique)
    
    # Phase 2: Greedy expansion
    if best_size >= 2:
        base_branches = list(best_result['branches'])
    else:
        u, v = P['edges'][0]
        base_branches = [u, v]
        best_size = 2
        best_result = {'branches': [u, v], 'paths': [[u, v]]}
    
    # Multiple restarts with different initial sets
    initial_sets = [list(base_branches)]
    for c in cliques[:10]:
        if len(c) >= 2:
            initial_sets.append(list(c))
    # Add some random pairs
    for _ in range(5):
        random.shuffle(P['edges'])
        e = P['edges'][0]
        initial_sets.append([e[0], e[1]])
    
    for init_set in initial_sets:
        if time.time() > deadline - 2:
            break
        
        cur = list(init_set)
        
        for iteration in range(30):
            if time.time() > deadline - 2:
                break
            
            candidates = [v for v in range(n) if v not in set(cur)]
            # Sort by degree (lower degree = more constrained = try first)
            candidates.sort(key=lambda v: len(adj[v]))
            random.shuffle(candidates)
            
            added = False
            for v in candidates:
                if time.time() > deadline - 2:
                    break
                new_branches = cur + [v]
                
                # Flow feasibility check first (fast reject)
                if not flow_feasible(new_branches, deadline):
                    continue
                
                result = max_flow_vertex_disjoint(new_branches, deadline)
                if result is not None:
                    cur = new_branches
                    if len(cur) > best_size:
                        best_size = len(cur)
                        best_result = {'branches': list(cur), 'paths': result}
                    added = True
                    break
            
            if not added:
                # Try swap: remove one, add another
                if len(cur) >= 3:
                    swapped = False
                    for i in range(len(cur)):
                        if time.time() > deadline - 2:
                            break
                        temp = cur[:i] + cur[i+1:]
                        cands2 = [v for v in range(n) if v not in set(temp)]
                        cands2.sort(key=lambda v: len(adj[v]))
                        random.shuffle(cands2)
                        for v in cands2[:15]:
                            if time.time() > deadline - 2:
                                break
                            new_b = temp + [v]
                            if not flow_feasible(new_b, deadline):
                                continue
                            result = max_flow_vertex_disjoint(new_b, deadline)
                            if result is not None:
                                cur = new_b
                                if len(cur) > best_size:
                                    best_size = len(cur)
                                    best_result = {'branches': list(cur), 'paths': result}
                                swapped = True
                                break
                        if swapped:
                            break
                else:
                    break
    
    if best_result is not None:
        branches = [int(b) for b in best_result['branches']]
        paths = [[int(x) for x in p] for p in best_result['paths']]
        return {'branches': branches, 'paths': paths}
    
    # Final fallback
    u, v = P['edges'][0]
    return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
# EVOLVE-BLOCK-END
