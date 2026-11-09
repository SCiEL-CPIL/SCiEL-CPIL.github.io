import time
import random
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

DEADLINE = time.time() + 55.0

def construct():
    t0 = time.time()
    n = P['n']
    edges = P['edges']

    adj = [set() for _ in range(n)]
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)

    def time_up():
        return time.time() - t0 > DEADLINE - 3.0

    def fallback():
        u, v = edges[0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}

    best = fallback()
    best_size = 2

    def max_flow_disjoint_paths(branches):
        """
        Use max-flow with vertex splitting to find k(k-1)/2 pairwise internally
        disjoint paths for all pairs of branch vertices.
        Returns list of paths or None.
        """
        bs = set(branches)
        sorted_bs = sorted(branches)
        pairs = list(combinations(sorted_bs, 2))
        m = len(pairs)
        if m == 0:
            return []

        # Build flow network with vertex splitting
        # For each vertex v: v_in -> v_out with capacity 1 (if non-branch) or infinity (if branch)
        # For each edge (u,v): u_out -> v_in and v_out -> u_in with capacity 1
        # For each pair (i,j): source_i -> i_in and j_out -> sink_ij
        # We need a super-source and super-sink approach, but since each path is independent,
        # we model it as: for each pair p=(u,v), create a flow from u to v.
        # Actually, we need ALL paths simultaneously. Use a multi-commodity flow? No.
        # Better: use the standard trick - create a flow network where we route all paths at once.
        # 
        # Alternative: Since paths must be internally disjoint, we can model this as:
        # - Split each non-branch vertex v into v_in and v_out with edge capacity 1
        # - Branch vertices get infinite capacity
        # - For each pair (u,v), we need 1 unit of flow from u to v
        # - This is a multi-commodity flow problem, which is hard in general.
        #
        # Simpler approach: Use successive shortest path with residual graph.
        # Route paths one at a time, updating residual capacities.
        
        # Node mapping: v -> (2*v) for v_in, (2*v+1) for v_out
        def vin(v): return 2 * v
        def vout(v): return 2 * v + 1
        def to_vertex(node): return node // 2
        def is_in(node): return node % 2 == 0

        total_nodes = 2 * n
        
        # Build adjacency for flow network
        # flow_adj[node] = list of (neighbor, capacity, edge_index)
        # We'll use a simple representation
        flow_adj = [[] for _ in range(total_nodes)]
        # edge_info[i] = (u, v, capacity)
        edge_info = []
        
        def add_edge(u, v, cap):
            """Add directed edge u->v with capacity cap. Returns edge index."""
            idx = len(edge_info)
            edge_info.append([u, v, cap])
            flow_adj[u].append((v, idx))
            # Reverse edge with 0 capacity
            idx_rev = len(edge_info)
            edge_info.append([v, u, 0])
            flow_adj[v].append((u, idx_rev))
            return idx

        # Add vertex split edges
        for v in range(n):
            if v in bs:
                cap = float('inf')
            else:
                cap = 1
            add_edge(vin(v), vout(v), cap)
        
        # Add graph edges (bidirectional)
        for a, b in edges:
            add_edge(vout(a), vin(b), 1)
            add_edge(vout(b), vin(a), 1)
        
        # For each pair, we need to route 1 unit from u to v
        # We'll do this greedily with BFS in residual graph
        
        # Residual graph tracking
        # edge_info[i][2] = current residual capacity
        # We need to track which edges belong to which path for reconstruction
        
        paths_found = []
        
        def bfs_residual(s, t):
            """Find path in residual graph from s to t using BFS. Returns list of edge indices or None."""
            parent_node = {s: None}
            parent_edge = {s: None}
            q = deque([s])
            while q:
                curr = q.popleft()
                if curr == t:
                    break
                for nb, eidx in flow_adj[curr]:
                    if edge_info[eidx][2] > 0 and nb not in parent_node:
                        parent_node[nb] = curr
                        parent_edge[nb] = eidx
                        if nb == t:
                            break
                        q.append(nb)
            if t not in parent_node:
                return None
            # Reconstruct path
            path_edges = []
            node = t
            while node != s:
                eidx = parent_edge[node]
                path_edges.append(eidx)
                node = edge_info[eidx][0]
            path_edges.reverse()
            return path_edges
        
        # Route paths one at a time
        for u, v in pairs:
            if time_up():
                return None
            s = vout(u)  # Start from u's output
            t = vin(v)   # End at v's input
            
            path_edges = bfs_residual(s, t)
            if path_edges is None:
                return None
            
            # Augment flow along this path
            for eidx in path_edges:
                edge_info[eidx][2] -= 1
                # Reverse edge
                rev_idx = eidx ^ 1
                edge_info[rev_idx][2] += 1
            
            # Extract the actual vertex path
            # The path in the flow network goes: u_out -> ... -> v_in
            # We need to extract the sequence of vertices
            vertex_path = [u]
            node = s
            for eidx in path_edges:
                nb = edge_info[eidx][1]
                # If we're going from v_out to w_in, we visit w
                # If we're going from v_in to v_out, we stay at v
                if is_in(nb):
                    w = to_vertex(nb)
                    if w != vertex_path[-1]:
                        vertex_path.append(w)
                node = nb
            # The last node should be v_in, and v should be the last vertex
            if vertex_path[-1] != v:
                vertex_path.append(v)
            
            # Verify the path
            if vertex_path[0] != u or vertex_path[-1] != v:
                return None
            # Check all internal vertices are non-branch and unique
            internal = set(vertex_path[1:-1])
            if internal & bs:
                return None
            # Check consecutive vertices are adjacent
            for i in range(len(vertex_path) - 1):
                if vertex_path[i+1] not in adj[vertex_path[i]]:
                    return None
            
            paths_found.append(vertex_path)
        
        return paths_found

    # Bron-Kerbosch to find maximal cliques
    def bron_kerbosch(R, P_set, X, result, time_limit):
        if time.time() > time_limit:
            return
        if not P_set and not X:
            if len(R) > best_size:
                result.append(list(R))
            return
        # Choose pivot from P_set union X
        candidates = P_set | X
        if candidates:
            pivot = max(candidates, key=lambda v: len(P_set & adj[v]))
            candidates_to_try = P_set - adj[pivot]
        else:
            candidates_to_try = P_set
        for v in list(candidates_to_try):
            bron_kerbosch(R | {v}, P_set & adj[v], X & adj[v], result, time_limit)
            P_set.discard(v)
            X.add(v)

    # Find all maximal cliques
    cliques = []
    initial_P = set(range(n))
    bron_kerbosch(set(), initial_P, set(), cliques, time.time() + 15.0)

    # Sort cliques by size descending
    cliques.sort(key=len, reverse=True)

    # Try each clique as a candidate branch set
    for clique in cliques:
        if time_up():
            break
        if len(clique) <= best_size:
            break
        paths = max_flow_disjoint_paths(clique)
        if paths is not None:
            best = {'branches': [int(x) for x in sorted(clique)], 'paths': [[int(x) for x in p] for p in paths]}
            best_size = len(clique)

    # Now try to extend the best clique by adding vertices
    if best_size < n and not time_up():
        current_branches = set(best['branches'])
        # Score non-branch vertices by how many current branches they connect to
        def score_vertex(v):
            return sum(1 for b in current_branches if v in adj[b])

        candidates = [v for v in range(n) if v not in current_branches]
        candidates.sort(key=score_vertex, reverse=True)

        # Try adding one vertex at a time, greedily
        for _ in range(15):  # up to 15 additions
            if time_up():
                break
            improved = False
            for v in candidates:
                if time_up():
                    break
                new_set = current_branches | {v}
                paths = max_flow_disjoint_paths(new_set)
                if paths is not None:
                    best = {'branches': [int(x) for x in sorted(new_set)], 'paths': [[int(x) for x in p] for p in paths]}
                    best_size = len(new_set)
                    current_branches = new_set
                    candidates = [x for x in range(n) if x not in current_branches]
                    candidates.sort(key=score_vertex, reverse=True)
                    improved = True
                    break
            if not improved:
                break

    # Try random subsets larger than best_size
    if not time_up():
        deg_sorted = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
        for k in range(best_size + 1, min(best_size + 6, n + 1)):
            if time_up():
                break
            for _ in range(80):
                if time_up():
                    break
                subset = random.sample(deg_sorted[:min(20, n)], k)
                paths = max_flow_disjoint_paths(subset)
                if paths is not None:
                    best = {'branches': [int(x) for x in sorted(subset)], 'paths': [[int(x) for x in p] for p in paths]}
                    best_size = k
                    break

    return best
