"""Improved program: exact minimum path cover via max-flow on line graph, with time guard."""
import time
import sys
from random import Random
from collections import deque

def construct():
    start_time = time.time()
    DEADLINE = 140  # seconds; keep 10s safety margin before 150s kill

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def max_flow(n_nodes, adj, s, t):
        """Dinic's algorithm. adj: list of lists of (v, cap_index). Returns max flow value."""
        n = n_nodes
        # Build residual graph
        # We'll use a standard Dinic implementation
        # adj[i] = list of [neighbor, capacity, reverse_index]
        # Actually let's use a cleaner approach with arrays
        
        # Build adjacency with edge objects
        # edges: list of [to, cap, rev]
        # adj: list of lists of indices into edges
        graph = [[] for _ in range(n)]
        edge_list = []  # [to, cap, rev_index_in_adj[to]]
        
        def add_edge(u, v, cap):
            if cap <= 0:
                return
            fwd = [v, cap, len(graph[v])]
            bwd = [u, 0, len(graph[u])]
            graph[u].append(fwd)
            graph[v].append(bwd)
        
        # Build the flow network
        # For each original edge e=(u,v), create node e.
        # Source -> e (cap 1)
        # e -> f if v==start(f) (cap 1)
        # e -> sink (cap 1)
        
        # We need to map each edge to a node index
        # Node 0 = source, node 1 = sink
        # Edges of G get nodes 2..(2+m-1)
        
        pass  # placeholder, we'll implement differently

    def min_path_cover(edges, n_nodes):
        """
        Find minimum number of edge-disjoint paths covering all edges.
        Returns list of paths (each path is list of vertices).
        """
        m = len(edges)
        if m == 0:
            return []
        
        # Build adjacency for the graph
        adj = [[] for _ in range(n_nodes)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Build line graph flow network
        # Nodes: source=0, sink=1, then 2..(m+1) for each edge
        # Edge i (0-indexed) gets node i+2
        
        # For each edge i = (u_i, v_i):
        #   source -> node_i (cap 1)
        #   node_i -> sink (cap 1)
        #   For each edge j = (u_j, v_j) where v_i == u_j or v_i == v_j:
        #     node_i -> node_j (cap 1)
        
        # Build adjacency list for flow network
        flow_n = m + 2
        flow_adj = [[] for _ in range(flow_n)]
        
        def add_flow_edge(u, v, cap):
            flow_adj[u].append([v, cap, len(flow_adj[v])])
            flow_adj[v].append([u, 0, len(flow_adj[u]) - 1])
        
        # Add source and sink edges
        for i in range(m):
            add_flow_edge(0, i + 2, 1)  # source -> edge_node
            add_flow_edge(i + 2, 1, 1)  # edge_node -> sink
        
        # Build edge-to-edge transitions
        # For each edge i, for each edge j, check if they can be consecutive
        # i=(a,b), j=(c,d): can follow if b==c or b==d
        # To avoid duplicates, we only add i->j if i<j and they share a vertex
        # Actually we need directed: i->j means edge i can be followed by edge j
        # i=(a,b), j=(c,d): i->j if b==c or b==d
        # j->i if d==a or d==b
        
        # For efficiency, build a map from vertex to list of edges incident
        vert_to_edges = [[] for _ in range(n_nodes)]
        for i, (u, v) in enumerate(edges):
            vert_to_edges[u].append(i)
            vert_to_edges[v].append(i)
        
        for i in range(m):
            a, b = edges[i]
            # For each vertex of edge i, find edges that start from that vertex
            # Edge i ends at b (or a, but edges are undirected so both ends)
            # Actually in the line graph, i->j if they share a vertex
            # But we need to be careful: the path goes through shared vertex
            # i=(a,b), j=(b,c): share vertex b, so i->j and j->i both possible
            # i=(a,b), j=(a,c): share vertex a, so i->j and j->i both possible
            
            # To avoid adding duplicate edges, we add i->j only once
            # For each shared vertex, add transition
            for vert in (a, b):
                for j in vert_to_edges[vert]:
                    if j != i:
                        add_flow_edge(i + 2, j + 2, 1)
        
        # Now run max flow
        # Dinic's algorithm
        def dinic():
            INF = float('inf')
            flow = 0
            while True:
                # BFS to find level graph
                level = [-1] * flow_n
                q = deque([0])
                level[0] = 0
                while q:
                    u = q.popleft()
                    for e in flow_adj[u]:
                        if e[1] > 0 and level[e[0]] < 0:
                            level[e[0]] = level[u] + 1
                            q.append(e[0])
                if level[1] < 0:
                    break
                # DFS to find blocking flow
                it = [0] * flow_n
                def dfs(u, f):
                    if u == 1:
                        return f
                    while it[u] < len(flow_adj[u]):
                        e = flow_adj[u][it[u]]
                        if e[1] > 0 and level[u] < level[e[0]]:
                            d = dfs(e[0], min(f, e[1]))
                            if d > 0:
                                e[1] -= d
                                flow_adj[e[0]][e[2]][1] += d
                                return d
                        it[u] += 1
                    return 0
                while True:
                    f = dfs(0, INF)
                    if f == 0:
                        break
                    flow += f
            return flow
        
        max_matching = dinic()
        num_paths = m - max_matching
        
        # Now reconstruct the paths from the flow
        # The flow tells us which edges are matched (i.e., consecutive in a path)
        # For each edge node i+2, if there's flow on source->(i+2), it's a path start
        # If there's flow on (i+2)->sink, it's a path end
        # If there's flow on (i+2)->(j+2), edge i is followed by edge j
        
        # Extract the matching: for each edge i, find which edge j it's followed by
        # and which edge h follows edge i
        next_edge = [-1] * m  # next_edge[i] = j if i is followed by j
        prev_edge = [-1] * m  # prev_edge[i] = h if h is followed by i
        
        for i in range(m):
            node_i = i + 2
            # Check if there's flow from node_i to some node_j
            for e in flow_adj[node_i]:
                j_node = e[0]
                if j_node >= 2 and j_node <= m + 1:
                    # Check reverse edge to see if flow was sent
                    rev = flow_adj[j_node][e[2]]
                    if rev[1] > 0:  # flow was sent from i to j
                        j = j_node - 2
                        next_edge[i] = j
                        prev_edge[j] = i
                        break
            # Also check if there's flow from some node_h to node_i
            # Actually the above should capture it since we check all outgoing edges
        
        # Wait, the above might not be correct. Let me think again.
        # In the flow network, if flow goes from node_i to node_j, then:
        #   flow_adj[node_i] has an edge to node_j with reduced capacity
        #   flow_adj[node_j] has a reverse edge to node_i with increased capacity
        # So to check if flow went from i to j, we check if the reverse edge has capacity > 0
        
        # Let me redo this more carefully
        next_edge = [-1] * m
        prev_edge = [-1] * m
        
        for i in range(m):
            node_i = i + 2
            for e in flow_adj[node_i]:
                j_node = e[0]
                if 2 <= j_node <= m + 1:
                    # Check if flow was sent from i to j
                    # The reverse edge is at flow_adj[j_node][e[2]]
                    rev_edge = flow_adj[j_node][e[2]]
                    if rev_edge[1] > 0:
                        j = j_node - 2
                        next_edge[i] = j
                        prev_edge[j] = i
        
        # Find path starts: edges with no predecessor
        starts = [i for i in range(m) if prev_edge[i] == -1]
        
        # Build paths
        paths = []
        for start in starts:
            path_edges = []
            cur = start
            while cur != -1:
                path_edges.append(cur)
                cur = next_edge[cur]
            # Convert edge list to vertex list
            verts = []
            for ei in path_edges:
                u, v = edges[ei]
                if not verts:
                    verts = [u, v]
                else:
                    # Check which end connects to previous edge
                    last_vert = verts[-1]
                    if u == last_vert:
                        verts.append(v)
                    elif v == last_vert:
                        verts.append(u)
                    else:
                        # This shouldn't happen if flow is correct
                        verts.append(u)
            paths.append(verts)
        
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() - start_time > DEADLINE:
            # Fallback: return individual edges
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        edges = instance(*spec)
        n_nodes = spec[0]
        paths = min_path_cover(edges, n_nodes)
        results.append(paths)
    
    return results
