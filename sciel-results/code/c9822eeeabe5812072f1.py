import time
import sys
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    from random import Random

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def hopcroft_karp(left_nodes, right_nodes, adj_left):
        """
        Hopcroft-Karp maximum bipartite matching.
        left_nodes: list of left-side node ids
        right_nodes: list of right-side node ids
        adj_left: dict mapping left node -> list of right nodes it connects to
        Returns: matching dict {left_node: right_node}
        """
        # Map right nodes to indices for arrays
        right_index = {r: i for i, r in enumerate(right_nodes)}
        n_right = len(right_nodes)
        
        # Match arrays
        match_left = {}   # left_node -> right_node
        match_right = [-1] * n_right  # right_index -> left_node
        
        # BFS: find shortest augmenting paths
        # dist[left_node] = distance from free left node in alternating BFS
        INF = float('inf')
        
        def bfs():
            # dist for left nodes
            dist = {}
            queue = deque()
            for ln in left_nodes:
                if ln not in match_left:
                    dist[ln] = 0
                    queue.append(ln)
                else:
                    dist[ln] = INF
            
            found = False
            while queue:
                u = queue.popleft()
                for v in adj_left.get(u, []):
                    vi = right_index[v]
                    matched_left = match_right[vi]
                    if matched_left == -1:
                        found = True
                    elif dist.get(matched_left, INF) > dist[u] + 1:
                        dist[matched_left] = dist[u] + 1
                        queue.append(matched_left)
            return dist, found
        
        def dfs(u, dist, visited_right):
            for v in adj_left.get(u, []):
                vi = right_index[v]
                if visited_right[v]:
                    continue
                visited_right[v] = True
                matched_left = match_right[vi]
                if matched_left == -1:
                    # Augmenting path found
                    match_left[u] = v
                    match_right[vi] = u
                    return True
                elif dist.get(matched_left, INF) == dist[u] + 1:
                    if dfs(matched_left, dist, visited_right):
                        match_left[u] = v
                        match_right[vi] = u
                        return True
            return False
        
        matching_size = 0
        while True:
            if time.time() > deadline:
                break
            dist, found = bfs()
            if not found:
                break
            visited_right = set()
            for ln in left_nodes:
                if ln not in match_left and dist.get(ln, INF) == 0:
                    if dfs(ln, dist, visited_right):
                        matching_size += 1
                    if time.time() > deadline:
                        break
        
        return match_left

    def decompose_graph(edges):
        """
        Minimum path cover via maximum matching on edge-splitting bipartite graph.
        
        Each edge e=(u,v) becomes:
          - Left node: (u, e_id)  representing "arriving at u via edge e"
          - Right node: (v, e_id) representing "leaving from v via edge e"
        
        Actually, the standard formulation for minimum path cover in a directed graph:
        For an undirected graph, we treat each undirected edge as two directed edges.
        But we need edge-disjoint paths, so each undirected edge can be used once.
        
        Better formulation: 
        - Left nodes: for each edge e=(u,v), a node representing "edge e ending at v" 
          i.e., the right endpoint of the edge when traversed u->v
        - Right nodes: for each edge e=(u,v), a node representing "edge e starting at u"
          i.e., the left endpoint when traversed u->v
        
        Wait, let me think more carefully.
        
        Standard minimum path cover in a DAG: create bipartite graph with V_left, V_right.
        For each directed edge (u,v), add bipartite edge u_left -> v_right.
        Max matching gives min path cover = |V| - |matching|.
        
        For our undirected edge-disjoint path cover:
        Each undirected edge can be oriented either way. We need to partition edges into 
        paths where consecutive edges share a vertex.
        
        Reformulation: Think of each edge as a "task". Two edges e1 and e2 can be 
        consecutive in a path if they share a vertex, and the shared vertex is an 
        endpoint of both. So edge e1=(a,b) and e2=(b,c) can form path a-b-c.
        
        Bipartite formulation:
        - Left side: one node per edge, representing the edge as a "segment"
        - Right side: one node per edge  
        - For edges e1=(u,v) and e2=(u,w) (sharing vertex u, u is endpoint of both):
          Add bipartite edge from e1_left to e2_right (meaning e1 comes before e2, 
          connected at vertex u)
          Also add e2_left to e1_right.
        - Similarly for e1=(u,v) and e2=(v,w): add e1_left->e2_right and e2_left->e1_right
        
        Then max matching on this bipartite graph: a matched pair (e1_left, e2_right) 
        means e1 is immediately followed by e2 in a path. The number of paths = 
        |E| - |max_matching|.
        
        This works because the bipartite graph is acyclic in the matching sense 
        (each edge appears once on left and once on right), so the matching defines 
        a set of disjoint chains (paths) over the edges.
        """
        if not edges:
            return []
        
        m = len(edges)
        edge_list = list(edges)
        
        # Build adjacency for the bipartite matching
        # Left node i = edge i (as predecessor)
        # Right node j = edge j (as successor)
        # Edge i_left -> j_right if edge i and edge j share a vertex AND 
        # the shared vertex is the "end" of edge i and "start" of edge j
        # 
        # Actually for undirected edges, we just need them to share a vertex.
        # If edge i = (a,b) and edge j = (b,c), they share vertex b.
        # We can chain i then j (going through b), or j then i.
        # 
        # In the bipartite graph: i_left connects to j_right if they share a vertex.
        # The matching will determine the direction/ordering.
        
        # Build: for each vertex, list of edges incident to it
        vert_edges = defaultdict(list)
        for idx, (u, v) in enumerate(edge_list):
            vert_edges[u].append(idx)
            vert_edges[v].append(idx)
        
        # Build bipartite adjacency: left node i -> set of right nodes j
        # i and j can be consecutive if they share a vertex (and i != j)
        adj_left = defaultdict(list)
        for v, eids in vert_edges.items():
            for i in eids:
                for j in eids:
                    if i != j:
                        adj_left[i].append(j)
        
        # Deduplicate
        for i in adj_left:
            adj_left[i] = list(set(adj_left[i]))
        
        left_nodes = list(range(m))
        right_nodes = list(range(m))
        
        # Run Hopcroft-Karp
        match_left = hopcroft_karp(left_nodes, right_nodes, adj_left)
        
        # Reconstruct paths
        # match_left[i] = j means edge i is immediately followed by edge j
        # Find path starts: edges i where i is NOT in match_left.values()
        # i.e., no edge points to i as its successor... 
        # Wait: match_left[i] = j means i -> j (i precedes j)
        # So the "next" of edge i is match_left.get(i)
        # The "prev" of edge j is the i such that match_left[i] = j
        
        next_edge = {}
        for i, j in match_left.items():
            next_edge[i] = j
        
        # Find starts: edges that are not a successor of any other edge
        successors = set(match_left.values())
        starts = [i for i in range(m) if i not in successors]
        
        paths = []
        for s in starts:
            path_edges = []
            cur = s
            visited = set()
            while cur is not None and cur not in visited:
                visited.add(cur)
                path_edges.append(cur)
                cur = next_edge.get(cur)
            # Convert edge indices to vertex path
            u, v = edge_list[path_edges[0]]
            # Determine direction: we need to figure out the traversal order
            # For a chain of edges e0, e1, ..., ek, we need to orient them consistently
            # e0 and e1 share a vertex, e1 and e2 share a vertex, etc.
            
            # Build the path by walking through shared vertices
            vertex_path = []
            # Start with first edge
            e0 = edge_list[path_edges[0]]
            # The next edge e1 shares a vertex with e0. The shared vertex is the connection point.
            # We need to determine the order of vertices.
            
            # Let's build it step by step
            # First edge: (a, b). The next edge shares either a or b.
            # If next edge shares b, path goes a -> b -> ...
            # If next edge shares a, path goes b -> a -> ...
            
            # Actually, let me just track the current "end" vertex
            # Start: we don't know which end is the "start" of the path
            # Let's just build the vertex sequence by finding shared vertices
            
            if len(path_edges) == 1:
                u, v = edge_list[path_edges[0]]
                paths.append([u, v])
            else:
                # Build vertex path
                # e0 = (u0, v0), e1 = (u1, v1), ..., ek = (uk, vk)
                # Shared vertex between e(i) and e(i+1) is the connection
                # The path is: one endpoint of e0, then shared(e0,e1), then shared(e1,e2), ..., other endpoint of ek
                
                verts = []
                # Determine the two endpoints of the path
                # Walk through edges, tracking which vertex is "used" as connection
                prev_edge = edge_list[path_edges[0]]
                # The path starts at one end of prev_edge
                # The connection to next edge tells us which end is the "middle"
                
                # Let's collect all vertices in order
                # For each consecutive pair, find the shared vertex
                shared_verts = []
                for idx in range(len(path_edges) - 1):
                    e_a = edge_list[path_edges[idx]]
                    e_b = edge_list[path_edges[idx + 1]]
                    # Find shared vertex
                    common = set(e_a) & set(e_b)
                    if len(common) == 1:
                        shared_verts.append(common.pop())
                    else:
                        # Shouldn't happen for simple graphs, but handle
                        shared_verts.append(e_a[1])
                
                # Now determine start and end
                # The path is: start_vertex -> shared[0] -> shared[1] -> ... -> end_vertex
                # start_vertex is the endpoint of e0 that is NOT shared[0]
                # end_vertex is the endpoint of ek that is NOT shared[-1]
                
                e_first = edge_list[path_edges[0]]
                e_last = edge_list[path_edges[-1]]
                
                if shared_verts:
                    start_v = e_first[0] if e_first[0] != shared_verts[0] else e_first[1]
                    end_v = e_last[0] if e_last[0] != shared_verts[-1] else e_last[1]
                else:
                    start_v = e_first[0]
                    end_v = e_first[1]
                
                vertex_path = [start_v] + shared_verts + [end_v]
                paths.append(vertex_path)
        
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline:
            break
        edges = instance(*spec)
        paths = decompose_graph(edges)
        results.append(paths)
    
    # If we ran out of time, fill remaining with trivial edge paths
    while len(results) < 3:
        spec = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))[len(results)]
        edges = instance(*spec)
        results.append([[u, v] for u, v in edges])
    
    return results
