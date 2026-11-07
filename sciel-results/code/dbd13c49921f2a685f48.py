import time
from collections import defaultdict, deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    adj = defaultdict(list)
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    
    # 1. Find Max Clique using Bron-Kerbosch with pivot
    def bron_kerbosch(R, P, X, pivot=None):
        if not P and not X:
            yield R
            return
        
        if pivot is None:
            if P:
                pivot = P[0]
            elif X:
                pivot = X[0]
            else:
                return
        
        # Choose pivot to maximize branching reduction
        # Simple heuristic: pick node in P U X with max degree in P
        best_pivot = None
        max_deg = -1
        for v in P:
            deg = len([u for u in P if u in adj[v]])
            if deg > max_deg:
                max_deg = deg
                best_pivot = v
        if best_pivot is not None:
            pivot = best_pivot
            
        # Iterate over P \ N(pivot)
        candidates = [v for v in P if v not in adj[pivot]] if pivot is not None else list(P)
        
        for v in candidates:
            yield from bron_kerbosch(R + [v], [u for u in P if u in adj[v]], [u for u in X if u in adj[v]])
            P.remove(v)
            X.append(v)
            
    max_clique = []
    # To speed up, we can stop if we find a large enough clique, but we want the max.
    # For n=24, it's fast.
    for clique in bron_kerbosch([], list(range(n)), []):
        if len(clique) > len(max_clique):
            max_clique = clique

    branches = list(max_clique)
    if len(branches) < 2:
        # Fallback: find any edge
        for u, v in P['edges']:
            branches = [u, v]
            break
        if len(branches) < 2:
            # Graph is empty or single node? Problem says at least 2 branches.
            # If impossible, return empty or minimal valid?
            # "There must be at least 2 branches".
            # If no edge, we can't form a path.
            # But the graph has edges.
            pass

    # Initialize paths
    paths = {}
    for i in range(len(branches)):
        for j in range(i+1, len(branches)):
            u, v = branches[i], branches[j]
            # Check if edge exists
            if v in adj[u]:
                paths[(u,v)] = [u, v]
            else:
                # This shouldn't happen for a clique
                # Find any path?
                # For now, assume clique
                pass

    used_internal = set()
    
    # Helper: Max Flow with vertex splitting
    def max_flow_disjoint_paths(source, sinks, forbidden):
        # forbidden: set of nodes that cannot be used as internal nodes
        # source: start node
        # sinks: list of target nodes
        # Returns True if |sinks| disjoint paths exist, and the paths
        
        # Nodes in flow graph:
        # source
        # sinks
        # internal nodes: V \ (branches U forbidden U {source} U sinks)
        # Actually, branches are sinks. Source is new branch.
        # Internal nodes are V \ (branches U forbidden U {source})
        
        # Mapping original node -> flow node(s)
        # For internal node x: x_in, x_out
        # For source: source
        # For sink s: s
        
        flow_nodes = set()
        flow_nodes.add(source)
        for s in sinks:
            flow_nodes.add(s)
            
        internal_nodes = [x for x in range(n) if x not in branches and x not in forbidden and x != source]
        for x in internal_nodes:
            flow_nodes.add((x, 'in'))
            flow_nodes.add((x, 'out'))
            
        # Build graph
        flow_adj = defaultdict(list)
        flow_cap = defaultdict(int)
        
        def add_edge(u, v, cap):
            flow_adj[u].append(v)
            flow_cap[(u, v)] = cap
            flow_adj[v].append(u)
            flow_cap[(v, u)] = 0
            
        # Add edges for internal nodes
        for x in internal_nodes:
            add_edge((x, 'in'), (x, 'out'), 1)
            
        # Add original graph edges
        for u in range(n):
            for v in adj[u]:
                if u == v: continue
                # Determine flow nodes for u and v
                # If u is source: u
                # If u is sink: u
                # If u is internal: (u, 'out') for outgoing, (u, 'in') for incoming
                # Wait, direction matters.
                # Edge u->v in flow graph.
                # If u is internal, we leave from u_out.
                # If v is internal, we enter v_in.
                
                u_flow = u
                if u in internal_nodes:
                    u_flow = (u, 'out')
                    
                v_flow = v
                if v in internal_nodes:
                    v_flow = (v, 'in')
                
                # Check if both endpoints are in flow_nodes
                if u_flow in flow_nodes and v_flow in flow_nodes:
                    # Capacity 1
                    add_edge(u_flow, v_flow, 1)
                    
        # Add Super Sink
        T = 'T'
        flow_nodes.add(T)
        for s in sinks:
            add_edge(s, T, 1)
            
        # Run Max Flow (Edmonds-Karp)
        # BFS for augmenting path
        total_flow = 0
        while True:
            # Find path from source to T
            parent = {source: None}
            queue = deque([source])
            found = False
            while queue and not found:
                u = queue.popleft()
                for v in flow_adj[u]:
                    if v not in parent and flow_cap[(u, v)] > 0:
                        parent[v] = u
                        if v == T:
                            found = True
                            break
                        queue.append(v)
                        
            if not found:
                break
                
            # Find min capacity on path
            path_cap = float('inf')
            v = T
            while v != source:
                u = parent[v]
                path_cap = min(path_cap, flow_cap[(u, v)])
                v = u
                
            # Update residual graph
            v = T
            while v != source:
                u = parent[v]
                flow_cap[(u, v)] -= path_cap
                flow_cap[(v, u)] += path_cap
                v = u
                
            total_flow += path_cap
            
            if total_flow == len(sinks):
                break
                
        if total_flow < len(sinks):
            return False, []
            
        # Extract paths
        # We need to find the actual paths in the original graph.
        # This is tricky because the flow might mix.
        # But since capacities are 1, we can trace back from each sink.
        # However, the residual graph is modified.
        # We should store the flow edges or re-trace.
        # Easier: Store the paths during BFS? No, BFS finds one path at a time.
        # We can collect the augmenting paths as we find them.
        
        # Let's restart and collect paths.
        # Or, since we just need to know IF it's possible for the greedy step,
        # we can just return True/False.
        # But we need the paths to update `used_internal`.
        # So we MUST extract them.
        
        # Let's modify the flow to store the augmenting paths.
        # Re-run flow, but save the path nodes.
        
        # Actually, let's just return False for now if we can't easily extract.
        # But we need the paths.
        
        # Alternative: Since the graph is small, we can use the fact that
        # the flow decomposes into paths.
        # We can do a DFS on the flow edges (where cap > 0 in original direction)
        # from source to sinks.
        
        # Let's build a "flow graph" of used edges.
        # An edge (u,v) is used if flow_cap[(v,u)] > 0 (since we added reverse edge with 0, and subtracted from forward).
        # Wait, initial cap was 1. If used, forward becomes 0, reverse becomes 1.
        # So used edges are those with flow_cap[(u,v)] == 0 and flow_cap[(v,u)] == 1?
        # No, if we sent 1 unit, forward is 0, reverse is 1.
        # So we can check if flow_cap[(v,u)] > 0.
        
        used_edges = set()
        for u in flow_nodes:
            for v in flow_adj[u]:
                if flow_cap[(v, u)] > 0: # Reverse has flow, so forward was used
                    used_edges.add((u, v))
                    
        # Now, find paths in the subgraph of used_edges.
        # We need |sinks| disjoint paths from source to sinks.
        # Since they are vertex-disjoint (except source/sinks), we can just trace.
        
        paths_found = []
        visited = {source}
        
        # We need to find paths to each sink.
        # Since they are disjoint, we can pick any path to any sink.
        # But we need to match them? No, just any set of disjoint paths.
        
        # Let's do a DFS from source to find a path to any unvisited sink.
        # Repeat len(sinks) times.
        
        remaining_sinks = set(sinks)
        
        for _ in range(len(sinks)):
            # Find path from source to any node in remaining_sinks
            # Using only used_edges
            # BFS
            parent = {source: None}
            queue = deque([source])
            target = None
            while queue and target is None:
                u = queue.popleft()
                for v in flow_adj[u]:
                    if (u, v) in used_edges and v not in parent:
                        parent[v] = u
                        if v in remaining_sinks:
                            target = v
                            break
                        queue.append(v)
                        
            if target is None:
                return False, [] # Should not happen if flow succeeded
                
            # Reconstruct path
            path = []
            v = target
            while v is not None:
                path.append(v)
                v = parent[v]
            path.reverse()
            
            # Map flow nodes to original nodes
            orig_path = []
            for node in path:
                if isinstance(node, tuple):
                    orig_path.append(node[0])
                else:
                    orig_path.append(node)
            
            # Remove consecutive duplicates (if any, e.g. x_in -> x_out -> x_in? No)
            # x_in -> x_out is an edge. x is one node.
            # Path: source -> x_in -> x_out -> y_in -> ... -> sink
            # Orig: source -> x -> x -> y -> ... -> sink
            # We need to deduplicate consecutive same nodes.
            deduped = []
            for i in range(len(orig_path)):
                if i == 0 or orig_path[i] != orig_path[i-1]:
                    deduped.append(orig_path[i])
                    
            paths_found.append(deduped)
            remaining_sinks.remove(target)
            # Mark internal nodes as visited?
            # The paths are vertex disjoint, so we don't need to mark,
            # but we should ensure we don't reuse edges/nodes in the next DFS.
            # Since we are tracing in the *used* subgraph, and the flow is 1,
            # the paths are disjoint.
            # However, the DFS might pick a path that shares nodes with a previously found path?
            # No, because the flow is vertex-disjoint. The set of used edges forms a set of disjoint paths.
            # So any path we find in the used subgraph is disjoint from others?
            # Yes, if we remove the edges of the found path from the used subgraph for the next search.
            
            # Remove edges of this path from used_edges
            for i in range(len(path)-1):
                used_edges.remove((path[i], path[i+1]))
                
        return True, paths_found

    improved = True
    while improved:
        improved = False
        for v in range(n):
            if v in branches or v in used_internal:
                continue
                
            sinks = branches
            # Check if we can add v
            success, new_paths = max_flow_disjoint_paths(v, sinks, used_internal)
            
            if success:
                # Add v to branches
                branches.append(v)
                # Add new paths
                for p in new_paths:
                    # p is from v to some b in branches_old
                    # The pair is (v, b) or (b, v)
                    # We need to store in paths dict
                    # Find the endpoint in branches_old
                    # p[0] is v. p[-1] is some b.
                    b = p[-1]
                    # Ensure key is sorted tuple
                    key = tuple(sorted([v, b]))
                    paths[key] = p
                    
                # Update used_internal
                for p in new_paths:
                    for node in p[1:-1]:
                        used_internal.add(node)
                        
                improved = True
                break # Restart loop with new branch set
                
    # Convert paths keys to lists? No, keys are tuples.
    # Return format: 'paths': list of paths?
    # "paths": one simple path for each unordered pair of branches.
    # The key in the dict is the pair.
    # The value is the path.
    # The prompt says: "Return a dict with exactly the keys 'branches' and 'paths'. Each path is a list..."
    # It doesn't specify if 'paths' is a list of lists or a dict.
    # "paths": one simple path for each unordered pair of branches.
    # Usually, this implies a list of paths, or a dict mapping pairs to paths.
    # The seed program returns: 'paths':[list(e) for e in combinations(row,2)]
    # This is a LIST of paths.
    # So I should return a list.
    
    final_paths = []
    for i in range(len(branches)):
        for j in range(i+1, len(branches)):
            u, v = branches[i], branches[j]
            key = (u, v)
            if key in paths:
                final_paths.append(paths[key])
            else:
                # Should not happen
                final_paths.append([u, v])
                
    return {'branches': branches, 'paths': final_paths}

if __name__ == "__main__":
    construct()
