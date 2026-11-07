"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    def max_flow_with_paths(s, t, graph, n_nodes):
        """
        Edmonds-Karp max flow that also tracks the flow on each edge.
        graph: dict of dict {u: {v: cap}}
        Returns max_flow_val and a dict of flows { (u, v): flow }
        """
        # Deep copy graph for residual
        residual = {u: {v: c for v, c in vs.items()} for u, vs in graph.items()}
        # Initialize flow dict
        flow = {}
        
        max_flow_val = 0
        
        while True:
            # BFS to find augmenting path
            visited = [False] * n_nodes
            queue = [s]
            visited[s] = True
            parent = {s: -1}
            
            found = False
            while queue and not found:
                u = queue.pop(0)
                for v, cap in residual[u].items():
                    if not visited[v] and cap > 0:
                        visited[v] = True
                        parent[v] = u
                        if v == t:
                            found = True
                            break
                        queue.append(v)
                        
            if not found:
                break
            
            # Find min capacity on path
            path_flow = float('inf')
            v = t
            while v != s:
                u = parent[v]
                path_flow = min(path_flow, residual[u][v])
                v = u
            
            max_flow_val += path_flow
            
            # Update residual capacities and record flow
            v = t
            while v != s:
                u = parent[v]
                residual[u][v] -= path_flow
                if v not in residual[u]:
                    residual[v][u] = 0
                residual[v][u] += path_flow
                
                # Record flow on original edge (u, v)
                # Note: The flow might be sent in reverse if residual[v][u] was increased
                # but for unit capacities and simple paths, we can just accumulate
                # Actually, it's safer to store net flow.
                # Since we start with 0 flow, and add path_flow each time,
                # if the path uses u->v, flow[u,v] increases.
                # If the path uses v->u (reverse edge in residual), that means
                # we are cancelling flow on u->v.
                # However, Edmonds-Karp on unit networks usually finds simple paths.
                # Let's just store the final residual to derive flow?
                # Flow(u,v) = Original_Cap(u,v) - Residual(u,v)
                # This is valid for the final state.
                
                v = u
                
        # Calculate final flows from residual
        # flow(u, v) = graph[u][v] - residual[u][v]
        # Note: graph might not have key v in residual[u] if cap was 0 initially?
        # We initialized residual with all edges from graph.
        for u in graph:
            for v, cap in graph[u].items():
                if residual[u].get(v, 0) < cap:
                    f = cap - residual[u].get(v, 0)
                    if f > 0:
                        flow[(u, v)] = f
        
        return max_flow_val, flow

    def try_construct(order):
        """
        Greedily add vertices in 'order' to branches.
        Returns (branches, paths_dict)
        """
        B = []
        U = set() # Used internal vertices
        paths = {} # (u, v) -> [u, ..., v]
        
        for v in order:
            if v in B or v in U:
                continue
            
            if len(B) == 0:
                B.append(v)
                continue
            
            # Build flow graph
            # Nodes: 0..2n-1. 
            # x_in = 2*x, x_out = 2*x+1
            n_flow_nodes = 2 * n
            flow_graph = {i: {} for i in range(n_flow_nodes)}
            
            # Add vertex split edges
            for x in range(n):
                if x in B or x in U:
                    cap = 0
                else:
                    cap = 1
                flow_graph[2*x][2*x+1] = cap
                
            # Add original edges
            for u, v_edge in edges:
                # u_out -> v_in
                flow_graph[2*u+1][2*v_edge] = 1
                # v_out -> u_in
                flow_graph[2*v_edge+1][2*u] = 1
            
            # Source: v_out
            s = 2 * v + 1
            
            # Sinks: u_in for u in B
            # Add super sink T
            T = n_flow_nodes
            flow_graph[T] = {}
            for u in B:
                flow_graph[2*u][T] = 1
                
            n_flow_nodes += 1 # T
            
            flow_val, flow_edges = max_flow_with_paths(s, T, flow_graph, n_flow_nodes)
            
            if flow_val == len(B):
                # Extract paths
                # We need to find the path from s to each u_in that has flow 1.
                # The flow edges are in the flow graph node space.
                # We need to map them back to original vertices.
                
                # Build a directed graph of used edges for path tracing
                used_edges = {}
                for (u_f, v_f), f in flow_edges.items():
                    if f > 0:
                        if u_f not in used_edges:
                            used_edges[u_f] = []
                        used_edges[u_f].append(v_f)
                        
                # For each u in B, trace path from s to 2*u
                # Note: The path ends at 2*u (u_in), then goes to T.
                # We want the path from v to u in original graph.
                # s is 2*v+1 (v_out).
                # The path in flow graph: 2*v+1 -> ... -> 2*u -> T.
                # We need to strip the node splitting.
                
                valid = True
                new_U = set(U)
                
                for u in B:
                    target = 2 * u
                    # Find path from s to target in used_edges
                    # Since capacities are 1, the path is unique in the flow decomposition
                    # We can just do a DFS from s following used_edges
                    # But used_edges might have multiple outgoing edges if flow is > 1?
                    # No, vertex capacity is 1, so flow out of any node is at most 1.
                    # Except source and sink?
                    # Source s has outflow = len(B).
                    # Intermediate nodes have inflow=outflow=1.
                    # Sink 2*u has inflow=1, outflow=1 (to T).
                    
                    # So we can just follow the unique edge out of each node?
                    # For s, there are len(B) outgoing edges.
                    # We need to match each u to its specific path.
                    
                    # Let's find the specific edge from s that leads to u.
                    # Or better, just trace back from T?
                    # T has incoming edges from 2*u for each u in B with flow 1.
                    # So for each u, the last node is 2*u.
                    # We can trace backwards from 2*u to s?
                    # In the flow graph, edges are directed.
                    # We can build a reverse adjacency list of used edges.
                    
                # Let's build reverse adjacency for used edges
                rev_used_edges = {}
                for (u_f, v_f), f in flow_edges.items():
                    if f > 0:
                        if v_f not in rev_used_edges:
                            rev_used_edges[v_f] = []
                        rev_used_edges[v_f].append(u_f)
                        
                # For each u in B, trace from 2*u back to s
                for u in B:
                    curr = 2 * u
                    path_nodes = [curr]
                    while curr != s:
                        # Find predecessor
                        preds = rev_used_edges.get(curr, [])
                        if not preds:
                            valid = False
                            break
                        # There should be exactly one predecessor in the flow path
                        # because inflow is 1 for non-s nodes.
                        prev_node = preds[0]
                        path_nodes.append(prev_node)
                        curr = prev_node
                    if not valid:
                        break
                    path_nodes.reverse()
                    
                    # Convert flow graph nodes to original vertices
                    # Node 2*x -> x (in), 2*x+1 -> x (out)
                    # The path is s (2*v+1) -> ... -> 2*u
                    # The sequence of original vertices:
                    # If node is 2*x, it represents entering x.
                    # If node is 2*x+1, it represents leaving x.
                    # The path in original graph:
                    # Start at v (from 2*v+1)
                    # Next node in flow path is some 2*y or 2*y+1?
                    # Edge 2*v+1 -> 2*y implies v-y edge.
                    # So the original vertex is y.
                    
                    orig_path = [v] # Start with v
                    for node in path_nodes[1:]:
                        # node is 2*y or 2*y+1
                        # If it's 2*y, it's the 'in' node of y.
                        # If it's 2*y+1, it's the 'out' node of y.
                        # The transition from prev_node to node:
                        # prev_node was either 2*x (in) or 2*x+1 (out)?
                        # Actually, the flow path alternates:
                        # out -> in -> out -> in ...
                        # s is out (2*v+1).
                        # Next is in (2*y).
                        # Then out (2*y+1).
                        # Then in (2*z).
                        # ...
                        # Last is in (2*u).
                        
                        # So the original vertices are determined by the 'in' nodes and the start 'out' node.
                        # The 'in' node 2*y corresponds to vertex y.
                        # The 'out' node 2*y+1 corresponds to vertex y.
                        # So just take node // 2.
                        
                        orig_vertex = node // 2
                        orig_path.append(orig_vertex)
                        
                    # Check if path is valid (simple, adjacent)
                    # The flow guarantees disjointness in terms of vertex capacity.
                    # We just need to store it.
                    
                    # Add internal vertices to new_U
                    for node in orig_path[1:-1]:
                        new_U.add(node)
                        
                    # Store path
                    # The path is from v to u
                    paths[(v, u)] = orig_path
                
                if valid:
                    B.append(v)
                    U = new_U
            else:
                pass
        
        return B, paths

    # Try different orderings
    best_B = []
    best_paths = {}
    
    # Ordering 1: Degree descending
    order1 = sorted(range(n), key=lambda x: -len(adj[x]))
    B1, P1 = try_construct(order1)
    if len(B1) > len(best_B):
        best_B = B1
        best_paths = P1
        
    # Ordering 2: Degree ascending
    order2 = sorted(range(n), key=lambda x: len(adj[x]))
    B2, P2 = try_construct(order2)
    if len(B2) > len(best_B):
        best_B = B2
        best_paths = P2
        
    # Ordering 3-10: Random
    random.seed(42)
    for i in range(10):
        order_rand = list(range(n))
        random.shuffle(order_rand)
        Br, Pr = try_construct(order_rand)
        if len(Br) > len(best_B):
            best_B = Br
            best_paths = Pr

    if len(best_B) < 2:
        # Fallback to edge
        u, v = edges[0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
    
    # Prepare output
    branches = [int(v) for v in best_B]
    out_paths = []
    for u, v in combinations(branches, 2):
        # Find path
        if (u, v) in best_paths:
            p = best_paths[(u, v)]
        elif (v, u) in best_paths:
            p = best_paths[(v, u)][::-1]
        else:
            # Should not happen
            p = [u, v]
        out_paths.append([int(x) for x in p])
        
    return {'branches': branches, 'paths': out_paths}

# EVOLVE-BLOCK-END
