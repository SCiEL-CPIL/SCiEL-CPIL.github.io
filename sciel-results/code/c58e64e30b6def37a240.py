"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Pre-compute edge list for flow graph construction
    # We need to map original (u,v) to flow nodes
    # Flow nodes: 
    #   in(x) = 2*x
    #   out(x) = 2*x + 1
    
    def find_disjoint_paths(new_v, branches, blocked_set):
        """
        Checks if we can add new_v to branches.
        Returns list of paths if successful, else None.
        Paths are from new_v to each b in branches.
        """
        if not branches:
            return []
            
        k = len(branches)
        # Nodes in flow graph: 0 to 2n-1 are vertex splits. 2n is super sink.
        # Total nodes: 2n + 1
        N_flow = 2 * n + 1
        sink = 2 * n
        
        # Initialize residual graph
        # residual[u][v] = capacity
        residual = [[0] * N_flow for _ in range(N_flow)]
        
        # 1. Vertex split edges
        # For each vertex x, edge in(x) -> out(x)
        for x in range(n):
            if x in blocked_set or x in branches:
                # Branches are endpoints, but internal nodes of paths must avoid them.
                # However, the path ends AT the branch.
                # The flow enters branch b via in(b).
                # So in(b) -> out(b) is NOT used for the path ending at b.
                # The path goes ... -> in(b) -> Sink.
                # So for branches, we don't need the split edge capacity 1 for traversal THROUGH them.
                # We only need to reach in(b).
                # So set capacity 0 for in->out for branches? 
                # Yes, because paths cannot pass through branches.
                residual[2*x][2*x+1] = 0
            else:
                residual[2*x][2*x+1] = 1
        
        # 2. Original edges
        # Edge u-v becomes out(u) -> in(v) and out(v) -> in(u)
        for u, v in edges:
            residual[2*u+1][2*v] = 1
            residual[2*v+1][2*u] = 1
            
        # 3. Source and Sink connections
        # Source: out(new_v)
        source = 2 * new_v + 1
        
        # Sinks: in(b) for each b in branches
        # Connect in(b) -> super_sink
        for b in branches:
            residual[2*b][sink] = 1
            
        # Max Flow (Edmonds-Karp)
        max_flow_val = 0
        parent = [0] * N_flow # Store parent for BFS
        
        while True:
            # BFS
            queue = deque()
            queue.append(source)
            visited = [False] * N_flow
            visited[source] = True
            parent[source] = -1
            
            found = False
            while queue and not found:
                u = queue.popleft()
                for v in range(N_flow):
                    if not visited[v] and residual[u][v] > 0:
                        visited[v] = True
                        parent[v] = u
                        if v == sink:
                            found = True
                            break
                        queue.append(v)
                        
            if not found:
                break
                
            # Find min capacity on path
            path_flow = float('inf')
            v = sink
            while v != source:
                u = parent[v]
                path_flow = min(path_flow, residual[u][v])
                v = u
                
            max_flow_val += path_flow
            
            # Update residual
            v = sink
            while v != source:
                u = parent[v]
                residual[u][v] -= path_flow
                residual[v][u] += path_flow
                v = u
                
        if max_flow_val != k:
            return None
            
        # Extract paths
        # The used edges are those where residual[v][u] > 0 (reverse edge has flow)
        # AND original capacity was > 0?
        # Actually, we can just trace back from sink for each branch?
        # No, the flow goes from source to sink.
        # We can reconstruct the paths by finding the edges with flow 1.
        
        # Build a graph of used edges
        used_graph = {i: [] for i in range(N_flow)}
        for u in range(N_flow):
            for v in range(N_flow):
                # If residual[v][u] > 0, it means there is flow from u to v
                # But we must ensure this flow is part of the current max flow.
                # Since capacities are 1, any reverse edge with capacity > 0 indicates a used forward edge.
                if residual[v][u] > 0:
                    # Check if this edge existed in original graph or was a sink edge
                    # Sink edges: in(b) -> sink. 
                    # Original edges: out(u) -> in(v).
                    # Split edges: in(x) -> out(x).
                    used_graph[u].append(v)
                    
        # Now find paths from source to each in(b)
        paths = []
        visited_nodes = set()
        
        # We need to find k disjoint paths.
        # Since they are disjoint, we can just DFS from source for each target in(b)
        # But we must ensure we don't reuse edges.
        # The `used_graph` contains exactly the edges with flow 1.
        # So we can just find paths in this subgraph.
        
        targets = [2*b for b in branches]
        
        # To ensure we get the correct paths, we can remove edges as we find them?
        # Or just DFS. Since paths are disjoint, DFS from source will find one path.
        # We need to be careful: The source has multiple outgoing edges.
        
        # Let's find paths one by one.
        # We will mark used edges in `used_graph` by removing them?
        # Since it's a list of lists, we can filter.
        
        # Better: Just DFS and track used edges.
        # Given the small size, we can do:
        
        current_used_edges = set()
        for u in range(N_flow):
            for v in used_graph[u]:
                current_used_edges.add((u, v))
                
        found_paths = []
        for target in targets:
            # DFS from source to target using only edges in current_used_edges
            stack = [(source, [source])]
            found = None
            while stack:
                node, path = stack.pop()
                if node == target:
                    found = path
                    break
                for neighbor in used_graph[node]:
                    if (node, neighbor) in current_used_edges:
                        stack.append((neighbor, path + [neighbor]))
                        
            if not found:
                return None # Should not happen if max flow is correct
                
            # Convert flow nodes back to original vertices
            # Flow node 2*x is in(x), 2*x+1 is out(x).
            # Path in flow graph: source(out(v)) -> ... -> target(in(b))
            # We need the sequence of original vertices.
            
            orig_path = []
            for node in found:
                # If node is even, it's in(x). If odd, out(x).
                # The vertex is node // 2.
                orig_path.append(node // 2)
                
            # The path starts at new_v and ends at b.
            # We need to remove duplicate consecutive vertices?
            # in(x) -> out(x) is a split edge.
            # out(u) -> in(v) is an original edge.
            # So the sequence of vertices might have duplicates like [x, x, y, y, z]?
            # Let's check:
            # Path: out(v) -> in(x) -> out(x) -> in(y) ...
            # Vertices: v, x, x, y ...
            # We should deduplicate consecutive identical vertices.
            
            clean_path = []
            for v in orig_path:
                if not clean_path or clean_path[-1] != v:
                    clean_path.append(v)
                    
            found_paths.append(clean_path)
            
            # Remove these edges from current_used_edges to ensure disjointness for next path
            for i in range(len(found) - 1):
                current_used_edges.remove((found[i], found[i+1]))
                
        return found_paths

    def solve_with_order(order):
        B = []
        paths_dict = {}
        blocked = set()
        
        for v in order:
            if v in B or v in blocked:
                continue
            
            if len(B) == 0:
                B.append(v)
                continue
            
            # Try to add v
            new_paths = find_disjoint_paths(v, B, blocked)
            if new_paths is not None:
                B.append(v)
                # Update paths_dict
                for p in new_paths:
                    # p is from v to some b in B
                    # Identify the target branch
                    target_b = p[-1]
                    # Store path for pair (v, target_b)
                    # Ensure consistent ordering for key? 
                    # Let's store as tuple (min, max) or just dict with set of branches?
                    # The task requires paths for each unordered pair.
                    # We can store in a dict keyed by frozenset or tuple.
                    key = (v, target_b)
                    paths_dict[key] = p
                    # Also update blocked set
                    for node in p[1:-1]:
                        blocked.add(node)
                        
        return B, paths_dict

    best_B = []
    best_paths = {}
    
    # Generate multiple orders
    orders = []
    
    # 1. Degree Descending
    orders.append(sorted(range(n), key=lambda x: -len(adj[x])))
    
    # 2. Degree Ascending
    orders.append(sorted(range(n), key=lambda x: len(adj[x])))
    
    # 3. Randomized orders
    for seed in range(20):
        random.seed(seed)
        order = list(range(n))
        random.shuffle(order)
        orders.append(order)
        
    # 4. Core-Periphery heuristic? 
    # Just use the above.
    
    for order in orders:
        B, paths = solve_with_order(order)
        if len(B) > len(best_B):
            best_B = B
            best_paths = paths
            
    # Fallback
    if len(best_B) < 2:
        u, v = edges[0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
        
    # Construct final output
    branches = [int(v) for v in best_B]
    out_paths = []
    from itertools import combinations
    
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
