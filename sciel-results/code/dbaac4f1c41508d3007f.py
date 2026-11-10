import time
import random
from itertools import combinations
from collections import defaultdict

# Use scipy for max flow if available, otherwise implement simple one.
# The prompt says SciPy is available.
try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_flow
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # 5s margin
    
    n = P['n']
    edges = P['edges']
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Helper to check if a set of branches is valid and return paths
    # We will use a greedy expansion strategy.
    
    def get_initial_seeds():
        seeds = []
        # All triangles
        for i, j, k in combinations(range(n), 3):
            if (j in adj[i] and k in adj[i] and k in adj[j]):
                seeds.append([i, j, k])
        # All pairs (sorted by degree)
        deg = [len(adj[i]) for i in range(n)]
        pairs = list(combinations(range(n), 2))
        pairs.sort(key=lambda p: (deg[p[0]] + deg[p[1]]), reverse=True)
        seeds.extend(pairs)
        return seeds

    best_branches = []
    best_paths = {}
    
    seeds = get_initial_seeds()
    
    # Shuffle seeds to add some randomness if time permits, or just iterate
    # Given n=24, iterating all triangles and top pairs is fast.
    
    for seed in seeds:
        if time.time() > deadline:
            break
            
        current_branches = list(seed)
        # Initialize paths for the seed
        # For a pair, path is [u, v]
        # For a triangle, paths are edges [u,v], [u,w], [v,w]
        # Note: For a triangle, internal vertices are empty.
        paths_dict = {}
        used_interiors = set()
        
        # If seed is a pair, just one path
        if len(seed) == 2:
            u, v = seed
            paths_dict[(u, v)] = [u, v]
        elif len(seed) == 3:
            u, v, w = seed
            paths_dict[(u, v)] = [u, v]
            paths_dict[(u, w)] = [u, w]
            paths_dict[(v, w)] = [v, w]
            
        # Try to expand
        expanded = True
        while expanded:
            expanded = False
            candidates = [v for v in range(n) if v not in current_branches and v not in used_interiors]
            
            # Sort candidates by connectivity to current branches
            # Prefer candidates that have many neighbors in the "available" space
            # or are close to branches
            scored_candidates = []
            for c in candidates:
                # Simple heuristic: count neighbors in adj[c] that are not blocked
                # blocked = current_branches U used_interiors U {c}
                blocked = set(current_branches) | used_interiors | {c}
                score = sum(1 for nb in adj[c] if nb not in blocked)
                scored_candidates.append((score, c))
            
            scored_candidates.sort(reverse=True)
            
            for score, c in scored_candidates:
                if time.time() > deadline:
                    break
                    
                # Try to add c
                new_paths, new_used = try_add_vertex(c, current_branches, used_interiors, paths_dict, adj, n)
                if new_paths is not None:
                    # Success
                    current_branches.append(c)
                    paths_dict = new_paths
                    used_interiors = new_used
                    expanded = True
                    break # Restart candidate loop with new state
        
        # Update best
        if len(current_branches) > len(best_branches):
            best_branches = current_branches
            best_paths = paths_dict

    # If no triangle or pair found (shouldn't happen for this graph), fallback
    if not best_branches:
         u, v = edges[0]
         best_branches = [u, v]
         best_paths = {(u, v): [u, v]}
         
    # Format output
    out_paths = []
    for u, v in combinations(best_branches, 2):
         # Ensure key exists
         key = (u, v)
         if key not in best_paths:
             key = (v, u)
         out_paths.append(best_paths[key])
         
    return {
        'branches': list(best_branches),
        'paths': out_paths
    }

def try_add_vertex(c, branches, used_interiors, paths_dict, adj, n):
    """
    Attempts to add vertex c to the set of branches.
    Returns (new_paths_dict, new_used_interiors) if successful, else (None, None).
    """
    k = len(branches)
    if k == 0:
        return None, None
        
    # We need k paths from c to each b in branches.
    # These paths must be internally disjoint from each other and from used_interiors.
    # They must avoid branches (except endpoints) and c (except start).
    
    # Build Flow Network
    # Nodes:
    # 0: Source S
    # 1: Sink T
    # For each vertex v in 0..n-1:
    #   v_in = 2 + 2*v
    #   v_out = 2 + 2*v + 1
    # Total nodes: 2 + 2*n
    
    num_nodes = 2 + 2 * n
    S = 0
    T = 1
    
    # Adjacency list for flow graph
    # We will build a matrix for scipy or a list of lists for custom flow.
    # Since n is small, let's use a simple max flow implementation to avoid dependency issues or overhead.
    # Actually, scipy is fast. Let's use scipy.
    
    row = []
    col = []
    data = []
    
    def add_edge(u, v, cap):
        if cap <= 0:
            return
        row.append(u)
        col.append(v)
        data.append(cap)
        
    # Vertex capacities
    for v in range(n):
        vin = 2 + 2 * v
        vout = 2 + 2 * v + 1
        
        if v == c:
            # c is the source of all paths. Capacity k.
            add_edge(S, vin, k)
            add_edge(vin, vout, k)
        elif v in branches:
            # v is a sink. Capacity 1.
            # Note: v is in branches, so it's an endpoint.
            # We don't add S->v or v->T here directly, 
            # but we need to allow flow to enter v_in and leave v_out to T.
            # Wait, the path ends at v. So flow goes ... -> v_in -> v_out -> T?
            # Or ... -> v_in -> T?
            # Standard vertex capacity: v_in -> v_out (cap 1).
            # If v is a sink, we connect v_out -> T with cap 1.
            # But we must ensure flow doesn't pass *through* v to another branch.
            # Since v is in branches, it's a terminal.
            add_edge(vin, vout, 1)
            add_edge(vout, T, 1)
        else:
            # Intermediate vertex.
            if v in used_interiors:
                # Blocked
                pass
            else:
                add_edge(vin, vout, 1)
                
    # Graph edges
    for u, v in P['edges']:
        uout = 2 + 2 * u + 1
        vin = 2 + 2 * v
        vout = 2 + 2 * v + 1
        uin = 2 + 2 * u
        
        # u -> v
        add_edge(uout, vin, 1)
        # v -> u
        add_edge(vout, uin, 1)
        
    # Run Max Flow
    try:
        A = csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
        flow_value, _, _, _ = maximum_flow(A, S, T)
    except Exception:
        return None, None
        
    if flow_value < k:
        return None, None
        
    # Extract paths
    # We need to trace the flow from S to T.
    # Since we used scipy, we need to get the flow matrix.
    # maximum_flow returns (flow_value, flow_success, node_flow, edge_flow)
    # edge_flow is a matrix where edge_flow[i,j] is the flow from i to j.
    
    # Re-run to get edge_flow if not returned properly or just use the returned one.
    # The signature is: flow_value, flow_success, node_flow, edge_flow
    # Let's re-capture it properly.
    
    # Actually, the previous call might not have captured edge_flow.
    # Let's do it in one go.
    
    # Re-do flow to get edge_flow
    try:
        A = csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
        flow_value, flow_success, node_flow, edge_flow = maximum_flow(A, S, T)
    except Exception:
        return None, None
        
    if flow_value < k:
        return None, None
        
    # Trace paths
    # We have k units of flow from S to T.
    # Each unit corresponds to a path from c to a branch b.
    # We can find paths by walking from S.
    
    new_paths = dict(paths_dict) # Copy existing
    new_used = set(used_interiors)
    
    # We need to assign each path to a specific branch.
    # The flow into T comes from branch_out nodes.
    # We can trace back from T.
    
    paths_found = []
    
    # Create a reverse adjacency list of the flow
    # flow[u, v] > 0 means edge u->v carries flow.
    # We want to find k disjoint paths from c_in to branch_outs.
    # Actually, the flow starts at S -> c_in.
    
    # Let's trace from S.
    # We know k paths start at S.
    # We can use DFS/BFS on the flow graph.
    
    # To ensure we get distinct paths, we can just pick any path, remove its flow, and repeat?
    # No, the flow is already a valid decomposition.
    # We can just trace the edges with positive flow.
    
    # Since the graph is small, we can just do a simple trace.
    # Start at S.
    # Next must be c_in.
    # Then c_out.
    # Then ...
    
    # Let's identify the k paths.
    # We can use a simple algorithm:
    # 1. Start at S.
    # 2. Follow edges with positive flow.
    # 3. When we hit a branch_out, we have a path.
    # 4. Record the vertices.
    # 5. Subtract the flow from the edges used? No, we just need the vertex sequence.
    # Since vertex capacities are 1 (except c), the paths are vertex-disjoint (except at c).
    
    # We need to map node indices back to original vertices.
    # node_index = 2 + 2*v (+1 for out)
    # v = (node_index - 2) // 2
    
    def node_to_vertex(node):
        if node == S or node == T:
            return None
        return (node - 2) // 2

    # Trace k paths
    # We can do this by repeatedly finding a path from S to T in the residual graph of the flow?
    # Or just walk the flow edges.
    
    # Since the flow is integral, we can just pick an edge with flow 1.
    
    # Let's store the flow edges
    flow_edges = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if edge_flow[i, j] > 0:
                flow_edges.append((i, j, int(edge_flow[i, j])))
                
    # We need to decompose the flow into k paths.
    # A simple way:
    # While there is flow from S:
    #   Start at S
    #   Follow any edge with flow > 0
    #   Decrement flow on that edge
    #   Continue until T
    #   This gives one path.
    
    # This modifies edge_flow, which is fine.
    
    for _ in range(k):
        current = S
        path_nodes = []
        while current != T:
            found_next = False
            for i in range(num_nodes):
                if edge_flow[current, i] > 0:
                    edge_flow[current, i] -= 1
                    current = i
                    path_nodes.append(current)
                    found_next = True
                    break
            if not found_next:
                # Should not happen if flow is valid
                break
        if current == T:
            # Extract vertex sequence
            verts = []
            for node in path_nodes:
                v = node_to_vertex(node)
                if v is not None:
                    # Avoid duplicates from in/out
                    if not verts or verts[-1] != v:
                        verts.append(v)
            # The path should start with c and end with a branch
            # verts[0] should be c
            # verts[-1] should be a branch
            if verts and verts[0] == c:
                b = verts[-1]
                # Add to new_paths
                # Key is (c, b) or (b, c)
                key = (c, b)
                new_paths[key] = verts
                # Update used_interiors
                for v in verts[1:-1]:
                    new_used.add(v)
                
    # Verify all branches were connected
    # If any branch is missing, it means the flow didn't reach it?
    # But max flow was k, and each branch has cap 1 to T.
    # So all k branches must have been reached.
    
    return new_paths, new_used
