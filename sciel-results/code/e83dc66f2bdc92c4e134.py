"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

N = P['n']
EDGES = P['edges']
EDGE_SET = set(frozenset(e) for e in EDGES)
ADJ = {i: set() for i in range(N)}
for u, v in EDGES:
    ADJ[u].add(v)
    ADJ[v].add(u)

def find_2regular_spanning_subgraph(available_edges):
    """
    Find a 2-regular spanning subgraph (cycle cover) using max-flow.
    
    We model this as: each vertex v has an in-node v_in and out-node v_out.
    - Source -> v_in: capacity 2
    - v_out -> sink: capacity 2
    - v_in -> v_out: capacity 2 (ensures vertex is used)
    - For each edge (u,v) in available_edges:
      - u_out -> v_in: capacity 1
      - v_out -> u_in: capacity 1
    
    A feasible flow of value 2*N means we have a 2-regular spanning subgraph.
    """
    if not available_edges:
        return None
    
    # Build adjacency for degree check
    adj = {i: set() for i in range(N)}
    for e in available_edges:
        u, v = tuple(e)
        adj[u].add(v)
        adj[v].add(u)
    
    # All vertices must have degree >= 2
    for i in range(N):
        if len(adj[i]) < 2:
            return None
    
    # Node numbering:
    # 0: source
    # 1: sink
    # 2 + 2*i: i_in
    # 2 + 2*i + 1: i_out
    # Total nodes: 2 + 2*N
    
    src = 0
    sink = 1
    def in_node(i):
        return 2 + 2*i
    def out_node(i):
        return 2 + 2*i + 1
    
    total_nodes = 2 + 2*N
    
    # Build flow network: adjacency list with (neighbor, capacity, reverse_index)
    # We'll use a simple max-flow implementation
    graph = [[] for _ in range(total_nodes)]
    
    def add_edge(u, v, cap):
        """Add edge u->v with capacity cap. Returns index of forward edge."""
        forward = [v, cap, len(graph[v])]
        backward = [u, 0, len(graph[u])]
        graph[u].append(forward)
        graph[v].append(backward)
        return len(graph[u]) - 1
    
    # Source to in-nodes
    for i in range(N):
        add_edge(src, in_node(i), 2)
    
    # In-nodes to out-nodes
    for i in range(N):
        add_edge(in_node(i), out_node(i), 2)
    
    # Out-nodes to sink
    for i in range(N):
        add_edge(out_node(i), sink, 2)
    
    # For each available edge (u,v), add directed edges u_out->v_in and v_out->u_in
    for e in available_edges:
        u, v = tuple(e)
        add_edge(out_node(u), in_node(v), 1)
        add_edge(out_node(v), in_node(u), 1)
    
    # Max-flow using Edmonds-Karp (BFS-based)
    from collections import deque
    
    def max_flow():
        total_flow = 0
        while True:
            # BFS to find augmenting path
            parent = [-1] * total_nodes
            parent_edge = [-1] * total_nodes
            parent[src] = src
            queue = deque([src])
            
            while queue:
                u = queue.popleft()
                if u == sink:
                    break
                for idx, (v, cap, rev) in enumerate(graph[u]):
                    if cap > 0 and parent[v] == -1:
                        parent[v] = u
                        parent_edge[v] = idx
                        queue.append(v)
            
            if parent[sink] == -1:
                break  # No more augmenting paths
            
            # Find bottleneck
            path_flow = float('inf')
            v = sink
            while v != src:
                u = parent[v]
                idx = parent_edge[v]
                path_flow = min(path_flow, graph[u][idx][1])
                v = u
            
            # Update residual capacities
            v = sink
            while v != src:
                u = parent[v]
                idx = parent_edge[v]
                graph[u][idx][1] -= path_flow
                graph[v][idx][2] += path_flow  # Wait, this is wrong. Let me fix.
                # Actually, the reverse edge is at graph[v][graph[u][idx][2]]
                # Let me redo this properly
                v = u
            
            total_flow += path_flow
        
        return total_flow
    
    # Let me rewrite max-flow more carefully
    def max_flow_v2():
        total_flow = 0
        while True:
            # BFS to find augmenting path
            parent = [-1] * total_nodes
            parent_edge = [-1] * total_nodes
            parent[src] = src
            queue = deque([src])
            
            while queue:
                u = queue.popleft()
                if u == sink:
                    break
                for idx, (v, cap, rev_idx) in enumerate(graph[u]):
                    if cap > 0 and parent[v] == -1:
                        parent[v] = u
                        parent_edge[v] = idx
                        queue.append(v)
            
            if parent[sink] == -1:
                break  # No more augmenting paths
            
            # Find bottleneck
            path_flow = float('inf')
            v = sink
            while v != src:
                u = parent[v]
                idx = parent_edge[v]
                path_flow = min(path_flow, graph[u][idx][1])
                v = u
            
            # Update residual capacities
            v = sink
            while v != src:
                u = parent[v]
                idx = parent_edge[v]
                # Forward edge: graph[u][idx]
                graph[u][idx][1] -= path_flow
                # Reverse edge: graph[v][graph[u][idx][2]]
                rev_idx = graph[u][idx][2]
                graph[v][rev_idx][1] += path_flow
                v = u
            
            total_flow += path_flow
        
        return total_flow
    
    flow_value = max_flow_v2()
    
    if flow_value != 2 * N:
        return None
    
    # Extract the selected edges
    # For each original edge (u,v), check if flow went through u_out->v_in or v_out->u_in
    selected_edges = set()
    
    # We need to check which edges have flow
    # Rebuild the graph structure to identify original edges
    # Actually, let's just check the residual graph
    
    # For each pair (u,v) in available_edges, check if there's flow on u_out->v_in or v_out->u_in
    # We can do this by looking at the residual capacities
    
    # Actually, it's easier to just reconstruct from the flow
    # For each vertex i, look at which in-node edges have flow (i.e., which out-nodes sent flow to i_in)
    
    # Let me just check: for each edge (u,v) in available_edges,
    # check if the residual capacity of u_out->v_in is 0 (meaning flow was sent)
    # OR if the residual capacity of v_out->u_in is 0
    
    # But we need to be careful: the edge might have been used in reverse
    # Actually, since we added both directions, we need to check both
    
    # Let's track which edges were used by checking the flow
    # For simplicity, let's just check all possible directed edges
    
    # Actually, the cleanest way: after max-flow, for each original undirected edge {u,v},
    # check if flow(u_out, v_in) + flow(v_out, u_in) == 1 (exactly one direction used)
    # But we don't have direct access to flow values easily.
    
    # Alternative: rebuild the flow network and check residual capacities
    # If original capacity was 1 and residual is 0, then flow = 1
    # If original capacity was 1 and residual is 1, then flow = 0
    
    # Let me just iterate through available_edges and check
    
    for e in available_edges:
        u, v = tuple(e)
        # Check u_out -> v_in
        # Find this edge in graph[out_node(u)]
        found_uv = False
        found_vu = False
        
        for (v_node, cap, rev) in graph[out_node(u)]:
            if v_node == in_node(v):
                # Original capacity was 1, current residual is cap
                # If cap == 0, flow was 1
                if cap == 0:
                    found_uv = True
                break
        
        for (v_node, cap, rev) in graph[out_node(v)]:
            if v_node == in_node(u):
                if cap == 0:
                    found_vu = True
                break
        
        if found_uv or found_vu:
            selected_edges.add(frozenset([u, v]))
    
    if len(selected_edges) != N:
        # Should have exactly N edges for a 2-regular spanning subgraph
        return None
    
    return selected_edges

# Pre-compute all simple cycles of length 3 to 7
# We represent a cycle as a tuple of vertices in order, normalized so the smallest vertex is first
# and the second smallest is also fixed to break rotational symmetry? 
# Actually, for edge checking, we just need the set of edges.
# For uniqueness in lists, we can use a canonical form.

def decompose_2regular_into_cycles(edge_set):
    """
    Decompose a 2-regular spanning subgraph into cycles.
    Returns a list of cycles, each as a list of vertices.
    """
    if not edge_set:
        return []
    
    # Build adjacency
    adj = {i: [] for i in range(N)}
    for e in edge_set:
        u, v = tuple(e)
        adj[u].append(v)
        adj[v].append(u)
    
    # Each vertex should have degree 2
    for i in range(N):
        if len(adj[i]) != 2:
            return None
    
    visited_edges = set()
    cycles = []
    
    for start in range(N):
        # Find an unvisited edge from start
        if all(frozenset([start, nb]) in visited_edges for nb in adj[start]):
            continue
        
        # Start a cycle from start
        path = [start]
        current = start
        prev = None
        
        while True:
            # Find next vertex
            next_v = None
            for nb in adj[current]:
                edge = frozenset([current, nb])
                if edge not in visited_edges:
                    next_v = nb
                    break
            
            if next_v is None:
                break
            
            visited_edges.add(frozenset([current, next_v]))
            path.append(next_v)
            prev = current
            current = next_v
            
            # Check if we've returned to start
            if current == start:
                break
            # Check if we've completed a cycle (should only happen at start)
            if len(path) > N:
                break
        
        if len(path) >= 3 and path[0] == start:
            # Check if last vertex connects to start
            if start in adj[path[-1]]:
                cycles.append(path)
    
    return cycles if cycles else None

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s margin
    
    best_count = 0
    best_cycles = []
    
    # Strategy: Use flow-based approach to find 2-regular spanning subgraphs
    # Then decompose into cycles, keeping only Hamiltonian cycles (length N)
    
    def try_flow_based_packing(available_edges, max_cycles=3):
        """
        Try to find edge-disjoint Hamiltonian cycles using flow-based approach.
        Returns a list of Hamiltonian cycles (as vertex lists) or None.
        """
        cycles_found = []
        current_available = set(available_edges)
        
        for _ in range(max_cycles):
            if time.time() > deadline:
                break
            
            # Find a 2-regular spanning subgraph
            subgraph = find_2regular_spanning_subgraph(current_available)
            if subgraph is None:
                break
            
            # Decompose into cycles
            cycles = decompose_2regular_into_cycles(subgraph)
            if cycles is None:
                break
            
            # Check if any cycle is Hamiltonian (length N)
            ham_cycle = None
            for c in cycles:
                if len(c) == N:
                    ham_cycle = c
                    break
            
            if ham_cycle is None:
                # No Hamiltonian cycle found in this decomposition
                # Try a different subgraph? For now, break
                break
            
            cycles_found.append(ham_cycle)
            
            # Remove edges of this Hamiltonian cycle from available
            for i in range(N):
                u, v = ham_cycle[i], ham_cycle[(i+1) % N]
                e = frozenset([u, v])
                if e in current_available:
                    current_available.remove(e)
            
            if not current_available:
                break
        
        return cycles_found if cycles_found else None
    
    # First, try the flow-based approach
    result = try_flow_based_packing(EDGE_SET, max_cycles=3)
    
    if result:
        best_count = len(result)
        best_cycles = result
    
    # If flow-based didn't find 3, try randomized DFS as fallback
    if best_count < 3:
        def find_hamiltonian_cycle_dfs(available_edges_set, attempts=50):
            """
            Find a Hamiltonian cycle using randomized DFS with Warnsdorff's heuristic.
            """
            adj = {i: set() for i in range(N)}
            for e in available_edges_set:
                u, v = tuple(e)
                adj[u].add(v)
                adj[v].add(u)
            
            for i in range(N):
                if len(adj[i]) < 2:
                    return None
            
            for attempt in range(attempts):
                if time.time() > deadline:
                    return None
                
                start = 0
                path = [start]
                visited = {start}
                
                def warnsdorff_order(current):
                    def score(neighbor):
                        count = 0
                        for nb in adj[neighbor]:
                            if nb not in visited and nb != current:
                                count += 1
                        return count
                    candidates = [n for n in adj[current] if n not in visited]
                    if not candidates:
                        return []
                    candidates.sort(key=lambda x: (score(x), x))
                    return candidates
                
                def dfs(current):
                    if len(path) == N:
                        if start in adj[current]:
                            return True
                        return False
                    
                    neighbors = warnsdorff_order(current)
                    
                    if neighbors and random.random() < 0.15:
                        neighbors = neighbors[:]
                        random.shuffle(neighbors)
                    
                    for neighbor in neighbors:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            path.append(neighbor)
                            if dfs(neighbor):
                                return True
                            path.pop()
                            visited.remove(neighbor)
                    return False
                
                if dfs(start):
                    return path
                path = [start]
                visited = {start}
            
            return None
        
        # Try to find additional Hamiltonian cycles
        used_edges = set()
        for cycle in best_cycles:
            for i in range(N):
                u, v = cycle[i], cycle[(i+1) % N]
                used_edges.add(frozenset([u, v]))
        
        available = set(EDGE_SET - used_edges)
        
        for _ in range(3 - best_count):
            if time.time() > deadline:
                break
            
            h_cycle = find_hamiltonian_cycle_dfs(available, attempts=100)
            if h_cycle is None:
                break
            
            best_cycles.append(h_cycle)
            best_count += 1
            
            for i in range(N):
                u, v = h_cycle[i], h_cycle[(i+1) % N]
                e = frozenset([u, v])
                available.remove(e)
            
            if best_count >= 3:
                break
    
    # Validate and return
    if not best_cycles:
        return [list(range(N))]
    
    # Verify all cycles are valid
    final_result = []
    seen_edges = set()
    for cycle in best_cycles:
        if len(cycle) != N:
            continue
        if len(set(cycle)) != N:
            continue
        
        valid = True
        for i in range(N):
            u, v = cycle[i], cycle[(i+1) % N]
            e = frozenset([u, v])
            if e not in EDGE_SET:
                valid = False
                break
            if e in seen_edges:
                valid = False
                break
            seen_edges.add(e)
        
        if valid:
            final_result.append(cycle)
    
    if not final_result:
        return [list(range(N))]
    
    return final_result

# EVOLVE-BLOCK-END
