"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _edge_set(cycle):
    """Get the set of undirected edges in a cycle."""
    edges = set()
    for i in range(len(cycle)):
        u = cycle[i]
        v = cycle[(i + 1) % len(cycle)]
        edges.add((min(u, v), max(u, v)))
    return edges

def _is_valid_cycle(cycle, adj, n):
    """Check if cycle is a valid simple cycle using only edges in adj."""
    if len(cycle) != n:
        return False
    if len(set(cycle)) != n:
        return False
    for i in range(n):
        u = cycle[i]
        v = cycle[(i + 1) % n]
        if v not in adj[u]:
            return False
    return True

def _find_ham_cycle(n, adj, start, deadline, max_nodes=200000):
    """Find a Hamiltonian cycle starting at 'start' using Warnsdorff's heuristic with backtracking."""
    path = [start]
    visited = [False] * n
    visited[start] = True
    nodes_explored = [0]
    
    def dfs():
        if time.time() > deadline:
            return False
        nodes_explored[0] += 1
        if nodes_explored[0] > max_nodes:
            return False
        if len(path) == n:
            if start in adj[path[-1]]:
                return True
            return False
        
        cur = path[-1]
        neighbors = [v for v in adj[cur] if not visited[v]]
        if not neighbors:
            return False
        
        # Warnsdorff: sort by number of unvisited neighbors (ascending)
        neighbors.sort(key=lambda v: sum(1 for w in adj[v] if not visited[w]))
        
        for v in neighbors:
            path.append(v)
            visited[v] = True
            if dfs():
                return True
            path.pop()
            visited[v] = False
        
        return False
    
    if dfs():
        return list(path)
    return None

def _shift_cycle(cycle, shift, n):
    """Create a cyclic shift of the cycle by 'shift' positions in vertex labels."""
    # The cycle visits vertices in order cycle[0], cycle[1], ..., cycle[n-1]
    # A "shift" means we relabel: vertex cycle[i] becomes cycle[(i+shift) % n]
    # Actually, for edge-disjointness via symmetry, we want to rotate the vertex labels
    # If the graph has rotational symmetry (vertex i -> i+1 mod n), then shifting
    # the vertex labels in the cycle gives another valid cycle.
    # Let's define: new_cycle[i] = (cycle[i] + shift) % n
    # This works if the graph is invariant under vertex relabeling i -> i+shift mod n
    shifted = [(v + shift) % n for v in cycle]
    return shifted

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin for 90s limit
    n = P['n']
    base_adj = _build_adj(n, P['edges'])
    
    # Step 1: Find one Hamiltonian cycle
    ham1 = None
    for s in range(n):
        if time.time() > deadline:
            break
        ham1 = _find_ham_cycle(n, base_adj, s, deadline)
        if ham1 is not None:
            break
    
    if ham1 is None:
        # Fallback: try a simple cycle
        return [[0, 1, 2]]
    
    # Step 2: Generate shifted versions and check edge-disjointness
    # The graph appears to have a "band" structure. Let's try all shifts 0..n-1
    # and collect valid, edge-disjoint cycles.
    
    candidate_cycles = []
    used_edges = set()
    
    for shift in range(n):
        if time.time() > deadline:
            break
        shifted = _shift_cycle(ham1, shift, n)
        # Check if shifted is a valid cycle in the original graph
        if not _is_valid_cycle(shifted, base_adj, n):
            continue
        edges = _edge_set(shifted)
        # Check edge-disjointness with already selected cycles
        if edges & used_edges:
            continue
        candidate_cycles.append(shifted)
        used_edges |= edges
        if len(candidate_cycles) >= 10:
            break
    
    # If we got 3+ cycles from shifts, we're done
    if len(candidate_cycles) >= 3:
        result = []
        for c in candidate_cycles[:10]:
            result.append([int(v) for v in c])
        return result
    
    # Step 3: If shifts didn't give us enough, use DFS on the remaining graph
    # to find additional edge-disjoint Hamiltonian cycles
    adj = [set(s) for s in base_adj]
    for c in candidate_cycles:
        for i in range(len(c)):
            u = c[i]
            v = c[(i + 1) % len(c)]
            adj[u].discard(v)
            adj[v].discard(u)
    
    # Try to find more cycles in the residual graph
    for _ in range(20):
        if time.time() > deadline:
            break
        if len(candidate_cycles) >= 10:
            break
        
        found = False
        # Try different starting vertices
        starts = list(range(n))
        starts.sort(key=lambda v: -len(adj[v]))
        
        for s in starts:
            if time.time() > deadline:
                break
            ham = _find_ham_cycle(n, adj, s, deadline)
            if ham is not None:
                # Remove edges
                for i in range(len(ham)):
                    u = ham[i]
                    v = ham[(i + 1) % len(ham)]
                    adj[u].discard(v)
                    adj[v].discard(u)
                candidate_cycles.append(ham)
                found = True
                break
        
        if not found:
            break
    
    # Ensure we return at least 1 cycle
    if not candidate_cycles:
        candidate_cycles = [ham1]
    
    # Convert to lists of Python ints
    result = []
    for c in candidate_cycles[:10]:
        result.append([int(v) for v in c])
    
    return result
# EVOLVE-BLOCK-END
