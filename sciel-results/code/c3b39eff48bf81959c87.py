import time
from random import Random
from collections import deque
import sys

# Check for scipy availability
try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import csr_matrix
    HAS_SCIPY_OPT = True
except ImportError:
    HAS_SCIPY_OPT = False

def instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def decompose_graph_ip(edges, n, time_limit=10):
    """
    Decompose edges into cycles and single edges using Integer Programming.
    Objective: Maximize the number of edges covered by cycles.
    Constraint: For every vertex, the number of incident edges selected for cycles
    must be even (0, 2, 4, ...). This ensures the subgraph decomposes into cycles.
    """
    if not HAS_SCIPY_OPT:
        # Fallback to simple greedy if no scipy
        return decompose_graph_greedy(edges, n)

    m = len(edges)
    if m == 0:
        return []
    
    # Map edges to indices
    edge_to_idx = {e: i for i, e in enumerate(edges)}
    idx_to_edge = {i: e for i, e in enumerate(edges)}
    
    # Build incidence matrix A (n x m)
    # A[u, i] = 1 if vertex u is incident to edge i
    rows = []
    cols = []
    data = []
    
    for i, (u, v) in enumerate(edges):
        rows.append(u)
        cols.append(i)
        data.append(1)
        rows.append(v)
        cols.append(i)
        data.append(1)
        
    A = csr_matrix((data, (rows, cols)), shape=(n, m))
    
    # Objective: maximize sum(x_i) -> minimize -sum(x_i)
    c = -1.0 * ones(m)
    
    # Constraints:
    # 1. 0 <= x_i <= 1 (integrality handled by milp)
    # 2. A @ x is even for all vertices.
    #    This is a parity constraint, which is non-linear for standard LP.
    #    However, we can relax this. 
    #    Actually, the condition "subgraph decomposes into cycles" is equivalent to:
    #    1. Every vertex has even degree in the subgraph.
    #    2. No isolated vertices? No, just even degree.
    #    
    #    Standard MILP solvers don't handle parity directly.
    #    Alternative approach:
    #    We can iterate over possible degrees? No.
    #    
    #    Let's look at the structure.
    #    If we just maximize edges subject to even degree, we get a 2-factor (or union of cycles).
    #    Is there a way to linearize parity?
    #    x_u = sum_{e incident to u} x_e. We want x_u % 2 == 0.
    #    
    #    This specific problem (Max Even Subgraph) is solvable via matching or flow, 
    #    but since we want to use IP, let's try a different formulation or a heuristic IP.
    #    
    #    Actually, for small n (20-30), we can use a different strategy:
    #    The "even degree" constraint is the only topological constraint for being a union of cycles.
    #    We can solve the Max Even Subgraph problem.
    #    
    #    Wait, scipy.milp requires linear constraints. Parity is not linear.
    #    
    #    Alternative Strategy:
    #    Use the fact that the complement of a 2-factor is a set of edges that can be removed.
    #    Or, use a heuristic: 
    #    1. Find a maximum spanning subgraph where all degrees are even.
    #    
    #    Let's switch to a different structural approach if IP parity is hard.
    #    
    #    NEW STRATEGY:
    #    1. Compute a 2-factor (max cycles) using a matching-based heuristic or flow if available.
    #    Since scipy doesn't have a direct max-flow-min-cost that is easy to call for this specific 
    #    "b-matching" without constructing a large network, let's use a simpler, robust heuristic 
    #    that is structurally different from "greedy longest cycle".
    #    
    #    Let's use **Local Search / Simulated Annealing** on the set of edges included in cycles.
    #    State: Binary vector x of length m.
    #    Fitness: Number of edges in x, subject to all degrees being even.
    #    If a degree is odd, penalty.
    #    
    #    This is a structural change: Global Optimization via Local Search on the Edge Set.
    
    pass

def decompose_graph_greedy(edges, n):
    """Fallback greedy method if IP fails or is not used."""
    remaining = set(tuple(sorted(e)) for e in edges)
    pieces = []
    adj = {u: set() for u in range(n)}
    for u, v in remaining:
        adj[u].add(v)
        adj[v].add(u)
        
    while remaining:
        start = None
        for u in range(n):
            if len(adj[u]) >= 2:
                start = u
                break
        if start is None:
            break
            
        # Find any cycle
        visited = {start}
        path = [start]
        stack = [(start, iter(adj[start]))]
        found_cycle = None
        
        while stack:
            node, neighbors = stack[-1]
            advanced = False
            for nb in neighbors:
                e = (node, nb) if node < nb else (nb, node)
                if e not in remaining:
                    continue
                if nb == start and len(path) >= 3:
                    found_cycle = path[:]
                    break
                if nb not in visited:
                    visited.add(nb)
                    path.append(nb)
                    stack.append((nb, iter(adj[nb])))
                    advanced = True
                    break
            if found_cycle:
                break
            if not advanced:
                stack.pop()
                if path:
                    visited.discard(path.pop())
        
        if found_cycle:
            pieces.append(found_cycle)
            for i in range(len(found_cycle)):
                u = found_cycle[i]
                v = found_cycle[(i+1)%len(found_cycle)]
                e = (u, v) if u < v else (v, u)
                remaining.discard(e)
                adj[u].discard(v)
                adj[v].discard(u)
        else:
            break
            
    for e in remaining:
        pieces.append([e[0], e[1]])
    return pieces

def solve_max_even_subgraph(edges, n, time_limit=10):
    """
    Finds a subgraph of edges where every vertex has even degree, maximizing the number of edges.
    Uses Simulated Annealing / Local Search.
    """
    m = len(edges)
    if m == 0:
        return set()
    
    # Initialize with empty set or random?
    # Start with all edges, then remove to fix parity?
    # Or start with empty, add edges?
    
    # Let's start with a random feasible solution or empty.
    # Empty is feasible (0 is even).
    x = [0] * m
    deg = [0] * n
    
    # Precompute incident edges for each vertex
    inc = {u: [] for u in range(n)}
    for i, (u, v) in enumerate(edges):
        inc[u].append(i)
        inc[v].append(i)
        
    def flip(i):
        nonlocal x, deg
        u, v = edges[i]
        if x[i]:
            x[i] = 0
            deg[u] -= 1
            deg[v] -= 1
        else:
            x[i] = 1
            deg[u] += 1
            deg[v] += 1
            
    def cost():
        # We want to maximize sum(x) subject to deg[u] % 2 == 0.
        # Penalty for odd degrees.
        odd_count = sum(1 for d in deg if d % 2 != 0)
        return -sum(x) + 100 * odd_count # Minimize this
        
    # Simulated Annealing
    import random
    rng = random.Random(42)
    
    # Initial state: Try to build a large subgraph.
    # Greedy initialization: Add edges randomly, if it creates odd parity, keep it but remember to fix later?
    # Better: Start with empty.
    
    current_cost = cost()
    best_x = x[:]
    best_cost = current_cost
    
    T = 10.0
    T_min = 0.01
    alpha = 0.99
    
    start_time = time.time()
    
    while T > T_min and time.time() - start_time < time_limit:
        # Pick a random edge to flip
        i = rng.randint(0, m - 1)
        flip(i)
        new_cost = cost()
        
        delta = new_cost - current_cost
        if delta < 0 or rng.random() < pow(2.718, -delta / T):
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_x = x[:]
        else:
            flip(i) # Revert
            
        T *= alpha
        
    # Restore best solution
    x = best_x[:]
    # Recalculate deg for best_x
    deg = [0] * n
    for i in range(m):
        if x[i]:
            u, v = edges[i]
            deg[u] += 1
            deg[v] += 1
            
    # The selected edges form a union of cycles.
    # We need to extract the cycles.
    selected_edges = [edges[i] for i in range(m) if x[i]]
    return set(selected_edges)

def extract_cycles_from_subgraph(selected_edges, n):
    """
    Given a set of edges where every vertex has even degree,
    decompose it into disjoint cycles.
    """
    if not selected_edges:
        return []
        
    adj = {u: set() for u in range(n)}
    for u, v in selected_edges:
        adj[u].add(v)
        adj[v].add(u)
        
    cycles = []
    remaining_edges = set(selected_edges)
    
    while remaining_edges:
        # Pick any edge
        u, v = next(iter(remaining_edges))
        # Walk to form a cycle
        path = [u, v]
        remaining_edges.discard((u, v) if u < v else (v, u))
        current = v
        prev = u
        
        while current != u:
            # Find next neighbor in remaining edges
            next_node = None
            for nb in adj[current]:
                e = (current, nb) if current < nb else (nb, current)
                if e in remaining_edges:
                    next_node = nb
                    remaining_edges.discard(e)
                    break
            if next_node is None:
                # Should not happen if degrees are even and we started with an edge
                # But if the subgraph is not connected properly?
                # In a 2-regular graph, it's a cycle.
                # If degrees > 2, we might have intersecting cycles.
                # The walk might get stuck if we don't manage the "remaining" set correctly.
                # Actually, if we just remove edges as we go, and all degrees are even,
                # we will eventually return to the start?
                # Yes, because we are traversing a path in a graph where all vertices have even degree.
                # Wait, if a vertex has degree 4, we might enter and leave, but we removed one edge.
                # The "remaining" graph might not have even degrees anymore.
                # 
                # Correct approach for extracting cycles from an even subgraph:
                # 1. While edges remain:
                #    a. Find any vertex with degree > 0 in the CURRENT remaining subgraph.
                #    b. Walk along edges, removing them as we go, until we return to start.
                #    This works because the subgraph is a union of edge-disjoint cycles.
                #    Removing a cycle leaves a graph that still has even degrees.
                
                # Let's restart the walk logic more carefully.
                break
        
        if next_node is not None or current == u:
             # This logic is getting messy. Let's use a standard cycle extraction.
             pass
             
    # Let's rewrite extraction properly
    cycles = []
    remaining_edges = set(selected_edges)
    adj = {u: set() for u in range(n)}
    for u, v in remaining_edges:
        adj[u].add(v)
        adj[v].add(u)
        
    while remaining_edges:
        # Find a start vertex
        start = None
        for u in range(n):
            if adj[u]:
                start = u
                break
        if start is None:
            break
            
        # Walk
        path = [start]
        prev = None
        curr = start
        while True:
            # Find an edge (curr, next) in remaining
            next_node = None
            for nb in list(adj[curr]):
                e = (curr, nb) if curr < nb else (nb, curr)
                if e in remaining_edges:
                    next_node = nb
                    break
                    
            if next_node is None:
                break # Should not happen if connected component is a cycle union
                
            # Remove edge
            e = (curr, next_node) if curr < next_node else (next_node, curr)
            remaining_edges.discard(e)
            adj[curr].discard(next_node)
            adj[next_node].discard(curr)
            
            path.append(next_node)
            prev = curr
            curr = next_node
            
            if curr == start:
                break
                
        if len(path) >= 3:
            cycles.append(path)
            
    return cycles

def decompose_graph(edges, n, time_limit=15):
    if not edges:
        return []
        
    # 1. Find max even subgraph
    selected_edges = solve_max_even_subgraph(edges, n, time_limit)
    
    # 2. Extract cycles from selected edges
    cycles = extract_cycles_from_subgraph(selected_edges, n)
    
    # 3. Remaining edges are single edges
    remaining_edges = set(tuple(sorted(e)) for e in edges) - set(tuple(sorted(e)) for e in selected_edges)
    
    pieces = cycles
    for e in remaining_edges:
        pieces.append([e[0], e[1]])
        
    return pieces

def construct():
    start_time = time.time()
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    
    for n, p, seed in specs:
        edges = instance(n, p, seed)
        
        # Time budget per graph: 40s total / 3 graphs = ~13s each.
        # Leave safety margin.
        time_remaining = 140 - (time.time() - start_time)
        if time_remaining < 5:
            # Fallback to all single edges
            results.append([list(ed) for ed in edges])
            continue
            
        t_limit = max(1, min(10, time_remaining / 3))
        
        pieces = decompose_graph(edges, n, t_limit)
        results.append(pieces)
        
    return results[:3]

if __name__ == "__main__":
    construct()
