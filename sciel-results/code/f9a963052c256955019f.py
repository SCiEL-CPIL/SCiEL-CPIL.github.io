import time
import random
from collections import defaultdict, deque

def construct():
    deadline = time.time() + 135
    results = []
    
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline - 5:
            # Emergency fallback: trivial decomposition (each edge as a path of length 1)
            edges = _instance(*spec)
            results.append([list(e) for e in edges])
            continue
            
        edges = _instance(*spec)
        paths = _solve_instance(edges, deadline)
        results.append(paths)
        
    return results

def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _solve_instance(edges, deadline):
    if not edges:
        return []
        
    # Build graph structure
    adj = defaultdict(set)
    degree = defaultdict(int)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        degree[u] += 1
        degree[v] += 1
        
    odd_vertices = [v for v in adj if degree[v] % 2 == 1]
    k = len(odd_vertices) // 2
    
    # Lower bound on number of paths is max(1, k) if edges exist
    # Gallai's conjecture: for connected graph, <= floor(n/2) paths
    # But we want to minimize. The theoretical lower bound is ceil(odd/2).
    
    # Strategy:
    # 1. If k == 0 (Eulerian), we need at least 1 path if edges exist. 
    #    Actually, an Eulerian graph can be decomposed into 1 cycle.
    #    But a cycle is not a simple path. 
    #    However, if we allow "path" to mean a trail? No, "simple path" means distinct vertices.
    #    A cycle v1-v2-...-vk-v1 is not a simple path because v1 appears twice.
    #    So an Eulerian graph must be broken into paths.
    #    Min paths for Eulerian graph? 
    #    If the graph is a single cycle, we can take v1-v2-...-vk as a path (k-1 edges) and vk-v1 as a path (1 edge). Total 2 paths.
    #    Can we do better? No, because we can't have a cycle as a simple path.
    #    Wait, if the graph is a single edge? No, that's odd degree.
    #    So if k==0, we target 2 paths? Or maybe more if structure is complex.
    #    Actually, any Eulerian graph can be decomposed into simple paths.
    #    The minimum number is related to the cycle space.
    #    Let's just use the general algorithm with k=0? 
    #    If k=0, we can add 0 virtual edges. The Eulerian circuit is a cycle.
    #    We can't represent a cycle as a simple path.
    #    So we must break it.
    #    My _solve_with_pairs expects pairs. If k=0, pairs is empty.
    #    Then it finds Eulerian circuit. It splits by virtual edges (none).
    #    So it returns the whole circuit as one path.
    #    BUT the circuit is a cycle! v0...v_{m-1}v0.
    #    The list of vertices will have v0 at start and end.
    #    Is that a simple path? No.
    #    So I need to handle k=0 specially.
    #    If k=0, I can pick any edge (u,v), remove it, making u and v odd.
    #    Then I have k=1. I pair (u,v).
    #    This forces the decomposition to use that edge as a "virtual" edge in the pairing sense?
    #    No, the virtual edges are added to the graph for the Eulerian circuit.
    #    If I add (u,v) as virtual, and it's already in the graph, I have a multigraph?
    #    No, I should just pick a pairing that works.
    #    Actually, if k=0, I can just pick any two vertices u,v, treat them as "odd" by adding a virtual edge (u,v).
    #    Then find Eulerian circuit in G + (u,v).
    #    Split by (u,v). This gives 2 paths.
    #    One path is the "rest" of the cycle, the other is just the edge? No.
    #    Let's trace: G is a cycle 1-2-3-1. Add virtual (1,2).
    #    Graph has edges (1,2),(2,3),(3,1) and virtual (1,2).
    #    Degrees: 1:3, 2:3, 3:2.
    #    Eulerian circuit: 1-2-3-1-2.
    #    Split by virtual (1,2).
    #    Path 1: 1-2 (virtual). Just vertices [1,2]. Edge (1,2) is real? Yes.
    #    Path 2: 2-3-1-2. Vertices [2,3,1,2]. Not simple (2 repeated).
    #    This approach is tricky for k=0.
    
    # Better approach for k=0:
    # Find a Hamiltonian path? No.
    # Just use the greedy method which handles it naturally?
    # Or, force k=1 by picking an edge, removing it, solving for k=1, then merging?
    
    # Let's try a unified approach:
    # We want to find a set of pairs (matching on odd vertices, or arbitrary if k=0)
    # such that the resulting paths are simple.
    
    best_paths = None
    best_count = float('inf')
    
    # If k == 0, we can try to find a decomposition into 2 paths?
    # Or just use greedy which will likely produce few paths.
    
    # Let's use the pairing method.
    # If k == 0, we can pick any vertex u and its neighbor v, and treat (u,v) as the "pair"
    # by adding a virtual edge? No, that changes the graph.
    
    # Alternative:
    # If k > 0, try random pairings of odd vertices.
    # If k == 0, try to find a long path covering most edges, then handle the rest.
    # Actually, if k==0, the graph is Eulerian.
    # We can decompose it into cycles. Each cycle can be turned into 2 paths?
    # Or we can just use the greedy algorithm which is robust.
    
    # Let's stick to the pairing method for k >= 1.
    # For k == 0, we will use a specific heuristic:
    # Pick an edge (u,v). Remove it. Now u,v are odd.
    # Solve for k=1 with pair (u,v).
    # This will produce 1 path? No, k=1 means 1 pair.
    # The Eulerian circuit of G+(u,v) split by (u,v) gives 2 paths.
    # One path will contain the edge (u,v) as its first/last step?
    # Let's assume the pairing method works if we can handle the k=0 case by forcing a pair.
    
    if k == 0:
        # Pick an arbitrary edge to "break" the Eulerian property
        # We add a virtual edge between two vertices to make them odd?
        # No, we just pick a pair (u,v) from existing edges?
        # Let's pick a random edge (u,v) and treat it as the "virtual" edge for splitting.
        # We don't actually add it to the graph, we just use it to split the Eulerian circuit.
        # But if we don't add it, the circuit is a cycle.
        # If we split a cycle by an edge (u,v) that is IN the cycle, we get 2 paths.
        # Path 1: u...v
        # Path 2: v...u
        # Both are simple if the cycle is simple?
        # A cycle is a simple path if we break it?
        # Yes, breaking a simple cycle at any edge yields two simple paths (one being the edge itself? No).
        # Cycle 1-2-3-4-1. Break at (1,2).
        # Path 1: 1-2.
        # Path 2: 2-3-4-1.
        # Both simple.
        # So for k=0, we can just find an Eulerian circuit (which is a cycle if connected)
        # and break it at any edge.
        # But what if the Eulerian circuit is not a simple cycle?
        # It's a closed trail. It might repeat vertices.
        # E.g. Figure 8.
        # Then breaking it might not yield simple paths.
        # So for k=0, we might need more than 2 paths.
        # Let's just use the greedy method for k=0 as a fallback, or try to find a simple cycle decomposition.
        
        # For now, let's try the pairing method with a dummy pair?
        # If I pick (u,v) and treat it as virtual, I must add it to the graph for the Eulerian circuit.
        # If (u,v) is already an edge, I have a multigraph.
        # Let's just use _greedy_decompose for k=0 to be safe, as it's robust.
        # Or, try to find a Hamiltonian path?
        
        # Actually, let's just use the general solver which handles k>=1.
        # For k=0, I will pick a random edge (u,v), remove it from the set of edges to be covered by the "pairing logic"
        # and instead treat (u,v) as a fixed path of length 1?
        # No, that's messy.
        
        # Let's use _greedy_decompose for k=0.
        greedy_paths = _greedy_decompose(edges, deadline)
        if greedy_paths:
            return greedy_paths
        return []

    # For k >= 1
    # Try multiple random pairings
    # We want to minimize the number of paths, which is fixed at k for this method.
    # So we just need to find ANY valid simple path decomposition with k paths.
    
    attempts = 50
    rng = random.Random(42)
    
    for _ in range(attempts):
        if time.time() > deadline - 5:
            break
            
        perm = odd_vertices[:]
        rng.shuffle(perm)
        pairs = [(perm[i], perm[i+1]) for i in range(0, len(perm), 2)]
        
        paths = _solve_with_pairs(edges, pairs, deadline)
        if paths is not None:
            # Verify simplicity and coverage
            if _verify(paths, edges):
                return paths
                
    # Fallback to greedy
    return _greedy_decompose(edges, deadline)

def _verify(paths, edges):
    # Check if paths are simple, use edges, and cover all edges exactly once
    covered = set()
    for path in paths:
        if len(path) < 2:
            # Single vertex path? Invalid for edge cover unless no edges.
            # Task says "lists of paths (vertex lists)".
            # Usually a path has at least 1 edge.
            if path:
                return False
            continue
        # Check simple
        if len(set(path)) != len(path):
            return False
        # Check edges
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            e = tuple(sorted((u, v)))
            if e in covered:
                return False
            if e not in set(tuple(sorted(e)) for e in edges):
                return False
            covered.add(e)
            
    if covered != set(tuple(sorted(e)) for e in edges):
        return False
    return True

def _solve_with_pairs(edges, pairs, deadline):
    # Add virtual edges
    full_edges = set(tuple(sorted(e)) for e in edges)
    virtual_edges = set(tuple(sorted(e)) for e in pairs)
    full_edges.update(virtual_edges)
    
    # Find Eulerian Circuit
    circuit = _find_eulerian_circuit(full_edges)
    if circuit is None:
        return None
        
    # Split circuit by virtual edges
    paths = []
    current_path = []
    
    # Circuit is [v0, v1, ..., vk]
    # We iterate over edges (v_i, v_{i+1})
    for i in range(len(circuit) - 1):
        u, v = circuit[i], circuit[i+1]
        edge = tuple(sorted((u, v)))
        
        if not current_path:
            current_path.append(u)
            
        current_path.append(v)
        
        if edge in virtual_edges:
            # End of path
            if len(set(current_path)) != len(current_path):
                return None
            paths.append(current_path)
            current_path = []
            
    if current_path:
        if len(set(current_path)) != len(current_path):
            return None
        paths.append(current_path)
        
    if len(paths) != len(pairs):
        return None
        
    return paths

def _find_eulerian_circuit(edges):
    if not edges:
        return []
        
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    visited_edges = set()
    start = next(iter(adj))
    
    stack = [start]
    circuit = []
    
    while stack:
        u = stack[-1]
        if adj[u]:
            v = adj[u].pop()
            edge = tuple(sorted((u, v)))
            if edge not in visited_edges:
                visited_edges.add(edge)
                stack.append(v)
        else:
            circuit.append(stack.pop())
            
    return circuit[::-1]

def _greedy_decompose(edges, deadline):
    remaining = set(tuple(sorted(e)) for e in edges)
    adj = defaultdict(set)
    degree = defaultdict(int)
    for u, v in remaining:
        adj[u].add(v)
        adj[v].add(u)
        degree[u] += 1
        degree[v] += 1
        
    paths = []
    rng = random.Random(123)
    
    while remaining:
        if time.time() > deadline - 2:
            # Emergency: add remaining edges as single-edge paths
            for e in remaining:
                paths.append(list(e))
            break
            
        # Pick start vertex
        # Prefer odd degree vertices to help with future simpleness?
        # Or just pick any vertex with degree > 0
        candidates = [v for v in adj if degree[v] > 0]
        if not candidates:
            break
            
        # Heuristic: pick vertex with highest degree? Or random?
        # Let's pick random to diversify
        start = rng.choice(candidates)
        
        path = [start]
        visited = {start}
        current = start
        
        while True:
            # Find next vertex
            neighbors = [n for n in adj[current] if n not in visited]
            if not neighbors:
                break
                
            # Warnsdorff's rule: pick neighbor with fewest unvisited neighbors
            neighbors.sort(key=lambda x: sum(1 for y in adj[x] if y not in visited))
            next_v = neighbors[0]
            
            path.append(next_v)
            visited.add(next_v)
            
            # Remove edge
            edge = tuple(sorted((current, next_v)))
            remaining.discard(edge)
            adj[current].discard(next_v)
            adj[next_v].discard(current)
            degree[current] -= 1
            degree[next_v] -= 1
            
            if not adj[current]:
                del adj[current]
            if not adj[next_v]:
                del adj[next_v]
                
            current = next_v
            
        paths.append(path)
        
    return paths
