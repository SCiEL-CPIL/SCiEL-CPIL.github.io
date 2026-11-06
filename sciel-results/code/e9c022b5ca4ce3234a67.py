import time
import random
from collections import defaultdict, deque
from random import Random

def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _build_adjacency(edges):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_biconnected_components(adj, edges_set):
    """
    Returns a list of biconnected components, where each component is a set of edges.
    Uses Tarjan's algorithm.
    """
    if not edges_set:
        return []
    
    # We need to run on the subgraph defined by edges_set
    # Build a local adjacency for just these edges
    local_adj = defaultdict(set)
    for u, v in edges_set:
        local_adj[u].add(v)
        local_adj[v].add(u)
    
    # Filter vertices that are actually present
    vertices = set()
    for u, v in edges_set:
        vertices.add(u)
        vertices.add(v)
    
    if not vertices:
        return []
    
    time_start = time.time()
    index_counter = [0]
    stack = []
    low = {}
    index = {}
    root = None
    result_components = []
    
    # Tarjan's algorithm for biconnected components
    def strongconnect(v):
        nonlocal root
        # Recursion count
        index[v] = index_counter[0]
        low[v] = index_counter[0]
        index_counter[0] += 1
        children = 0
        
        for w in local_adj[v]:
            if time.time() - time_start > 5: # Safety break
                return
            
            if w not in index:
                stack.append((v, w))
                children += 1
                strongconnect(w)
                low[v] = min(low[v], low[w])
                
                # (v, w) is a bridge in the biconnected component tree
                if low[w] >= index[v]:
                    comp = set()
                    while True:
                        x, y = stack.pop()
                        edge = (min(x, y), max(x, y))
                        comp.add(edge)
                        if x == v and y == w:
                            break
                    if comp:
                        result_components.append(comp)
            elif w != v: # (v, w) is a back edge
                low[v] = min(low[v], index[w])
                stack.append((v, w))
                
    # Run for all vertices
    for v in vertices:
        if v not in index:
            root = v
            strongconnect(v)
            # If root, pop any remaining edges in stack?
            # Actually, the condition low[w] >= index[v] handles the root's children.
            # If the graph is disconnected, we handle each component.
            
    return result_components

def _find_longest_cycle_randomized(adj, edges_set, num_trials=50):
    """
    Finds a long cycle in the graph defined by edges_set using randomized DFS.
    """
    if not edges_set:
        return None
    
    # Get vertices
    vertices = set()
    for u, v in edges_set:
        vertices.add(u)
        vertices.add(v)
        
    if not vertices:
        return None
    
    # Build local adjacency
    local_adj = defaultdict(set)
    for u, v in edges_set:
        local_adj[u].add(v)
        local_adj[v].add(u)
        
    best_cycle = None
    best_len = 0
    
    # Sort vertices by degree for better start candidates
    sorted_verts = sorted(vertices, key=lambda x: len(local_adj[x]), reverse=True)
    
    for _ in range(num_trials):
        # Pick a random start vertex, biased towards high degree
        if random.random() < 0.5:
            start = random.choice(sorted_verts[:max(1, len(sorted_verts)//3)])
        else:
            start = random.choice(sorted_verts)
            
        visited = {start}
        path = [start]
        
        def dfs(v):
            nonlocal best_cycle, best_len
            for neighbor in local_adj[v]:
                if neighbor == start and len(path) >= 3:
                    # Found a cycle
                    if len(path) > best_len:
                        best_len = len(path)
                        best_cycle = path[:]
                elif neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    dfs(neighbor)
                    path.pop()
                    visited.remove(neighbor)
                    
        dfs(start)
        
        # Early stop if we found a Hamiltonian cycle (len == num vertices)
        if best_len == len(vertices):
            break
            
    return best_cycle

def _decompose_graph(edges):
    if not edges:
        return []
    
    edges_set = set(tuple(sorted(e)) for e in edges)
    pieces = []
    
    deadline = time.time() + 10 # 10 seconds per graph
    
    while edges_set and time.time() < deadline:
        # Find biconnected components
        bccs = _find_biconnected_components({}, edges_set)
        
        # Filter out components that are just single edges or trees (no cycles)
        # A biconnected component with >= 3 edges must contain a cycle.
        # Actually, a biconnected component is 2-vertex-connected.
        # The smallest biconnected component is a triangle (3 edges).
        # So any bcc with edges >= 3 has a cycle.
        
        found_cycle = False
        best_cycle_in_bcc = None
        best_bcc = None
        
        for bcc in bccs:
            if len(bcc) < 3:
                continue
            
            # Try to find a cycle in this bcc
            # We pass the global adj, but we only care about edges in bcc
            # _find_longest_cycle_randomized needs adj and edges_set
            # We can just pass the full adj and the specific bcc edges
            cycle = _find_longest_cycle_randomized({}, bcc, num_trials=20)
            
            if cycle and (best_cycle_in_bcc is None or len(cycle) > len(best_cycle_in_bcc)):
                best_cycle_in_bcc = cycle
                best_bcc = bcc
                
        if best_cycle_in_bcc:
            # Add cycle to pieces
            pieces.append(best_cycle_in_bcc)
            
            # Remove edges of this cycle from edges_set
            for i in range(len(best_cycle_in_bcc)):
                u = best_cycle_in_bcc[i]
                v = best_cycle_in_bcc[(i + 1) % len(best_cycle_in_bcc)]
                edge = (min(u, v), max(u, v))
                edges_set.discard(edge)
                
            found_cycle = True
        else:
            # No more cycles can be found
            break
            
    # Add remaining edges as single edges
    for u, v in edges_set:
        pieces.append([u, v])
        
    return pieces

def construct():
    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    
    for n, p, seed in specs:
        edges = _instance(n, p, seed)
        pieces = _decompose_graph(edges)
        results.append(pieces)
        
    return results
