import time
import random
import sys
from collections import defaultdict
import heapq

def _instance(n, p, seed):
    """Exact copy of the verifier's instance generator."""
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _build_graph(edges):
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _find_eulerian_circuit(adj, edges_set):
    """
    Finds an Eulerian circuit in an Eulerian graph.
    Returns a list of edges in order.
    """
    if not edges_set:
        return []
    
    # Hierholzer's algorithm
    stack = [next(iter(edges_set))[0]] # Start at any vertex
    circuit = []
    used_edges = set()
    
    # We need to track remaining edges
    # Since it's Eulerian, we can just pick any unused edge from the current vertex
    
    # Optimization: Use a dict of sets for remaining edges
    rem_adj = defaultdict(set)
    for u, v in edges_set:
        rem_adj[u].add(v)
        rem_adj[v].add(u)
        
    start_v = next(iter(edges_set))[0]
    curr_v = start_v
    path = [start_v]
    
    while rem_adj:
        # Find next unused edge from curr_v
        if not rem_adj[curr_v]:
            # Backtrack
            if len(path) == 1:
                break # Should not happen if connected and Eulerian
            # Pop last vertex
            last_v = path.pop()
            curr_v = path[-1]
            continue
            
        # Pick an unused neighbor
        # To make it deterministic, pick min
        next_v = min(rem_adj[curr_v])
        
        # Remove edge
        rem_adj[curr_v].remove(next_v)
        rem_adj[next_v].remove(curr_v)
        
        path.append(next_v)
        curr_v = next_v
        
    # The path found is the circuit
    # But Hierholzer's usually builds the circuit by splicing.
    # The simple stack approach above might fail to visit all edges if it gets stuck in a sub-loop.
    # Let's use the standard splicing method.
    
    def hierholzer(start):
        stack = [start]
        circuit = []
        while stack:
            v = stack[-1]
            if rem_adj[v]:
                # Pick neighbor
                u = min(rem_adj[v])
                rem_adj[v].remove(u)
                rem_adj[u].remove(v)
                stack.append(u)
            else:
                circuit.append(stack.pop())
        return circuit[::-1]
        
    # Check if graph is connected (it should be)
    # If not, we'd need to handle components, but instance graphs are connected.
    
    # Re-initialize rem_adj for the actual run
    rem_adj = defaultdict(set)
    for u, v in edges_set:
        rem_adj[u].add(v)
        rem_adj[v].add(u)
        
    if not rem_adj:
        return []
        
    start_v = next(iter(rem_adj.keys()))
    # Ensure start_v has edges
    while not rem_adj[start_v]:
        start_v = next(iter(rem_adj.keys()))
        
    circuit = hierholzer(start_v)
    
    # Convert vertex sequence to edge sequence
    edges_seq = []
    for i in range(len(circuit) - 1):
        u, v = circuit[i], circuit[i+1]
        edges_seq.append((min(u,v), max(u,v)))
        
    return edges_seq

def _construct_via_eulerian(edges, deadline, rng):
    """
    Try to construct a path cover of minimal size using Eulerian augmentation.
    """
    adj = _build_graph(edges)
    deg = defaultdict(int)
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
        
    odd_verts = [v for v in deg if deg[v] % 2 != 0]
    
    if not odd_verts:
        # Eulerian graph
        # Target: 2 paths (or 1 if it's a single edge? No, single edge is not Eulerian)
        # If it's a cycle, 2 paths.
        # If it's a single vertex with self-loop? No simple graphs.
        # So target is 2.
        target_paths = 2
        # Add 2 "virtual" edges to make it non-Eulerian? No.
        # Just find Eulerian circuit and split into 2 paths.
        edges_seq = _find_eulerian_circuit(adj, set(map(tuple, edges)))
        if not edges_seq:
            return []
        
        # Split the circuit into 2 paths
        # Pick a random index to break
        # Actually, breaking a cycle at 1 point gives 1 path (which is not a cycle).
        # But a path must be simple.
        # If the Eulerian circuit is simple (a cycle), breaking it gives 1 path.
        # But we need to cover all edges.
        # If we break at 1 point, we get 1 path covering all edges.
        # Is that allowed?
        # "Paths must be simple".
        # A cycle broken at 1 point is a path.
        # So if the graph is a simple cycle, 1 path is sufficient.
        # But wait, if the graph is Eulerian but not a simple cycle (e.g., figure 8),
        # the Eulerian circuit revisits vertices.
        # Breaking it at 1 point gives a trail that revisits vertices -> not a simple path.
        # So we must split it into multiple simple paths.
        
        # For now, let's just return the greedy result if this gets complicated.
        # But we can try to decompose the circuit into simple paths.
        
        pass # Fall back to greedy for Eulerian case initially, or handle carefully.
        
    if len(odd_verts) == 2:
        target_paths = 1
        # Add 1 virtual edge between the two odd vertices
        u, v = odd_verts
        virtual_edge = (min(u,v), max(u,v))
        
        # Check if virtual edge exists in graph. If so, we can't "add" it in a way that distinguishes it?
        # Actually, if the edge exists, adding it makes degrees even.
        # Then we find Eulerian circuit in G + {virtual_edge}.
        # Remove virtual_edge from circuit.
        # We get 1 path.
        
        all_edges = set(map(tuple, edges))
        all_edges.add(virtual_edge)
        
        edges_seq = _find_eulerian_circuit(adj, all_edges)
        
        # Remove virtual edge
        # The circuit is a list of edges.
        # Find the virtual edge in the sequence.
        # Split the sequence at that edge.
        
        # Since it's a circuit, removing one edge leaves a path.
        # The path is the sequence of edges excluding the virtual one.
        # We need to order them correctly.
        
        # Find index of virtual_edge
        try:
            idx = edges_seq.index(virtual_edge)
        except ValueError:
            return [] # Should not happen
            
        # The path is edges_seq[idx+1:] + edges_seq[:idx]
        path_edges = edges_seq[idx+1:] + edges_seq[:idx]
        
        # Reconstruct vertex path
        if not path_edges:
            return [[odd_verts[0]]] # Degenerate case?
            
        v0 = odd_verts[0] if odd_verts[0] in [e[0] for e in path_edges] + [e[1] for e in path_edges] else odd_verts[1]
        # Actually, just trace from one end.
        # The ends of the path are the two odd vertices.
        
        # Simple check: Is it a simple path?
        # If yes, return [path].
        # If no, fall back to greedy.
        
        # For simplicity and safety, if we can't easily verify simplicity, use greedy.
        # But this construction is very strong.
        
        pass # Fall back to greedy for now to ensure correctness.

    # Fallback: Greedy
    return None

def _greedy_walk(adj, remaining, deg, start_v, rng=None, randomize=False, mode="low_deg"):
    path = [start_v]
    visited = {start_v}
    
    while True:
        v = path[-1]
        valid_neighbors = []
        for u in adj[v]:
            if u not in visited:
                e = (min(v, u), max(v, u))
                if e in remaining:
                    valid_neighbors.append(u)
        
        if not valid_neighbors:
            break
        
        if randomize and rng is not None:
            if mode == "high_deg":
                weights = [float(deg.get(u, 0) + 1) for u in valid_neighbors]
            elif mode == "low_deg":
                weights = [1.0 / (deg.get(u, 0) + 1) for u in valid_neighbors]
            else:
                weights = [1.0 + deg.get(u, 0) * 0.5 for u in valid_neighbors]
            
            total_w = sum(weights)
            if total_w == 0:
                next_v = valid_neighbors[0]
            else:
                r = rng.random() * total_w
                cum = 0
                next_v = valid_neighbors[-1]
                for i, u in enumerate(valid_neighbors):
                    cum += weights[i]
                    if r <= cum:
                        next_v = u
                        break
        else:
            if mode == "high_deg":
                next_v = max(valid_neighbors, key=lambda u: deg.get(u, 0))
            elif mode == "low_deg":
                next_v = min(valid_neighbors, key=lambda u: deg.get(u, 0))
            else:
                next_v = max(valid_neighbors, key=lambda u: (deg.get(u, 0), -u))
        
        e = (min(v, next_v), max(v, next_v))
        remaining.discard(e)
        deg[v] -= 1
        deg[next_v] -= 1
        path.append(next_v)
        visited.add(next_v)
        
    return path

def _path_cover_single(edges, deadline, rng=None, randomize=False, mode="low_deg", start_strategy="leaf", shuffle_adj=False):
    adj = _build_graph(edges)
    
    if shuffle_adj and rng is not None:
        for v in adj:
            rng.shuffle(adj[v])

    remaining = set(map(tuple, edges))
    paths = []
    
    deg = defaultdict(int)
    for e in remaining:
        deg[e[0]] += 1
        deg[e[1]] += 1
    
    while remaining and time.time() < deadline:
        start_candidates = set()
        for e in remaining:
            start_candidates.add(e[0])
            start_candidates.add(e[1])
        
        if start_strategy == "leaf":
            leaves = [v for v in start_candidates if deg[v] == 1]
            if leaves:
                if randomize and rng is not None:
                    start_v = rng.choice(leaves)
                else:
                    start_v = leaves[0]
            else:
                min_deg = min(deg[v] for v in start_candidates)
                candidates = [v for v in start_candidates if deg[v] == min_deg]
                if randomize and rng is not None:
                    start_v = rng.choice(candidates)
                else:
                    start_v = min(candidates)
        elif start_strategy == "max_deg":
            max_deg = max(deg[v] for v in start_candidates)
            candidates = [v for v in start_candidates if deg[v] == max_deg]
            if randomize and rng is not None:
                start_v = rng.choice(candidates)
            else:
                start_v = min(candidates)
        elif start_strategy == "min_deg":
            min_deg = min(deg[v] for v in start_candidates)
            candidates = [v for v in start_candidates if deg[v] == min_deg]
            if randomize and rng is not None:
                start_v = rng.choice(candidates)
            else:
                start_v = min(candidates)
        else:
            if randomize and rng is not None:
                start_v = rng.choice(list(start_candidates))
            else:
                start_v = min(start_candidates)
        
        path = _greedy_walk(adj, remaining, deg, start_v, rng, randomize, mode)
        paths.append(path)
    
    for e in remaining:
        paths.append(list(e))
        
    return paths

def _path_cover_with_restarts(edges, deadline):
    best_paths = None
    best_count = len(edges)
    start = time.time()
    
    # 1. Deterministic baseline
    paths = _path_cover_single(edges, deadline, randomize=False, mode="low_deg", start_strategy="leaf")
    best_paths = paths
    best_count = len(paths)
    
    # 2. Deterministic variant
    if time.time() < deadline - 10:
        paths2 = _path_cover_single(edges, deadline, randomize=False, mode="low_deg", start_strategy="min_deg")
        if len(paths2) < best_count:
            best_count = len(paths2)
            best_paths = paths2
            
    # 3. Randomized restarts
    rng = random.Random(42)
    modes = ["low_deg", "low_deg", "balanced", "high_deg"]
    start_strats = ["leaf", "min_deg", "random"]
    
    trial = 0
    while time.time() < deadline - 5:
        mode = modes[trial % len(modes)]
        strat = start_strats[trial % len(start_strats)]
        
        paths = _path_cover_single(edges, deadline - 5, rng, randomize=True, mode=mode, start_strategy=strat, shuffle_adj=True)
        
        if len(paths) < best_count:
            best_count = len(paths)
            best_paths = paths
            if best_count <= 10:
                break
        
        trial += 1
    
    return best_paths

def construct():
    total_deadline = time.time() + 145
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []
    
    all_edges = [_instance(n, p, seed) for n, p, seed in specs]
    total_edges = sum(len(e) for e in all_edges)
    
    for i, (n, p, seed) in enumerate(specs):
        edges = all_edges[i]
        frac = len(edges) / total_edges
        time_alloc = max(15.0, frac * 140.0)
        remaining_time = total_deadline - time.time()
        time_alloc = min(time_alloc, remaining_time - 2.0)
        if time_alloc < 5.0:
            time_alloc = remaining_time - 1.0
        
        graph_deadline = time.time() + time_alloc
        paths = _path_cover_with_restarts(edges, graph_deadline)
        results.append(paths)
    
    return results

if __name__ == "__main__":
    result = construct()
    for i, paths in enumerate(result):
        print(f"Graph {i}: {len(paths)} paths")
    print(f"Total: {sum(len(p) for p in result)}")
