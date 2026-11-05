import time
import random
from random import Random

DEADLINE_MARGIN = 10  # seconds before 150s

def instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def build_adj(edges):
    adj = {}
    for (u, v) in edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    return adj

def find_2_factor(remaining, adj, rng, max_iter=20000):
    """
    Greedy randomized algorithm to find a 2-regular spanning subgraph (2-factor)
    on as many vertices as possible. A 2-factor is a union of edge-disjoint cycles.
    We build it by repeatedly selecting edges so that no vertex exceeds degree 2,
    prioritizing vertices with lower current degree and higher remaining connectivity.
    """
    if not remaining:
        return []
    
    # Track degree in the selected 2-factor
    deg = {}
    for e in remaining:
        for v in e:
            deg.setdefault(v, 0)
    
    selected_edges = set()
    
    # Work on a copy of adjacency for the remaining graph
    rem_adj = {v: set() for v in deg}
    for (u, v) in remaining:
        rem_adj[u].add(v)
        rem_adj[v].add(u)
    
    # Greedy: repeatedly pick an edge that connects two vertices of degree < 2,
    # preferring to close cycles or extend paths to maximize eventual cycle count.
    # To find a good 2-factor, we use a randomized greedy with backtracking-lite.
    
    # Strategy: 
    # 1. Identify vertices that can have degree 2 (all vertices in remaining graph).
    # 2. Greedily add edges to form a 2-regular subgraph.
    #    We prefer edges that connect vertices with degree 0 or 1 to avoid isolating vertices.
    
    # Sort edges randomly for tie-breaking
    edge_list = list(remaining)
    rng.shuffle(edge_list)
    
    # We'll try to build a 2-factor by adding edges one by one.
    # An edge (u,v) can be added if deg[u] < 2 and deg[v] < 2.
    # To maximize the number of vertices in the 2-factor, we should prefer edges
    # that involve vertices with deg 0 first, then deg 1.
    
    # Priority: edges where both endpoints have deg 0, then one deg 0 and one deg 1,
    # then both deg 1. Avoid edges that would create a component that can't be closed.
    
    # Simple greedy with randomization:
    added = set()
    for e in edge_list:
        u, v = e
        if deg.get(u, 0) < 2 and deg.get(v, 0) < 2:
            # Check if adding this edge would create a "dead end" component
            # that can't be part of a cycle. This is complex, so we use a simpler heuristic:
            # Just add it if degrees allow.
            selected_edges.add(e)
            deg[u] = deg.get(u, 0) + 1
            deg[v] = deg.get(v, 0) + 1
    
    # Now, selected_edges may not form a 2-factor (it might have paths and cycles).
    # We need to extract only the cycle components.
    # Build the graph from selected_edges and find all cycles.
    
    if not selected_edges:
        return []
    
    # Build adjacency for selected edges
    sel_adj = {}
    for (u, v) in selected_edges:
        sel_adj.setdefault(u, set()).add(v)
        sel_adj.setdefault(v, set()).add(u)
    
    # Find all cycles in the selected subgraph
    visited = set()
    cycles = []
    
    for node in sel_adj:
        if node in visited:
            continue
        # BFS/DFS to find the component
        component = []
        stack = [node]
        visited.add(node)
        while stack:
            n = stack.pop()
            component.append(n)
            for nb in sel_adj[n]:
                if nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        
        # Check if this component is a cycle (all vertices have degree 2 in selected_edges)
        is_cycle = True
        for v in component:
            if len(sel_adj[v]) != 2:
                is_cycle = False
                break
        
        if is_cycle and len(component) >= 3:
            # Extract the cycle order
            # Start from any node, follow neighbors
            cycle = [component[0]]
            prev = None
            curr = component[0]
            while True:
                cycle.append(curr)
                nexts = [nb for nb in sel_adj[curr] if nb != prev]
                if not nexts:
                    break
                next_node = nexts[0]
                if next_node == component[0]:
                    break
                prev = curr
                curr = next_node
            # Remove duplicate start if present
            if len(cycle) > 1 and cycle[0] == cycle[-1]:
                cycle = cycle[:-1]
            if len(cycle) >= 3:
                cycles.append(cycle)
    
    return cycles

def decompose_2factor(edges, max_time=40.0):
    """
    Decompose edges by repeatedly extracting a 2-factor (union of cycles),
    then decomposing the 2-factor into individual cycles.
    Remaining edges become single-edge pieces.
    """
    start = time.time()
    remaining = set(tuple(sorted(e)) for e in edges)
    pieces = []
    rng = Random(42)
    
    while remaining:
        cur_time = time.time()
        if cur_time - start > max_time:
            break
        
        adj = build_adj(list(remaining))
        
        # Try to find a 2-factor
        cycles = find_2_factor(remaining, adj, rng)
        
        if cycles:
            for cycle in cycles:
                # Verify the cycle edges are in remaining
                cyc_edges = set()
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    e = (u, v) if u < v else (v, u)
                    cyc_edges.add(e)
                
                if cyc_edges <= remaining:
                    pieces.append(cycle[:])
                    for e in cyc_edges:
                        remaining.discard(e)
        
        if not remaining:
            break
        
        # If no cycle was found or some edges remain, try to extract any simple cycle
        # as a fallback, otherwise take single edges
        found_any = False
        # Try a simple random walk to find a cycle
        nodes = list(set([u for (u, v) in remaining] + [v for (u, v) in remaining]))
        if len(nodes) >= 3:
            for _ in range(10):
                start_node = rng.choice(nodes)
                visited = {start_node}
                path = [start_node]
                stack = [(start_node, iter(sorted(adj.get(start_node, []), key=lambda x: rng.random())))]
                found_cycle = False
                while stack:
                    node, neighbors = stack[-1]
                    advanced = False
                    for nb in neighbors:
                        e = (node, nb) if node < nb else (nb, node)
                        if e not in remaining:
                            continue
                        if nb == start_node and len(path) >= 3:
                            # Found a cycle
                            cyc_edges = set()
                            for i in range(len(path)):
                                u = path[i]
                                v = path[(i + 1) % len(path)]
                                ce = (u, v) if u < v else (v, u)
                                cyc_edges.add(ce)
                            if cyc_edges <= remaining:
                                pieces.append(path[:])
                                for ce in cyc_edges:
                                    remaining.discard(ce)
                                found_any = True
                                found_cycle = True
                                break
                        if nb not in visited:
                            visited.add(nb)
                            path.append(nb)
                            stack.append((nb, iter(sorted(adj.get(nb, []), key=lambda x: rng.random()))))
                            advanced = True
                            break
                    if found_cycle:
                        break
                    if not advanced:
                        stack.pop()
                        if path:
                            last = path.pop()
                            visited.discard(last)
                if found_cycle:
                    break
        
        if not found_any:
            # No cycle found, take a single edge
            if remaining:
                e = next(iter(remaining))
                pieces.append([e[0], e[1]])
                remaining.discard(e)
    
    # Add remaining edges as single-edge pieces
    for e in remaining:
        pieces.append([e[0], e[1]])
    
    return pieces

def construct():
    start = time.time()
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    per_graph_time = max(1.0, (DEADLINE_MARGIN - 5) / 3)
    
    for i, spec in enumerate(specs):
        edges = instance(*spec)
        cur_time = time.time()
        remaining_time = min(per_graph_time, DEADLINE_MARGIN - (cur_time - start))
        if remaining_time < 1.0:
            # No time left, return all edges as single pieces
            results.append([list(e) for e in edges])
            continue
        
        pieces = decompose_2factor(edges, max_time=remaining_time)
        results.append(pieces)
        
        # Check deadline
        cur_time = time.time()
        if cur_time - start > DEADLINE_MARGIN - 3:
            # Return remaining as individual edges
            for remaining_spec in specs[i + 1:]:
                e = instance(*remaining_spec)
                results.append([list(e) for e in e])
            break
    
    # Ensure we have exactly 3 results
    while len(results) < 3:
        results.append([])
    
    return results[:3]
