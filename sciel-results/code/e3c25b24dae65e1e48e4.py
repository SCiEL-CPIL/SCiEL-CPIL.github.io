import time
from random import Random
import sys

def construct():
    t0 = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def canonical(e):
        a, b = e
        return (a, b) if a < b else (b, a)

    def decompose(edges):
        if not edges:
            return []
        remaining = set(canonical(e) for e in edges)
        pieces = []
        
        def build_adj(rem):
            adj = {}
            for u, v in rem:
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
            return adj

        def find_longest_cycle(rem, adj, rng, max_tries=200):
            """
            Finds a long simple cycle in the graph defined by adj.
            Uses randomized DFS with backtracking.
            """
            if not adj:
                return None
            
            best_cycle = None
            
            # Vertices with degree >= 2 are candidates for start
            candidates = [u for u in adj if len(adj[u]) >= 2]
            if not candidates:
                return None
            
            for _ in range(max_tries):
                start = rng.choice(candidates)
                path = [start]
                visited = {start}
                node = start
                
                # Randomized DFS
                # We want to find a cycle, so we stop if we hit a neighbor of start
                # We prefer to move to unvisited nodes.
                
                # To encourage longer cycles, we can avoid closing too early if possible
                # but we must close eventually.
                
                # Simple strategy: Random walk, if we can close and length >= 3, close.
                # If we get stuck, backtrack (not implemented in simple version, just restart).
                # Restarting is cheap for n=30.
                
                while True:
                    # Get unvisited neighbors
                    nbrs = [x for x in adj.get(node, ()) if x not in visited]
                    
                    # Check if we can close the cycle
                    if start in adj.get(node, ()) and len(path) >= 2:
                        # We found a cycle
                        # Note: path contains start...node. The edge node-start closes it.
                        # The cycle vertices are path.
                        if best_cycle is None or len(path) > len(best_cycle):
                            best_cycle = list(path)
                        # Break to restart with new random choices
                        break
                    
                    if not nbrs:
                        # Dead end, restart
                        break
                    
                    # Choose next node
                    # Heuristic: Prefer neighbors that have more unvisited neighbors?
                    # Or just random.
                    # Let's try random first, it's robust.
                    node = rng.choice(nbrs)
                    path.append(node)
                    visited.add(node)
                    
                    # Safety check: if path gets too long, it's fine, but n is small
                    if len(path) > len(adj):
                        break

            return best_cycle

        # Main decomposition loop
        while remaining:
            if time.time() - t0 > DEADLINE - 5:
                break
            
            adj = build_adj(remaining)
            
            # Try to find a cycle
            # Use a deterministic seed based on the current set of edges for reproducibility in logic, 
            # but random for exploration.
            rng = Random(hash(frozenset(remaining)) & 0xffffffff)
            
            # Try harder for the first few cycles?
            # Just use a fixed number of tries.
            cyc = find_longest_cycle(remaining, adj, rng, max_tries=100)
            
            if cyc and len(cyc) >= 3:
                # Validate cycle edges are in remaining
                cycle_edges = set()
                for i in range(len(cyc)):
                    u, v = cyc[i], cyc[(i + 1) % len(cyc)]
                    cycle_edges.add(canonical((u, v)))
                
                if cycle_edges <= remaining:
                    pieces.append(list(cyc))
                    remaining -= cycle_edges
                    continue
            else:
                # No cycle found, break to avoid infinite loop
                break
        
        # Add remaining edges as single pieces
        for e in sorted(remaining):
            pieces.append(list(e))
            
        return pieces

    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    
    for spec in specs:
        if time.time() - t0 > DEADLINE - 10:
            # Fallback: just return all edges as pieces
            edges = instance(*spec)
            results.append([list(e) for e in edges])
        else:
            edges = instance(*spec)
            results.append(decompose(edges))

    return results
