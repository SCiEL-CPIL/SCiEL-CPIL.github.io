"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

def construct():
    """
    Decompose three fixed random graphs into few cycles and edges.
    """
    deadline = time.time() + 110

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = set()
        for i in range(len(order) - 1):
            u, v = order[i], order[i + 1]
            edges.add((min(u, v), max(u, v)))
        for u in range(n):
            for v in range(u + 1, n):
                if (u, v) not in edges and rng.random() < p:
                    edges.add((u, v))
        return sorted(edges)

    def find_cycle(adj, remaining):
        """Find a cycle in the graph defined by adj. Returns list of vertices or None."""
        # Find a vertex with degree >= 2
        candidates = [v for v in adj if len(adj[v]) >= 2]
        if not candidates:
            return None
        random.shuffle(candidates)
        
        for start in candidates:
            # DFS to find a cycle
            visited = {start: -1}  # vertex -> parent, -1 for start
            stack = [start]
            
            while stack:
                curr = stack[-1]
                if not adj[curr]:
                    stack.pop()
                    continue
                
                # Try neighbors
                found = False
                for nb in sorted(adj[curr]):
                    if nb == visited.get(curr, -2):  # skip parent
                        continue
                    if nb in visited:
                        # Found a back edge to an already visited node
                        # Reconstruct cycle
                        if nb == start:
                            # Cycle back to start
                            cycle = [curr]
                            node = curr
                            while node != start:
                                node = visited[node]
                                if node == -1:
                                    break
                                cycle.append(node)
                            if len(cycle) >= 3:
                                return cycle
                        else:
                            # Back edge to some other visited node
                            cycle = [curr]
                            node = curr
                            while node != nb:
                                node = visited[node]
                                if node == -1:
                                    break
                                cycle.append(node)
                            if len(cycle) >= 3:
                                return cycle
                    else:
                        visited[nb] = curr
                        stack.append(nb)
                        found = True
                        break
                
                if not found:
                    stack.pop()
        
        return None

    def decompose(edges, n):
        """Decompose edges into cycles and single edges."""
        if not edges:
            return []
        
        remaining = set(edges)
        pieces = []
        
        adj = defaultdict(set)
        for u, v in remaining:
            adj[u].add(v)
            adj[v].add(u)
        
        # Repeatedly find cycles
        for _ in range(500):
            if not remaining:
                break
            
            cycle = find_cycle(adj, remaining)
            if cycle is None:
                break
            
            # Verify all edges in cycle exist
            valid = True
            for i in range(len(cycle)):
                u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                edge = (min(u, v), max(u, v))
                if edge not in remaining:
                    valid = False
                    break
            
            if not valid:
                break
            
            pieces.append(cycle)
            for i in range(len(cycle)):
                u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                edge = (min(u, v), max(u, v))
                remaining.discard(edge)
                adj[u].discard(v)
                adj[v].discard(u)
        
        # Add remaining edges as single-edge pieces
        for u, v in remaining:
            pieces.append([u, v])
        
        return pieces

    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    
    for n, p, seed in specs:
        if time.time() > deadline:
            edges = instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue
        
        edges = instance(n, p, seed)
        pieces = decompose(edges, n)
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
