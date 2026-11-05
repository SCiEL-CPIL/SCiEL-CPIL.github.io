import time
import random
from collections import defaultdict

def construct():
    """
    Decomposes the edges of three fixed instance graphs into the minimum number of
    edge-disjoint simple paths.
    
    Returns:
        A list of three lists of paths. Each path is a list of vertices.
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def decompose(edges):
        if not edges:
            return []
        
        edges_set = set(edges)
        n_vertices = max(max(u, v) for u, v in edges) + 1
        
        # Build adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        def edge_key(u, v):
            return (min(u, v), max(u, v))
        
        def path_edges(path):
            return {edge_key(path[i], path[i+1]) for i in range(len(path)-1)}
        
        def validate(paths):
            """Check if paths are valid: simple, edge-disjoint, cover all edges."""
            used = set()
            for path in paths:
                if len(set(path)) != len(path):
                    return False
                for i in range(len(path)-1):
                    e = edge_key(path[i], path[i+1])
                    if e not in edges_set:
                        return False
                    if e in used:
                        return False
                    used.add(e)
            return used == edges_set
        
        def find_trails():
            """
            Decompose the graph into trails using Hierholzer's algorithm.
            Returns a list of trails (lists of vertices).
            """
            remaining_edges = set(edges)
            adj_rem = defaultdict(set)
            for u, v in edges:
                adj_rem[u].add(v)
                adj_rem[v].add(u)
            
            def get_odd_vertices():
                odds = []
                for u in range(n_vertices):
                    if len(adj_rem[u]) % 2 != 0:
                        odds.append(u)
                return odds
            
            trails = []
            
            while remaining_edges:
                # Pick a start vertex
                odds = get_odd_vertices()
                if odds:
                    start = odds[0]
                else:
                    # Pick any vertex with remaining edges
                    start = next(iter(adj_rem.keys()))
                    if not adj_rem[start]:
                        continue

                # Hierholzer's algorithm to find a trail
                trail = []
                stack = [start]
                visited_edges = set()
                
                while stack:
                    v = stack[-1]
                    if adj_rem[v]:
                        # Pick a neighbor
                        # To ensure we cover all edges, we just pick one
                        # Randomization helps find different decompositions
                        neighbors = list(adj_rem[v])
                        random.shuffle(neighbors)
                        u = neighbors[0]
                        e = edge_key(v, u)
                        
                        # Remove edge from remaining
                        adj_rem[v].remove(u)
                        adj_rem[u].remove(v)
                        remaining_edges.discard(e)
                        
                        stack.append(u)
                        trail.append(v)
                    else:
                        # Backtrack
                        v = stack.pop()
                        if stack:
                            trail.append(v)
                if stack:
                     trail.append(stack[0]) # This logic is slightly off for standard Hierholzer implementation returning the circuit.
                
                # Standard Hierholzer returns the circuit in reverse or specific order.
                # Let's implement a simpler DFS trail finder for clarity and correctness with edge removal.
                
                # Re-implementing trail finding properly
                def find_trail_from(start):
                    trail = []
                    stack = [start]
                    while stack:
                        v = stack[-1]
                        if adj_rem[v]:
                            u = next(iter(adj_rem[v]))
                            e = edge_key(v, u)
                            adj_rem[v].remove(u)
                            adj_rem[u].remove(v)
                            remaining_edges.discard(e)
                            stack.append(u)
                            trail.append(v)
                        else:
                            stack.pop()
                    return trail[::-1] # Vertices in order

                # The above simple stack approach doesn't handle branches correctly for Eulerian trails in one go if we don't backtrack correctly.
                # Correct Hierholzer:
                def hierholzer(start):
                    circuit = []
                    stack = [start]
                    while stack:
                        v = stack[-1]
                        if adj_rem[v]:
                            u = next(iter(adj_rem[v]))
                            e = edge_key(v, u)
                            adj_rem[v].remove(u)
                            adj_rem[u].remove(v)
                            remaining_edges.discard(e)
                            stack.append(u)
                        else:
                            circuit.append(stack.pop())
                    return circuit[::-1]
                
                t = hierholzer(start)
                if t:
                    trails.append(t)
            
            return trails

        def split_trail(trail):
            """
            Split a trail (which may repeat vertices) into simple paths.
            """
            paths = []
            if not trail:
                return paths
            
            current_path = [trail[0]]
            visited_in_path = {trail[0]}
            
            for i in range(1, len(trail)):
                v = trail[i]
                if v in visited_in_path:
                    # Close current path
                    paths.append(current_path)
                    # Start new path with the vertex that caused the conflict?
                    # The edge was (trail[i-1], v).
                    # So the new path starts with trail[i-1] and v?
                    # No, the edge is already "used" in the sense that we are traversing it.
                    # If we close current_path at trail[i-1], we have used edges up to trail[i-1].
                    # The next edge is (trail[i-1], v).
                    # So we start a new path with [trail[i-1], v].
                    current_path = [trail[i-1], v]
                    visited_in_path = {trail[i-1], v}
                else:
                    current_path.append(v)
                    visited_in_path.add(v)
            
            if current_path:
                paths.append(current_path)
            
            return paths

        def try_merge(paths, i, j):
            """
            Try to merge paths[i] and paths[j] into a single simple path.
            """
            p1 = paths[i]
            p2 = paths[j]
            
            # Check vertex disjointness
            if set(p1) & set(p2):
                return None
            
            # Try all 4 endpoint combinations
            endpoints1 = [(0, p1[0]), (len(p1)-1, p1[-1])]
            endpoints2 = [(0, p2[0]), (len(p2)-1, p2[-1])]
            
            for idx1, v1 in endpoints1:
                for idx2, v2 in endpoints2:
                    if edge_key(v1, v2) in edges_set:
                        # Check if v1 and v2 are connected
                        # Orient p1 so v1 is at the end
                        # Orient p2 so v2 is at the start
                        p1_oriented = p1 if p1[-1] == v1 else p1[::-1]
                        p2_oriented = p2 if p2[0] == v2 else p2[::-1]
                        
                        new_path = p1_oriented + p2_oriented[1:]
                        
                        # Since p1 and p2 are vertex-disjoint, and individually simple, 
                        # new_path is simple.
                        # We just need to ensure the edge exists (checked above).
                        
                        new_paths = [paths[k] for k in range(len(paths)) if k not in (i, j)]
                        new_paths.append(new_path)
                        return new_paths
            
            return None

        def local_search(paths, max_iters=10000):
            paths = list(paths)
            improved = True
            iters = 0
            while improved and iters < max_iters:
                improved = False
                iters += 1
                n = len(paths)
                for i in range(n):
                    for j in range(i+1, n):
                        new_paths = try_merge(paths, i, j)
                        if new_paths is not None:
                            paths = new_paths
                            improved = True
                            break
                    if improved:
                        break
            return paths

        # 1. Generate initial trails
        trails = find_trails()
        
        # 2. Split trails into simple paths
        initial_paths = []
        for t in trails:
            initial_paths.extend(split_trail(t))
        
        # 3. Local search to merge
        best_paths = local_search(initial_paths)
        
        # 4. Validation
        if not validate(best_paths):
            # Fallback
            best_paths = [[u, v] for u, v in edges]
            
        return best_paths

    def validate_paths(paths, edges):
        """Validate that paths are simple, edge-disjoint, and cover all edges."""
        if not paths:
            return len(edges) == 0
        
        edge_set = set(edges)
        used_edges = set()
        
        for path in paths:
            if len(set(path)) != len(path):
                return False
            
            for j in range(len(path) - 1):
                u, v = path[j], path[j+1]
                e = tuple(sorted((u, v)))
                if e not in edge_set:
                    return False
                if e in used_edges:
                    return False
                used_edges.add(e)
        
        if used_edges != edge_set:
            return False
        
        return True

    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    results = []
    
    for spec in specs:
        n, p, seed = spec
        edges = instance(n, p, seed)
        paths = decompose(edges)
        
        # Validate
        if not validate_paths(paths, edges):
            # Fallback: each edge as a path
            paths = [[u, v] for u, v in edges]
        
        results.append(paths)
        
        if time.time() > deadline - 1:
            break
    
    while len(results) < 3:
        n, p, seed = specs[len(results)]
        edges = instance(n, p, seed)
        results.append([[u, v] for u, v in edges])
    
    return results

if __name__ == "__main__":
    res = construct()
    total = sum(len(r) for r in res)
    print(f"Total paths: {total}")
    for i, r in enumerate(res):
        print(f"Instance {i}: {len(r)} paths")
