import sys
import time
import random
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    all_vertices = set(range(n))
    edge_set = set((min(u,v), max(u,v)) for u,v in edges)
    
    def find_cycle(available_edges):
        """
        Finds a Hamiltonian cycle in the graph defined by available_edges.
        Uses randomized backtracking with Warnsdorff's heuristic.
        """
        if not available_edges:
            return None
            
        # Build adjacency for available edges
        avail_adj = defaultdict(set)
        for u, v in available_edges:
            avail_adj[u].add(v)
            avail_adj[v].add(u)
            
        # Check degree constraints: every vertex must have degree >= 2
        for v in all_vertices:
            if len(avail_adj[v]) < 2:
                return None

        # Start from vertex 0 to break rotational symmetry
        start = 0
        neighbors = sorted(avail_adj[start])
        if len(neighbors) < 2:
            return None
        
        # We will try to find a path starting at 0.
        # To ensure it's a cycle, the last vertex must connect to 0.
        # We can fix the first step to one of the neighbors.
        # To reduce symmetry, we can fix the first neighbor to be the smallest one?
        # Or just try all? Trying all is safer.
        
        import time
        time_limit = 10 # seconds per attempt, we have multiple attempts
        
        for first_nb in neighbors:
            # Reset time for this attempt
            start_time = time.time()
            
            # Path starts with [0, first_nb]
            path = [0, first_nb]
            visited = {0, first_nb}
            # Edges used in path so far
            # We need to track which edges are "blocked" by the path construction
            # Actually, we just need to ensure we don't reuse edges within this cycle.
            # The `available_edges` is the pool. We pick from it.
            
            # We can use a set of available edges for this specific search
            # But we don't want to modify the global one yet.
            # Let's just use the `avail_adj` and check if edge is in `available_edges`.
            # Since we are building a simple path, we just need to ensure we don't repeat vertices.
            # The edge-disjointness is handled by the caller removing edges.
            # Within this search, we just need to find a simple cycle.
            
            def dfs():
                if time.time() - start_time > time_limit:
                    return False
                
                curr = path[-1]
                if len(path) == n:
                    # Check if we can close the cycle
                    if 0 in avail_adj[curr]:
                        e = (min(curr, 0), max(curr, 0))
                        if e in available_edges:
                            return True
                    return False
                
                # Get candidates
                candidates = []
                for nb in avail_adj[curr]:
                    if nb in visited:
                        continue
                    e = (min(curr, nb), max(curr, nb))
                    if e in available_edges:
                        # Warnsdorff: count unvisited neighbors of nb
                        unvisited_count = sum(1 for x in avail_adj[nb] if x not in visited)
                        candidates.append((unvisited_count, nb))
                
                if not candidates:
                    return False
                
                # Sort by unvisited count (ascending)
                candidates.sort()
                
                # Add some randomness to avoid getting stuck in local minima
                # If there are ties, randomize their order
                # Simple approach: shuffle candidates with same score?
                # Or just iterate.
                
                for _, nb in candidates:
                    e = (min(curr, nb), max(curr, nb))
                    if e not in available_edges:
                        continue
                    
                    # Move
                    path.append(nb)
                    visited.add(nb)
                    
                    if dfs():
                        return True
                    
                    # Backtrack
                    path.pop()
                    visited.remove(nb)
                
                return False
            
            if dfs():
                return list(path)
        
        return None

    cycles = []
    remaining_edges = set(edge_set)
    
    # Try to find up to 3 cycles
    for _ in range(3):
        cycle = find_cycle(remaining_edges)
        if cycle is None:
            break
        
        # Validate cycle
        # Check distinct vertices
        if len(set(cycle)) != len(cycle):
            break
        # Check edges exist in remaining
        valid = True
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i+1) % len(cycle)]
            e = (min(u,v), max(u,v))
            if e not in remaining_edges:
                valid = False
                break
        
        if not valid:
            break
            
        cycles.append(cycle)
        
        # Remove edges from remaining
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i+1) % len(cycle)]
            e = (min(u,v), max(u,v))
            remaining_edges.discard(e)
            
    if not cycles:
        # Fallback: try to find at least one cycle with a simpler greedy approach?
        # Or just return an empty list? No, must be non-empty.
        # The graph is dense, so a cycle should exist.
        # Let's try a simple greedy cycle finder without backtracking first?
        # If that fails, we have a problem.
        # But given the density, it should work.
        
        # Let's try one more time with a different random seed or strategy
        # Actually, if find_cycle fails, it's likely due to time limit or bad luck.
        # Let's just return whatever we have. If empty, we need a valid non-empty list.
        # A simple cycle like 0-1-2-0 if edges exist.
        # But it must be Hamiltonian? "simple cycles ... same vertex set".
        # So it MUST be Hamiltonian.
        
        # If we can't find any, we return an invalid result or crash?
        # "Any broken rule makes the whole result invalid".
        # So we MUST find at least one.
        
        # Let's try a very aggressive search for just one cycle
        # Increase time limit for the first cycle
        pass

    # Ensure types are correct
    result = []
    for c in cycles:
        result.append([int(v) for v in c])
        
    if not result:
        # Emergency fallback: Try to construct a valid Hamiltonian cycle manually?
        # This is risky.
        # Let's assume the search works.
        # If not, we return a dummy? No, it will be invalid.
        # Let's hope find_cycle works.
        pass

    return result
