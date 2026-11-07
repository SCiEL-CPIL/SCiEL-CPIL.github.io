"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build edge set and adjacency
    edge_set = set()
    adj = defaultdict(set)
    for u, v in edges:
        key = (min(u, v), max(u, v))
        edge_set.add(key)
        adj[u].add(v)
        adj[v].add(u)
    
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    def get_cycle_edges(cycle):
        """Get set of edges (as frozensets) for a cycle."""
        es = set()
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            es.add(frozenset([u, v]))
        return es
    
    def find_hc(adj_sets, start, max_depth=21):
        """
        Find Hamiltonian cycle starting from 'start' in graph with adjacency as dict of sets.
        Uses Warnsdorff's heuristic with randomization.
        Returns list of vertices or None.
        """
        path = [start]
        visited = {start}
        
        def dfs(u):
            if time.time() > deadline - 2:
                return None
            
            if len(path) == n:
                if start in adj_sets[u]:
                    return list(path)
                return None
            
            # Get unvisited neighbors
            unvisited = [v for v in adj_sets[u] if v not in visited]
            if not unvisited:
                return None
            
            # Warnsdorff: prefer neighbors with fewer onward options
            scored = []
            for v in unvisited:
                onward = sum(1 for w in adj_sets[v] if w not in visited and w != u)
                # Add small random perturbation for diversity
                scored.append((onward + random.random() * 0.5, v))
            
            scored.sort()
            
            for _, v in scored:
                path.append(v)
                visited.add(v)
                res = dfs(v)
                if res is not None:
                    return res
                path.pop()
                visited.remove(v)
            
            return None
        
        return dfs(start)
    
    def find_hc_multi(adj_sets, max_attempts=8):
        """Try multiple random orderings to find a Hamiltonian cycle."""
        for attempt in range(max_attempts):
            if time.time() > deadline - 2:
                return None
            
            # Shuffle adjacency for diversity
            adj_shuffled = {u: list(v) for u, v in adj_sets.items()}
            for u in adj_shuffled:
                random.shuffle(adj_shuffled[u])
            
            # Convert back to sets for the finder (or use lists)
            # Actually, let's just try different starts
            start = random.choice(list(adj_sets.keys()))
            hc = find_hc(adj_sets, start)
            if hc is not None:
                return hc
        
        return None
    
    def verify_cycles(cycles):
        """Verify cycles are valid, edge-disjoint, and use all vertices."""
        all_vertices = set()
        all_edges = set()
        
        for cycle in cycles:
            if len(cycle) < 3:
                return False
            if len(set(cycle)) != len(cycle):
                return False
            all_vertices.update(cycle)
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                e = frozenset([u, v])
                if e in all_edges:
                    return False
                if e not in edge_set:
                    return False
                all_edges.add(e)
        
        if all_vertices != set(range(n)):
            return False
        
        return True
    
    def find_three_cycles():
        """Find 3 edge-disjoint Hamiltonian cycles."""
        for attempt in range(300):
            if time.time() > deadline - 5:
                break
            
            # Shuffle adjacency
            adj_cur = {u: set(v) for u, v in adj.items()}
            
            # Find first HC
            hc1 = find_hc_multi(adj_cur, max_attempts=6)
            if hc1 is None:
                continue
            
            # Remove hc1 edges
            used = get_cycle_edges(hc1)
            
            # Build remaining graph
            adj_rem = defaultdict(set)
            for u in range(n):
                for v in adj[u]:
                    if frozenset([u, v]) not in used:
                        adj_rem[u].add(v)
            
            # Check all vertices have degree 4
            if any(len(adj_rem[i]) != 4 for i in range(n)):
                continue
            
            # Find second HC
            hc2 = find_hc_multi(adj_rem, max_attempts=6)
            if hc2 is None:
                continue
            
            # Remove hc2 edges
            used.update(get_cycle_edges(hc2))
            
            # Build final remaining graph
            adj_final = defaultdict(set)
            for u in range(n):
                for v in adj[u]:
                    if frozenset([u, v]) not in used:
                        adj_final[u].add(v)
            
            # Check all vertices have degree 2
            if any(len(adj_final[i]) != 2 for i in range(n)):
                continue
            
            # Remaining graph is 2-regular: union of cycles
            # Check if it's a single Hamiltonian cycle
            visited_final = set()
            cycle = [0]
            visited_final.add(0)
            current = 0
            
            while True:
                nexts = [v for v in adj_final[current] if v not in visited_final]
                if not nexts:
                    if 0 in adj_final[current] and len(cycle) == n:
                        break
                    else:
                        cycle = None
                        break
                current = nexts[0]
                cycle.append(current)
                visited_final.add(current)
                if len(cycle) > n:
                    cycle = None
                    break
            
            if cycle is not None and len(cycle) == n and 0 in adj_final[cycle[-1]]:
                if verify_cycles([hc1, hc2, cycle]):
                    return [list(hc1), list(hc2), list(cycle)]
        
        return None
    
    def find_two_cycles():
        """Find 2 edge-disjoint Hamiltonian cycles."""
        for attempt in range(200):
            if time.time() > deadline - 3:
                break
            
            adj_cur = {u: set(v) for u, v in adj.items()}
            
            hc1 = find_hc_multi(adj_cur, max_attempts=6)
            if hc1 is None:
                continue
            
            used = get_cycle_edges(hc1)
            
            adj_rem = defaultdict(set)
            for u in range(n):
                for v in adj[u]:
                    if frozenset([u, v]) not in used:
                        adj_rem[u].add(v)
            
            hc2 = find_hc_multi(adj_rem, max_attempts=6)
            if hc2 is None:
                continue
            
            if verify_cycles([hc1, hc2]):
                return [list(hc1), list(hc2)]
        
        return None
    
    def find_one_cycle():
        """Find one Hamiltonian cycle."""
        adj_cur = {u: set(v) for u, v in adj.items()}
        hc = find_hc_multi(adj_cur, max_attempts=10)
        if hc is not None:
            return [hc]
        
        # Fallback: find any triangle
        for u, v in edges:
            for w in adj[u]:
                if w != v and frozenset([v, w]) in edge_set:
                    if u < v < w:
                        return [[int(u), int(v), int(w)]]
        
        return None
    
    # Try to find 3 cycles first
    result = find_three_cycles()
    if result is not None:
        return [[int(v) for v in c] for c in result]
    
    # Fallback to 2 cycles
    result = find_two_cycles()
    if result is not None:
        return [[int(v) for v in c] for c in result]
    
    # Fallback to 1 cycle
    result = find_one_cycle()
    if result is not None:
        return [[int(v) for v in c] for c in result]
    
    # Last resort
    return [[0, 1, 2]]

# EVOLVE-BLOCK-END
