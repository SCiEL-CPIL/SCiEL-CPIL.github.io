import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin for 90s limit
    
    n = 21
    edges = P['edges']
    edge_set = set()
    adj = defaultdict(set)
    for u, v in edges:
        edge_set.add((min(u, v), max(u, v)))
        adj[u].add(v)
        adj[v].add(u)
    
    best_cycles = []
    best_count = 0
    
    def find_hamiltonian_cycle(vertices, used_edges, deadline):
        """Find a Hamiltonian cycle using DFS with Warnsdorff heuristic and pruning."""
        k = len(vertices)
        if k < 3:
            return None
        
        # Precompute available neighbors as sets for fast lookup
        avail = {}
        for v in vertices:
            avail[v] = set()
            for u in adj[v]:
                if u in vertices and (min(u,v), max(u,v)) not in used_edges:
                    avail[v].add(u)
            if len(avail[v]) < 2:
                return None
        
        # Choose start vertex with minimum available degree
        start = min(vertices, key=lambda v: len(avail[v]))
        remaining = set(vertices) - {start}
        path = [start]
        
        # Precompute edge lookup as a set for O(1) checks
        used_set = set(used_edges)
        
        def backtrack(path, remaining):
            nonlocal deadline
            if time.time() > deadline:
                return None
            if len(path) == k:
                last = path[-1]
                if last in avail[start]:
                    return path[:]
                return None
            
            last = path[-1]
            # Get candidates: neighbors of last that are in remaining
            candidates = [v for v in remaining if v in avail[last]]
            
            if not candidates:
                return None
            
            # Warnsdorff: sort by number of remaining options (most constrained first)
            # Also check if any candidate would leave another vertex with < 2 options
            scored = []
            for v in candidates:
                # After moving to v, how many options does v have among remaining - {v}?
                opts = 0
                for w in avail[v]:
                    if w != last and w in remaining:
                        opts += 1
                # Check if this move would strand any other vertex
                stranded = False
                if len(remaining) > 1:
                    for w in remaining:
                        if w == v:
                            continue
                        # w's options after removing v from remaining
                        w_opts = 0
                        for x in avail[w]:
                            if x != last and x in remaining and x != v:
                                w_opts += 1
                        if w_opts < 1:  # w would have no path forward
                            stranded = True
                            break
                score = opts + (100 if stranded else 0)
                scored.append((score, v))
            
            scored.sort()
            
            for _, v in scored:
                if time.time() > deadline:
                    return None
                path.append(v)
                remaining.remove(v)
                result = backtrack(path, remaining)
                if result is not None:
                    return result
                path.pop()
                remaining.add(v)
            return None
        
        return backtrack(path, remaining)
    
    def find_cycle_random(vertices, used_edges, deadline, num_restarts=50):
        """Try multiple random starts to find a Hamiltonian cycle."""
        k = len(vertices)
        if k < 3:
            return None
        
        # Precompute available neighbors
        avail = {}
        valid = True
        for v in vertices:
            avail[v] = set()
            for u in adj[v]:
                if u in vertices and (min(u,v), max(u,v)) not in used_edges:
                    avail[v].add(u)
            if len(avail[v]) < 2:
                valid = False
                break
        if not valid:
            return None
        
        seed = hash(frozenset(vertices)) ^ hash(tuple(sorted(used_edges)))
        rng = random.Random(seed & 0xFFFFFFFF)
        
        for attempt in range(num_restarts):
            if time.time() > deadline:
                return None
            
            start = rng.choice(list(vertices))
            remaining = set(vertices) - {start}
            path = [start]
            
            def backtrack(path, remaining):
                nonlocal deadline
                if time.time() > deadline:
                    return None
                if len(path) == k:
                    last = path[-1]
                    if last in avail[start]:
                        return path[:]
                    return None
                
                last = path[-1]
                candidates = [v for v in remaining if v in avail[last]]
                
                if not candidates:
                    return None
                
                # Warnsdorff with randomization for diversity
                scored = []
                for v in candidates:
                    opts = sum(1 for w in avail[v] if w != last and w in remaining)
                    scored.append((opts, v))
                
                scored.sort()
                # Take top 3 most constrained, shuffle them for diversity
                if len(scored) > 3:
                    top = scored[:3]
                    rng.shuffle(top)
                    scored = top + scored[3:]
                
                for _, v in scored:
                    if time.time() > deadline:
                        return None
                    path.append(v)
                    remaining.remove(v)
                    result = backtrack(path, remaining)
                    if result is not None:
                        return result
                    path.pop()
                    remaining.add(v)
                return None
            
            result = backtrack(path, remaining)
            if result is not None:
                return result
        
        return None
    
    def try_vertex_subset(vertices, deadline):
        """Try to find max edge-disjoint Hamiltonian cycles on this vertex set."""
        nonlocal best_cycles, best_count
        if time.time() > deadline:
            return
        
        vertices = frozenset(vertices)
        k = len(vertices)
        if k < 3:
            return
        
        # Count edges in induced subgraph
        induced_edges = set()
        for u in vertices:
            for v in adj[u]:
                if v in vertices:
                    edge = (min(u, v), max(u, v))
                    induced_edges.add(edge)
        
        max_possible = len(induced_edges) // k
        if max_possible <= best_count:
            return
        
        used_edges = set()
        cycles = []
        
        # Try to find cycles greedily
        max_attempts = min(max_possible, 10)
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Try deterministic first, then random
            cycle = find_hamiltonian_cycle(vertices, used_edges, deadline)
            if cycle is None:
                cycle = find_cycle_random(vertices, used_edges, deadline, num_restarts=50)
            
            if cycle is None:
                break
            
            # Add edges to used
            for i in range(k):
                u = cycle[i]
                v = cycle[(i + 1) % k]
                edge = (min(u, v), max(u, v))
                used_edges.add(edge)
            cycles.append(cycle)
        
        if len(cycles) > best_count:
            best_count = len(cycles)
            best_cycles = [list(c) for c in cycles]
    
    all_vertices = set(range(n))
    
    # Priority order: full set, then all single-vertex removals
    subsets_to_try = [all_vertices]
    for removed in range(n):
        subsets_to_try.append(all_vertices - {removed})
    
    # Try each subset with time check
    for subset in subsets_to_try:
        if time.time() > deadline:
            break
        try_vertex_subset(subset, deadline)
    
    # Return best result
    if best_cycles:
        return [[int(v) for v in cycle] for cycle in best_cycles]
    else:
        # Fallback: try to find any cycle on full vertex set
        cycle = find_hamiltonian_cycle(all_vertices, set(), deadline)
        if cycle:
            return [[int(v) for v in cycle]]
        return [[0, 1, 2]]

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}
