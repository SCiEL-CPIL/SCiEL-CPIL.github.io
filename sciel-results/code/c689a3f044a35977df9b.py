import time
import random
from collections import defaultdict
from itertools import combinations

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
            
            # Warnsdorff: sort by number of remaining options
            scored = []
            for v in candidates:
                opts = sum(1 for w in avail[v] if w != last and w in remaining)
                scored.append((opts, v))
            
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
    
    def find_cycle_random(vertices, used_edges, deadline, num_restarts=200):
        """Try multiple random starts to find a Hamiltonian cycle."""
        k = len(vertices)
        if k < 3:
            return None
        
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
                
                scored = []
                for v in candidates:
                    opts = sum(1 for w in avail[v] if w != last and w in remaining)
                    scored.append((opts, v))
                
                scored.sort()
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
        
        max_attempts = min(max_possible, 10)
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            cycle = find_hamiltonian_cycle(vertices, used_edges, deadline)
            if cycle is None:
                cycle = find_cycle_random(vertices, used_edges, deadline, num_restarts=200)
            
            if cycle is None:
                break
            
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
    
    # Generate candidate subsets: full set, then smaller dense subsets
    # For >3 cycles, we need smaller vertex sets with high edge density
    # k vertices need at least (best_count+1)*k edges
    # Try all subsets of size 5-10 that are dense
    
    subsets_to_try = [all_vertices]
    
    # Try removing 1-2 vertices from full set
    for removed in range(n):
        subsets_to_try.append(all_vertices - {removed})
    
    # For 4+ cycles, need smaller sets. Try all 6-8 vertex subsets
    # that have enough edges
    target_size_min = 5
    target_size_max = min(10, n)
    
    # Generate promising subsets by greedy density
    all_v_list = list(range(n))
    
    # Try subsets of various sizes
    for size in range(target_size_min, target_size_max + 1):
        if time.time() > deadline:
            break
        # Use a few random subsets plus structured ones
        rng = random.Random(42)
        tried = set()
        for _ in range(200):
            if time.time() > deadline:
                break
            subset = set(rng.sample(all_v_list, size))
            frozen = frozenset(subset)
            if frozen in tried:
                continue
            tried.add(frozen)
            
            # Check edge count
            induced_edges = set()
            for u in subset:
                for v in adj[u]:
                    if v in subset:
                        induced_edges.add((min(u, v), max(u, v)))
            if len(induced_edges) // size > best_count:
                try_vertex_subset(subset, deadline)
    
    # Also try structured subsets: consecutive vertices, etc.
    for start_idx in range(n - 9):
        for size in range(5, 11):
            if start_idx + size > n:
                break
            subset = set(range(start_idx, start_idx + size))
            induced_edges = set()
            for u in subset:
                for v in adj[u]:
                    if v in subset:
                        induced_edges.add((min(u, v), max(u, v)))
            if len(induced_edges) // size > best_count:
                try_vertex_subset(subset, deadline)
    
    # Return best result
    if best_cycles:
        return [[int(v) for v in cycle] for cycle in best_cycles]
    else:
        cycle = find_hamiltonian_cycle(all_vertices, set(), deadline)
        if cycle:
            return [[int(v) for v in cycle]]
        return [[0, 1, 2]]

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}
