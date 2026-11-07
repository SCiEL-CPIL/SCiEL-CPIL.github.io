"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    deadline = time.time() + 85
    n = P['n']
    all_vertices = set(range(n))
    
    # Build adjacency
    adj = defaultdict(set)
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    def find_hamiltonian_cycle(used_edges, start=None):
        """Find a Hamiltonian cycle avoiding used_edges."""
        if start is None:
            start = 0
        visited = {start}
        path = [start]
        
        def backtrack():
            if len(path) == n:
                # Check if edge back to start exists and is unused
                last = path[-1]
                if start in adj[last] and (min(last, start), max(last, start)) not in used_edges:
                    return True
                return False
            
            cur = path[-1]
            neighbors = list(adj[cur])
            random.shuffle(neighbors)
            
            for nb in neighbors:
                if nb not in visited:
                    edge = (min(cur, nb), max(cur, nb))
                    if edge not in used_edges:
                        visited.add(nb)
                        path.append(nb)
                        if backtrack():
                            return True
                        path.pop()
                        visited.remove(nb)
            return False
        
        if backtrack():
            return path
        return None
    
    def try_greedy_packing():
        """Greedy: repeatedly find Hamiltonian cycles avoiding used edges."""
        used_edges = set()
        cycles = []
        
        for attempt in range(20):
            if time.time() > deadline:
                break
            # Try multiple random starts
            found = False
            for start in range(n):
                random.shuffle(list(adj.keys()))
                cycle = find_hamiltonian_cycle(used_edges, start)
                if cycle is not None:
                    # Mark all edges in this cycle as used
                    for i in range(len(cycle)):
                        u = cycle[i]
                        v = cycle[(i + 1) % len(cycle)]
                        used_edges.add((min(u, v), max(u, v)))
                    cycles.append(cycle)
                    found = True
                    break
            if not found:
                break
        
        return cycles
    
    # First try a structured approach using the circulant nature
    # The graph has edges at distances: 1, 2, 5, 4 (i.e., ±1, ±2, ±5, ±4 mod 21)
    # Let's verify
    dist_edges = defaultdict(list)
    for u, v in P['edges']:
        d = (v - u) % n
        if d > n // 2:
            d = n - d
        dist_edges[d].append((u, v))
    
    # Try to find cycles using specific distance patterns
    # A Hamiltonian cycle using only distance-d edges would be a "jump" cycle
    # For d=1: 0,1,2,...,20,0 - uses all consecutive edges
    # For d=2: 0,2,4,...,20,0 - uses all even-odd skip edges
    
    def make_jump_cycle(step):
        """Create a cycle visiting vertices 0, step, 2*step, ... mod 21"""
        if gcd(step, n) != 1:
            return None
        cycle = []
        cur = 0
        for _ in range(n):
            cycle.append(cur)
            cur = (cur + step) % n
        # Verify all edges exist
        for i in range(n):
            u = cycle[i]
            v = cycle[(i + 1) % n]
            if v not in adj[u]:
                return None
        return cycle
    
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    # Find valid jump steps
    valid_steps = []
    for s in range(1, n):
        if gcd(s, n) == 1:
            c = make_jump_cycle(s)
            if c is not None:
                valid_steps.append(s)
    
    # Greedily pick edge-disjoint jump cycles
    used_edges = set()
    cycles = []
    for s in valid_steps:
        if time.time() > deadline:
            break
        c = make_jump_cycle(s)
        if c is None:
            continue
        # Check if all edges are unused
        ok = True
        for i in range(n):
            u = c[i]
            v = c[(i + 1) % n]
            e = (min(u, v), max(u, v))
            if e in used_edges:
                ok = False
                break
        if ok:
            for i in range(n):
                u = c[i]
                v = c[(i + 1) % n]
                used_edges.add((min(u, v), max(u, v)))
            cycles.append(c)
    
    # Try to add more via backtracking search on remaining edges
    if time.time() < deadline - 5:
        extra = try_greedy_packing()
        # Merge: use jump cycles first, then try to add extras
        all_used = set()
        for c in cycles:
            for i in range(n):
                u = c[i]
                v = c[(i + 1) % n]
                all_used.add((min(u, v), max(u, v)))
        
        # Try adding extras that don't conflict
        for c in extra:
            if time.time() > deadline:
                break
            conflict = False
            for i in range(n):
                u = c[i]
                v = c[(i + 1) % n]
                e = (min(u, v), max(u, v))
                if e in all_used:
                    conflict = True
                    break
            if not conflict:
                for i in range(n):
                    u = c[i]
                    v = c[(i + 1) % n]
                    all_used.add((min(u, v), max(u, v)))
                cycles.append(c)
    
    # Fallback: if we have nothing, return a single cycle
    if not cycles:
        cycles = [list(range(n))]
    
    # Ensure at most 10 cycles
    cycles = cycles[:10]
    
    # Convert to lists of ints
    result = []
    for c in cycles:
        result.append([int(v) for v in c])
    
    return result
# EVOLVE-BLOCK-END
