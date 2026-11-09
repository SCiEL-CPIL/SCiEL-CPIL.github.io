"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
sys.setrecursionlimit(10000)
import random

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    def edge_key(u, v):
        return (min(u, v), max(u, v))

    def find_ham_cycle(used_edges_set, rng):
        """Find a Hamiltonian cycle avoiding edges in used_edges_set using randomized backtracking."""
        start = 0
        path = [start]
        visited = [False] * n
        visited[start] = True
        
        def backtrack():
            if len(path) == n:
                last = path[-1]
                if edge_key(last, start) not in used_edges_set and start in adj[last]:
                    return True
                return False
            
            last = path[-1]
            candidates = []
            for nxt in adj[last]:
                if nxt == start:
                    if len(path) == n - 1:
                        if edge_key(last, start) not in used_edges_set:
                            candidates.append(nxt)
                else:
                    if not visited[nxt] and edge_key(last, nxt) not in used_edges_set:
                        candidates.append(nxt)
            
            if not candidates:
                return False
                
            rng.shuffle(candidates)
            
            for nxt in candidates:
                path.append(nxt)
                visited[nxt] = True
                if backtrack():
                    return True
                path.pop()
                visited[nxt] = False
            
            return False
        
        if backtrack():
            return path[:]
        return None

    best_cycles = []
    max_attempts = 100  # Try multiple random restarts to ensure we find the max set
    
    for attempt in range(max_attempts):
        rng = random.Random(attempt * 12345 + 999)
        used_edges = set()
        cycles = []
        
        success = True
        for _ in range(3):
            cycle = find_ham_cycle(used_edges, rng)
            if cycle is None:
                success = False
                break
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                used_edges.add(edge_key(u, v))
            cycles.append(cycle)
            
        if success and len(cycles) == 3:
            return [list(c) for c in cycles]
            
        if len(cycles) > len(best_cycles):
            best_cycles = cycles
            
    if not best_cycles:
        # Fallback: Try deterministic once more
        rng = random.Random(0)
        cycle = find_ham_cycle(set(), rng)
        if cycle:
            return [cycle]
        return [list(range(n))]
        
    return [list(c) for c in best_cycles]
# EVOLVE-BLOCK-END
