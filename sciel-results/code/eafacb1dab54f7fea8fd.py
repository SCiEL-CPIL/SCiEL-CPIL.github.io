"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import permutations

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_ham_cycle(adj, used_edges, start=0):
    """Find a Hamiltonian cycle using DFS with Warnsdorff heuristic."""
    n = P['n']
    # used_edges is a set of frozensets or sorted tuples
    def edge_key(u, v):
        return (min(u,v), max(u,v))
    
    def dfs(path, visited, remaining):
        if len(path) == n:
            # Check if last connects to start
            last = path[-1]
            e = edge_key(last, start)
            if e not in used_edges and start in adj[last]:
                return path
            return None
        cur = path[-1]
        # Get unvisited neighbors not blocked by used edges
        candidates = []
        for nb in adj[cur]:
            if not visited[nb]:
                e = edge_key(cur, nb)
                if e not in used_edges:
                    candidates.append(nb)
        if not candidates:
            return None
        # Warnsdorff: sort by number of onward options
        def score(v):
            cnt = 0
            for nb2 in adj[v]:
                if not visited[nb2]:
                    e2 = edge_key(v, nb2)
                    if e2 not in used_edges:
                        cnt += 1
            return cnt
        candidates.sort(key=score)
        for nb in candidates:
            visited.add(nb)
            result = dfs(path + [nb], visited, remaining - 1)
            if result is not None:
                return result
            visited.remove(nb)
        return None
    
    visited = {start}
    return dfs([start], visited, n - 1)

def _try_find_cycles(adj, max_cycles=3, time_limit=85.0):
    """Try to find up to max_cycles edge-disjoint Hamiltonian cycles."""
    start_time = time.time()
    used_edges = set()
    cycles = []
    
    for _ in range(max_cycles):
        if time.time() - start_time > time_limit:
            break
        # Try multiple starting points and randomizations
        found = False
        for start in range(P['n']):
            if time.time() - start_time > time_limit:
                break
            result = _find_ham_cycle(adj, used_edges, start)
            if result is not None:
                # Extract edges from this cycle
                for i in range(len(result)):
                    u = result[i]
                    v = result[(i+1) % len(result)]
                    used_edges.add((min(u,v), max(u,v)))
                cycles.append(result)
                found = True
                break
        if not found:
            break
    
    return cycles

def construct():
    start_time = time.time()
    adj = _build_adj()
    
    # Try to find 3 edge-disjoint Hamiltonian cycles
    cycles = _try_find_cycles(adj, max_cycles=3, time_limit=85.0)
    
    # If we only found fewer, try with randomized restarts
    if len(cycles) < 3 and time.time() - start_time < 85.0:
        # Try different orderings
        for trial in range(20):
            if time.time() - start_time > 85.0:
                break
            used_edges = set()
            new_cycles = []
            success = True
            for _ in range(3):
                if time.time() - start_time > 85.0:
                    success = False
                    break
                found = False
                # Shuffle starts
                starts = list(range(P['n']))
                random.shuffle(starts)
                for start in starts:
                    if time.time() - start_time > 85.0:
                        success = False
                        break
                    result = _find_ham_cycle(adj, used_edges, start)
                    if result is not None:
                        for i in range(len(result)):
                            u = result[i]
                            v = result[(i+1) % len(result)]
                            used_edges.add((min(u,v), max(u,v)))
                        new_cycles.append(result)
                        found = True
                        break
                if not found:
                    success = False
                    break
            if success and len(new_cycles) > len(cycles):
                cycles = new_cycles
                if len(cycles) == 3:
                    break
    
    # Ensure valid result
    if not cycles:
        # Fallback: return the seed's single cycle
        return [list(range(P['n']))]
    
    # Validate and convert types
    result = []
    for c in cycles:
        if isinstance(c, (list, tuple)):
            result.append([int(x) for x in c])
    
    if not result:
        return [list(range(P['n']))]
    
    return result
# EVOLVE-BLOCK-END
