import sys
import time
import random

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    deadline = time.time() + 50  # Safety margin
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    edge_set = set()
    for u, v in P['edges']:
        edge_set.add(tuple(sorted((u, v))))
    
    forbidden = set()
    
    def find_cycle_backtracking(start_vertex, randomize=False):
        """Find a Hamiltonian cycle using backtracking with strong pruning."""
        path = [start_vertex]
        visited = [False] * n
        visited[start_vertex] = True
        
        def avail_neighbors(u):
            result = []
            for v in adj[u]:
                if visited[v]:
                    continue
                e = tuple(sorted((u, v)))
                if e in forbidden:
                    continue
                if e not in edge_set:
                    continue
                # Count onward options for v
                opts = 0
                for w in adj[v]:
                    if not visited[w] and w != u:
                        ew = tuple(sorted((v, w)))
                        if ew not in forbidden and ew in edge_set:
                            opts += 1
                result.append((opts, v))
            return result
        
        def dfs(u, depth):
            if time.time() > deadline:
                return False
            
            if depth == n:
                e = tuple(sorted((u, start_vertex)))
                if e not in forbidden and e in edge_set:
                    return True
                return False
            
            neighbors = avail_neighbors(u)
            
            if not neighbors:
                return False

            if randomize:
                random.shuffle(neighbors)
            else:
                neighbors.sort(key=lambda x: x[0])
            
            for _, v in neighbors:
                visited[v] = True
                path.append(v)
                
                if dfs(v, depth + 1):
                    return True
                
                visited[v] = False
                path.pop()
            
            return False
        
        if dfs(start_vertex, 1):
            return path[:]
        return None

    cycles = []
    max_possible = 3
    
    # Try to find 3 cycles
    # We use a strategy of finding one, removing edges, and repeating.
    # If we get stuck, we might need to restart with a different first cycle.
    # Given the density, simple greedy should work.
    
    for attempt in range(10):
        if time.time() > deadline - 5:
            break
        
        forbidden.clear()
        cycles = []
        
        success = True
        for i in range(max_possible):
            if time.time() > deadline - 5:
                success = False
                break
            
            # Edge budget check
            remaining_edges = len(edge_set) - len(forbidden)
            if remaining_edges < n:
                success = False
                break
            
            # Try to find a cycle
            # Use randomization in later attempts to escape local optima
            cycle = None
            for sv in range(n):
                if time.time() > deadline - 5:
                    break
                cycle = find_cycle_backtracking(sv, randomize=(attempt > 0))
                if cycle is not None:
                    break
            
            if cycle is None:
                success = False
                break
            
            # Validate cycle
            valid = True
            for j in range(len(cycle)):
                u = cycle[j]
                v = cycle[(j + 1) % len(cycle)]
                e = tuple(sorted((u, v)))
                if e not in edge_set or e in forbidden:
                    valid = False
                    break
            if not valid:
                success = False
                break
            
            cycles.append(cycle)
            
            # Mark edges as forbidden
            for j in range(len(cycle)):
                u = cycle[j]
                v = cycle[(j + 1) % len(cycle)]
                forbidden.add(tuple(sorted((u, v))))
        
        if success and len(cycles) == max_possible:
            break
    
    if not cycles:
        # Fallback to a single cycle if we can't find 3
        forbidden.clear()
        cycle = find_cycle_backtracking(0, randomize=False)
        if cycle:
            cycles.append(cycle)
    
    if not cycles:
        return []
    
    # Convert to lists of ints
    return [list(c) for c in cycles]

# EVOLVE-BLOCK-END
