import time
import random
import sys
from collections import defaultdict, deque

sys.setrecursionlimit(10000)

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Function to find a Hamiltonian cycle in the current graph (defined by adj)
    # Returns list of vertices or None
    def find_hc(adj):
        # Check degree constraints
        for v in range(n):
            if len(adj[v]) < 2:
                return None
        
        # DFS with heuristics
        visited = [False] * n
        path = [0]
        visited[0] = True
        
        def dfs(current, depth):
            if depth == n:
                # Check if we can close the cycle
                if 0 in adj[current]:
                    return True
                return False
            
            # Get unvisited neighbors
            unvisited = [v for v in adj[current] if not visited[v]]
            if not unvisited:
                return False
            
            # Heuristic: Sort by degree in remaining graph (Warnsdorff's rule)
            unvisited.sort(key=lambda v: len([x for x in adj[v] if not visited[x]]))
            
            for nxt in unvisited:
                visited[nxt] = True
                path.append(nxt)
                if dfs(nxt, depth + 1):
                    return True
                path.pop()
                visited[nxt] = False
            
            return False
        
        if dfs(0, 1):
            return list(path)
        return None

    # Function to find a Hamiltonian cycle with randomized neighbor ordering
    def find_hc_random(adj, seed):
        for v in range(n):
            if len(adj[v]) < 2:
                return None
        
        visited = [False] * n
        path = [0]
        visited[0] = True
        rng = random.Random(seed)
        
        def dfs(current, depth):
            if depth == n:
                if 0 in adj[current]:
                    return True
                return False
            
            unvisited = [v for v in adj[current] if not visited[v]]
            if not unvisited:
                return False
            
            # Shuffle for randomization
            rng.shuffle(unvisited)
            
            for nxt in unvisited:
                visited[nxt] = True
                path.append(nxt)
                if dfs(nxt, depth + 1):
                    return True
                path.pop()
                visited[nxt] = False
            
            return False
        
        if dfs(0, 1):
            return list(path)
        return None

    # Function to extract a simple cycle from a set of edges
    def extract_cycle(edge_set):
        # Build adjacency from edge_set
        local_adj = defaultdict(list)
        for u, v in edge_set:
            local_adj[u].append(v)
            local_adj[v].append(u)
        
        # Find a cycle using DFS
        visited = set()
        
        def dfs_cycle(start):
            stack = [(start, [start])]
            visited_stack = {start}
            while stack:
                node, path = stack.pop()
                if len(path) >= 3 and start in local_adj[node]:
                    return path
                for nb in local_adj[node]:
                    if nb == start and len(path) >= 3:
                        return path
                    if nb not in visited_stack:
                        visited_stack.add(nb)
                        stack.append((nb, path + [nb]))
            return None
        
        # Try to find a cycle starting from each node
        for node in local_adj:
            visited = set()
            cycle = dfs_cycle(node)
            if cycle:
                return cycle
        return None

    # CSP-based approach: Assign edges to 3 groups such that each vertex has degree 2 in each group
    # This is equivalent to finding 3 edge-disjoint Hamiltonian cycles if the graph is 6-regular
    # But our graph is 5-regular, so this won't work directly.
    # Instead, we use the greedy approach with backtracking.

    best_cycles = []
    deadline = time.time() + 80  # 80s safety margin for 90s limit
    
    # Strategy 1: Greedy with deterministic HC finder
    current_adj = [set(s) for s in adj]
    cycles = []
    for i in range(3):
        hc = find_hc(current_adj)
        if hc is None:
            break
        cycles.append(hc)
        for j in range(len(hc)):
            u = hc[j]
            v = hc[(j + 1) % len(hc)]
            current_adj[u].remove(v)
            current_adj[v].remove(u)
    
    if len(cycles) > len(best_cycles):
        best_cycles = cycles
    
    # Strategy 2: Greedy with randomized HC finder
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        current_adj = [set(s) for s in adj]
        cycles = []
        for i in range(3):
            hc = find_hc_random(current_adj, attempt * 100 + i)
            if hc is None:
                break
            cycles.append(hc)
            for j in range(len(hc)):
                u = hc[j]
                v = hc[(j + 1) % len(hc)]
                current_adj[u].remove(v)
                current_adj[v].remove(u)
        
        if len(cycles) > len(best_cycles):
            best_cycles = cycles
    
    # Strategy 3: Try to find a 4th cycle if we have 3
    if len(best_cycles) == 3:
        if time.time() < deadline:
            current_adj = [set(s) for s in adj]
            for cycle in best_cycles:
                for j in range(len(cycle)):
                    u = cycle[j]
                    v = cycle[(j + 1) % len(cycle)]
                    current_adj[u].remove(v)
                    current_adj[v].remove(u)
            
            # Try to find any simple cycle in the remaining graph
            remaining_edges = set()
            for u in range(n):
                for v in current_adj[u]:
                    if u < v:
                        remaining_edges.add((u, v))
            
            # Try to find a Hamiltonian cycle in the remaining graph
            hc = find_hc(current_adj)
            if hc:
                best_cycles.append(hc)
            
            # If no Hamiltonian cycle, try to find any cycle and extend
            # For now, just stick with what we have
    
    # Ensure types are correct
    result = []
    for cycle in best_cycles:
        result.append([int(v) for v in cycle])
    
    return result

# Test the program
if __name__ == '__main__':
    start = time.time()
    result = construct()
    elapsed = time.time() - start
    print(f"Found {len(result)} cycles in {elapsed:.2f}s")
    for i, cycle in enumerate(result):
        print(f"Cycle {i+1}: {cycle}")
