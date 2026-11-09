"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import sys

# Increase recursion limit for DFS
sys.setrecursionlimit(10000)

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    best_path = []
    best_len = 0
    
    # Time limit
    start_time = time.time()
    max_time = 55  # seconds, leave buffer
    
    # Helper to find connected components of a set of nodes
    def get_components(nodes):
        if not nodes:
            return []
        nodes = set(nodes)
        components = []
        while nodes:
            node = next(iter(nodes))
            stack = [node]
            comp = set()
            while stack:
                curr = stack.pop()
                if curr in comp:
                    continue
                comp.add(curr)
                nodes.discard(curr)
                for neighbor in adj[curr]:
                    if neighbor in nodes:
                        stack.append(neighbor)
            components.append(comp)
        return components

    # Helper to get the component size reachable from a start node within a set
    def get_reachable_component_size(start, nodes):
        if start not in nodes:
            return 0
        nodes = set(nodes)
        stack = [start]
        comp = set()
        while stack:
            curr = stack.pop()
            if curr in comp:
                continue
            comp.add(curr)
            nodes.discard(curr)
            for neighbor in adj[curr]:
                if neighbor in nodes:
                    stack.append(neighbor)
        return len(comp)

    def dfs(u, path, visited):
        nonlocal best_path, best_len
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
            
        if time.time() - start_time > max_time:
            return

        # Pruning: if we've visited all nodes, we can't do better
        if len(path) == n:
            return

        unvisited = set(range(n)) - visited
        if not unvisited:
            return
            
        # Calculate upper bound
        # We can only extend into the component of unvisited nodes connected to u
        # Actually, we move to a neighbor v. Then we are in v's component of (unvisited - {v}).
        # Let's just check if any neighbor leads to a component that could beat best_len.
        
        candidates = [v for v in adj[u] if v in unvisited]
        if not candidates:
            return
            
        # Sort candidates by Warnsdorff's rule: fewest onward moves
        # But also consider component size for pruning potential
        # We want to try "dead end" like moves first? Or "safe" moves?
        # Usually, for Hamiltonian path, Warnsdorff (min degree) works well.
        
        def score_candidate(v):
            # Number of unvisited neighbors of v
            unvisited_neighbors = len(adj[v] & unvisited)
            return unvisited_neighbors
            
        candidates.sort(key=score_candidate)
        
        for v in candidates:
            if time.time() - start_time > max_time:
                return
                
            new_visited = visited | {v}
            new_unvisited = unvisited - {v}
            
            # Pruning check:
            # If we go to v, we can only continue in the component of new_unvisited connected to v.
            # Let comp_size be the size of that component.
            # Max possible path length = len(path) + 1 + comp_size
            if new_unvisited:
                comp_size = get_reachable_component_size(v, new_unvisited)
            else:
                comp_size = 0
                
            upper_bound = len(path) + 1 + comp_size
            if upper_bound <= best_len:
                continue
                
            dfs(v, path + [v], new_visited)
            
    # Try all starting vertices
    # To speed up, we can start with high-degree vertices
    starts = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
    
    for start in starts:
        if time.time() - start_time > max_time:
            break
        dfs(start, [start], {start})
        
    # Final greedy extension on the best path found
    if best_path:
        seen = set(best_path)
        # Try to extend from both ends
        for _ in range(100):
            extended = False
            # Extend from end
            for v in adj[best_path[-1]]:
                if v not in seen:
                    best_path.append(v)
                    seen.add(v)
                    extended = True
                    break
            # Extend from start
            for v in adj[best_path[0]]:
                if v not in seen:
                    best_path.insert(0, v)
                    seen.add(v)
                    extended = True
                    break
            if not extended:
                break
                
    return best_path if best_path else [0]
# EVOLVE-BLOCK-END
