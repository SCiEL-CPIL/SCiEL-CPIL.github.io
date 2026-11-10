import time
import random
import sys

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    original_edges = P['edges']
    
    def build_graph():
        remaining_edges = set()
        adj = {i: set() for i in range(n)}
        for u, v in original_edges:
            if u > v: u, v = v, u
            remaining_edges.add((u, v))
            adj[u].add(v)
            adj[v].add(u)
        return remaining_edges, adj

    def find_hamiltonian_cycle(adj, remaining_edges, deadline, randomize=False):
        path = [0]
        visited = {0}
        
        def backtrack():
            if time.time() > deadline:
                return None
            
            if len(path) == n:
                last = path[-1]
                edge_key = (0, last) if 0 < last else (last, 0)
                if edge_key in remaining_edges:
                    return path[:]
                return None
            
            curr = path[-1]
            neighbors = list(adj[curr])
            if randomize:
                random.shuffle(neighbors)
            else:
                neighbors.sort(key=lambda x: len(adj[x]))
            
            for nxt in neighbors:
                if nxt not in visited:
                    edge_key = (curr, nxt) if curr < nxt else (nxt, curr)
                    if edge_key in remaining_edges:
                        path.append(nxt)
                        visited.add(nxt)
                        
                        dead_end = False
                        unvisited = [v for v in range(n) if v not in visited]
                        if len(unvisited) > 1:
                            for v in unvisited:
                                has_unvisited_neighbor = False
                                for u in adj[v]:
                                    if u in unvisited:
                                        has_unvisited_neighbor = True
                                        break
                                if not has_unvisited_neighbor:
                                    dead_end = True
                                    break
                        
                        if not dead_end:
                            res = backtrack()
                            if res is not None:
                                return res
                        
                        path.pop()
                        visited.remove(nxt)
            
            return None
        
        return backtrack()

    start_time = time.time()
    deadline = start_time + 55.0
    
    best_cycles = []
    
    # Try deterministic first
    remaining_edges, adj = build_graph()
    cycles_found = []
    
    while True:
        if time.time() > deadline:
            break
        cycle = find_hamiltonian_cycle(adj, remaining_edges, deadline, randomize=False)
        if cycle is None:
            break
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            edge_key = (u, v) if u < v else (v, u)
            if edge_key in remaining_edges:
                remaining_edges.remove(edge_key)
                adj[u].remove(v)
                adj[v].remove(u)
        cycles_found.append(cycle)
        if len(cycles_found) >= 3:
            break
            
    if len(cycles_found) > len(best_cycles):
        best_cycles = cycles_found
        
    # If we didn't get 3, try random restarts
    if len(best_cycles) < 3 and time.time() < deadline - 5.0:
        for _ in range(10):
            if time.time() > deadline - 2.0:
                break
            remaining_edges, adj = build_graph()
            cycles_found = []
            while True:
                if time.time() > deadline - 1.0:
                    break
                cycle = find_hamiltonian_cycle(adj, remaining_edges, deadline - 1.0, randomize=True)
                if cycle is None:
                    break
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    edge_key = (u, v) if u < v else (v, u)
                    if edge_key in remaining_edges:
                        remaining_edges.remove(edge_key)
                        adj[u].remove(v)
                        adj[v].remove(u)
                cycles_found.append(cycle)
                if len(cycles_found) >= 3:
                    break
                    
            if len(cycles_found) > len(best_cycles):
                best_cycles = cycles_found
                if len(best_cycles) == 3:
                    break

    if not best_cycles:
        return [list(range(n))]
        
    result = []
    for c in best_cycles:
        result.append([int(v) for v in c])
        
    return result
# EVOLVE-BLOCK-END
