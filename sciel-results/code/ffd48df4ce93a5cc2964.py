import time
import random

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        u, v = e
        base_edges.add((u, v))
        base_edges.add((v, u))
        
    def solve_once():
        adj = {i: set() for i in range(n)}
        for u, v in P['edges']:
            adj[u].add(v)
            adj[v].add(u)
            
        added_edges = []
        added_set = set()
        
        def get_diameter():
            max_diam = 0
            for i in range(n):
                dist = [-1] * n
                dist[i] = 0
                queue = [i]
                while queue:
                    curr = queue.pop(0)
                    for nei in adj[curr]:
                        if dist[nei] == -1:
                            dist[nei] = dist[curr] + 1
                            queue.append(nei)
                max_diam = max(max_diam, max(dist))
            return max_diam

        def is_valid_add(u, v):
            if (u, v) in base_edges or (v, u) in base_edges:
                return False
            if (u, v) in added_set or (v, u) in added_set:
                return False
            if adj[u].intersection(adj[v]):
                return False
            return True

        current_diam = get_diameter()
        
        # Greedy Phase
        while current_diam > 4:
            best_edge = None
            best_new_diam = current_diam + 1
            
            candidates = []
            for u in range(n):
                for v in range(u + 1, n):
                    if is_valid_add(u, v):
                        candidates.append((u, v))
                        
            random.shuffle(candidates)
            
            for u, v in candidates:
                adj[u].add(v)
                adj[v].add(u)
                added_edges.append((u, v))
                added_set.add((u, v))
                added_set.add((v, u))
                
                new_diam = get_diameter()
                
                if new_diam < best_new_diam:
                    best_new_diam = new_diam
                    best_edge = (u, v)
                
                added_edges.pop()
                added_set.remove((u, v))
                added_set.remove((v, u))
                adj[u].remove(v)
                adj[v].remove(u)
                
            if best_edge is None:
                break
                
            u, v = best_edge
            adj[u].add(v)
            adj[v].add(u)
            added_edges.append((u, v))
            added_set.add((u, v))
            added_set.add((v, u))
            current_diam = best_new_diam

        # Local Search Phase
        changed = True
        while changed:
            changed = False
            for i in range(len(added_edges) - 1, -1, -1):
                u, v = added_edges[i]
                adj[u].remove(v)
                adj[v].remove(u)
                added_edges.pop(i)
                added_set.remove((u, v))
                added_set.remove((v, u))
                
                if get_diameter() <= 4:
                    changed = True
                else:
                    added_edges.insert(i, (u, v))
                    added_set.add((u, v))
                    added_set.add((v, u))
                    adj[u].add(v)
                    adj[v].add(u)

        return added_edges

    best_solution = None
    start_time = time.time()
    
    # Run multiple restarts
    while time.time() - start_time < 50:
        solution = solve_once()
        if best_solution is None or len(solution) < len(best_solution):
            best_solution = solution

    if best_solution is None:
        # Fallback to seed-like solution if something goes wrong
        return [[u, v] for u in range(n) for v in range(u+1, n) if (u-v)%2 and (u,v) not in base_edges]

    return [[u, v] for u, v in best_solution]
