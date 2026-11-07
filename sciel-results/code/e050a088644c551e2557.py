import time
import random

P = {'n': 50, 'edges': [[0,1],[0,7],[0,43],[0,49],[1,2],[1,8],[1,44],[2,3],[2,9],[2,45],[3,4],[3,10],[3,46],[4,5],[4,11],[4,47],[5,6],[5,12],[5,48],[6,7],[6,13],[6,49],[7,8],[7,14],[8,9],[8,15],[9,10],[9,16],[10,11],[10,17],[11,12],[11,18],[12,13],[12,19],[13,14],[13,20],[14,15],[14,21],[15,16],[15,22],[16,17],[16,23],[17,18],[17,24],[18,19],[18,25],[19,20],[19,26],[20,21],[20,27],[21,22],[21,28],[22,23],[22,29],[23,24],[23,30],[24,25],[24,31],[25,26],[25,32],[26,27],[26,33],[27,28],[27,34],[28,29],[28,35],[29,30],[29,36],[30,31],[30,37],[31,32],[31,38],[32,33],[32,39],[33,34],[33,40],[34,35],[34,41],[35,36],[35,42],[36,37],[36,43],[37,38],[37,44],[38,39],[38,45],[39,40],[39,46],[40,41],[40,47],[41,42],[41,48],[42,43],[42,49],[43,44],[44,45],[45,46],[46,47],[47,48],[48,49]]}

def construct():
    start = time.time()
    deadline = start + 80
    
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    adj_list = [sorted(adj[i]) for i in range(n)]
    
    # Try k from 3 upward
    for k in range(3, 10):
        if time.time() > deadline - 5:
            break
            
        # Use backtracking with DSU
        solution = _backtrack(n, adj_list, k, deadline)
        if solution is not None:
            # Verify solution
            if _is_valid(n, adj_list, solution):
                return [int(x) for x in solution]
        
    # Fallback: trivial valid solution (unique color per vertex)
    return list(range(n))

def _backtrack(n, adj_list, k, deadline):
    colors = [-1] * n
    
    def backtrack(idx):
        if time.time() > deadline:
            return False
        if idx == n:
            return True
            
        # Heuristic: Choose unassigned vertex with most constraints
        unassigned = [i for i in range(n) if colors[i] == -1]
        if not unassigned:
            return True
            
        best_v = -1
        best_avail = []
        
        for v in unassigned:
            avail = []
            for c in range(k):
                # Check adjacency
                ok_adj = True
                for u in adj_list[v]:
                    if colors[u] == c:
                        ok_adj = False
                        break
                if not ok_adj:
                    continue
                
                # Check bicolored cycles
                valid = True
                for d in range(k):
                    if d == c:
                        continue
                    neigh_d = [u for u in adj_list[v] if colors[u] == d]
                    if len(neigh_d) >= 2:
                        # Check if any two are connected in {c, d} subgraph
                        visited = set()
                        queue = [neigh_d[0]]
                        visited.add(neigh_d[0])
                        while queue:
                            cur = queue.pop(0)
                            for w in adj_list[cur]:
                                if w in visited:
                                    continue
                                if colors[w] in (c, d):
                                    visited.add(w)
                                    queue.append(w)
                        for u in neigh_d[1:]:
                            if u in visited:
                                valid = False
                                break
                    if not valid:
                        break
                if valid:
                    avail.append(c)
            
            if not avail:
                return False # Dead end
            
            if best_v == -1 or len(avail) < len(best_avail):
                best_v = v
                best_avail = avail
        
        if best_v == -1:
            return False
            
        for c in best_avail:
            colors[best_v] = c
            if backtrack(idx + 1):
                return True
            colors[best_v] = -1
            
        return False

    if backtrack(0):
        return colors
    return None

def _is_valid(n, adj_list, colors):
    # Check adjacency
    for u, v in P['edges']:
        if colors[u] == colors[v]:
            return False
    
    # Check bicolored cycles
    used_colors = set(colors)
    for c1 in used_colors:
        for c2 in used_colors:
            if c1 >= c2:
                continue
            
            # Induced subgraph on vertices with color c1 or c2
            verts = set()
            for i in range(n):
                if colors[i] in (c1, c2):
                    verts.add(i)
            
            if not verts:
                continue
                
            # Check for cycle in this induced subgraph
            visited = set()
            has_cycle = False
            
            for start in verts:
                if start in visited:
                    continue
                stack = [(start, -1)]
                visited.add(start)
                while stack:
                    node, parent = stack.pop()
                    for nei in adj_list[node]:
                        if colors[nei] not in (c1, c2):
                            continue
                        if nei == parent:
                            continue
                        if nei in visited:
                            has_cycle = True
                            break
                        visited.add(nei)
                        stack.append((nei, node))
                    if has_cycle:
                        break
                if has_cycle:
                    break
            if has_cycle:
                return False
                
    return True
