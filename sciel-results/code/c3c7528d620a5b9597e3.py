import time
import random

def construct(k, g):
    if k == 4 and g == 10:
        return _solve_4_10()
    return _seed_construct(k, g)

def _solve_4_10():
    N = 320
    deadline = time.time() + 110
    
    # Helper: Generate random 4-regular graph
    def gen_graph():
        # Use configuration model with retries for simplicity
        while True:
            stubs = []
            for i in range(N):
                stubs += [i, i, i, i]
            random.shuffle(stubs)
            adj = {i: [] for i in range(N)}
            ok = True
            for i in range(0, N * 4, 2):
                u, v = stubs[i], stubs[i+1]
                if u == v:
                    ok = False
                    break
                if v in adj[u]:
                    ok = False
                    break
                adj[u].append(v)
                adj[v].append(u)
            if ok:
                return adj

    # Helper: Count 4-cycles
    def count_4_cycles(adj):
        adj_set = {i: set(adj[i]) for i in range(N)}
        count = 0
        for u in range(N):
            nbrs = adj[u]
            for i in range(len(nbrs)):
                v = nbrs[i]
                for j in range(i+1, len(nbrs)):
                    x = nbrs[j]
                    # Common neighbors of v and x
                    # u is one. Any other?
                    common = adj_set[v].intersection(adj_set[x])
                    count += max(0, len(common) - 1)
        return count // 4

    # Helper: Check girth
    def check_girth(adj):
        # BFS from each node
        for start in range(N):
            visited = {start: 0}
            queue = [(start, -1)]
            while queue:
                node, parent = queue.pop(0)
                dist = visited[node]
                if dist >= 5: # girth 10 means radius 5
                    continue
                for nb in adj[node]:
                    if nb == parent:
                        continue
                    if nb in visited:
                        # Cycle found
                        # Length is dist + visited[nb] + 1?
                        # Actually, if nb is visited, we have a cycle.
                        # The length is dist + 1 + visited[nb] if the path is simple?
                        # BFS tree: if nb is already visited, the cycle length is dist + 1 + visited[nb]
                        # But this assumes the path from start to nb is the shortest.
                        # Yes.
                        cycle_len = dist + 1 + visited[nb]
                        if cycle_len < 10:
                            return False
                    else:
                        visited[nb] = dist + 1
                        queue.append((nb, node))
        return True

    best_adj = None
    best_score = float('inf')
    
    while time.time() < deadline:
        adj = gen_graph()
        score = count_4_cycles(adj)
        
        if score < best_score:
            best_score = score
            best_adj = adj
            
        if score == 0:
            if check_girth(adj):
                edges = []
                for u in range(N):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return N, edges
        
        # Local search: try to break 4-cycles
        # Find a 4-cycle
        # ... (simplified: just restart if score doesn't improve quickly)
        # For speed, we will just do many random restarts.
        # If we find a graph with 0 4-cycles, we check girth.
        # If girth < 10, we try local search on 6/8 cycles.
        
        # Given time, random restarts might be enough if the basin is large.
        # But 320 is specific.
        
    # Fallback
    return _seed_construct(4, 10)

def _seed_construct(k, g):
    # ... (same as before)
