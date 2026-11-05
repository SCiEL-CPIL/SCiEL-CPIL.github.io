import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    def get_edges(adj):
        n = len(adj)
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    edges.append([i, j])
        return edges

    def find_violation(n, adj):
        # 1. Check K4
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    # Common neighbors
                    common = []
                    for k in range(n):
                        if adj[i][k] and adj[j][k]:
                            common.append(k)
                    # Check triangles in common
                    for a in range(len(common)):
                        ca = common[a]
                        for b in range(a+1, len(common)):
                            cb = common[b]
                            if adj[ca][cb]:
                                return 'K4', [i, j, ca, cb]
        
        # 2. Check IndSet6
        # Backtracking
        # To speed up, we can use a bitset or just list.
        # n=35 is small enough for list.
        
        def find_ind(cands, cur):
            if len(cur) == 6:
                return cur
            if len(cands) < 6 - len(cur):
                return None
            
            # Heuristic: pick vertex with min degree in induced subgraph?
            # Or just first.
            # Let's try to pick the one with fewest connections to other cands.
            # This helps prune.
            
            # For speed, just iterate.
            for idx, v in enumerate(cands):
                new_cands = []
                for u in cands[idx+1:]:
                    if adj[v][u] == 0:
                        new_cands.append(u)
                res = find_ind(new_cands, cur + [v])
                if res:
                    return res
            return None
            
        res = find_ind(list(range(n)), [])
        if res:
            return 'IND6', res
        return None, None

    def try_n(n, time_limit):
        t_start = time.time()
        while time.time() - t_start < time_limit:
            # Random init
            adj = [[0]*n for _ in range(n)]
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.5:
                        adj[i][j] = 1
                        adj[j][i] = 1
            
            # Repair
            repair_start = time.time()
            while time.time() - repair_start < 5: # 5s per restart
                v_type, v_verts = find_violation(n, adj)
                if v_type is None:
                    return adj
                
                if v_type == 'K4':
                    verts = v_verts
                    # Remove random edge
                    pairs = [(verts[a], verts[b]) for a in range(4) for b in range(a+1, 4)]
                    u, v = random.choice(pairs)
                    adj[u][v] = 0
                    adj[v][u] = 0
                elif v_type == 'IND6':
                    verts = v_verts
                    # Add random edge
                    pairs = [(verts[a], verts[b]) for a in range(6) for b in range(a+1, 6)]
                    u, v = random.choice(pairs)
                    adj[u][v] = 1
                    adj[v][u] = 1
                    
        return None

    # Try n=35
    # Allocate 120s
    t_limit_35 = min(120, deadline - time.time())
    if t_limit_35 > 5:
        adj = try_n(35, t_limit_35)
        if adj:
            return {'n': 35, 'edges': get_edges(adj)}
            
    # Try n=34
    t_limit_34 = min(10, deadline - time.time())
    if t_limit_34 > 2:
        adj = try_n(34, t_limit_34)
        if adj:
            return {'n': 34, 'edges': get_edges(adj)}
            
    # Try n=33
    t_limit_33 = min(5, deadline - time.time())
    if t_limit_33 > 1:
        adj = try_n(33, t_limit_33)
        if adj:
            return {'n': 33, 'edges': get_edges(adj)}
            
    # Fallback
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}

