import math

def is_square(n):
    if n < 0:
        return False
    r = int(math.isqrt(n))
    return r * r == n

def construct():
    # 1. Filter valid vertices (2*v not square)
    valid_vertices = []
    for i in range(1, 257):
        if not is_square(2 * i):
            valid_vertices.append(i)
            
    V = set(valid_vertices)
    
    # 2. Build adjacency list
    adj = {v: set() for v in V}
    v_list = list(V)
    for i in range(len(v_list)):
        u = v_list[i]
        for j in range(i + 1, len(v_list)):
            v = v_list[j]
            if is_square(u + v):
                adj[u].add(v)
                adj[v].add(u)
                
    # Heuristic 1: Greedy Min-Degree Add
    def greedy_add():
        U = set(V)
        S = []
        while U:
            # Find min degree in U
            best_v = None
            min_deg = float('inf')
            for v in U:
                deg = len(adj[v] & U)
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
            S.append(best_v)
            # Remove best_v and neighbors
            U -= (adj[best_v] | {best_v})
        return S

    # Heuristic 2: Greedy Max-Degree Remove
    def greedy_remove():
        S = set(V)
        # Calculate initial degrees within S
        while True:
            # Find vertex with max conflicts in S
            max_conf = -1
            worst_v = None
            for v in S:
                conf = len(adj[v] & S)
                if conf > max_conf:
                    max_conf = conf
                    worst_v = v
            if max_conf == 0:
                break
            S.remove(worst_v)
        return list(S)

    sol1 = greedy_add()
    sol2 = greedy_remove()
    
    if len(sol1) >= len(sol2):
        return sol1
    else:
        return sol2
