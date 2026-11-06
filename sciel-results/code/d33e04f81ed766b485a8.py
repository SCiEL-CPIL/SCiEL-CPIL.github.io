"""Rewritten program with 2-opt + multi-start + insertion search."""
import random

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    adj_list = [sorted(adj[i]) for i in range(n)]
    
    def is_valid(path):
        for i in range(len(path) - 1):
            if path[i + 1] not in adj[path[i]]:
                return False
        return True
    
    def try_extend(path):
        changed = True
        while changed:
            changed = False
            end = path[-1]
            for v in adj_list[end]:
                if v not in path:
                    path.append(v)
                    changed = True
                    break
            start = path[0]
            for v in adj_list[start]:
                if v not in path:
                    path.insert(0, v)
                    changed = True
                    break
        return path
    
    def two_opt(path):
        n_path = len(path)
        if n_path < 4:
            return path
        improved = True
        while improved:
            improved = False
            for i in range(n_path - 2):
                for j in range(i + 2, n_path):
                    if path[j] in adj[path[i]]:
                        new_path = path[:i+1] + path[i+1:j+1][::-1] + path[j+1:]
                        if is_valid(new_path):
                            path = new_path
                            improved = True
                            break
                if improved:
                    break
        return path
    
    def try_insert(path):
        in_path = set(path)
        unvisited = [v for v in range(n) if v not in in_path]
        best_new_path = path
        for v in unvisited:
            for i in range(len(path) - 1):
                a, b = path[i], path[i+1]
                if a in adj[v] and b in adj[v]:
                    new_path = path[:i+1] + [v] + path[i+1:]
                    if len(new_path) > len(best_new_path):
                        best_new_path = new_path
            if path and path[0] in adj[v]:
                new_path = [v] + path
                if len(new_path) > len(best_new_path):
                    best_new_path = new_path
            if path and path[-1] in adj[v]:
                new_path = path + [v]
                if len(new_path) > len(best_new_path):
                    best_new_path = new_path
        return best_new_path
    
    def improve(path):
        path = try_extend(path)
        path = two_opt(path)
        prev_len = -1
        while len(path) > prev_len:
            prev_len = len(path)
            path = try_insert(path)
            path = try_extend(path)
            path = two_opt(path)
        return path
    
    rng = random.Random(42)
    best_path = []
    
    # Random walk starts
    for trial in range(300):
        start = rng.randint(0, n - 1)
        path = [start]
        visited = {start}
        while True:
            end = path[-1]
            cands = [v for v in adj_list[end] if v not in visited]
            if not cands:
                break
            nxt = rng.choice(cands)
            path.append(nxt)
            visited.add(nxt)
        path = improve(path)
        if len(path) > len(best_path):
            best_path = path[:]
    
    # Deterministic starts from each vertex
    for start in range(n):
        path = [start]
        visited = {start}
        while True:
            end = path[-1]
            cands = [v for v in adj_list[end] if v not in visited]
            if not cands:
                break
            path.append(cands[0])
            visited.add(cands[0])
        path = improve(path)
        if len(path) > len(best_path):
            best_path = path[:]
    
    if not best_path:
        best_path = [0]
    
    return best_path
