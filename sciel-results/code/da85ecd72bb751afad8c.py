import time

N = 9
SIZE = 1 << N
DEADLINE = 115.0

neighbors = [[] for _ in range(SIZE)]
for v in range(SIZE):
    for b in range(N):
        neighbors[v].append(v ^ (1 << b))

def construct():
    start_time = time.time()
    
    # 1. Quick Greedy to get a baseline
    path_g = [0]
    used_g = [False] * SIZE
    used_g[0] = True
    C_g = [0] * SIZE
    for w in neighbors[0]:
        C_g[w] = 1
        
    while True:
        cur = path_g[-1]
        nxt = None
        # Sort by lowest index for determinism or just pick first valid
        for v in neighbors[cur]:
            if not used_g[v] and C_g[v] == 1:
                nxt = v
                break
        if nxt is None:
            break
        path_g.append(nxt)
        used_g[nxt] = True
        for w in neighbors[nxt]:
            C_g[w] += 1
            
    best_path = path_g[:]
    best_len = len(path_g)
    
    # 2. Backtracking with Warnsdorff
    path = [0]
    used = [False] * SIZE
    used[0] = True
    C = [0] * SIZE
    for w in neighbors[0]:
        C[w] = 1
        
    def backtrack():
        nonlocal best_path, best_len, start_time
        
        if time.time() - start_time > DEADLINE:
            return
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
            
        cur = path[-1]
        
        # Get valid candidates
        candidates = []
        for v in neighbors[cur]:
            if not used[v] and C[v] == 1:
                candidates.append(v)
                
        if not candidates:
            return
            
        # Warnsdorff's heuristic
        def exit_count(v):
            count = 0
            for w in neighbors[v]:
                if not used[w] and C[w] == 1:
                    count += 1
            return count
            
        candidates.sort(key=exit_count)
        
        for v in candidates:
            if time.time() - start_time > DEADLINE:
                return
                
            # Move
            path.append(v)
            used[v] = True
            for w in neighbors[v]:
                C[w] += 1
                
            backtrack()
            
            # Undo
            path.pop()
            used[v] = False
            for w in neighbors[v]:
                C[w] -= 1

    backtrack()
    
    return best_path
