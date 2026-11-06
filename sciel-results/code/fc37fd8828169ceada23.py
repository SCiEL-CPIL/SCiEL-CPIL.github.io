def construct():
    import time
    from math import gcd
    import random
    
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    N = 128
    # Precompute GCD matrix
    g = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i+1, N+1):
            g[i][j] = g[j][i] = gcd(i, j)
    
    def can_add(x, S):
        """Check if x can be added to S without creating a bad triple."""
        for i in range(len(S)):
            a = S[i]
            g_ax = g[a][x]
            for j in range(i+1, len(S)):
                b = S[j]
                if g[a][b] == g_ax == g[b][x]:
                    return False
        return True

    def greedy_fill(S, S_set, order):
        """Add elements from 'order' to S if valid."""
        for x in order:
            if x in S_set:
                continue
            if can_add(x, S):
                S.append(x)
                S_set.add(x)

    def local_search(S, S_set):
        """Try to add any missing element until stuck."""
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for x in range(1, N+1):
                if x in S_set:
                    continue
                if can_add(x, S):
                    S.append(x)
                    S_set.add(x)
                    improved = True
                    break # Restart to check new possibilities

    best_S = []
    best_size = 0

    # Initial deterministic greedy to get a baseline
    S = []
    S_set = set()
    greedy_fill(S, S_set, list(range(1, N+1)))
    local_search(S, S_set)
    if len(S) > best_size:
        best_size = len(S)
        best_S = list(S)
        
    # Randomized Restarts with Kicks
    num_restarts = 0
    while time.time() < deadline - 5:
        num_restarts += 1
        if num_restarts % 2 == 0:
            # Start from scratch with random order
            S = []
            S_set = set()
            order = list(range(1, N+1))
            random.shuffle(order)
            greedy_fill(S, S_set, order)
        else:
            # Start from best_S and kick
            S = list(best_S)
            S_set = set(S)
            if S:
                k = random.randint(1, 2)
                for _ in range(k):
                    if S:
                        idx = random.randint(0, len(S)-1)
                        x = S.pop(idx)
                        S_set.remove(x)
            
        # Local search to fill up
        local_search(S, S_set)
        
        # Try some "Swaps" to potentially increase size or escape local optima
        for _ in range(10):
            if time.time() > deadline - 2:
                break
            if not S:
                break
            
            x = random.choice(S)
            candidates = [y for y in range(1, N+1) if y not in S_set]
            if not candidates:
                break
            y = random.choice(candidates)
            
            # Check validity of S' = S - {x} + {y}
            valid_swap = True
            # Iterate pairs in S \ {x}
            # Optimization: just iterate S and skip x
            for i in range(len(S)):
                a = S[i]
                if a == x:
                    continue
                g_ay = g[a][y]
                for j in range(i+1, len(S)):
                    b = S[j]
                    if b == x:
                        continue
                    if g[a][b] == g_ay == g[b][y]:
                        valid_swap = False
                        break
                if not valid_swap:
                    break
            
            if valid_swap:
                # Perform swap
                S.remove(x)
                S_set.remove(x)
                S.append(y)
                S_set.add(y)
                # Re-polish
                local_search(S, S_set)
                
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = list(S)

    return best_S if best_S else [1]
