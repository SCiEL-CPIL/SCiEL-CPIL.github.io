import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    N = 12
    period = 27720  # lcm(1..12)
    
    def can_match(m, L):
        # Build candidate lists for i=1..12 using NumPy
        cands = []
        for i in range(1, N + 1):
            start = m + 1
            rem = start % i
            if rem == 0:
                first = start
            else:
                first = start + (i - rem)
            
            if first > m + L:
                return False
                
            # Generate candidates using NumPy
            lst = np.arange(first, m + L + 1, i)
            if lst.size == 0:
                return False
            cands.append(lst)
        
        # Convert to lists of Python ints for matching
        val_set = set()
        for lst in cands:
            val_set.update(lst.tolist())
        
        if len(val_set) < N:
            return False
            
        val_list = sorted(val_set)
        val_to_idx = {v: idx for idx, v in enumerate(val_list)}
        num_vals = len(val_list)
        
        adj = []
        for lst in cands:
            adj.append([val_to_idx[x] for x in lst.tolist()])
        
        matchR = [-1] * num_vals
        
        def dfs(i, visited):
            for j in adj[i]:
                if visited[j]:
                    continue
                visited[j] = True
                if matchR[j] == -1 or dfs(matchR[j], visited):
                    matchR[j] = i
                    return True
            return False
        
        result = 0
        for i in range(N):
            visited = [False] * num_vals
            if dfs(i, visited):
                result += 1
                
        return result == N

    # Search for maximum L
    # Iterate L from 144 down to 1
    # For each L, check if any m works
    # If yes, return that m (since we want to maximize L, and this is the highest L)
    
    for L in range(144, 0, -1):
        # Check if time is running out
        if time.time() > deadline - 5:
            break
            
        for m in range(period):
            if m % 100 == 0:
                if time.time() > deadline - 2:
                    break
                    
            if can_match(m, L):
                return int(m)
                
    # Fallback: return 0 if no high L found in time
    return int(0)
