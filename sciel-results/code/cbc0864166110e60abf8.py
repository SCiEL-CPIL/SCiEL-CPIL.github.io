import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    period = 27720
    
    best_L = 0
    best_m = 0
    
    for m in range(period):
        if time.time() > deadline:
            break
            
        # We want to find the minimum L such that a matching exists.
        # We check L from 12 to 144.
        # We can stop early if we find a match.
        
        # To speed up, we can check if L=144 works first? No, we need the MINIMUM L.
        # So we must check L=12, 13, ...
        
        # Optimization: 
        # If the current best_L is 144, we can't do better. 
        # But we need to find the m that gives 144.
        # Actually, if we find an m with L=144, that's the max possible.
        # We can break out of the m loop if we find L=144?
        # Yes, because f(12,m) <= 144 always.
        
        found_L = -1
        # Initialize structures for incremental matching?
        # It's easier to just call a function that takes m and L.
        # But to save time, let's use the incremental idea inside the L loop.
        
        adj = [[] for _ in range(12)]
        # Pre-calculate start_k for each i
        # For a given m, the multiples of i in (m, m+L] are k*i.
        # k ranges from floor(m/i)+1 to floor((m+L)/i).
        
        # We will iterate L from 12 to 144.
        # For each L, we update adj and check matching.
        
        # To avoid rebuilding adj from scratch for each L, 
        # we can maintain the current max k for each i.
        # But the matching state (match_to_right) depends on L (size of right set).
        # So we have to re-run matching anyway.
        
        # Let's just stick to the simple rebuild if it's fast enough.
        # The previous program did that and got 1.375.
        # Maybe it timed out?
        
        # Let's try to optimize the matching check itself.
        # Use bitmasks for adjacency?
        # Left nodes: 12. Right nodes: up to 144.
        # Adjacency for left node i is a bitmask of right nodes.
        # Matching: 
        # We can use a standard Hopcroft-Karp or just Kuhn.
        # With N=12, Kuhn is very fast.
        
        # The bottleneck is likely the Python loop overhead.
        
        # Let's try to vectorize or use a faster method.
        # Or just ensure we don't waste time on L values that are clearly too small?
        
        # Heuristic: 
        # If L < 12, impossible.
        # We know L >= 12.
        
        # Let's try the incremental adj update.
        
        # For fixed m:
        # current_L = 12
        # adj = []
        # For i in 1..12:
        #   find all multiples of i in (m, m+12]
        #   add to adj[i]
        # 
        # check matching(m, 12, adj)
        # if match: found_L = 12, break
        # 
        # for current_L in 13..144:
        #   for i in 1..12:
        #     new_v = m + current_L
        #     if new_v % i == 0:
        #       adj[i].append(new_v - (m+1)) # index in right set
        #   check matching(m, current_L, adj)
        #   if match: found_L = current_L, break
        
        # This avoids rebuilding adj.
        
        adj = [[] for _ in range(12)]
        # Populate for L=12
        for i in range(1, 13):
            start_k = (m // i) + 1
            end_k = (m + 12) // i
            for k in range(start_k, end_k + 1):
                v = k * i
                if m < v <= m + 12:
                    j = v - (m + 1)
                    adj[i-1].append(j)
        
        found_L = -1
        # Check L=12
        if _has_matching(adj, 12):
            found_L = 12
        else:
            for L in range(13, 145):
                # Update adj for new L
                # The new number added to the interval (m, m+L] is v_new = m + L
                v_new = m + L
                j_new = v_new - (m + 1) # which is L-1
                for i in range(1, 13):
                    if v_new % i == 0:
                        adj[i-1].append(j_new)
                
                if _has_matching(adj, L):
                    found_L = L
                    break
        
        if found_L != -1:
            if found_L > best_L:
                best_L = found_L
                best_m = m
                if best_L == 144:
                    # Can't get better
                    break

    return best_m

def _has_matching(adj, L):
    # adj is list of lists of right node indices
    # L is the number of right nodes
    match_to_right = [-1] * L
    match_from_left = [-1] * 12
    
    def dfs(i, visited):
        for j in adj[i]:
            if visited[j]:
                continue
            visited[j] = True
            if match_to_right[j] == -1 or dfs(match_to_right[j], visited):
                match_to_right[j] = i
                match_from_left[i] = j
                return True
        return False
    
    for i in range(12):
        visited = [False] * L
        if not dfs(i, visited):
            return False
    return True
