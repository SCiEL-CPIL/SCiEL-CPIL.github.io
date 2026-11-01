import time
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # 110s limit to be safe
    
    word = [1]
    
    for k in range(2, n + 1):
        # Check time
        if time.time() > deadline - 5:
            # Fallback: use standard order
            perms, seen = [], set()
            for i in range(len(word) - k + 2):
                window = tuple(word[i:i + k - 1])
                if len(set(window)) == k - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [k] + list(p)
                if not new:
                    new = block
                else:
                    overlap = get_overlap(new, block)
                    new += block[overlap:]
            word = new
            continue

        # Extract permutations
        perms, seen = [], set()
        for i in range(len(word) - k + 2):
            window = tuple(word[i:i + k - 1])
            if len(set(window)) == k - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        N = len(perms)
        
        # Compute overlap matrix
        blocks = [list(p) + [k] + list(p) for p in perms]
        
        M = [[0]*N for _ in range(N)]
        for i in range(N):
            for j in range(N):
                if i != j:
                    M[i][j] = get_overlap(blocks[i], blocks[j])
        
        # Solve TSP
        if N <= 12:
            order = tsp_exact(perms, M, blocks)
        else:
            order = tsp_heuristic(perms, M, blocks, deadline - time.time())
            
        # Construct new word
        new = []
        prev_idx = None
        for idx in order:
            if not new:
                new = blocks[idx]
            else:
                ov = M[prev_idx][idx]
                new += blocks[idx][ov:]
            prev_idx = idx
            
        word = new
        
    return "".join(map(str, word))

def get_overlap(a, b):
    max_l = min(len(a), len(b))
    # Start from max possible overlap
    # Overlap cannot be len(a) or len(b) if they are distinct blocks
    # Max overlap is min(len(a), len(b)) - 1
    for l in range(max_l - 1, -1, -1):
        if l == 0:
            return 0
        if a[-l:] == b[:l]:
            return l
    return 0

def tsp_exact(perms, M, blocks):
    N = len(perms)
    # DP over subsets - maximize total overlap
    INF = float('inf')
    dp = [[INF]*N for _ in range(1<<N)]
    parent = [[-1]*N for _ in range(1<<N)]
    
    for i in range(N):
        dp[1<<i][i] = 0
        
    for mask in range(1, 1<<N):
        for last in range(N):
            if not (mask & (1<<last)):
                continue
            if dp[mask][last] == INF:
                continue
            for nxt in range(N):
                if mask & (1<<nxt):
                    continue
                new_mask = mask | (1<<nxt)
                new_cost = dp[mask][last] - M[last][nxt]
                if new_cost < dp[new_mask][nxt]:
                    dp[new_mask][nxt] = new_cost
                    parent[new_mask][nxt] = last
    
    # Find best ending
    full_mask = (1<<N) - 1
    best_last = 0
    best_cost = dp[full_mask][0]
    for i in range(1, N):
        if dp[full_mask][i] < best_cost:
            best_cost = dp[full_mask][i]
            best_last = i
    
    # Reconstruct path
    order = []
    mask = full_mask
    last = best_last
    while last != -1:
        order.append(last)
        prev = parent[mask][last]
        mask ^= (1<<last)
        last = prev
    
    order.reverse()
    return order

def tsp_heuristic(perms, M, blocks, time_limit):
    N = len(perms)
    start_time = time.time()
    
    # Greedy construction with multiple random starts
    best_order = None
    best_score = float('inf')
    
    for attempt in range(20):
        if time.time() - start_time > time_limit * 0.3:
            break
            
        # Random start
        start_node = random.randint(0, N-1)
        order = [start_node]
        visited = {start_node}
        
        while len(order) < N:
            last = order[-1]
            best_next = -1
            best_ov = -1
            for j in range(N):
                if j not in visited:
                    if M[last][j] > best_ov:
                        best_ov = M[last][j]
                        best_next = j
            order.append(best_next)
            visited.add(best_next)
        
        # Calculate score
        score = 0
        for i in range(N-1):
            score -= M[order[i]][order[i+1]]
        
        if score < best_score:
            best_score = score
            best_order = order[:]
    
    if best_order is None:
        return list(range(N))
    
    # Local search: 2-opt style moves
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        
        # Try moving each element to a different position
        for i in range(N):
            if time.time() - start_time > time_limit:
                break
            for j in range(N):
                if i == j or i == 0 and j == 0:
                    continue
                # Calculate delta
                old_score = 0
                for k in range(N-1):
                    old_score -= M[best_order[k]][best_order[k+1]]
                
                # Create new order
                new_order = best_order[:]
                elem = new_order.pop(i)
                new_order.insert(j, elem)
                
                new_score = 0
                for k in range(N-1):
                    new_score -= M[new_order[k]][new_order[k+1]]
                
                if new_score < old_score:
                    best_order = new_order
                    best_score = new_score
                    improved = True
    
    return best_order
