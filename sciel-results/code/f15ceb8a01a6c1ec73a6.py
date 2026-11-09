# Refined Logic
passes = 0
while _elapsed() < _DEADLINE - 5:
    passes += 1
    # New random seed for each pass
    random.seed(42 + passes)
    
    # Reset to base
    adj = [0] * 16
    for a in range(3):
        for b in range(3, 16):
            adj[a] |= (1 << b)
            adj[b] |= (1 << a)
            
    candidates = []
    for i in range(16):
        for j in range(i+1, 16):
            if not ((adj[i] >> j) & 1):
                candidates.append((i, j))
                
    random.shuffle(candidates)
    
    for u, v in candidates:
        if _elapsed() > _DEADLINE:
            break
        if _check_creates_q3(adj, u, v):
            continue
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        current_count = sum(bin(x).count("1") for x in adj) // 2
        if current_count > best_count:
            best_count = current_count
            best_adj = adj[:]
            
    # If we finished the pass quickly, continue to next pass
    # If we hit deadline, break
    if _elapsed() > _DEADLINE:
        break
