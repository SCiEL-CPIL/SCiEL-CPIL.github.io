import time

def construct():
    # Construct Andrasfai graph A_9
    k = 9
    n_full = k * k
    adj_full = [0] * n_full
    
    for i in range(n_full):
        x1 = i // k
        y1 = i % k
        for j in range(i + 1, n_full):
            x2 = j // k
            y2 = j % k
            if x1 != x2:
                if (y2 - y1) % k == (x1 * x2) % k:
                    adj_full[i] |= (1 << j)
                    adj_full[j] |= (1 << i)
    
    n_target = 40
    adj = [0] * n_target
    for i in range(n_target):
        adj[i] = adj_full[i] & ((1 << n_target) - 1)
        
    edges = []
    for i in range(n_target):
        for j in range(i + 1, n_target):
            if adj[i] & (1 << j):
                edges.append([i, j])
                
    return {'n': n_target, 'edges': edges}
