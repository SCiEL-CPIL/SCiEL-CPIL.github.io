import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    adj = np.asarray(adjacency, dtype=np.float64)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([])

    # Compute features
    deg = adj.sum(axis=1)
    sum_deg = adj @ deg
    
    # Optional: Triangle counts (might be slow for large N, but N=200 is fine)
    # tri = np.diag(adj @ adj @ adj)
    # For N=200, A^3 is ~8M ops, very fast.
    try:
        tri = np.diag(adj @ adj @ adj)
    except:
        tri = np.zeros(n)

    # Normalize features for random combinations
    d_norm = deg / (deg.max() + 1e-9)
    s_norm = sum_deg / (sum_deg.max() + 1e-9)
    t_norm = tri / (tri.max() + 1e-9)

    candidates = []

    # 1. Seed: Lowest degree first
    candidates.append(-deg)

    # 2. High sum of neighbor degrees (blocks hubs)
    candidates.append(sum_deg)

    # 3. Linear combinations: sum_deg - lambda * deg
    # This balances "blocking hubs" against "low degree"
    for lam in [1, 2, 5, 10, 20, 50, 100, 200, 500]:
        candidates.append(sum_deg - lam * deg)

    # 4. Linear combinations: -deg + lambda * sum_deg
    for lam in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]:
        candidates.append(-deg + lam * sum_deg)

    # 5. Average neighbor degree
    candidates.append(sum_deg / (deg + 1))

    # 6. Random linear combinations of normalized features
    # S = w1 * d_norm + w2 * s_norm + w3 * t_norm
    # We want to find weights that maximize the greedy set size
    np.random.seed(42) # Fixed seed for reproducibility in the search? 
    # Actually, using a fixed seed makes the random candidates deterministic.
    # This is good for consistent evaluation.
    
    for _ in range(50):
        w1 = np.random.uniform(-1, 1)
        w2 = np.random.uniform(-1, 1)
        w3 = np.random.uniform(-0.5, 0.5)
        # Ensure some signal
        if abs(w1) + abs(w2) + abs(w3) < 0.1:
            w2 = 1.0
        scores = w1 * d_norm + w2 * s_norm + w3 * t_norm
        candidates.append(scores)

    best_score = -np.inf
    best_scores = -deg  # Fallback to seed

    for scores in candidates:
        if time.time() > deadline - 5.0:
            break

        # Simulate greedy decoder
        # Descending order of scores
        # np.argsort is stable, so ties are broken by original index (ascending)
        # Reversing means ties are broken by original index (descending)
        # i.e., higher index picked first among ties
        order = np.argsort(scores)[::-1]
        
        selected = np.zeros(n, dtype=bool)
        count = 0
        
        for v in order:
            if not selected[v]:
                selected[v] = True
                count += 1
                # Mark neighbors as selected (blocked)
                # adj[v] is a row of 0s and 1s
                neighbors = np.nonzero(adj[v])[0]
                if len(neighbors) > 0:
                    selected[neighbors] = True

        if count > best_score:
            best_score = count
            best_scores = scores.copy()

    return best_scores
