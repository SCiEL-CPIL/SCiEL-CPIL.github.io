"""Heuristic based on consensus of multiple optimized tours (frequency + inverse distance)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before the 330s wall-clock kill
    D = np.asarray(distance_matrix, dtype=float)
    n = D.shape[0]

    def tour_len(t):
        idx = np.asarray(t, dtype=int)
        return float(np.sum(D[idx, np.roll(idx, -1)]))

    def two_opt(t, max_passes=50):
        t = list(t)
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            if time.time() > deadline:
                break
            for i in range(n - 1):
                for j in range(i + 2, n):
                    if i == 0 and j == n - 1:
                        continue
                    a, b, c, d = t[i], t[i + 1], t[j], t[(j + 1) % n]
                    delta = (D[a, c] + D[b, d]) - (D[a, b] + D[c, d])
                    if delta < -1e-12:
                        t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
                        improved = True
        return t

    def nearest_neighbor(start):
        visited = np.zeros(n, dtype=bool)
        visited[start] = True
        tour = [start]
        cur = start
        for _ in range(n - 1):
            d = D[cur, :].copy()
            d[visited] = np.inf
            nxt = int(np.argmin(d))
            tour.append(nxt)
            visited[nxt] = True
            cur = nxt
        return tour

    rng = np.random.default_rng(42)
    tours = []
    
    # Generate diverse tours
    # 1. Nearest neighbor from all nodes (or a subset if n is large, but n=50 is small)
    # We'll do 50 NN starts (one per node) to ensure diversity
    for s in range(n):
        if time.time() > deadline - 10.0:
            break
        t = nearest_neighbor(s)
        t = two_opt(t)
        tours.append(t)
        
    # 2. Add some random starts to catch different local optima
    num_random_starts = 50
    for _ in range(num_random_starts):
        if time.time() > deadline - 10.0:
            break
        s = int(rng.integers(n))
        t = nearest_neighbor(s)
        t = two_opt(t)
        tours.append(t)

    if not tours:
        # Fallback if we timed out immediately
        t = nearest_neighbor(0)
        t = two_opt(t)
        tours.append(t)

    # Count edge frequencies in the best 50% of tours (or all if fewer)
    # First, sort tours by length to keep only the best ones for the heuristic
    tours_with_len = [(tour_len(t), t) for t in tours]
    tours_with_len.sort(key=lambda x: x[0])
    
    # Use the top 50% of tours to build the consensus
    num_top = max(1, len(tours_with_len) // 2)
    top_tours = [t for _, t in tours_with_len[:num_top]]
    
    # Frequency matrix
    freq = np.zeros((n, n), dtype=float)
    for t in top_tours:
        for i in range(n):
            u = t[i]
            v = t[(i + 1) % n]
            freq[u, v] += 1.0

    # Base heuristic: inverse distance
    diag = np.eye(n, dtype=bool)
    safe_D = D.copy()
    safe_D[diag] = 1.0
    base = 1.0 / safe_D
    
    # Combine frequency and inverse distance
    # Heuristic = (freq + 1) * base^alpha
    # Using a power law to amplify good edges
    # Normalize freq to [0, 1] for better scaling? Or just use raw counts.
    # Let's use: H[i,j] = (freq[i,j] + 1.0) * (1.0 / D[i,j]) ** 1.5
    # This promotes edges that are both short and frequently used in good tours.
    
    alpha = 1.5
    H = (freq + 1.0) * (base ** alpha)
    
    # Ensure no zeros or non-finite values
    H[diag] = 1e-9
    H = np.where(np.isfinite(H), H, 1e-9)
    H = np.maximum(H, 1e-9)
    
    return H
# EVOLVE-BLOCK-END
