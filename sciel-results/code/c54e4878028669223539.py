"""Heuristic based on multi-start 2-opt with OR-opt and frequency-based consensus."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before the 330s wall-clock kill
    D = np.asarray(distance_matrix, dtype=float)
    n = D.shape[0]
    diag = np.eye(n, dtype=bool)

    def tour_len(t):
        idx = np.asarray(t, dtype=int)
        return float(np.sum(D[idx, np.roll(idx, -1)]))

    def two_opt(t, max_passes=200):
        """Vectorized 2-opt: find all improving reversals per pass, apply the best one."""
        t = np.asarray(t, dtype=int).copy()
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            if time.time() > deadline - 5.0:
                break
            # Precompute edge arrays
            edges = np.roll(t, -1)
            # For each pair (i, j) with j >= i+2, compute delta
            # delta(i,j) = D[t[i],t[j]] + D[t[i+1],t[(j+1)%n]] - D[t[i],t[i+1]] - D[t[j],t[(j+1)%n]]
            # Build index arrays for all valid (i,j) pairs
            i_idx, j_idx = np.where(np.triu(np.ones((n, n), dtype=bool), k=2))
            # Exclude the wrap-around pair (i=0, j=n-1)
            mask = ~((i_idx == 0) & (j_idx == n - 1))
            i_idx = i_idx[mask]
            j_idx = j_idx[mask]
            if len(i_idx) == 0:
                break
            a = t[i_idx]
            b = t[i_idx + 1]
            c = t[j_idx]
            d = t[(j_idx + 1) % n]
            deltas = D[a, c] + D[b, d] - D[a, b] - D[c, d]
            best_k = np.argmin(deltas)
            if deltas[best_k] < -1e-12:
                i_best = i_idx[best_k]
                j_best = j_idx[best_k]
                t[i_best + 1:j_best + 1] = t[i_best + 1:j_best + 1][::-1]
                improved = True
        return t

    def or_opt(t, max_passes=100):
        """Vectorized OR-opt: relocate single nodes to their best position."""
        t = np.asarray(t, dtype=int).copy()
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            if time.time() > deadline - 5.0:
                break
            best_delta = 0.0
            best_i = -1
            best_j = -1
            for i in range(n):
                node = t[i]
                prev = t[(i - 1) % n]
                next_ = t[(i + 1) % n]
                # Vectorize over all insertion positions j
                valid_j = np.array([j for j in range(n) if j != i and j != (i - 1) % n and j != (i + 1) % n])
                if len(valid_j) == 0:
                    continue
                a = t[valid_j]
                b = t[(valid_j + 1) % n]
                # delta = D[prev,node] + D[node,next_] + D[a,b] - D[prev,a] - D[a,node] - D[node,b]
                deltas = (D[prev, node] + D[node, next_] + D[a, b]) - (D[prev, a] + D[a, node] + D[node, b])
                k = np.argmin(deltas)
                if deltas[k] < best_delta - 1e-12:
                    best_delta = deltas[k]
                    best_i = i
                    best_j = valid_j[k]
            if best_i >= 0:
                node = t[best_i]
                np.delete(t, best_i)
                t = np.insert(t, best_j + 1, node)
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
        return np.array(tour, dtype=int)

    def random_tour(seed):
        rng = np.random.default_rng(seed)
        t = np.arange(n)
        rng.shuffle(t)
        return t

    rng = np.random.default_rng(12345)
    tours = []
    
    # Generate diverse tours
    # 1. Nearest neighbor from all nodes
    for s in range(n):
        if time.time() > deadline - 30.0:
            break
        t = nearest_neighbor(s)
        t = two_opt(t, max_passes=200)
        t = or_opt(t, max_passes=100)
        tours.append(t)
        
    # 2. Add random starts to catch different local optima
    num_random_starts = 120
    for i in range(num_random_starts):
        if time.time() > deadline - 30.0:
            break
        s = int(rng.integers(n))
        t = nearest_neighbor(s)
        t = two_opt(t, max_passes=200)
        t = or_opt(t, max_passes=100)
        tours.append(t)
        
    # 3. Add some fully random tours
    num_full_random = 50
    for i in range(num_full_random):
        if time.time() > deadline - 30.0:
            break
        t = random_tour(1000 + i)
        t = two_opt(t, max_passes=200)
        t = or_opt(t, max_passes=100)
        tours.append(t)

    if not tours:
        t = nearest_neighbor(0)
        t = two_opt(t, max_passes=200)
        t = or_opt(t, max_passes=100)
        tours.append(t)

    # Sort tours by length
    tours_with_len = [(tour_len(t), t) for t in tours]
    tours_with_len.sort(key=lambda x: x[0])
    
    # Use the top 60% of tours to build the consensus
    num_top = max(1, int(len(tours_with_len) * 0.6))
    top_tours = [t for _, t in tours_with_len[:num_top]]
    
    # Frequency matrix with weighted contribution (better tours contribute more)
    freq = np.zeros((n, n), dtype=float)
    for rank, t in enumerate(top_tours):
        weight = 1.0 / (1.0 + rank * 0.05)  # Better tours get higher weight, less decay
        for i in range(n):
            u = t[i]
            v = t[(i + 1) % n]
            freq[u, v] += weight

    # Base heuristic: inverse distance
    safe_D = D.copy()
    safe_D[diag] = 1.0
    base = 1.0 / safe_D
    
    # Combine frequency and inverse distance
    # Heuristic = (freq + 1) * base^alpha
    alpha = 1.5
    H = (freq + 1.0) * (base ** alpha)
    
    # Ensure no zeros or non-finite values
    H[diag] = 1e-9
    H = np.where(np.isfinite(H), H, 1e-9)
    H = np.maximum(H, 1e-9)
    
    return H
# EVOLVE-BLOCK-END
