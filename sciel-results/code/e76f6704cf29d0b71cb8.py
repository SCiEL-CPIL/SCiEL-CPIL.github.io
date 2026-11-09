"""Consensus heuristic: build multiple high-quality reference tours, then
promote edges that are frequently part of these tours, weighted by tour quality."""
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

    def two_opt(t, max_passes=100):
        """Optimized 2-opt with early termination."""
        t = list(t)
        improved = True
        passes = 0
        # Pre-compute some indices for speed? Or just keep it simple.
        while improved and passes < max_passes:
            improved = False
            passes += 1
            if time.time() > deadline:
                break
            # Iterate over pairs (i, j) with i < j-1
            # We can vectorize the delta calculation partially, but for n=50,
            # a simple loop is fast enough.
            for i in range(n - 1):
                a = t[i]
                b = t[i + 1]
                # Optimization: if i+1 and i+2 are close, skip? No, standard 2-opt.
                for j in range(i + 2, n):
                    if i == 0 and j == n - 1:
                        continue
                    c = t[j]
                    d = t[(j + 1) % n]
                    # Calculate delta
                    # Current: D[a,b] + D[c,d]
                    # New: D[a,c] + D[b,d]
                    delta = (D[a, c] + D[b, d]) - (D[a, b] + D[c, d])
                    if delta < -1e-12:
                        # Reverse segment i+1 to j
                        t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
                        improved = True
                        b = t[i + 1] # Update b for next iteration if we want to be strict, 
                                     # but standard 2-opt often just continues.
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

    # Store the best tours and their lengths
    tours = []
    lengths = []
    
    rng = np.random.default_rng(42)
    
    def add_tour(tour):
        t = two_opt(tour)
        L = tour_len(t)
        tours.append(t)
        lengths.append(L)

    # 1. Deterministic start
    add_tour(nearest_neighbor(0))
    
    # 2. Randomized starts
    # We want to spend most of the time here to get a good consensus.
    # n=50 is small, so 2-opt is fast.
    num_starts = 100
    for _ in range(num_starts):
        if time.time() > deadline - 10.0:
            break
        s = int(rng.integers(n))
        add_tour(nearest_neighbor(s))
        
    # 3. Or-opt / Random Perturbation on the best few tours
    # Sort tours by length
    if tours:
        indices = np.argsort(lengths)
        best_indices = indices[:min(10, len(indices))]
        
        for _ in range(20): # Additional refinement rounds
            if time.time() > deadline - 5.0:
                break
            idx = int(rng.choice(best_indices))
            t = list(tours[idx])
            # Perform a few random 2-opt moves to perturb
            for _ in range(5):
                i = int(rng.integers(n))
                j = int(rng.integers(n))
                if i > j:
                    i, j = j, i
                if j - i >= 1:
                    t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
            add_tour(t)
            
    if not tours:
        # Fallback if something went wrong
        t = nearest_neighbor(0)
        t = two_opt(t)
        tours.append(t)
        lengths.append(tour_len(t))

    # Build the consensus heuristic
    # H[i, j] = sum over tours k of (1 / length_k) * indicator(edge (i,j) in tour k)
    # Also blend with 1/distance to ensure geometric sense
    
    H = np.zeros((n, n), dtype=float)
    
    # We can vectorize the accumulation if we represent tours as pairs, 
    # but for n=50 and ~100 tours, a loop is fine.
    for t, L in zip(tours, lengths):
        weight = 1.0 / L
        for k in range(n):
            i = t[k]
            j = t[(k + 1) % n]
            H[i, j] += weight
            # The problem is directed? 
            # "probability proportional to pheromone[i, j] * heuristic[i, j]"
            # "adds 1/length to pheromone[i, j] for each step i to j ... in travel direction only"
            # So the heuristic should also respect direction or be symmetric?
            # Usually heuristics are symmetric (1/d). 
            # However, if we want to strongly guide the ants, we can make it directed 
            # to match the direction of the good tours.
            # Let's add to the reverse direction as well with a smaller weight or same weight?
            # If the tour is good, the reverse tour is also good.
            # But ACO is directed. Let's add to both to be safe, or just the direction found.
            # The current code adds to base[i, j] = 1e4.
            # Let's add to H[i, j] and H[j, i] to be symmetric, as 1/d is symmetric.
            H[j, i] += weight

    # Blend with inverse distance
    diag = np.eye(n, dtype=bool)
    safe = D.copy()
    safe[diag] = 1.0
    inv_dist = 1.0 / safe
    
    # Normalize H to be comparable to inv_dist
    # Max H value might be small. 
    # Let's scale H so that its max is comparable to inv_dist max.
    max_H = np.max(H)
    max_inv = np.max(inv_dist)
    
    if max_H > 0:
        H_scaled = H * (max_inv / max_H)
    else:
        H_scaled = H
        
    # Combine: 0.5 * inv_dist + 0.5 * H_scaled ?
    # Or H_scaled + inv_dist?
    # The consensus should be dominant if the tours are good.
    # Let's try: H_final = H_scaled + 0.1 * inv_dist
    # This ensures that even if an edge is not in any tour, it has some attractiveness based on distance.
    
    H_final = H_scaled + 0.1 * inv_dist
    
    # Ensure positive and finite
    H_final[diag] = 1e-9
    H_final = np.maximum(H_final, 1e-9)
    H_final = np.where(np.isfinite(H_final), H_final, 1e-9)
    
    return H_final
# EVOLVE-BLOCK-END
