import time
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "121312131"
    
    perms = list(permutations(range(1, n + 1)))
    N = len(perms)
    
    # Precompute Distance Matrix D
    # D[i][j] = n - overlap(perms[i], perms[j])
    D = [[0] * N for _ in range(N)]
    for i in range(N):
        pi = perms[i]
        for j in range(N):
            pj = perms[j]
            # Find max k < n such that pi[-k:] == pj[:k]
            # Since n is small (7), we can just check k from n-1 down to 0
            max_k = 0
            for k in range(n - 1, -1, -1):
                if k == 0:
                    break
                if pi[n-k:] == pj[:k]:
                    max_k = k
                    break
            D[i][j] = n - max_k
            
    # Initial Tour: Greedy Nearest Neighbor
    visited = [False] * N
    tour = [0]
    visited[0] = True
    curr = 0
    for _ in range(N - 1):
        best_next = -1
        best_d = float('inf')
        for j in range(N):
            if not visited[j]:
                d = D[curr][j]
                if d < best_d:
                    best_d = d
                    best_next = j
        tour.append(best_next)
        visited[best_next] = True
        curr = best_next
        
    # Calculate current cost
    def get_cost(tour):
        cost = n # First node
        for k in range(N - 1):
            cost += D[tour[k]][tour[k+1]]
        return cost
        
    current_cost = get_cost(tour)
    
    # 2-Opt Local Search
    # We try to reverse segment [i, j] for 1 <= i < j <= N-1
    # Note: tour[0] is fixed as start? No, we can rotate.
    # But for simplicity, let's assume start is fixed.
    # Actually, we can try all starts by rotating the tour after optimization.
    
    improved = True
    passes = 0
    max_passes = 20
    
    while improved and passes < max_passes and time.time() < deadline:
        improved = False
        passes += 1
        for i in range(1, N - 1):
            if time.time() > deadline:
                break
            # We need to calculate the gain of reversing tour[i..j]
            # Let A = tour[i-1], B = tour[i], C = tour[j], D_next = tour[j+1]
            # Old edges: (A, B) and (C, D_next)
            # New edges: (A, C) and (B, D_next)
            # Internal edges: (B, B+1)...(C-1, C) become (C, C-1)...(B+1, B)
            
            # To do this in O(1) per (i,j) pair, we need prefix sums of reverse distances.
            # Let R[k] = D[tour[k+1]][tour[k]] (reverse distance)
            # Let F[k] = D[tour[k]][tour[k+1]] (forward distance)
            
            # We can precompute prefix sums for F and R for the current tour.
            # But the tour changes.
            
            # Given N=5040, O(N^2) per pass is 25M.
            # If we do it in Python, it might be slow.
            # Let's try to optimize the inner loop.
            
            for j in range(i + 1, N):
                if time.time() > deadline:
                    break
                # Calculate gain
                # This requires summing internal edges.
                # If we don't have prefix sums, this is O(N) per (i,j), making it O(N^3).
                # O(N^3) is 125 * 10^9. Too slow.
                
                # We MUST use prefix sums.
                # Let's update prefix sums after each successful swap?
                # That's complex.
                
                # Alternative: Use "Or-opt" (move a single node).
                # This is O(N^2) per pass, and gain calculation is O(1).
                
                # Let's switch to Or-opt 1 (move single node).
                # Remove node at i, insert at j.
                
                # Break out of j loop to do Or-opt instead?
                # Let's restart the inner loop logic for Or-opt.
                pass
        
        # If we didn't implement Or-opt above, let's do a simple pass of Or-opt.
        for i in range(N):
            if time.time() > deadline:
                break
            node = tour[i]
            tour.pop(i)
            # Try inserting at all positions j
            best_j = i
            best_gain = 0
            
            # Cost before removal:
            # D[prev][node] + D[node][next]
            prev = tour[i-1] if i > 0 else None
            next = tour[i] if i < N else None # After pop, indices shifted
            
            # This is getting messy with indices.
            
            # Let's just use a simple random restart + greedy + few swaps.
            # Given the time, a robust but simple heuristic is better.
            
            tour.insert(i, node)
            
    # Fallback: Just return the greedy result if search takes too long or is not implemented well.
    # But we need to beat 5913.
    # Greedy NN is usually quite good.
    # Let's add a simple "2-opt" that only checks adjacent swaps (2-opt with j=i+1).
    # This is O(N) per pass.
    
    # Re-implementing a clean, fast local search:
    # 1. Precompute F (forward dist) and R (reverse dist) for current tour.
    # 2. 2-Opt:
    #    For i in 1..N-2:
    #       For j in i+1..N-1:
    #          Gain = D[A][C] + D[B][D] - D[A][B] - D[C][D] + Sum(R[i..j-1]) - Sum(F[i..j-1])
    #          Where A=tour[i-1], B=tour[i], C=tour[j], D=tour[j+1]
    
    # Let's implement this properly.
    
    def compute_prefix_sums(tour):
        F = [0] * N
        R = [0] * N
        for k in range(N - 1):
            F[k] = D[tour[k]][tour[k+1]]
            R[k] = D[tour[k+1]][tour[k]]
        # Prefix sums
        PF = [0] * (N + 1)
        PR = [0] * (N + 1)
        for k in range(N):
            PF[k+1] = PF[k] + F[k]
            PR[k+1] = PR[k] + R[k]
        return PF, PR

    PF, PR = compute_prefix_sums(tour)
    
    improved = True
    passes = 0
    max_passes = 50
    
    while improved and passes < max_passes and time.time() < deadline:
        improved = False
        passes += 1
        for i in range(1, N - 1):
            if time.time() > deadline:
                break
            A = tour[i-1]
            B = tour[i]
            # Sum F[i..j-1] = PF[j] - PF[i]
            # Sum R[i..j-1] = PR[j] - PR[i]
            
            for j in range(i + 1, N):
                if time.time() > deadline:
                    break
                C = tour[j]
                D_next = tour[j+1] if j + 1 < N else None
                
                # If D_next is None, the edge (C, D_next) doesn't exist.
                # The term D[C][D_next] is 0.
                # The term D[B][D_next] is 0.
                
                cost_old_external = D[A][B]
                if D_next is not None:
                    cost_old_external += D[C][D_next]
                    
                cost_new_external = D[A][C]
                if D_next is not None:
                    cost_new_external += D[B][D_next]
                    
                internal_old = PF[j] - PF[i]
                internal_new = PR[j] - PR[i]
                
                gain = (cost_new_external + internal_new) - (cost_old_external + internal_old)
                
                if gain < 0:
                    # Reverse tour[i..j]
                    tour[i:j+1] = tour[i:j+1][::-1]
                    improved = True
                    # Recompute prefix sums?
                    # This is O(N).
                    PF, PR = compute_prefix_sums(tour)
                    break # Restart i loop? Or continue?
                    # Continuing with old PF/PR is incorrect.
                    # So we must break and restart the outer loop or recompute.
                    # Let's break both loops and restart.
                    # To do this, we can use a flag.
                    # Or, since N=5040, recomputing PF/PR is fast.
                    # But we are inside a double loop.
                    # Let's just break out and let the while loop restart.
                    # But we need to signal that we made a change.
                    # We set improved=True.
                    # We break the j loop.
                    # We should also break the i loop to recompute PF/PR cleanly.
                    break
            if improved:
                break
                
    # Reconstruct
    word = list(perms[tour[0]])
    for k in range(1, N):
        dist = D[tour[k-1]][tour[k]]
        word.extend(perms[tour[k]][dist:])
        
    return "".join(map(str, word))
