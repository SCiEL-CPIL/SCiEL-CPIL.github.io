import time
import random
import math
import numpy as np
from itertools import permutations

def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"

    start_time = time.time()
    deadline = 110.0 # Safety margin

    # Generate all permutations
    perms = list(permutations(range(1, n + 1)))
    N = len(perms)
    
    # Precompute Overlap Matrix M
    # M[i][j] = max k such that perms[i][-k:] == perms[j][:k]
    M = np.zeros((N, N), dtype=np.uint8)
    
    # Build prefix maps for each k from 1 to n-1
    # prefix_maps[k] maps tuple of length k to list of indices j where perms[j][:k] == key
    prefix_maps = [{} for _ in range(n)] # index 0 unused, 1 to n-1
    
    for k in range(1, n):
        pm = prefix_maps[k]
        for j, p in enumerate(perms):
            pre = p[:k]
            if pre not in pm:
                pm[pre] = []
            pm[pre].append(j)
            
    # Fill M
    # We want the MAX overlap.
    # Iterate k from n-1 down to 1.
    # If M[i][j] is 0, it means we haven't found a larger overlap yet.
    for k in range(n - 1, 0, -1):
        pm = prefix_maps[k]
        for i in range(N):
            suf = perms[i][n - k:]
            if suf in pm:
                js = pm[suf]
                for j in js:
                    if M[i][j] == 0:
                        M[i][j] = k
                        
    # Greedy Initial Solution
    visited = [False] * N
    order = []
    current = 0
    visited[0] = True
    order.append(0)
    
    # Precompute unvisited lists? No, just iterate.
    # To speed up greedy, we can just scan. N=5040, N^2 = 25M.
    # 25M simple ops in Python is ~1-2s.
    
    for _ in range(N - 1):
        best_next = -1
        best_ov = -1
        row = M[current]
        for j in range(N):
            if not visited[j]:
                ov = row[j]
                if ov > best_ov:
                    best_ov = ov
                    best_next = j
        if best_next == -1:
            # Fallback: pick any unvisited
            for j in range(N):
                if not visited[j]:
                    best_next = j
                    break
        order.append(best_next)
        visited[best_next] = True
        current = best_next
        
    current_order = order
    # Calculate initial score (sum of overlaps)
    current_score = 0
    for i in range(N - 1):
        current_score += M[current_order[i]][current_order[i+1]]
        
    best_order = current_order.copy()
    best_score = current_score
    
    # Simulated Annealing
    T = 5.0
    min_T = 0.01
    cooling = 0.999
    
    # Use local variables for speed
    M_local = M
    order_local = current_order
    score_local = current_score
    
    while time.time() - start_time < deadline:
        # Random swap
        # Pick two distinct indices
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        
        # Calculate delta
        # Edges affected:
        # (i-1, i), (i, i+1)
        # (j-1, j), (j, j+1)
        # Note: if j == i+1, edges overlap.
        
        old_score = 0
        new_score = 0
        
        # Helper logic inline for speed
        
        # Get neighbors
        prev_i = order_local[i-1] if i > 0 else None
        next_i = order_local[i+1] if i < N-1 else None
        prev_j = order_local[j-1] if j > 0 else None
        next_j = order_local[j+1] if j < N-1 else None
        
        curr_i = order_local[i]
        curr_j = order_local[j]
        
        # Old edges
        if prev_i is not None:
            old_score += M_local[prev_i][curr_i]
        if next_i is not None:
            old_score += M_local[curr_i][next_i]
            
        if prev_j is not None:
            old_score += M_local[prev_j][curr_j]
        if next_j is not None:
            old_score += M_local[curr_j][next_j]
            
        # Perform swap
        order_local[i] = curr_j
        order_local[j] = curr_i
        
        # New edges
        if prev_i is not None:
            new_score += M_local[prev_i][curr_j]
        if next_i is not None:
            new_score += M_local[curr_j][next_i]
            
        if prev_j is not None:
            new_score += M_local[prev_j][curr_i]
        if next_j is not None:
            new_score += M_local[curr_i][next_j]
            
        delta = new_score - old_score
        
        # Acceptance
        if delta > 0 or (T > min_T and random.random() < math.exp(delta / T)):
            score_local += delta
            if score_local > best_score:
                best_score = score_local
                best_order = order_local.copy()
        else:
            # Revert
            order_local[i] = curr_i
            order_local[j] = curr_j
            
        T *= cooling
        if T < min_T:
            T = min_T

    # Construct final word
    final_word = list(perms[best_order[0]])
    for i in range(1, len(best_order)):
        prev_idx = best_order[i-1]
        curr_idx = best_order[i]
        prev = perms[prev_idx]
        curr = perms[curr_idx]
        
        max_ov = M[prev_idx][curr_idx]
        final_word += curr[max_ov:]
        
    return "".join(map(str, final_word))
