import numpy as np
from itertools import permutations
import time
import sys

def construct(n):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    # Classic construction as baseline
    classic_word = _classic_construct(n)
    best_word = classic_word
    best_len = len(classic_word)

    # For n <= 6, classic is optimal or very close; skip heavy TSP
    # For n=7, classic is 5913, shortest known is 5905.
    if n == 7:
        try:
            t = _tsp_construct(n, deadline)
            if t is not None:
                if len(t) < best_len:
                    best_word = t
                    best_len = len(t)
        except Exception:
            pass
    elif n < 7:
        # For smaller n, TSP might find optimal quickly
        try:
            t = _tsp_construct(n, deadline)
            if t is not None:
                if len(t) < best_len:
                    best_word = t
                    best_len = len(t)
        except Exception:
            pass
            
    return best_word

def _classic_construct(n):
    """Classic recursive superpermutation construction."""
    if n == 1:
        return "1"
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find maximum overlap
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return "".join(map(str, word))

def _tsp_construct(n, deadline):
    """
    TSP-based construction:
    1. Generate all n! permutations
    2. Compute overlap matrix efficiently using vectorized operations
    3. Solve TSP approximately using multiple initial solutions + 2-opt
    4. Build word from optimal order
    """
    N = int(_factorial(n))
    
    # Generate all permutations
    perms = list(permutations(range(1, n + 1)))
    perm_strs = ["".join(map(str, p)) for p in perms]
    
    # Precompute overlap matrix efficiently
    # overlap[i][j] = longest suffix of perm_strs[i] that is prefix of perm_strs[j]
    # Use integer encoding for fast comparison
    # Encode each permutation as an integer for fast suffix/prefix matching
    
    # Build lookup: for each k (1 to n-1), map prefix string -> list of indices
    prefix_maps = [{} for _ in range(n)]
    for j in range(N):
        s_j = perm_strs[j]
        for k in range(1, n):
            pref = s_j[:k]
            if pref not in prefix_maps[k]:
                prefix_maps[k][pref] = []
            prefix_maps[k][pref].append(j)
    
    # Fill overlap matrix - vectorized approach
    # For each i, find the max k such that suffix of s_i of length k is prefix of s_j
    overlap = np.zeros((N, N), dtype=np.int8)
    
    # Process each i
    for i in range(N):
        s_i = perm_strs[i]
        # Check overlaps from largest to smallest
        for k in range(n - 1, 0, -1):
            suffix = s_i[n-k:]
            if suffix in prefix_maps[k]:
                targets = prefix_maps[k][suffix]
                for j in targets:
                    if overlap[i, j] == 0:
                        overlap[i, j] = k
    
    if time.time() > deadline - 30:
        return None
    
    # Solve TSP: maximize sum of overlaps (minimize sum of n - overlap)
    # Cost matrix: cost[i][j] = n - overlap[i][j]
    cost = n - overlap.astype(np.int32)
    
    # Multiple restarts with different starting points
    best_path = None
    best_cost = float('inf')
    
    # Try several different starting nodes
    start_candidates = [0, N//2, N-1, 1, N//4, 3*N//4]
    start_candidates = list(set(start_candidates))
    
    time_per_start = (deadline - 10 - time.time()) / max(1, len(start_candidates))
    
    for start_idx in start_candidates:
        if time.time() > deadline - 10:
            break
        
        # Nearest Neighbor Heuristic from this start
        visited = np.zeros(N, dtype=bool)
        path = [start_idx]
        visited[start_idx] = True
        curr = start_idx
        
        for _ in range(N - 1):
            # Find unvisited j with min cost[curr][j]
            c = cost[curr].copy()
            c[visited] = 999999
            next_idx = np.argmin(c)
            
            visited[next_idx] = True
            path.append(next_idx)
            curr = next_idx
        
        # Convert to numpy array
        path_np = np.array(path)
        
        # 2-Opt Improvement
        path_np = _two_opt(cost, path_np, deadline - 10)
        
        # Calculate total cost
        total_cost = 0
        for i in range(N - 1):
            total_cost += cost[path_np[i], path_np[i+1]]
        
        if total_cost < best_cost:
            best_cost = total_cost
            best_path = path_np.copy()
    
    if best_path is None:
        return None
    
    # Build the word
    s = perm_strs[best_path[0]]
    for k in range(1, N):
        p_prev = perm_strs[best_path[k-1]]
        p_curr = perm_strs[best_path[k]]
        ov = overlap[best_path[k-1]][best_path[k]]
        s += p_curr[ov:]
    
    # Verify validity
    windows = set()
    for i in range(len(s) - n + 1):
        w = s[i:i+n]
        windows.add(w)
        
    if len(windows) == N:
        return s
    else:
        return None

def _two_opt(cost, path, deadline):
    """2-opt improvement for TSP path"""
    N = len(path)
    path = path.copy()
    improved = True
    max_iters = 50
    iter_count = 0
    
    while improved and iter_count < max_iters and time.time() < deadline:
        improved = False
        iter_count += 1
        
        for i in range(1, N - 1):
            if time.time() > deadline:
                return path
            
            # j ranges from i+1 to N-2
            js = np.arange(i + 1, N - 1)
            if len(js) == 0:
                continue
            
            idx_j = path[js]
            idx_jp1 = path[js + 1]
            
            # Old edges: (i-1, i) and (j, j+1)
            # New edges: (i-1, j) and (i, j+1)
            
            idx_im1 = path[i-1]
            idx_i = path[i]
            
            old_cost_im1_i = cost[idx_im1, idx_i]
            old_cost_j_jp1 = cost[idx_j, idx_jp1]
            new_cost_im1_j = cost[idx_im1, idx_j]
            new_cost_i_jp1 = cost[idx_i, idx_jp1]
            
            old_costs = old_cost_im1_i + old_cost_j_jp1
            new_costs = new_cost_im1_j + new_cost_i_jp1
            
            better_js = js[new_costs < old_costs]
            
            if len(better_js) > 0:
                # Apply the best improvement
                min_idx = np.argmin(new_costs[new_costs < old_costs])
                best_j = better_js[min_idx]
                
                # Reverse path[i : best_j+1]
                path[i : best_j+1] = path[i : best_j+1][::-1]
                improved = True
    
    return path

def _factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
