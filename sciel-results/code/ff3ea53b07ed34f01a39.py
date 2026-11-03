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
    2. Compute overlap matrix efficiently
    3. Solve TSP approximately using nearest-neighbor + 2-opt
    4. Build word from optimal order
    """
    N = int(_factorial(n))
    
    # Generate all permutations
    perms = list(permutations(range(1, n + 1)))
    perm_strs = ["".join(map(str, p)) for p in perms]
    
    # Precompute overlap matrix
    # overlap[i][j] = longest suffix of perm_strs[i] that is prefix of perm_strs[j]
    overlap = np.zeros((N, N), dtype=np.int8)
    
    # Build prefix maps for efficient lookup
    # prefix_maps[k] maps string of length k to list of indices
    prefix_maps = [{} for _ in range(n)]
    for j in range(N):
        s_j = perm_strs[j]
        for k in range(1, n):
            pref = s_j[:k]
            if pref not in prefix_maps[k]:
                prefix_maps[k][pref] = []
            prefix_maps[k][pref].append(j)
            
    # Fill overlap matrix
    for i in range(N):
        s_i = perm_strs[i]
        # Check overlaps from largest to smallest
        for k in range(n - 1, 0, -1):
            suffix = s_i[n-k:]
            if suffix in prefix_maps[k]:
                targets = prefix_maps[k][suffix]
                for j in targets:
                    # Since we iterate k downwards, the first match is the max overlap
                    # But we need to be careful: if overlap[i][j] is already set, it's larger
                    if overlap[i][j] == 0:
                        overlap[i][j] = k
                        
    if time.time() > deadline - 30:
        return None
    
    # Solve TSP: maximize sum of overlaps (minimize sum of n - overlap)
    
    # Nearest Neighbor Heuristic
    visited = np.zeros(N, dtype=bool)
    path = [0]
    visited[0] = True
    curr = 0
    
    for _ in range(N - 1):
        # Find unvisited j with max overlap[curr][j]
        ov = overlap[curr].copy()
        ov[visited] = -1
        next_idx = np.argmax(ov)
        
        visited[next_idx] = True
        path.append(next_idx)
        curr = next_idx
        
        if time.time() > deadline - 20:
            return None
            
    # Convert path to numpy array for faster indexing in 2-opt
    path_np = np.array(path)
    
    # 2-Opt Improvement
    # We iterate until no improvement or time runs out
    improved = True
    max_iters = 100
    iter_count = 0
    
    while improved and iter_count < max_iters and time.time() < deadline - 10:
        improved = False
        iter_count += 1
        
        for i in range(1, N - 1):
            if time.time() > deadline - 10:
                break
                
            # Vectorized 2-opt check for fixed i
            # Old edges: (i-1, i) and (j, j+1)
            # New edges: (i-1, j) and (i, j+1)
            
            # Indices
            idx_im1 = path_np[i-1]
            idx_i = path_np[i]
            
            # j ranges from i+1 to N-2 (since j+1 must exist)
            js = np.arange(i + 1, N - 1)
            if len(js) == 0:
                continue
                
            idx_j = path_np[js]
            idx_jp1 = path_np[js + 1]
            
            # Costs
            # Cost is n - overlap
            # We want new_cost < old_cost
            # (n - O[i-1][j]) + (n - O[i][j+1]) < (n - O[i-1][i]) + (n - O[j][j+1])
            
            old_cost_im1_i = n - overlap[idx_im1, idx_i]
            
            # Vectorized
            old_cost_j_jp1 = n - overlap[idx_j, idx_jp1]
            new_cost_im1_j = n - overlap[idx_im1, idx_j]
            new_cost_i_jp1 = n - overlap[idx_i, idx_jp1]
            
            old_costs = old_cost_im1_i + old_cost_j_jp1
            new_costs = new_cost_im1_j + new_cost_i_jp1
            
            # Find indices where new < old
            better_js = js[new_costs < old_costs]
            
            if len(better_js) > 0:
                # Apply the first improvement found (or best?)
                # Let's just apply one to keep it simple, or the best one
                # Applying the best one for this i
                min_new_cost = np.min(new_costs[new_costs < old_costs])
                # Actually, just taking the first one is fine for local search
                best_j = better_js[0]
                
                # Reverse path[i : best_j+1]
                path_np[i : best_j+1] = path_np[i : best_j+1][::-1]
                improved = True
                
    # Reconstruct path list from numpy array
    path = list(path_np)
    
    # Build the word
    s = perm_strs[path[0]]
    for k in range(1, N):
        p_prev = perm_strs[path[k-1]]
        p_curr = perm_strs[path[k]]
        # Find overlap
        ov = overlap[path[k-1]][path[k]]
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

def _factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
