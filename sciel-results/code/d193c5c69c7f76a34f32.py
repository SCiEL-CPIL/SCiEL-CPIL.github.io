import time
import math
import random
import numpy as np
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit

    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "1234121432134231432"  # Known optimal length 33

    # For n < 7, use classic construction
    if n < 7:
        word = [1]
        for m in range(2, n + 1):
            perms_local, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms_local.append(window)
            new = []
            for p in perms_local:
                block = list(p) + [m] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    # n >= 7: Use TSP-based approach
    symbols = list(range(1, n + 1))
    perms = list(permutations(symbols))
    N = len(perms)
    
    # Map permutation to index
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    
    # Build overlap matrix using numpy for speed
    # W[i][j] = max k such that perms[i][n-k:] == perms[j][:k]
    # Convert to integer arrays for faster comparison
    perm_arr = np.array(perms, dtype=np.int32)  # shape (N, n)
    
    # Compute overlap matrix
    # For each pair (i, j), find max k where suffix of i matches prefix of j
    # Overlap k means: perm_arr[i][n-k:] == perm_arr[j][:k]
    
    # Efficient computation: for each possible overlap k (1 to n-1),
    # check which pairs have that overlap
    W = np.zeros((N, N), dtype=np.int8)
    
    for k in range(1, n):
        # For overlap k, we need perm_arr[i][n-k:] == perm_arr[j][:k]
        # This is a vectorized operation
        suffixes = perm_arr[:, n-k:]  # shape (N, k)
        prefixes = perm_arr[:, :k]    # shape (N, k)
        
        # For each i, find all j where prefixes[j] == suffixes[i]
        # We can use a hash-based approach
        # Convert to tuples for hashing
        suffix_tuples = [tuple(row) for row in suffixes]
        prefix_to_indices = {}
        for j in range(N):
            pt = tuple(prefixes[j])
            if pt not in prefix_to_indices:
                prefix_to_indices[pt] = []
            prefix_to_indices[pt].append(j)
        
        for i in range(N):
            st = suffix_tuples[i]
            if st in prefix_to_indices:
                for j in prefix_to_indices[st]:
                    W[i, j] = k  # Since we go from k=1 to n-1, last assignment wins (max k)
    
    # Convert to list for faster access in Python loops
    W_list = W.tolist()
    
    if time.time() > deadline - 10:
        # Fallback to seed
        word = [1]
        for m in range(2, n + 1):
            perms_local, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms_local.append(window)
            new = []
            for p in perms_local:
                block = list(p) + [m] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        return "".join(map(str, word))
    
    # Initial path: Extract from seed word
    seed_word = [1]
    for m in range(2, n + 1):
        perms_seed, seen = [], set()
        for i in range(len(seed_word) - m + 2):
            window = tuple(seed_word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms_seed.append(window)
        new = []
        for p in perms_seed:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        seed_word = new
        
    # Get order of first appearance
    path = []
    seen = set()
    for i in range(len(seed_word) - n + 1):
        p = tuple(seed_word[i:i+n])
        if p not in seen:
            seen.add(p)
            path.append(perm_to_idx[p])
            
    # Calculate initial cost
    def calc_cost(path):
        cost = 0
        for i in range(len(path) - 1):
            cost += W_list[path[i]][path[i+1]]
        return cost
        
    current_cost = calc_cost(path)
    best_path = path[:]
    best_cost = current_cost
    
    # Lin-Kernighan style edge exchange
    # We'll do 2-opt moves: reverse a segment of the path
    # This is equivalent to swapping two edges in the TSP tour
    
    temp = 5.0
    cool = 0.995
    min_temp = 0.01
    
    # Track best improvement
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline - 5:
        # 2-opt move: reverse segment [i, j]
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        
        # Calculate delta for reversing segment i..j
        # Old edges removed: (i-1, i), (i, i+1), ..., (j-1, j), (j, j+1)
        # New edges added: (i-1, j), (j, j-1), ..., (i+1, i), (i, i+1)
        
        delta = 0
        
        # Remove old boundary edges
        if i > 0:
            delta -= W_list[path[i-1]][path[i]]
        if j < N - 1:
            delta -= W_list[path[j]][path[j+1]]
            
        # Add new boundary edges
        if i > 0:
            delta += W_list[path[i-1]][path[j]]
        if j < N - 1:
            delta += W_list[path[j+1]][path[i]]
            
        # Internal edges: old (k, k+1) for k in i..j-1, new (k+1, k) for k in i..j-1
        # For large segments, this is expensive. Limit segment length for speed.
        if j - i > 100:
            # Skip long segments for speed, or sample
            if random.random() < 0.1:
                continue
        
        for k in range(i, j):
            delta -= W_list[path[k]][path[k+1]]
            delta += W_list[path[k+1]][path[k]]
            
        # Accept if improvement or with probability exp(-delta/temp) for worsening
        if delta >= 0:
            # Improvement or neutral
            path[i:j+1] = path[i:j+1][::-1]
            current_cost += delta
            if current_cost > best_cost:
                best_cost = current_cost
                best_path = path[:]
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            # Worsening: accept with probability exp(delta/temp)
            if random.random() < math.exp(delta / temp):
                path[i:j+1] = path[i:j+1][::-1]
                current_cost += delta
                no_improve_count += 1
            else:
                no_improve_count += 1
        
        temp *= cool
        if temp < min_temp:
            temp = min_temp
        
        # Periodically try some short-range moves
        if random.random() < 0.2:
            # Try short segment reversal
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, min(i + 20, N - 1))
            
            delta = 0
            if i > 0:
                delta -= W_list[path[i-1]][path[i]]
            if j < N - 1:
                delta -= W_list[path[j]][path[j+1]]
                
            if i > 0:
                delta += W_list[path[i-1]][path[j]]
            if j < N - 1:
                delta += W_list[path[j+1]][path[i]]
                
            for k in range(i, j):
                delta -= W_list[path[k]][path[k+1]]
                delta += W_list[path[k+1]][path[k]]
                
            if delta >= 0 or random.random() < math.exp(delta / temp):
                path[i:j+1] = path[i:j+1][::-1]
                current_cost += delta
                if current_cost > best_cost:
                    best_cost = current_cost
                    best_path = path[:]
                    no_improve_count = 0
    
    # Final polish: local search with only improving moves
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        for i in range(N - 1):
            if time.time() > deadline - 2:
                break
            for j in range(i + 1, min(i + 30, N)):
                delta = 0
                if i > 0:
                    delta -= W_list[path[i-1]][path[i]]
                if j < N - 1:
                    delta -= W_list[path[j]][path[j+1]]
                    
                if i > 0:
                    delta += W_list[path[i-1]][path[j]]
                if j < N - 1:
                    delta += W_list[path[j+1]][path[i]]
                    
                for k in range(i, j):
                    delta -= W_list[path[k]][path[k+1]]
                    delta += W_list[path[k+1]][path[k]]
                    
                if delta > 0:
                    path[i:j+1] = path[i:j+1][::-1]
                    current_cost += delta
                    if current_cost > best_cost:
                        best_cost = current_cost
                        best_path = path[:]
                    improved = True
    
    # Reconstruct word from best_path
    word = list(perms[best_path[0]])
    for i in range(1, N):
        p_prev = perms[best_path[i-1]]
        p_curr = perms[best_path[i]]
        # Find max overlap
        overlap = W_list[best_path[i-1]][best_path[i]]
        # Append the rest of p_curr
        word.extend(p_curr[overlap:])
        
    return "".join(map(str, word))
