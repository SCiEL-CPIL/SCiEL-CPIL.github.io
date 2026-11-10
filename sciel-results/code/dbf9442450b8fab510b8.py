import time
import math
import random
from itertools import permutations

def construct(n):
    # Deadline guard
    start_time = time.time()
    deadline = 140.0  # 140s out of 150s
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        # Known shortest is 33
        # Seed gives 33? 1!+2!+3!+4! = 1+2+6+24 = 33.
        # So seed is optimal for n=4.
        pass
        
    # For n=7, we do the TSP heuristic.
    # For smaller n, we can just use the seed or known optimal.
    
    if n < 7:
        # Use seed for n < 7 as it's fast and likely optimal or near-optimal
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
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    # n = 7 (or larger, but task is n=7)
    symbols = list(range(1, n + 1))
    perms = list(permutations(symbols))
    N = len(perms)
    
    # Map permutation to index
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    
    # Build overlap matrix
    # W[i][j] = overlap of perms[i] and perms[j]
    # Overlap: max k such that perms[i][n-k:] == perms[j][:k]
    W = [[0]*N for _ in range(N)]
    
    # Precompute suffixes and prefixes?
    # Or just compute directly.
    # N = 5040. N^2 = 25M.
    # Direct computation in Python loop might be slow.
    # Let's optimize.
    
    # We can group permutations by their prefix/suffix.
    # But for simplicity and speed in Python, we might need to be careful.
    # 25M iterations in Python is ~5-10s.
    # Let's try to speed it up.
    
    # Actually, we can compute W on the fly if we do many restarts?
    # No, precomputation is better.
    
    # Optimization:
    # For each i, we can check overlaps with all j.
    # We can use string matching?
    # Convert perms to strings.
    perm_strs = ["".join(map(str, p)) for p in perms]
    
    # Compute W
    for i in range(N):
        s_i = perm_strs[i]
        for j in range(N):
            s_j = perm_strs[j]
            # Check overlap
            # Max possible overlap is n-1
            found = 0
            for k in range(n-1, 0, -1):
                if s_i[n-k:] == s_j[:k]:
                    found = k
                    break
            W[i][j] = found
            
    if time.time() > deadline:
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
            
    # path should have length N
    # If not, something went wrong, but it should.
    
    # Calculate initial cost
    def calc_cost(path):
        cost = 0
        for i in range(len(path) - 1):
            cost += W[path[i]][path[i+1]]
        return cost
        
    current_cost = calc_cost(path)
    best_path = path[:]
    best_cost = current_cost
    
    # Simulated Annealing
    temp = 10.0
    cool = 0.99
    
    while time.time() < deadline:
        # 2-opt move
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        
        # Reverse segment i..j
        # Edges affected:
        # (i-1, i), (i, i+1) ... (j-1, j), (j, j+1)
        # New:
        # (i-1, j), (j, j-1) ... (i, i+1), (i+1, i)
        
        # Calculate delta
        delta = 0
        
        # Remove old edges
        if i > 0:
            delta -= W[path[i-1]][path[i]]
        if j < N - 1:
            delta -= W[path[j]][path[j+1]]
            
        # Add new edges
        if i > 0:
            delta += W[path[i-1]][path[j]]
        if j < N - 1:
            delta += W[path[j+1]][path[i]]
            
        # Internal edges of the reversed segment
        # Old: (k, k+1) for k in i..j-1
        # New: (k+1, k) for k in i..j-1
        for k in range(i, j):
            delta -= W[path[k]][path[k+1]]
            delta += W[path[k+1]][path[k]]
            
        if delta > 0 or random.random() < math.exp(delta / temp):
            # Accept
            path[i:j+1] = reversed(path[i:j+1])
            current_cost += delta
            if current_cost > best_cost:
                best_cost = current_cost
                best_path = path[:]
                
        temp *= cool
        if temp < 0.01:
            break

    # Reconstruct word from best_path
    word = list(perms[best_path[0]])
    for i in range(1, N):
        p_prev = perms[best_path[i-1]]
        p_curr = perms[best_path[i]]
        # Find max overlap
        overlap = W[best_path[i-1]][best_path[i]]
        # Append the rest of p_curr
        word.extend(p_curr[overlap:])
        
    return "".join(map(str, word))
