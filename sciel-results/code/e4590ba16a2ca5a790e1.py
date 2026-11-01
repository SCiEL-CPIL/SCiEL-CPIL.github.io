import time
import itertools
import random
import math
import numpy as np

def construct(n):
    start_time = time.time()
    timeout = 115.0  # Safety margin before 150s kill (evaluator might have overhead)
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "1234123142312431234123142312431234"
    if n == 5:
        # Recursive is optimal for n=5 (153)
        # 1!+2!+3!+4!+5! = 1+2+6+24+120 = 153
        pass

    # Generate all permutations
    perms = list(itertools.permutations(range(1, n+1)))
    N = len(perms)
    
    # Precompute overlap/cost matrix
    # Cost C[i][j] = n - overlap(perms[i], perms[j])
    # overlap is max k < n such that perms[i][n-k:] == perms[j][:k]
    
    # For n=7, N=5040. N^2 = 25,401,600.
    # We can store this in a numpy array of uint8.
    # 25M bytes = 25MB. OK.
    
    costs = np.zeros((N, N), dtype=np.uint8)
    
    # Optimization: 
    # For a fixed i, we can compute overlaps with all j?
    # Or just loop all pairs.
    # Python loops are slow. 25M iterations might take ~10-20s.
    # Let's try to optimize the inner loop.
    
    # Precompute suffixes?
    # For each perm p, and each k in 1..n-1, suffix p[n-k:] is a tuple.
    # We can map (suffix_tuple) to something?
    # Actually, for a fixed i, we want to find for each j the max k.
    # This is equivalent to: for each k from n-1 down to 1,
    #   find all j such that perms[j][:k] == perms[i][n-k:]
    #   and set cost[i][j] = n-k if not already set (since we go down, first hit is max k).
    
    # Let's build a dictionary for each prefix length k?
    # No, let's just use the direct comparison but optimized.
    
    # Alternative: 
    # For each k in 1..n-1:
    #   Create a dict: prefix_k -> list of indices j
    #   For each i:
    #      suffix = perms[i][n-k:]
    #      if suffix in dict:
    #          for j in dict[suffix]:
    #              if costs[i][j] == 0: # not set yet, meaning no larger overlap found
    #                  costs[i][j] = n - k
    #      # If no match, cost remains 0? No, cost is n - 0 = n.
    #      # Wait, if overlap is 0, cost is n.
    #      # So initialize costs to n.
    #      # Then if we find overlap k, set to n-k.
    #      # Since we iterate k from n-1 down to 1, the first match is the max overlap.
    #      # So we can just set it and break? No, we need to update all j.
    #      # Actually, for a fixed i and k, the suffix is unique.
    #      # The set of j matching that prefix is fixed.
    #      # We want to set costs[i][j] = n-k for all j in that set, ONLY IF costs[i][j] is still n (meaning no larger overlap was found).
    
    costs.fill(n) # Default cost is n (overlap 0)
    
    # Precompute prefix dictionaries for all k
    # prefix_dicts[k] = { prefix_tuple: [indices] }
    prefix_dicts = [{} for _ in range(n)]
    
    for k in range(1, n):
        d = prefix_dicts[k]
        for j, p in enumerate(perms):
            prefix = p[:k]
            if prefix not in d:
                d[prefix] = []
            d[prefix].append(j)
            
    # Now fill costs
    # For each k from n-1 down to 1:
    for k in range(n-1, 0, -1):
        d = prefix_dicts[k]
        for i, p in enumerate(perms):
            suffix = p[n-k:]
            if suffix in d:
                js = d[suffix]
                for j in js:
                    # If cost is still n, it means no larger overlap was found
                    # So we set it to n-k
                    if costs[i][j] == n:
                        costs[i][j] = n - k
                        
    # Check time
    if time.time() - start_time > timeout - 20:
        # Fallback to recursive
        def get_recursive_superpermutation(k):
            if k == 1:
                return [1]
            prev = get_recursive_superpermutation(k - 1)
            perms_prev = []
            seen = set()
            for i in range(len(prev) - (k - 1) + 1):
                window = tuple(prev[i:i + k - 1])
                if len(set(window)) == k - 1 and window not in seen:
                    seen.add(window)
                    perms_prev.append(window)
            new_word = []
            for p in perms_prev:
                block = list(p) + [k] + list(p)
                max_overlap = min(len(new_word), len(block))
                overlap = 0
                for o in range(max_overlap, -1, -1):
                    if o == 0:
                        overlap = 0
                        break
                    if new_word[len(new_word) - o:] == block[:o]:
                        overlap = o
                        break
                new_word += block[overlap:]
            return new_word
        rec_word = get_recursive_superpermutation(n)
        return "".join(map(str, rec_word))

    # Initialize order with recursive order
    # We need to map permutations to indices
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    
    def get_recursive_order(k):
        if k == 1:
            return [(1,)]
        prev_order = get_recursive_order(k - 1)
        new_order = []
        for p in prev_order:
            # p is a tuple of length k-1
            # In the recursive construction S_k, we insert k.
            # The order of permutations in S_k is derived from S_{k-1}.
            # Actually, we can just extract the order from the recursive word.
            pass
        # Easier: just use the recursive word extraction
        rec_word = get_recursive_superpermutation(k)
        order = []
        seen = set()
        for i in range(len(rec_word) - k + 1):
            w = tuple(rec_word[i:i+k])
            if w not in seen:
                seen.add(w)
                order.append(perm_to_idx[w])
        return order
        
    initial_order = get_recursive_order(n)
    
    # Calculate initial cost
    def calc_cost(order):
        c = 0
        for i in range(len(order) - 1):
            c += costs[order[i]][order[i+1]]
        return c
        
    current_order = initial_order[:]
    current_cost = calc_cost(current_order)
    best_order = current_order[:]
    best_cost = current_cost
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    cooling = 0.99999
    
    # We need to be careful with the time.
    # Each move involves:
    # 1. Pick 2 random indices
    # 2. Calculate delta
    # 3. Update
    
    # To speed up, we can use a list for order.
    # Accessing costs[order[i]] is fast.
    
    while time.time() - start_time < timeout - 5:
        if T < T_min:
            break
            
        # Perform a batch of moves
        # How many moves? 
        # Let's do 1000 moves per time check.
        for _ in range(1000):
            # Pick random i, j
            i = random.randint(0, N - 1)
            j = random.randint(0, N - 1)
            if i == j:
                continue
                
            # Calculate delta
            # Edges affected:
            # (i-1, i), (i, i+1) -> (i-1, j), (j, i+1)
            # (j-1, j), (j, j+1) -> (j-1, i), (i, j+1)
            # Need to handle boundaries (0 and N-1)
            
            old_cost = 0
            new_cost = 0
            
            # Left of i
            if i > 0:
                old_cost += costs[current_order[i-1]][current_order[i]]
                new_cost += costs[current_order[i-1]][current_order[j]]
            # Right of i
            if i < N - 1:
                old_cost += costs[current_order[i]][current_order[i+1]]
                new_cost += costs[current_order[j]][current_order[i+1]]
                
            # Left of j
            if j > 0:
                old_cost += costs[current_order[j-1]][current_order[j]]
                new_cost += costs[current_order[j-1]][current_order[i]]
            # Right of j
            if j < N - 1:
                old_cost += costs[current_order[j]][current_order[j+1]]
                new_cost += costs[current_order[i]][current_order[j+1]]
                
            delta = new_cost - old_cost
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                # Accept
                current_order[i], current_order[j] = current_order[j], current_order[i]
                current_cost += delta
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_order = current_order[:]
                    
        T *= cooling
        
    # Reconstruct word from best_order
    word = list(perms[best_order[0]])
    for idx in range(1, N):
        prev_p = perms[best_order[idx-1]]
        curr_p = perms[best_order[idx]]
        ov = n - costs[best_order[idx-1]][best_order[idx]]
        word.extend(curr_p[ov:])
        
    return "".join(map(str, word))
