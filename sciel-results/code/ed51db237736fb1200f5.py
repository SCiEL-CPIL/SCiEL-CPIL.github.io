import time
from math import gcd
import sys

def construct():
    deadline = time.time() + 55  # 55s safety margin for 60s limit
    
    N = 256
    # weights[i] = 1.0 / i for i in 2..256
    # We use bit positions 0..254 corresponding to numbers 2..256
    
    # Precompute conflict masks
    # mask[v] has bit (u-2) set if lcm(v, u) <= 256
    masks = [0] * (N + 1)
    for v in range(2, N + 1):
        m = 0
        for u in range(2, N + 1):
            if u == v:
                continue
            g = gcd(v, u)
            if v * u <= 256 * g:
                m |= (1 << (u - 2))
        masks[v] = m
        
    # Weight array indexed by bit position
    weights_bit = [0.0] * (N - 1)
    for p in range(N - 1):
        v = p + 2
        weights_bit[p] = 1.0 / v
        
    best_score = 0.0
    best_set = []
    
    # Initial candidates: all bits set for 2..256
    initial_mask = (1 << (N - 1)) - 1
    
    # Heuristic initialization to provide a strong lower bound
    # Candidate: {2} U {odd in [129, 256]}
    heuristic_set = [2] + [x for x in range(129, 257, 2)]
    
    # Also try: all primes in [129, 256] plus 2
    # Primes have no small factors, so lcm with others is large
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for d in range(3, int(n**0.5) + 1, 2):
            if n % d == 0:
                return False
        return True
    
    prime_set = [2] + [x for x in range(129, 257) if is_prime(x)]
    valid_p = True
    for i in range(len(prime_set)):
        for j in range(i+1, len(prime_set)):
            a, b = prime_set[i], prime_set[j]
            if a * b <= 256 * gcd(a, b):
                valid_p = False
                break
        if not valid_p:
            break
    if valid_p:
        score_p = sum(1.0/x for x in prime_set)
        if score_p > best_score:
            best_score = score_p
            best_set = list(prime_set)
    
    # Verify validity
    valid = True
    for i in range(len(heuristic_set)):
        for j in range(i+1, len(heuristic_set)):
            a, b = heuristic_set[i], heuristic_set[j]
            if a * b <= 256 * gcd(a, b):
                valid = False
                break
        if not valid:
            break
            
    if valid:
        best_score = sum(1.0/x for x in heuristic_set)
        best_set = list(heuristic_set)
    else:
        # Fallback: Greedy smallest first
        greedy_set = []
        for x in range(2, N + 1):
            ok = True
            for y in greedy_set:
                if x * y <= 256 * gcd(x, y):
                    ok = False
                    break
            if ok:
                greedy_set.append(x)
        if greedy_set:
            best_score = sum(1.0/x for x in greedy_set)
            best_set = list(greedy_set)
            
    sys.setrecursionlimit(5000)
    
    # Precompute weight sums for 8-bit chunks to speed up bound calculation
    # There are 255 bits, so we need ceil(255/8) = 32 chunks
    num_chunks = (N - 1 + 7) // 8
    chunk_weight_sums = [[0.0] * 256 for _ in range(num_chunks)]
    for chunk_idx in range(num_chunks):
        for subset in range(256):
            s = 0.0
            for bit in range(8):
                if subset & (1 << bit):
                    global_idx = chunk_idx * 8 + bit
                    if global_idx < N - 1:
                        s += weights_bit[global_idx]
            chunk_weight_sums[chunk_idx][subset] = s
    
    def bound(mask):
        s = 0.0
        for chunk_idx in range(num_chunks):
            subset = (mask >> (chunk_idx * 8)) & 0xFF
            s += chunk_weight_sums[chunk_idx][subset]
        return s
    
    # Let's implement the solver with improved branching strategy
    
    def solve(mask, current_score, current_set):
        nonlocal best_score, best_set
        
        if time.time() > deadline:
            return
            
        if mask == 0:
            if current_score > best_score:
                best_score = current_score
                best_set = list(current_set)
            return
            
        # Upper bound
        ub = current_score + bound(mask)
        if ub <= best_score:
            return
            
        # Choose branching variable: pick the element with highest weight among available
        # Find the lowest set bit (smallest value = highest weight)
        lsb = mask & -mask
        idx = lsb.bit_length() - 1
        v = idx + 2
        
        # Prune: if including v would exceed deadline, skip
        if time.time() > deadline:
            return
        
        # Option 1: Include v
        # Remove v and its neighbors
        new_mask_include = mask & ~(masks[v] | (1 << idx))
        
        # Try Include first (higher weight)
        solve(new_mask_include, current_score + weights_bit[idx], current_set + [v])
        
        # Option 2: Exclude v
        new_mask_exclude = mask & ~(1 << idx)
        
        # Pruning check for exclude branch
        if current_score + bound(new_mask_exclude) > best_score:
            solve(new_mask_exclude, current_score, current_set)
        
    # Run solver
    solve(initial_mask, 0.0, [])
    
    # Local search improvement: try to add elements that fit, or swap
    # Convert best_set to a set for fast lookup
    best_set_list = list(best_set)
    best_set_s = set(best_set_list)
    
    # Try to add any element not in the set that doesn't conflict
    for x in range(2, N + 1):
        if x in best_set_s:
            continue
        ok = True
        for y in best_set_list:
            if x * y <= 256 * gcd(x, y):
                ok = False
                break
        if ok:
            best_set_list.append(x)
            best_set_s.add(x)
    
    # Try swaps: replace one element with another that gives higher weight
    improved = True
    while improved:
        improved = False
        for i in range(len(best_set_list)):
            for x in range(2, N + 1):
                if x in best_set_s:
                    continue
                # Try replacing best_set_list[i] with x
                old = best_set_list[i]
                if 1.0 / x <= 1.0 / old:
                    continue  # No improvement
                # Check if x conflicts with any other element
                ok = True
                for j in range(len(best_set_list)):
                    if j == i:
                        continue
                    y = best_set_list[j]
                    if x * y <= 256 * gcd(x, y):
                        ok = False
                        break
                if ok:
                    best_set_list[i] = x
                    best_set_s.remove(old)
                    best_set_s.add(x)
                    improved = True
                    break
            if improved:
                break
    
    # Try to add elements again after swaps
    for x in range(2, N + 1):
        if x in best_set_s:
            continue
        ok = True
        for y in best_set_list:
            if x * y <= 256 * gcd(x, y):
                ok = False
                break
        if ok:
            best_set_list.append(x)
            best_set_s.add(x)
    
    return best_set_list
