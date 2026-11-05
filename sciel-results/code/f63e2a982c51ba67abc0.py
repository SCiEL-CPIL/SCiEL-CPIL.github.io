"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    start_time = time.time()
    time_limit = 115.0  # 120s limit, keep 5s margin

    N = 512
    
    # 1. Precomputation
    # Sieve for primes
    is_prime = [True] * N
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N, i):
                is_prime[j] = False
            
    primes = [i for i in range(2, N) if is_prime[i]]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    masks = [0] * N
    scores = [0.0] * N
    for a in range(1, N):
        scores[a] = 1.0 / (N - a)
        m = a
        for p in primes:
            if p * p > m:
                break
            if m % p == 0:
                masks[a] |= (1 << prime_to_idx[p])
                while m % p == 0:
                    m //= p
        if m > 1:
            masks[a] |= (1 << prime_to_idx[m])
            
    # 2. Greedy Initialization
    selected = [] # List of [a, mask, score]
    
    for a in range(N-1, 0, -1):
        ma = masks[a]
        compatible = True
        for _, mb, _ in selected:
            if ma & mb:
                compatible = False
                break
        if compatible:
            selected.append([a, ma, scores[a]])
            
    current_score = sum(s for _, _, s in selected)
    selected_set = set(a for a, _, _ in selected)
    
    # 3. Local Search
    improved = True
    while improved:
        if time.time() - start_time > time_limit:
            break
        improved = False
        
        # Try 1-for-1 swaps
        for i in range(len(selected)):
            if time.time() - start_time > time_limit:
                break
            
            a_out, mask_out, score_out = selected[i]
            best_gain = 0
            best_a_in = -1
            
            # Iterate unselected
            for a_in in range(1, N):
                if a_in in selected_set:
                    continue
                
                ma_in = masks[a_in]
                # Check compatibility with all except i
                compat = True
                for j in range(len(selected)):
                    if j == i:
                        continue
                    if ma_in & selected[j][1]:
                        compat = False
                        break
                
                if compat:
                    gain = scores[a_in] - score_out
                    if gain > best_gain:
                        best_gain = gain
                        best_a_in = a_in
            
            if best_a_in != -1 and best_gain > 0:
                # Perform swap
                selected_set.remove(a_out)
                selected_set.add(best_a_in)
                selected[i] = [best_a_in, masks[best_a_in], scores[best_a_in]]
                current_score += best_gain
                improved = True
                break # Restart outer loop to update state
                
        if improved:
            continue
            
        # Try 1-for-2 swaps
        # Remove one, add two
        for i in range(len(selected)):
            if time.time() - start_time > time_limit:
                break
            
            a_out, mask_out, score_out = selected[i]
            
            # Find all unselected compatible with selected \ {i}
            candidates = []
            for a_in in range(1, N):
                if a_in in selected_set:
                    continue
                ma_in = masks[a_in]
                compat = True
                for j in range(len(selected)):
                    if j == i:
                        continue
                    if ma_in & selected[j][1]:
                        compat = False
                        break
                if compat:
                    candidates.append(a_in)
            
            # Find best pair in candidates
            best_gain = 0
            best_pair = (-1, -1)
            
            # Optimization: Sort candidates by score descending?
            # Or just iterate. Size of candidates is ~400. 400^2/2 = 80k pairs.
            # 80k checks per i. 90 i's. 7.2M checks.
            # Each check is AND of two ints. Fast.
            
            for u_idx in range(len(candidates)):
                a_u = candidates[u_idx]
                ma_u = masks[a_u]
                score_u = scores[a_u]
                
                for v_idx in range(u_idx + 1, len(candidates)):
                    a_v = candidates[v_idx]
                    ma_v = masks[a_v]
                    
                    if ma_u & ma_v:
                        continue
                    
                    gain = score_u + scores[a_v] - score_out
                    if gain > best_gain:
                        best_gain = gain
                        best_pair = (a_u, a_v)
            
            if best_pair != (-1, -1) and best_gain > 0:
                a_u, a_v = best_pair
                # Remove a_out, add a_u, a_v
                selected_set.remove(a_out)
                selected_set.add(a_u)
                selected_set.add(a_v)
                
                # Replace selected[i] with a_u
                selected[i] = [a_u, masks[a_u], scores[a_u]]
                # Append a_v
                selected.append([a_v, masks[a_v], scores[a_v]])
                
                current_score += best_gain
                improved = True
                break # Restart outer loop

    return [a for a, _, _ in selected]

# EVOLVE-BLOCK-END
