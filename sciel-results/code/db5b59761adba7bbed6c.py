import time
from math import gcd

def construct():
    """
    Find a set of distinct pairwise coprime integers in 1..511 that maximizes
    the sum of 1/(512 - a) using branch-and-bound with a tight clique-partition
    upper bound and greedy + local search initialization.
    """
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s limit
    
    N = 512
    
    # Get all primes < 512
    def get_primes(n):
        sieve = [True] * n
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n, i):
                    sieve[j] = False
        return [i for i, is_p in enumerate(sieve) if is_p]
    
    primes = get_primes(N)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    # Precompute prime factor masks for all candidates 1..511
    def get_factor_mask(n):
        mask = 0
        temp = n
        for i, p in enumerate(primes):
            if p * p > temp:
                break
            if temp % p == 0:
                mask |= (1 << i)
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            mask |= (1 << prime_to_idx[temp])
        return mask
    
    factor_masks = [0] * N
    spf = [0] * N  # Smallest Prime Factor
    for a in range(1, N):
        factor_masks[a] = get_factor_mask(a)
        temp = a
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                spf[a] = p
                break
        if spf[a] == 0 and a > 1:
            spf[a] = a
        if a == 1:
            spf[a] = 0
    
    # Precompute scores
    scores = [0.0] * N
    for a in range(1, N):
        scores[a] = 1.0 / (N - a)
    
    # Multiple greedy initializations with different orderings
    def greedy_construct(order):
        chosen = []
        chosen_mask = 0
        for a in order:
            if (factor_masks[a] & chosen_mask) == 0:
                chosen.append(a)
                chosen_mask |= factor_masks[a]
        return chosen, chosen_mask, sum(scores[a] for a in chosen)
    
    # Try several different greedy orderings
    orders = [
        list(range(N-1, 0, -1)),
        list(range(1, N)),
        sorted(range(1, N), key=lambda x: -scores[x]),
        sorted(range(1, N), key=lambda x: scores[x]),
        sorted(range(1, N), key=lambda x: factor_masks[x].bit_count(), reverse=True),
        sorted(range(1, N), key=lambda x: factor_masks[x].bit_count()),
        # Additional heuristics: prefer primes (fewest conflicts)
        sorted(range(1, N), key=lambda x: (factor_masks[x].bit_count(), -scores[x])),
        sorted(range(1, N), key=lambda x: (-scores[x], factor_masks[x].bit_count())),
    ]
    
    best_chosen = []
    best_score = 0.0
    best_mask = 0
    
    for order in orders:
        chosen, mask, score = greedy_construct(order)
        if score > best_score:
            best_score = score
            best_chosen = chosen
            best_mask = mask
    
    # Enhanced local search: try add, remove, and swap operations
    def local_search(chosen, mask, score):
        chosen_set = set(chosen)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try adding a new element that fits
            for b in range(1, N):
                if b in chosen_set:
                    continue
                if (factor_masks[b] & mask) == 0:
                    chosen.append(b)
                    chosen_set.add(b)
                    mask |= factor_masks[b]
                    score += scores[b]
                    improved = True
            
            # Try swapping elements to improve
            for i, a in enumerate(list(chosen)):
                new_mask = mask & ~(factor_masks[a])
                best_b = -1
                best_gain = 0.0
                for b in range(1, N):
                    if b in chosen_set:
                        continue
                    if (factor_masks[b] & new_mask) == 0:
                        gain = scores[b] - scores[a]
                        if gain > best_gain:
                            best_gain = gain
                            best_b = b
                if best_b != -1 and best_gain > 1e-12:
                    chosen[i] = best_b
                    chosen_set.discard(a)
                    chosen_set.add(best_b)
                    mask = new_mask | factor_masks[best_b]
                    score += best_gain
                    improved = True
        return chosen, mask, score
    
    best_chosen, best_mask, best_score = local_search(best_chosen, best_mask, best_score)
    
    # Branch and bound with tight clique-partition upper bound
    nodes = list(range(1, N))
    nodes.sort(key=lambda x: -scores[x])
    n = len(nodes)
    
    # Precompute conflict masks for each node
    conflict_mask = [0] * N
    for i, a in enumerate(nodes):
        for j in range(i+1, len(nodes)):
            b = nodes[j]
            if (factor_masks[a] & factor_masks[b]) != 0:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
    
    # Precompute SPF group membership for tighter bound
    # Group indices by their SPF value
    spf_groups = {}
    for i in range(n):
        p = spf[nodes[i]]
        if p not in spf_groups:
            spf_groups[p] = []
        spf_groups[p].append(i)
    
    # Precompute for each group: list of (index, score) sorted by score desc
    group_sorted = {}
    for p, indices in spf_groups.items():
        group_sorted[p] = sorted(indices, key=lambda i: -scores[nodes[i]])
    
    best_solution = list(best_chosen)
    best_val = best_score
    
    # Tight upper bound: for each SPF group, take the max score among available nodes
    def compute_upper_bound(available_mask, current_score):
        bound = current_score
        for p, sorted_indices in group_sorted.items():
            for i in sorted_indices:
                if (available_mask >> i) & 1:
                    bound += scores[nodes[i]]
                    break  # Only the highest-scoring available in this group
        return bound
    
    # Branch and bound search
    def branch_and_bound(available_mask, current_score, chosen_indices):
        nonlocal best_solution, best_val
        
        if time.time() > deadline:
            return
        
        # Update best if we have a valid solution
        if current_score > best_val:
            best_val = current_score
            best_solution = [nodes[i] for i in chosen_indices]
        
        if available_mask == 0:
            return
        
        # Compute upper bound
        ub = compute_upper_bound(available_mask, current_score)
        if ub <= best_val + 1e-12:
            return
        
        # Pick the node with highest score from available
        best_i = -1
        best_score_i = -1.0
        for i in range(n):
            if (available_mask >> i) & 1:
                if scores[nodes[i]] > best_score_i:
                    best_score_i = scores[nodes[i]]
                    best_i = i
        
        if best_i == -1:
            return
        
        # Branch 1: include best_i
        new_available = available_mask & ~(1 << best_i)
        cm = conflict_mask[best_i]
        j = 0
        while cm:
            if cm & 1:
                new_available &= ~(1 << j)
            cm >>= 1
            j += 1
        
        branch_and_bound(new_available, current_score + best_score_i,
                        chosen_indices + [best_i])
        
        # Branch 2: exclude best_i
        new_available = available_mask & ~(1 << best_i)
        branch_and_bound(new_available, current_score, chosen_indices)
    
    # Start branch and bound from the best greedy solution
    chosen_set = set(best_chosen)
    initial_mask = 0
    for i, a in enumerate(nodes):
        if a not in chosen_set:
            initial_mask |= (1 << i)
    
    for a in best_chosen:
        for i, b in enumerate(nodes):
            if b in chosen_set:
                continue
            if (factor_masks[a] & factor_masks[b]) != 0:
                initial_mask &= ~(1 << i)
    
    branch_and_bound(initial_mask, best_val, [])
    
    # Final local search on the best solution
    if time.time() < deadline:
        final_mask = 0
        for a in best_solution:
            final_mask |= factor_masks[a]
        best_solution, final_mask, best_val = local_search(best_solution, final_mask, best_val)
    
    return best_solution
