import time
import random
from math import gcd
from fractions import Fraction

def _prime_factors(n):
    """Return the set of prime factors of n."""
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

def construct():
    """
    Find distinct pairwise coprime integers a in 1..511 maximizing sum 1/(512 - a).
    Uses bitset-based local search with greedy initialization and simulated annealing.
    """
    deadline = time.time() + 140.0  # Safety margin
    N = 512
    n_vars = N - 1  # 1..511
    
    # Precompute weights
    weights = [0.0] * N
    for a in range(1, N):
        weights[a] = 1.0 / (N - a)
    
    # Precompute prime factors and conflict masks
    # conflict_masks[i] is an integer where bit j is 1 if gcd(i, j) > 1
    conflict_masks = [0] * N
    pfactors = [set() for _ in range(N)]
    for a in range(2, N):
        pfactors[a] = _prime_factors(a)
    
    for a in range(1, N):
        mask = 0
        if a == 1:
            # 1 is coprime to everything
            continue
        for b in range(1, N):
            if a != b and gcd(a, b) > 1:
                mask |= (1 << b)
        conflict_masks[a] = mask
    
    # Initialize with Greedy
    # Sort by weight descending (largest a first)
    candidates = sorted(range(1, N), key=lambda x: -weights[x])
    
    in_set_mask = 0
    score = 0.0
    in_set_list = []
    
    for a in candidates:
        # Check if a conflicts with any in in_set_mask
        if (in_set_mask & conflict_masks[a]) == 0:
            in_set_mask |= (1 << a)
            in_set_list.append(a)
            score += weights[a]
    
    best_score = score
    best_mask = in_set_mask
    best_list = list(in_set_list)
    
    # Local Search
    # We will use a hill-climbing approach with random restarts if stuck
    # Moves:
    # 1. Add a vertex
    # 2. Remove a vertex
    # 3. Swap 1 for 1
    # 4. Swap 1 for 2
    # 5. Swap 2 for 1
    
    def get_list_from_mask(mask):
        return [i for i in range(1, N) if (mask >> i) & 1]
    
    improved = True
    no_improve_count = 0
    max_no_improve = 50
    
    while improved and time.time() < deadline:
        improved = False
        best_move_delta = 0
        best_move_type = None
        best_move_args = None
        
        current_set_list = get_list_from_mask(in_set_mask)
        
        # 1. Try Additions
        for a in candidates:
            if (in_set_mask >> a) & 1:
                continue
            if (in_set_mask & conflict_masks[a]) == 0:
                delta = weights[a]
                if delta > best_move_delta + 1e-12:
                    best_move_delta = delta
                    best_move_type = 'add'
                    best_move_args = (a,)
        
        # 2. Try 1-for-1 Swaps
        for u in current_set_list:
            # Remove u temporarily
            mask_without_u = in_set_mask ^ (1 << u)
            for v in candidates:
                if v == u or (in_set_mask >> v) & 1:
                    continue
                # Check if v conflicts with mask_without_u
                if (mask_without_u & conflict_masks[v]) == 0:
                    delta = weights[v] - weights[u]
                    if delta > best_move_delta + 1e-12:
                        best_move_delta = delta
                        best_move_type = 'swap11'
                        best_move_args = (u, v)
        
        # 3. Try 1-for-2 Swaps (Remove u, Add v, w)
        # This is O(N^3) which is too slow for 511 if done naively every step.
        # We will only do this if we are stuck or periodically.
        if no_improve_count > 10 and time.time() < deadline - 10:
            for u in current_set_list:
                mask_without_u = in_set_mask ^ (1 << u)
                # Find all v, w that are coprime to mask_without_u and each other
                # And v, w not in set
                avail = []
                for v in candidates:
                    if v == u or (in_set_mask >> v) & 1:
                        continue
                    if (mask_without_u & conflict_masks[v]) == 0:
                        avail.append(v)
                
                # Check pairs in avail
                for i in range(len(avail)):
                    v = avail[i]
                    for j in range(i + 1, len(avail)):
                        w = avail[j]
                        if (conflict_masks[v] >> w) & 1:
                            continue
                        delta = weights[v] + weights[w] - weights[u]
                        if delta > best_move_delta + 1e-12:
                            best_move_delta = delta
                            best_move_type = 'swap12'
                            best_move_args = (u, v, w)
                            break
                    if best_move_type == 'swap12' and best_move_args[0] == u:
                        break
                if best_move_type == 'swap12' and best_move_args[0] == u:
                    break

        if best_move_type:
            # Apply move
            if best_move_type == 'add':
                a = best_move_args[0]
                in_set_mask |= (1 << a)
                score += weights[a]
                in_set_list.append(a)
            elif best_move_type == 'swap11':
                u, v = best_move_args
                in_set_mask ^= (1 << u)
                in_set_mask |= (1 << v)
                score += weights[v] - weights[u]
                in_set_list[in_set_list.index(u)] = v
            elif best_move_type == 'swap12':
                u, v, w = best_move_args
                in_set_mask ^= (1 << u)
                in_set_mask |= (1 << v)
                in_set_mask |= (1 << w)
                score += weights[v] + weights[w] - weights[u]
                in_set_list[in_set_list.index(u)] = v
                in_set_list.append(w)
            
            improved = True
            no_improve_count = 0
            
            if score > best_score:
                best_score = score
                best_mask = in_set_mask
                best_list = list(in_set_list)
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Random restart or perturbation
                if time.time() > deadline - 5:
                    break
                # Perturb: remove a random element and add a random non-conflicting one
                if in_set_list:
                    u = random.choice(in_set_list)
                    in_set_mask ^= (1 << u)
                    score -= weights[u]
                    in_set_list.remove(u)
                    
                    # Try to add a random high-weight element
                    added = False
                    for v in candidates:
                        if (in_set_mask >> v) & 1:
                            continue
                        if (in_set_mask & conflict_masks[v]) == 0:
                            in_set_mask |= (1 << v)
                            score += weights[v]
                            in_set_list.append(v)
                            added = True
                            break
                    if not added:
                        # Revert if we couldn't add anything? Or just continue
                        pass
                    
                    no_improve_count = 0
                    improved = True # To continue loop
                    if score > best_score:
                        best_score = score
                        best_mask = in_set_mask
                        best_list = list(in_set_list)

    # Final Validation
    final_set = get_list_from_mask(best_mask)
    for i in range(len(final_set)):
        for j in range(i + 1, len(final_set)):
            if gcd(final_set[i], final_set[j]) != 1:
                # Fallback to greedy if invalid (should not happen)
                return get_list_from_mask(in_set_mask)
    
    return final_set

# EVOLVE-BLOCK-END
