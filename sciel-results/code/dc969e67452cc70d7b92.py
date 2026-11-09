"""
Rewritten program: Optimized local search with bitmask coprimality and aggressive 2-swap moves.
"""
# EVOLVE-BLOCK-START
import time
import math
import random

_DEADLINE = 145.0
_START = None

def _elapsed():
    return time.time() - _START if _START is not None else 0.0

def _deadline_guard(limit=_DEADLINE, margin=2.0):
    return _elapsed() > (limit - margin)

def _sieve_primes(n):
    """Return list of primes < n."""
    if n < 2:
        return []
    sieve = bytearray([1]) * n
    sieve[0] = sieve[1] = 0
    for i in range(2, int(math.isqrt(n - 1)) + 1):
        if sieve[i]:
            for j in range(i * i, n, i):
                sieve[j] = 0
    return [i for i in range(2, n) if sieve[i]]

def _build_prime_masks(primes):
    """For each number 1..511, compute bitmask of its prime factors."""
    masks = [0] * 512
    # Precompute index for each prime
    prime_idx = {p: i for i, p in enumerate(primes)}
    
    for a in range(2, 512):
        x = a
        mask = 0
        for idx, p in enumerate(primes):
            if p * p > x:
                break
            if x % p == 0:
                mask |= (1 << idx)
                while x % p == 0:
                    x //= p
        if x > 1:
            if x in prime_idx:
                mask |= (1 << prime_idx[x])
            else:
                # Should not happen if sieve is correct for 512
                # But handle just in case
                pass
        masks[a] = mask
    return masks

def _greedy_select(masks, order):
    """Greedy select in given order using bitmask coprimality."""
    chosen = []
    chosen_mask = 0
    for a in order:
        if masks[a] & chosen_mask:
            continue
        chosen.append(a)
        chosen_mask |= masks[a]
    return chosen

def _hill_climb(chosen, masks, weights):
    """
    Optimized Hill-climbing: 1-add, 1-swap, and 2-swap moves.
    Uses bitmasks for fast conflict checking.
    """
    chosen_set = set(chosen)
    cur_score = sum(weights[a] for a in chosen)
    n = 512
    
    # Precompute mask for current set
    cur_mask = 0
    for a in chosen:
        cur_mask |= masks[a]
        
    improved = True
    while improved:
        if _deadline_guard():
            break
        improved = False
        best_gain = 0.0
        best_action = None
        
        # 1. Try 1-Add
        # Find the best a not in chosen_set such that masks[a] & cur_mask == 0
        best_add_a = -1
        best_add_gain = 0.0
        for a in range(511, 0, -1):
            if a in chosen_set:
                continue
            if masks[a] & cur_mask:
                continue
            gain = weights[a]
            if gain > best_add_gain:
                best_add_gain = gain
                best_add_a = a
        
        if best_add_a != -1 and best_add_gain > best_gain:
            best_gain = best_add_gain
            best_action = ('add', best_add_a)
            
        # 2. Try 1-Swap (remove r, add a)
        # For each r in chosen, check if there is a better a
        for r in chosen:
            if _deadline_guard():
                break
            # Mask of others
            mask_others = cur_mask ^ masks[r] # Since r is unique in set, XOR works if we assume no duplicate bits? 
            # Wait, masks are bitmasks of prime factors. If two numbers share a prime, they conflict.
            # cur_mask is OR of all masks.
            # mask_others should be OR of masks of (chosen - {r}).
            # We can't just XOR if bits overlap with other elements? 
            # Actually, if r has a prime p, and another element s has p, then r and s would conflict, which is impossible.
            # So masks in chosen set are pairwise disjoint.
            # Therefore, cur_mask = masks[r] | mask_others, and masks[r] & mask_others == 0.
            # So mask_others = cur_mask ^ masks[r] is correct.
            
            best_swap_a = -1
            best_swap_gain = 0.0
            # We want a such that masks[a] & mask_others == 0
            # And gain = weights[a] - weights[r] > 0 => weights[a] > weights[r]
            # So a must have higher weight than r.
            # Since weights decrease as a increases, a must be LARGER than r? No.
            # weight(a) = 1/(512-a).
            # If a > r, then 512-a < 512-r, so 1/(512-a) > 1/(512-r).
            # So we only need to check a > r?
            # Not necessarily. r could be a composite with small weight, and a could be a prime with larger weight but smaller index?
            # Example: r=10 (weight 1/502), a=9 (weight 1/503). 1/502 > 1/503. So larger index -> larger weight.
            # Yes, weight is strictly decreasing with a.
            # So if we remove r, we can only improve by adding a > r.
            
            for a in range(511, r, -1):
                if a in chosen_set:
                    continue
                if masks[a] & mask_others:
                    continue
                gain = weights[a] - weights[r]
                if gain > best_swap_gain:
                    best_swap_gain = gain
                    best_swap_a = a
                    
            if best_swap_a != -1 and best_swap_gain > best_gain:
                best_gain = best_swap_gain
                best_action = ('swap', r, best_swap_a)

        # 3. Try 2-Swap (remove r1, r2, add a1, a2)
        # This is expensive. We limit the search space.
        # We look for pairs (r1, r2) and (a1, a2).
        # Gain = w(a1) + w(a2) - w(r1) - w(r2)
        # To be positive, w(a1)+w(a2) > w(r1)+w(r2).
        # Since weights are sorted by index, a1, a2 should be larger than r1, r2 on average.
        
        # Optimization: Only consider small sets of candidates for a1, a2.
        # For a given pair (r1, r2), the allowed a's are those coprime to mask_others = cur_mask ^ masks[r1] ^ masks[r2].
        # We pick the top 2 allowed a's by weight.
        
        # To speed up, we only try 2-swaps involving the "worst" elements in the current set?
        # Or just iterate over all pairs if the set is small?
        # The set size is likely < 50. 50*50 = 2500 pairs. For each, scan 512 elements. 2500*512 = 1.2M ops.
        # This might be too slow for many iterations.
        # Let's restrict to a subset of r's.
        
        # Strategy: Try 2-swaps where we remove two elements and add two.
        # We'll iterate r1, r2 from the set.
        # To save time, we only do this if the set size is small enough or we haven't found an improvement in 1-moves.
        
        if len(chosen) > 2:
            # Limit r1, r2 to the last few added or smallest weights?
            # Let's just iterate all pairs but break early if time is tight.
            # We can optimize by pre-computing the "best available a" for a given mask.
            
            # For efficiency, we will only attempt 2-swaps if no 1-swap or 1-add was found.
            if best_action is None:
                best_2swap_gain = 0.0
                best_2swap_action = None
                
                # Iterate over pairs of removals
                # To reduce complexity, we can limit the candidates for r1, r2.
                # But for correctness in local search, we should check more.
                # Let's check all pairs but with a time check inside.
                
                for i in range(len(chosen)):
                    if _deadline_guard():
                        break
                    r1 = chosen[i]
                    m1 = masks[r1]
                    for j in range(i + 1, len(chosen)):
                        r2 = chosen[j]
                        m2 = masks[r2]
                        
                        mask_others = cur_mask ^ m1 ^ m2
                        
                        # Find top 2 candidates a1, a2
                        # a1, a2 must be coprime to mask_others and each other.
                        # And not in chosen_set.
                        
                        cand_a = []
                        # We want high weights, so iterate a from 511 down
                        # We can stop once we have 2 valid candidates that are coprime to each other?
                        # Not necessarily the top 2 by weight are coprime.
                        # We need the best pair (a1, a2) such that a1, a2 coprime to mask_others and to each other.
                        
                        # Heuristic: Pick the best a1, then best a2 coprime to a1 and mask_others.
                        # This is not exhaustive for 2-swap but good heuristic.
                        
                        best_a1 = -1
                        best_a1_w = 0.0
                        for a in range(511, 0, -1):
                            if a in chosen_set:
                                continue
                            if masks[a] & mask_others:
                                continue
                            # Potential a1
                            # Check if there is a valid a2
                            best_a2 = -1
                            best_a2_w = 0.0
                            for b in range(511, 0, -1):
                                if b in chosen_set or b == a:
                                    continue
                                if masks[b] & mask_others:
                                    continue
                                if masks[b] & masks[a]:
                                    continue
                                best_a2 = b
                                best_a2_w = weights[b]
                                break # Found the best b for this a
                                
                            if best_a2 != -1:
                                gain = weights[a] + best_a2_w - weights[r1] - weights[r2]
                                if gain > best_2swap_gain:
                                    best_2swap_gain = gain
                                    best_2swap_action = ('swap2', r1, r2, a, best_a2)
                                    best_a1 = a
                                    best_a1_w = weights[a]
                                    # We can break early if gain is very high? 
                                    # No, keep looking for max gain.
                                    # But we can skip subsequent a if weights[a] + max_possible_b - w(r1) - w(r2) <= best_2swap_gain
                                    # max_possible_b is weights[511] = 1.0.
                                    # If weights[a] + 1.0 - w(r1) - w(r2) <= best_2swap_gain, break.
                                    if weights[a] + 1.0 - weights[r1] - weights[r2] <= best_2swap_gain:
                                        break
                        # End a loop
                    # End j loop
                # End i loop
                
                if best_2swap_action is not None and best_2swap_gain > 1e-15:
                    best_gain = best_2swap_gain
                    best_action = best_2swap_action

        if best_action is not None and best_gain > 1e-15:
            if best_action[0] == 'add':
                a = best_action[1]
                chosen.append(a)
                chosen_set.add(a)
                cur_score += weights[a]
                cur_mask |= masks[a]
            elif best_action[0] == 'swap':
                r, a = best_action[1], best_action[2]
                chosen.remove(r)
                chosen_set.remove(r)
                chosen.append(a)
                chosen_set.add(a)
                cur_score += weights[a] - weights[r]
                cur_mask = (cur_mask ^ masks[r]) | masks[a]
            elif best_action[0] == 'swap2':
                r1, r2, a1, a2 = best_action[1], best_action[2], best_action[3], best_action[4]
                chosen.remove(r1)
                chosen.remove(r2)
                chosen_set.remove(r1)
                chosen_set.remove(r2)
                chosen.append(a1)
                chosen.append(a2)
                chosen_set.add(a1)
                chosen_set.add(a2)
                cur_score += weights[a1] + weights[a2] - weights[r1] - weights[r2]
                cur_mask = (cur_mask ^ masks[r1] ^ masks[r2]) | masks[a1] | masks[a2]
            improved = True
            
    return chosen, cur_score

def construct():
    global _START
    _START = time.time()
    
    N = 512
    primes = _sieve_primes(N)
    masks = _build_prime_masks(primes)
    weights = [0.0] * N
    for a in range(1, N):
        weights[a] = 1.0 / (N - a)
    
    best_chosen = []
    best_score = 0.0
    rng = random.Random(42)
    
    # Multiple greedy initializations with different orderings
    n_restarts = 100 # Reduced restarts to allow deeper hill climbing per restart
    
    for restart in range(n_restarts):
        if _deadline_guard():
            break
        
        if restart == 0:
            # Descending weight (largest a first)
            order = list(range(511, 0, -1))
        elif restart == 1:
            # Primes first, then others
            order = [p for p in primes if p < N]
            order += [a for a in range(511, 0, -1) if a not in order]
        elif restart == 2:
            # Random
            order = list(range(511, 0, -1))
            rng.shuffle(order)
        else:
            # Perturbed descending
            order = list(range(511, 0, -1))
            n_swaps = rng.randint(10, 100)
            for _ in range(n_swaps):
                i, j = rng.randint(0, 510), rng.randint(0, 510)
                order[i], order[j] = order[j], order[i]
        
        chosen = _greedy_select(masks, order)
        
        # Hill-climb to local optimum
        chosen, score = _hill_climb(chosen, masks, weights)
        
        if score > best_score:
            best_score = score
            best_chosen = chosen[:]
    
    # Final Polish: Run hill climb again on the best solution found, maybe with more time
    if best_chosen and not _deadline_guard():
        best_chosen, best_score = _hill_climb(best_chosen, masks, weights)

    # Verify
    assert len(set(best_chosen)) == len(best_chosen)
    assert all(1 <= x <= 511 for x in best_chosen)
    for i in range(len(best_chosen)):
        for j in range(i + 1, len(best_chosen)):
            assert math.gcd(best_chosen[i], best_chosen[j]) == 1
    
    return best_chosen
# EVOLVE-BLOCK-END
