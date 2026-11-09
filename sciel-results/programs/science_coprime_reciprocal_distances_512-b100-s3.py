"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    """
    Find a set of distinct pairwise coprime integers in 1..511 that maximizes
    the sum of 1/(512 - a).
    
    Strategy:
    - Use bitmask-based coprimality checks for speed
    - Multiple greedy initializations with different orderings
    - Local search with 1-swap and 2-swap moves
    - Simulated annealing to escape local optima
    """
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin before 120s limit
    
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
        for i, p in enumerate(primes):
            if p * p > n:
                break
            if n % p == 0:
                mask |= (1 << i)
                while n % p == 0:
                    n //= p
        if n > 1:
            mask |= (1 << prime_to_idx[n])
        return mask
    
    factor_masks = [0] * (N)  # index by a (1..511)
    for a in range(1, N):
        factor_masks[a] = get_factor_mask(a)
    
    # Precompute scores
    scores = [0.0] * N
    for a in range(1, N):
        scores[a] = 1.0 / (N - a)
    
    # Check if two candidates are coprime using masks
    def are_coprime(a, b):
        return (factor_masks[a] & factor_masks[b]) == 0
    
    # Check if candidate a is coprime to all in chosen set
    def is_coprime_to_all(a, chosen_set, chosen_mask):
        # Check against the combined mask of chosen
        if factor_masks[a] & chosen_mask:
            return False
        return True
    
    # Compute score of a set
    def compute_score(chosen):
        return sum(scores[a] for a in chosen)
    
    # Greedy construction with given ordering
    def greedy_construct(ordering):
        chosen = []
        chosen_set = set()
        chosen_mask = 0
        for a in ordering:
            if factor_masks[a] & chosen_mask == 0:
                chosen.append(a)
                chosen_set.add(a)
                chosen_mask |= factor_masks[a]
        return chosen, chosen_set, chosen_mask
    
    # Try multiple greedy orderings
    best_chosen = None
    best_score = -1
    best_chosen_set = None
    best_chosen_mask = 0
    
    # Ordering 1: descending a (highest score first)
    ordering1 = list(range(511, 0, -1))
    chosen1, set1, mask1 = greedy_construct(ordering1)
    score1 = compute_score(chosen1)
    if score1 > best_score:
        best_score = score1
        best_chosen = chosen1
        best_chosen_set = set1
        best_chosen_mask = mask1
    
    # Ordering 2: random permutations
    for _ in range(20):
        if time.time() > deadline:
            break
        ordering = list(range(1, 512))
        random.shuffle(ordering)
        # Bias towards larger values
        ordering.sort(key=lambda x: random.random() * 0.3 + scores[x], reverse=True)
        chosen, cset, cmask = greedy_construct(ordering)
        s = compute_score(chosen)
        if s > best_score:
            best_score = s
            best_chosen = chosen
            best_chosen_set = cset
            best_chosen_mask = cmask
    
    # Ordering 3: Sort by score / number_of_prime_factors (prefer numbers with fewer prime factors)
    def num_prime_factors(a):
        return bin(factor_masks[a]).count('1')
    
    ordering3 = sorted(range(1, 512), key=lambda x: scores[x] / max(1, num_prime_factors(x)), reverse=True)
    chosen3, set3, mask3 = greedy_construct(ordering3)
    score3 = compute_score(chosen3)
    if score3 > best_score:
        best_score = score3
        best_chosen = chosen3
        best_chosen_set = set3
        best_chosen_mask = mask3
    
    # Local search: 1-swap moves
    def local_search_1swap(chosen, chosen_set, chosen_mask, current_score):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for a in list(chosen):
                # Find best b to replace a
                best_b = None
                best_gain = 0
                for b in range(511, 0, -1):
                    if b in chosen_set:
                        continue
                    # Check if b is coprime to all chosen except a
                    temp_mask = chosen_mask & ~factor_masks[a]
                    if factor_masks[b] & temp_mask:
                        continue
                    gain = scores[b] - scores[a]
                    if gain > best_gain + 1e-12:
                        best_gain = gain
                        best_b = b
                if best_b is not None:
                    # Perform swap
                    chosen.remove(a)
                    chosen.append(best_b)
                    chosen_set.remove(a)
                    chosen_set.add(best_b)
                    chosen_mask = (chosen_mask & ~factor_masks[a]) | factor_masks[best_b]
                    current_score += best_gain
                    improved = True
        return chosen, chosen_set, chosen_mask, current_score
    
    best_chosen, best_chosen_set, best_chosen_mask, best_score = local_search_1swap(
        best_chosen, best_chosen_set, best_chosen_mask, best_score
    )
    
    # Local search: 2-swap moves (remove two, add two)
    def local_search_2swap(chosen, chosen_set, chosen_mask, current_score):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            n = len(chosen)
            if n < 2:
                break
            for i in range(n):
                for j in range(i+1, n):
                    a1 = chosen[i]
                    a2 = chosen[j]
                    # Find best pair (b1, b2) to replace (a1, a2)
                    best_b1 = None
                    best_b2 = None
                    best_gain = 0
                    temp_mask = chosen_mask & ~factor_masks[a1] & ~factor_masks[a2]
                    for b1 in range(511, 0, -1):
                        if b1 in chosen_set:
                            continue
                        if factor_masks[b1] & temp_mask:
                            continue
                        for b2 in range(511, 0, -1):
                            if b2 == b1 or b2 in chosen_set:
                                continue
                            if factor_masks[b2] & temp_mask:
                                continue
                            if factor_masks[b1] & factor_masks[b2]:
                                continue
                            gain = scores[b1] + scores[b2] - scores[a1] - scores[a2]
                            if gain > best_gain + 1e-12:
                                best_gain = gain
                                best_b1 = b1
                                best_b2 = b2
                    if best_b1 is not None:
                        # Perform swap
                        chosen.remove(a1)
                        chosen.remove(a2)
                        chosen.append(best_b1)
                        chosen.append(best_b2)
                        chosen_set.remove(a1)
                        chosen_set.remove(a2)
                        chosen_set.add(best_b1)
                        chosen_set.add(best_b2)
                        chosen_mask = (chosen_mask & ~factor_masks[a1] & ~factor_masks[a2]) | factor_masks[best_b1] | factor_masks[best_b2]
                        current_score += best_gain
                        improved = True
                        break
                if improved:
                    break
        return chosen, chosen_set, chosen_mask, current_score
    
    best_chosen, best_chosen_set, best_chosen_mask, best_score = local_search_2swap(
        best_chosen, best_chosen_set, best_chosen_mask, best_score
    )
    
    # Simulated annealing
    def simulated_annealing(chosen, chosen_set, chosen_mask, current_score):
        temperature = 0.01
        cooling_rate = 0.995
        min_temperature = 0.0001
        
        while temperature > min_temperature and time.time() < deadline:
            # Generate a random move
            move_type = random.random()
            if move_type < 0.6 and len(chosen) > 0:
                # 1-swap: remove one, add one
                a = random.choice(chosen)
                candidates_for_add = [b for b in range(1, 512) if b not in chosen_set and 
                                       (factor_masks[b] & (chosen_mask & ~factor_masks[a])) == 0]
                if not candidates_for_add:
                    temperature *= cooling_rate
                    continue
                b = random.choice(candidates_for_add)
                delta = scores[b] - scores[a]
                if delta > 0 or random.random() < pow(2.718281828, delta / temperature):
                    # Accept
                    chosen.remove(a)
                    chosen.append(b)
                    chosen_set.remove(a)
                    chosen_set.add(b)
                    chosen_mask = (chosen_mask & ~factor_masks[a]) | factor_masks[b]
                    current_score += delta
            elif move_type < 0.85 and len(chosen) >= 2:
                # 2-swap: remove two, add two
                a1, a2 = random.sample(chosen, 2)
                temp_mask = chosen_mask & ~factor_masks[a1] & ~factor_masks[a2]
                candidates_for_add = [b for b in range(1, 512) if b not in chosen_set and 
                                       (factor_masks[b] & temp_mask) == 0]
                if len(candidates_for_add) < 2:
                    temperature *= cooling_rate
                    continue
                b1, b2 = random.sample(candidates_for_add, 2)
                if (factor_masks[b1] & factor_masks[b2]) != 0:
                    temperature *= cooling_rate
                    continue
                delta = scores[b1] + scores[b2] - scores[a1] - scores[a2]
                if delta > 0 or random.random() < pow(2.718281828, delta / temperature):
                    # Accept
                    chosen.remove(a1)
                    chosen.remove(a2)
                    chosen.append(b1)
                    chosen.append(b2)
                    chosen_set.remove(a1)
                    chosen_set.remove(a2)
                    chosen_set.add(b1)
                    chosen_set.add(b2)
                    chosen_mask = (chosen_mask & ~factor_masks[a1] & ~factor_masks[a2]) | factor_masks[b1] | factor_masks[b2]
                    current_score += delta
            else:
                # Just cool down
                pass
            
            temperature *= cooling_rate
        
        return chosen, chosen_set, chosen_mask, current_score
    
    # Run simulated annealing
    best_chosen, best_chosen_set, best_chosen_mask, best_score = simulated_annealing(
        best_chosen, best_chosen_set, best_chosen_mask, best_score
    )
    
    # Final local search pass
    best_chosen, best_chosen_set, best_chosen_mask, best_score = local_search_1swap(
        best_chosen, best_chosen_set, best_chosen_mask, best_score
    )
    best_chosen, best_chosen_set, best_chosen_mask, best_score = local_search_2swap(
        best_chosen, best_chosen_set, best_chosen_mask, best_score
    )
    
    return best_chosen
# EVOLVE-BLOCK-END
