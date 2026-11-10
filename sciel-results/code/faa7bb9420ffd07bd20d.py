"""Rewritten program: maximize sum 1/(512 - a) over pairwise coprime a in 1..511."""
# EVOLVE-BLOCK-START
import math
import time
import sys
from math import gcd

def construct():
    t0 = time.time()
    deadline = t0 + 130.0  # safety margin before 150s kill
    
    # Candidate values: a in 1..511. We want to maximize sum 1/(512-a).
    # So we prefer larger a (smaller 512-a).
    # We need pairwise coprime selection.
    
    # Strategy: Greedy with local search / simulated annealing style improvements.
    # Start with a greedy construction, then try swaps to improve.
    
    def score_of(selected):
        return sum(1.0 / (512.0 - a) for a in selected)
    
    # Greedy init: pick highest a first, ensuring pairwise coprime
    def greedy_init():
        chosen = []
        for a in range(511, 0, -1):
            if all(gcd(a, b) == 1 for b in chosen):
                chosen.append(a)
        return chosen
    
    # Greedy after removal: try to add back better elements
    def greedy_after_removal(chosen):
        chosen = list(chosen)
        chosen_set = set(chosen)
        improved = True
        while improved:
            improved = False
            # Try adding any a not in chosen that is coprime to all chosen
            best_a = None
            best_gain = 0.0
            for a in range(511, 0, -1):
                if a in chosen_set:
                    continue
                if all(gcd(a, b) == 1 for b in chosen):
                    gain = 1.0 / (512.0 - a)
                    if gain > best_gain:
                        best_gain = gain
                        best_a = a
            if best_a is not None:
                chosen.append(best_a)
                chosen_set.add(best_a)
                improved = True
        return chosen
    
    # Initial solution
    best = greedy_init()
    best_score = score_of(best)
    
    # Local search: try swaps - remove one element, then greedily re-add
    # Also try single-element removal if it allows better additions
    # Also try pairwise swaps
    
    # Precompute gcd for speed
    # Use a set for membership checks
    
    best_set = set(best)
    current_score = best_score
    
    # Simulated annealing / hill climbing with swaps
    # Temperature schedule
    T = 0.1
    T_min = 0.001
    cooling = 0.9995
    
    # Precompute prime factors for each number to speed up gcd checks
    # Actually, for speed, let's precompute for each a, the set of primes dividing a
    def prime_factors(n):
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
    
    # Precompute prime factors for all candidates
    max_val = 511
    pf_cache = [None] * (max_val + 1)
    for i in range(1, max_val + 1):
        pf_cache[i] = prime_factors(i)
    
    def is_coprime_to_all(a, chosen_set, pf_cache):
        """Check if a is coprime to all in chosen_set using prime factor cache."""
        pa = pf_cache[a]
        if not pa:
            return True
        for b in chosen_set:
            pb = pf_cache[b]
            if not pb:
                continue
            # Check intersection
            if pa & pb:
                return False
        return True
    
    # Local search with moves:
    # 1. Add a new element if coprime to all
    # 2. Remove an element and replace with one or more better elements
    # 3. Swap: remove one, add another
    
    def try_improve(chosen, chosen_set, current_score, pf_cache, deadline):
        """Try local improvements. Returns (chosen, chosen_set, score)."""
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            current_score = score_of(chosen)
            
            # Try adding elements (greedy)
            best_a = None
            best_gain = 0.0
            for a in range(511, 0, -1):
                if a in chosen_set:
                    continue
                if is_coprime_to_all(a, chosen_set, pf_cache):
                    gain = 1.0 / (512.0 - a)
                    if gain > best_gain:
                        best_gain = gain
                        best_a = a
            if best_a is not None and best_a not in chosen_set:
                chosen.append(best_a)
                chosen_set.add(best_a)
                current_score += best_gain
                improved = True
                continue
            
            # Try swap: remove one element i, add element a
            # Net gain = 1/(512-a) - 1/(512-b) where b is removed
            # Must be coprime to all others after removing b
            best_swap_gain = 0.0
            best_swap_out = None
            best_swap_in = None
            
            for b_idx, b in enumerate(chosen):
                if time.time() > deadline:
                    break
                # After removing b, check which new a can be added
                remaining = set(chosen)
                remaining.discard(b)
                for a in range(511, 0, -1):
                    if a in remaining:
                        continue
                    if is_coprime_to_all(a, remaining, pf_cache):
                        gain = 1.0 / (512.0 - a) - 1.0 / (512.0 - b)
                        if gain > best_swap_gain:
                            best_swap_gain = gain
                            best_swap_out = b
                            best_swap_in = a
            
            if best_swap_out is not None and best_swap_gain > 0:
                # Perform swap
                chosen.remove(best_swap_out)
                chosen_set.discard(best_swap_out)
                chosen.append(best_swap_in)
                chosen_set.add(best_swap_in)
                current_score += best_swap_gain
                improved = True
                continue
            
            # Try multi-swap: remove one, add multiple
            # This is more complex; skip for now to keep runtime bounded
            # Try removing a low-value element and adding several high-value ones
            for b_idx, b in enumerate(chosen):
                if time.time() > deadline:
                    break
                remaining = list(chosen)
                remaining.pop(b_idx)
                rem_set = set(remaining)
                
                # Greedily add best coprime elements to remaining
                additions = []
                add_score = 0.0
                for a in range(511, 0, -1):
                    if a in rem_set:
                        continue
                    if is_coprime_to_all(a, rem_set | set(additions), pf_cache):
                        additions.append(a)
                        add_score += 1.0 / (512.0 - a)
                
                net_gain = add_score - 1.0 / (512.0 - b)
                if net_gain > best_swap_gain:
                    best_swap_gain = net_gain
                    best_swap_out = b
                    best_swap_in = additions  # list of additions
            
            if best_swap_out is not None and best_swap_gain > 0 and isinstance(best_swap_in, list):
                chosen.remove(best_swap_out)
                chosen_set.discard(best_swap_out)
                for a in best_swap_in:
                    if a not in chosen_set:
                        chosen.append(a)
                        chosen_set.add(a)
                current_score += best_swap_gain
                improved = True
                continue
        
        return chosen, chosen_set, current_score
    
    # Run local search
    best, best_set, current_score = try_improve(best, best_set, current_score, pf_cache, deadline)
    best_score = current_score
    
    # Multiple restarts with different greedy orders if time permits
    if time.time() < deadline - 20.0:
        # Try reverse greedy (smallest first) as another starting point
        alt = greedy_init()  # Already high-to-low
        
        # Try a randomized greedy
        import random
        random.seed(42)
        for restart in range(5):
            if time.time() > deadline - 10.0:
                break
            random_order = list(range(511, 0, -1))
            random.shuffle(random_order)
            # Bias towards larger values
            random_order.sort(key=lambda x: -x + random.random() * 100)
            alt = []
            alt_set = set()
            for a in random_order:
                if is_coprime_to_all(a, alt_set, pf_cache):
                    alt.append(a)
                    alt_set.add(a)
            alt, alt_set, alt_score = try_improve(alt, alt_set, score_of(alt), pf_cache, deadline - 5.0)
            if alt_score > best_score:
                best = alt
                best_set = alt_set
                best_score = alt_score
    
    return best
# EVOLVE-BLOCK-END
