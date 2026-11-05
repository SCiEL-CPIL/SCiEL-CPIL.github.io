import time
import math
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    N = 511
    
    # 1. Precompute prime factors for all numbers 1..511
    factors = [set() for _ in range(N + 1)]
    for i in range(2, N + 1):
        if not factors[i]:  # i is prime
            for j in range(i, N + 1, i):
                factors[j].add(i)
    
    # 2. Prepare candidates and weights
    candidates = list(range(N, 0, -1))
    weights = [0.0] * (N + 1)
    for a in candidates:
        weights[a] = 1.0 / (512 - a)
    
    # 3. Lagrangian Relaxation Branch-and-Bound
    # Formulate as weighted set packing:
    # Ground set: primes < 512
    # Each a in [1,511] corresponds to set of its prime factors with weight 1/(512-a)
    # Maximize sum of weights subject to no prime factor appearing in more than one chosen set
    
    # Get all primes < 512
    primes = [p for p in range(2, N + 1) if not factors[p-1] and p <= N]
    # Actually, let's get primes properly
    primes = []
    for p in range(2, N + 1):
        is_prime = True
        for q in primes:
            if q * q > p:
                break
            if p % q == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(p)
    
    # Map each prime to an index
    prime_index = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    # For each candidate a, store its prime factor bitmask
    # Since we have at most ~97 primes < 512, we can use a set or tuple
    candidate_factors = [frozenset(factors[a]) for a in range(N + 1)]
    
    # Greedy initial solution
    def greedy_solution():
        chosen = []
        used_primes = set()
        for a in candidates:
            if not (candidate_factors[a] & used_primes):
                chosen.append(a)
                used_primes |= candidate_factors[a]
        return chosen
    
    best_chosen = greedy_solution()
    best_score = sum(weights[a] for a in best_chosen)
    
    # Lagrangian relaxation: dual variables for each prime constraint
    # L(lambda) = sum_a w_a * x_a + sum_p lambda_p * (sum_{a: p|a} x_a - 1)
    # = sum_a (w_a + sum_{p|a} lambda_p) * x_a - sum_p lambda_p
    # For fixed lambda, optimal x_a = 1 if reduced cost > 0 and feasible (no conflicts)
    
    # Branch and bound with Lagrangian bounds
    def compute_bound(chosen_set, remaining_candidates, lambda_vals):
        """
        Compute upper bound using Lagrangian relaxation.
        chosen_set: set of already chosen numbers
        remaining_candidates: list of candidates not yet decided
        lambda_vals: dict mapping prime to dual variable
        """
        # Current score from chosen
        current_score = sum(weights[a] for a in chosen_set)
        
        # For remaining candidates, compute reduced costs
        # and find a feasible subset that maximizes reduced cost sum
        # This is still a weighted independent set, but we use a greedy upper bound
        
        used_primes = set()
        for a in chosen_set:
            used_primes |= candidate_factors[a]
        
        bound = current_score
        for a in remaining_candidates:
            if a in chosen_set:
                continue
            # Check if a conflicts with any chosen
            if candidate_factors[a] & used_primes:
                continue
            # Reduced cost
            rc = weights[a]
            for p in candidate_factors[a]:
                rc += lambda_vals.get(p, 0.0)
            if rc > 0:
                bound += rc
                used_primes |= candidate_factors[a]
        
        # Subtract the dual objective
        for p in lambda_vals:
            if lambda_vals[p] > 0:
                bound -= lambda_vals[p]
        
        return bound
    
    # Simpler approach: use beam search with better pruning based on Lagrangian bounds
    # But let's try a direct branch and bound with good bounds
    
    # Actually, let's use a hybrid: beam search with Lagrangian-based pruning
    beam_width = 30000
    
    # State: (score, frozenset_of_factors, tuple_of_chosen)
    states = [(0.0, frozenset(), ())]
    
    # Track best
    best_score_bb = 0.0
    best_chosen_bb = ()
    
    for idx, a in enumerate(candidates):
        if time.time() > deadline:
            break
            
        w_a = weights[a]
        fac_a = candidate_factors[a]
        
        new_states = []
        
        for score, fac_set, chosen in states:
            # Option 1: Exclude a
            new_states.append((score, fac_set, chosen))
            
            # Option 2: Include a
            if not (fac_a & fac_set):
                new_score = score + w_a
                if new_score > best_score_bb:
                    best_score_bb = new_score
                    best_chosen_bb = chosen + (a,)
                
                new_fac = frozenset(fac_set | fac_a)
                new_chosen = chosen + (a,)
                new_states.append((new_score, new_fac, new_chosen))
        
        # Prune: keep top beam_width states by score
        new_states.sort(key=lambda x: x[0], reverse=True)
        states = new_states[:beam_width]
        
        if time.time() > deadline:
            break
    
    if best_score_bb > best_score:
        best_score = best_score_bb
        best_chosen = best_chosen_bb
    
    # 4. Local Refinement
    if best_chosen:
        current_set = list(best_chosen)
        current_score = best_score
        in_set = set(current_set)
        
        def can_add(a, set_list, set_obj):
            fac_a = candidate_factors[a]
            for b in set_list:
                if fac_a & candidate_factors[b]:
                    return False
            return True
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(current_set)):
                if time.time() > deadline:
                    break
                P = current_set[i]
                remaining = [x for k, x in enumerate(current_set) if k != i]
                remaining_set = set(remaining)
                
                best_add = None
                best_add_score = 0.0
                for a in range(N, 0, -1):
                    if a in remaining_set:
                        continue
                    if can_add(a, remaining, remaining_set):
                        best_add = a
                        best_add_score = weights[a]
                        break
                
                if best_add is not None:
                    delta = best_add_score - weights[P]
                    if delta > 1e-9:
                        current_set[i] = best_add
                        current_score += delta
                        in_set = set(current_set)
                        improved = True
                        break
            
            if not improved:
                for a in range(N, 0, -1):
                    if a in in_set:
                        continue
                    if can_add(a, current_set, in_set):
                        current_set.append(a)
                        current_score += weights[a]
                        in_set.add(a)
                        improved = True
                        break
                        
        if current_score > best_score:
            best_score = current_score
            best_chosen = tuple(current_set)

    return list(best_chosen)
