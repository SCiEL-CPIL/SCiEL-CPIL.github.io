"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def _prime_factors(n):
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
    t0 = time.time()
    deadline = t0 + 140.0
    
    N = 512
    pf = {}
    for a in range(1, N):
        pf[a] = _prime_factors(a)
    
    def score_val(a):
        return 1.0 / (N - a)
    
    def total_score(chosen):
        return sum(score_val(a) for a in chosen)
    
    def is_coprime_to_all(a, chosen_set):
        pa = pf[a]
        for b in chosen_set:
            if pa & pf[b]:
                return False
        return True
    
    # Greedy largest-first
    def greedy():
        chosen = []
        chosen_set = set()
        for a in range(N - 1, 0, -1):
            if is_coprime_to_all(a, chosen_set):
                chosen.append(a)
                chosen_set.add(a)
        return chosen
    
    best = greedy()
    best_score = total_score(best)
    
    # Local search: single swaps
    def local_search(chosen, time_limit):
        chosen = list(chosen)
        chosen_set = set(chosen)
        cur_score = total_score(chosen)
        t_start = time.time()
        
        while True:
            if time.time() > time_limit:
                break
            improved = False
            for i in range(len(chosen)):
                a = chosen[i]
                remaining = [x for j, x in enumerate(chosen) if j != i]
                rem_set = set(remaining)
                best_b = -1
                best_gain = 0
                for b in range(N - 1, 0, -1):
                    if b in chosen_set:
                        continue
                    if not is_coprime_to_all(b, rem_set):
                        continue
                    gain = score_val(b) - score_val(a)
                    if gain > best_gain:
                        best_gain = gain
                        best_b = b
                if best_b > 0:
                    # Replace a with best_b
                    chosen[i] = best_b
                    chosen_set.discard(a)
                    chosen_set.add(best_b)
                    cur_score += best_gain
                    improved = True
                    break
            if not improved:
                break
        return chosen, cur_score
    
    best, best_score = local_search(best, deadline - 1.0)
    
    # Multiple restarts with perturbation
    for restart in range(100):
        if time.time() > deadline - 2.0:
            break
        
        # Perturb best solution
        perturbed = list(best)
        k = random.randint(1, min(4, len(perturbed) - 1))
        if len(perturbed) > k:
            to_remove = random.sample(range(len(perturbed)), k)
            to_remove_set = set(to_remove)
            perturbed = [x for i, x in enumerate(perturbed) if i not in to_remove_set]
        
        # Greedily fill
        pert_set = set(perturbed)
        for a in range(N - 1, 0, -1):
            if a in pert_set:
                continue
            if is_coprime_to_all(a, pert_set):
                perturbed.append(a)
                pert_set.add(a)
        
        # Local search
        result, score = local_search(perturbed, deadline - 2.0)
        if score > best_score:
            best = result
            best_score = score
    
    return best
# EVOLVE-BLOCK-END
