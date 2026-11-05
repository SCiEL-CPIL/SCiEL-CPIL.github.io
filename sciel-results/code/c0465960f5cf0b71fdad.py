"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    DEADLINE = 110  # safety margin before 120s limit
    
    def get_score(chosen):
        return sum(1.0 / (512 - a) for a in chosen)
    
    def is_coprime_set(chosen):
        for i in range(len(chosen)):
            for j in range(i+1, len(chosen)):
                if gcd(chosen[i], chosen[j]) != 1:
                    return False
        return True
    
    def is_coprime_to_all(a, chosen):
        return all(gcd(a, b) == 1 for b in chosen)
    
    # Greedy initialization: pick from 511 down
    chosen = []
    for a in range(511, 0, -1):
        if is_coprime_to_all(a, chosen):
            chosen.append(a)
    
    best_score = get_score(chosen)
    best_set = list(chosen)
    
    # Local search: try single replacements
    # For each element in chosen, try replacing with a number not in chosen
    # that is coprime to all others and gives better score
    
    # Precompute: for each candidate a, its weight
    weights = {a: 1.0 / (512 - a) for a in range(1, 512)}
    
    # Build a set of chosen for fast lookup
    chosen_set = set(chosen)
    
    # Try single swaps: remove one, add one
    improved = True
    while improved:
        improved = False
        if time.time() - start_time > DEADLINE:
            break
        
        for i in range(len(chosen)):
            if time.time() - start_time > DEADLINE:
                break
            
            removed = chosen[i]
            others = [chosen[j] for j in range(len(chosen)) if j != i]
            
            # Try adding a better number
            for a in range(511, 0, -1):
                if a in chosen_set or a == removed:
                    continue
                if not is_coprime_to_all(a, others):
                    continue
                # Check if this improves score
                new_score = best_score - weights[removed] + weights[a]
                if new_score > best_score + 1e-12:
                    # Make the swap
                    chosen[i] = a
                    chosen_set.remove(removed)
                    chosen_set.add(a)
                    best_score = new_score
                    improved = True
                    break  # restart with new set
    
    # Try 2-swaps: remove two, add two
    if time.time() - start_time < DEADLINE:
        improved2 = True
        while improved2:
            improved2 = False
            if time.time() - start_time > DEADLINE:
                break
            
            n = len(chosen)
            for i in range(n):
                if time.time() - start_time > DEADLINE:
                    break
                for j in range(i+1, n):
                    if time.time() - start_time > DEADLINE:
                        break
                    
                    removed1 = chosen[i]
                    removed2 = chosen[j]
                    others = [chosen[k] for k in range(n) if k != i and k != j]
                    
                    # Try pairs of replacements
                    for a1 in range(511, 0, -1):
                        if a1 in chosen_set or a1 == removed1 or a1 == removed2:
                            continue
                        if not is_coprime_to_all(a1, others):
                            continue
                        for a2 in range(a1+1, 512):
                            if a2 in chosen_set or a2 == removed1 or a2 == removed2:
                                continue
                            if a2 == a1:
                                continue
                            if not is_coprime_to_all(a2, others + [a1]):
                                continue
                            
                            new_score = best_score - weights[removed1] - weights[removed2] + weights[a1] + weights[a2]
                            if new_score > best_score + 1e-12:
                                # Make the swap
                                chosen[i] = a1
                                chosen[j] = a2
                                chosen_set.remove(removed1)
                                chosen_set.remove(removed2)
                                chosen_set.add(a1)
                                chosen_set.add(a2)
                                best_score = new_score
                                improved2 = True
                                break
                        if improved2:
                            break
                    if improved2:
                        break
                if improved2:
                    break
    
    # Return the best set found
    return best_set
# EVOLVE-BLOCK-END
