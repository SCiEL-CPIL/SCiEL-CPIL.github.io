"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import lcm
from fractions import Fraction

def construct():
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Compute LCM of denominators 2..48
    L = 1
    for d in denominators:
        L = lcm(L, d)
    
    # Convert to integer weights: weight[i] = L / denominators[i]
    # Sum of selected weights must be < L
    weights = [L // d for d in denominators]
    
    # Precompute suffix sums of weights for pruning
    suffix_sums = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + weights[i]
    
    # Greedy initial solution
    best_deficit = L  # deficit = L - sum, we want to minimize positive deficit
    best_indices = []
    current_sum = 0
    for i in range(n):
        if current_sum + weights[i] < L:
            current_sum += weights[i]
            best_indices.append(i)
    if best_indices:
        best_deficit = L - current_sum
    
    start_time = time.time()
    deadline = start_time + 55.0  # Leave 5s margin before 60s limit
    
    # Use a set to track visited states (sum, index) to avoid redundant work
    # But the state space might be too large, so we use a different approach:
    # Local search / hill climbing with random restarts
    
    def greedy_with_start(start_idx, order):
        """Greedy: try to add items in given order, starting from start_idx"""
        total = 0
        indices = []
        for i in order:
            if i < start_idx:
                continue
            if total + weights[i] < L:
                total += weights[i]
                indices.append(i)
        return total, indices
    
    # Strategy 1: Randomized greedy with different orderings
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        # Create a random permutation of indices
        order = list(range(n))
        random.shuffle(order)
        
        total, indices = greedy_with_start(0, order)
        
        if indices and (L - total) < best_deficit:
            best_deficit = L - total
            best_indices = indices
    
    # Strategy 2: Local search - try to swap or add/remove elements
    if best_indices:
        best_set = set(best_indices)
        best_sum = L - best_deficit
        
        for iteration in range(50000):
            if time.time() > deadline:
                break
            
            # Try adding a random element
            candidates_to_add = [i for i in range(n) if i not in best_set]
            if candidates_to_add:
                i = random.choice(candidates_to_add)
                if best_sum + weights[i] < L:
                    new_sum = best_sum + weights[i]
                    new_deficit = L - new_sum
                    if new_deficit < best_deficit:
                        best_deficit = new_deficit
                        best_sum = new_sum
                        best_indices = list(best_set) + [i]
                        best_set.add(i)
            
            # Try removing a random element and adding a different one
            if best_set:
                i = random.choice(list(best_set))
                j_candidates = [j for j in range(n) if j not in best_set and j != i]
                if j_candidates:
                    j = random.choice(j_candidates)
                    new_sum = best_sum - weights[i] + weights[j]
                    if 0 < new_sum < L:
                        new_deficit = L - new_sum
                        if new_deficit < best_deficit:
                            best_deficit = new_deficit
                            best_sum = new_sum
                            best_indices = [x for x in best_indices if x != i] + [j]
                            best_set.remove(i)
                            best_set.add(j)
    
    return best_indices
# EVOLVE-BLOCK-END
