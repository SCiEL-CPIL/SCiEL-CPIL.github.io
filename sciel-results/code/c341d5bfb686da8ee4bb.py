"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 60s kill
    
    N = 256
    
    # Precompute prime factors for 1..N
    prime_factors = [set() for _ in range(N + 1)]
    for i in range(2, N + 1):
        if not prime_factors[i]:
            for j in range(i, N + 1, i):
                prime_factors[j].add(i)
    # 1 has no prime factors, prime_factors[1] remains empty
    
    def is_compatible(a, b):
        return not (prime_factors[a] & prime_factors[b])
    
    # Greedy initialization: start with 1, then add numbers in descending order
    # if compatible with all existing
    solution = [1]
    for a in range(N, 1, -1):
        if all(is_compatible(a, b) for b in solution):
            solution.append(a)
    
    current_sum = sum(solution)
    solution_set = set(solution)
    
    def can_add(a):
        return all(is_compatible(a, b) for b in solution_set)
    
    def can_replace(a, b):
        # Check if replacing b with a is valid
        for c in solution_set:
            if c == b:
                continue
            if not is_compatible(a, c):
                return False
        return True
    
    # Local search: 1-for-1 swaps
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(solution)):
            a = solution[i]
            # Try to replace a with a larger number not in solution
            for b in range(N, a, -1):
                if b in solution_set:
                    continue
                if can_replace(b, a):
                    solution_set.remove(a)
                    solution_set.add(b)
                    solution[i] = b
                    current_sum += b - a
                    improved = True
                    break
            # If we made a swap, restart the loop for this position
            # (but we already broke, so continue to next i)
    
    # Local search: 1-for-2 swaps (remove two, add one larger)
    # This is more complex, so we do a simpler version: try to add a number
    # that is compatible with all, but we already did that in greedy.
    # Instead, try to find a number that can replace one number and allow adding another.
    # For simplicity, we'll do a few rounds of random restarts if time permits.
    
    best_sum = current_sum
    best_solution = solution[:]
    
    # Random restarts
    num_restarts = 0
    while time.time() < deadline - 5:
        # Random perturbation
        if len(solution) < 2:
            break
        
        # Remove a random element
        idx = random.randint(0, len(solution) - 1)
        removed = solution.pop(idx)
        solution_set.remove(removed)
        
        # Try to re-add it or something else
        # Greedily add back the best compatible numbers
        for a in range(N, 1, -1):
            if a not in solution_set and can_add(a):
                solution.append(a)
                solution_set.add(a)
        
        current_sum = sum(solution)
        
        if current_sum > best_sum:
            best_sum = current_sum
            best_solution = solution[:]
        
        # Restore the removed element if it's compatible
        if removed not in solution_set and can_add(removed):
            solution.append(removed)
            solution_set.add(removed)
        
        num_restarts += 1
        if num_restarts > 100:  # Limit restarts
            break
    
    # Final polish: try 1-for-1 swaps again on the best solution
    solution = best_solution[:]
    solution_set = set(solution)
    current_sum = best_sum
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(solution)):
            a = solution[i]
            for b in range(N, a, -1):
                if b in solution_set:
                    continue
                if can_replace(b, a):
                    solution_set.remove(a)
                    solution_set.add(b)
                    solution[i] = b
                    current_sum += b - a
                    improved = True
                    break
    
    return [int(x) for x in solution]
# EVOLVE-BLOCK-END
