"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random

def construct():
    N = 256
    deadline = time.time() + 85  # Leave 5s safety margin
    
    # Precompute GCD table for 1..256
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            gcd_table[i][j] = math.gcd(i, j)
    
    def is_pairwise_coprime(lst):
        n = len(lst)
        for i in range(n):
            for j in range(i + 1, n):
                if gcd_table[lst[i]][lst[j]] != 1:
                    return False
        return True
    
    # Sieve of Eratosthenes to find primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
                
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # Initial solution: largest prime power for each prime
    best_result = [1]
    for p in primes:
        power = p
        while power * p <= N:
            power *= p
        best_result.append(power)
    
    best_sum = sum(best_result)
    
    # Local search: try to improve by swapping elements
    all_candidates = list(range(1, N + 1))
    
    # Precompute which candidates are coprime with each other
    coprime_with = [set() for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            if gcd_table[i][j] == 1:
                coprime_with[i].add(j)
    
    def can_add(lst, x):
        for y in lst:
            if x != y and gcd_table[x][y] != 1:
                return False
        return True
    
    def can_swap(lst, remove_idx, add_x):
        for i, y in enumerate(lst):
            if i == remove_idx:
                continue
            if gcd_table[add_x][y] != 1:
                return False
        return True
    
    # Iterative local search
    current = list(best_result)
    current_sum = best_sum
    improved = True
    max_iterations = 1000
    
    while improved and time.time() < deadline:
        improved = False
        # Try adding a new element
        for x in all_candidates:
            if x in current:
                continue
            if can_add(current, x):
                new_sum = current_sum + x
                if new_sum > best_sum:
                    current.append(x)
                    current_sum = new_sum
                    best_sum = new_sum
                    improved = True
                    break
        
        # Try swapping: remove one element, add another
        if not improved:
            for i in range(len(current)):
                removed = current[i]
                for x in all_candidates:
                    if x in current or x == removed:
                        continue
                    if can_swap(current, i, x):
                        new_sum = current_sum - removed + x
                        if new_sum > best_sum:
                            current[i] = x
                            current_sum = new_sum
                            best_sum = new_sum
                            improved = True
                            break
                if improved:
                    break
        
        # Try 2-for-1 swaps: remove two, add one
        if not improved:
            n = len(current)
            for i in range(n):
                for j in range(i + 1, n):
                    removed1 = current[i]
                    removed2 = current[j]
                    for x in all_candidates:
                        if x in current or x == removed1 or x == removed2:
                            continue
                        # Check if x is coprime with all remaining
                        ok = True
                        for k in range(n):
                            if k == i or k == j:
                                continue
                            if gcd_table[x][current[k]] != 1:
                                ok = False
                                break
                        if ok:
                            new_sum = current_sum - removed1 - removed2 + x
                            if new_sum > best_sum:
                                # Remove i and j, add x
                                new_current = [y for k, y in enumerate(current) if k != i and k != j]
                                new_current.append(x)
                                current = new_current
                                current_sum = new_sum
                                best_sum = new_sum
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
        
        # Try 1-for-2 swaps: remove one, add two
        if not improved:
            for i in range(len(current)):
                removed = current[i]
                for x1 in all_candidates:
                    if x1 in current or x1 == removed:
                        continue
                    for x2 in all_candidates:
                        if x2 in current or x2 == removed or x2 == x1:
                            continue
                        if gcd_table[x1][x2] != 1:
                            continue
                        # Check if both are coprime with all remaining
                        ok = True
                        for k in range(len(current)):
                            if k == i:
                                continue
                            if gcd_table[x1][current[k]] != 1 or gcd_table[x2][current[k]] != 1:
                                ok = False
                                break
                        if ok:
                            new_sum = current_sum - removed + x1 + x2
                            if new_sum > best_sum:
                                new_current = [y for k, y in enumerate(current) if k != i]
                                new_current.extend([x1, x2])
                                current = new_current
                                current_sum = new_sum
                                best_sum = new_sum
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
    
    # Final polish: try to add any remaining coprime elements
    for x in all_candidates:
        if x in current:
            continue
        if can_add(current, x):
            current.append(x)
    
    # Ensure we return a list of Python ints
    return [int(x) for x in current]
# EVOLVE-BLOCK-END
