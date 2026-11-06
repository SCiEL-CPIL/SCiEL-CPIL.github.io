"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    N = 128
    # Precompute GCD table
    gcd_table = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i+1, N+1):
            gcd_table[i][j] = gcd(i, j)
            gcd_table[j][i] = gcd_table[i][j]
    
    def is_valid_triple(a, b, c):
        """Check if three distinct elements have all three pairwise GCDs equal."""
        ga = gcd_table[a][b]
        gb = gcd_table[a][c]
        gc = gcd_table[b][c]
        return not (ga == gb == gc)
    
    def can_add(x, selected_list):
        """Check if x can be added without creating a bad triple."""
        for i in range(len(selected_list)):
            for j in range(i+1, len(selected_list)):
                a = selected_list[i]
                b = selected_list[j]
                if not is_valid_triple(a, b, x):
                    return False
        return True
    
    def greedy(order):
        """Run greedy algorithm with given order."""
        selected = []
        for x in order:
            if can_add(x, selected):
                selected.append(x)
        return selected
    
    def local_search(selected, time_limit):
        """Try to improve the solution via local search."""
        best = list(selected)
        best_set = set(best)
        
        while time.time() < time_limit:
            improved = False
            # Try adding elements not in the set
            for x in range(1, N+1):
                if x in best_set:
                    continue
                if can_add(x, best):
                    best.append(x)
                    best_set.add(x)
                    improved = True
                    break
            if improved:
                continue
            
            # Try swapping: remove one, try to add two
            for i in range(len(best)):
                removed = best[i]
                temp_best = best[:i] + best[i+1:]
                temp_set = set(temp_best)
                
                added_count = 0
                for x in range(1, N+1):
                    if x in temp_set or x == removed:
                        continue
                    if can_add(x, temp_best):
                        temp_best.append(x)
                        temp_set.add(x)
                        added_count += 1
                        if added_count >= 2:
                            break
                
                if added_count > 1:
                    best = temp_best
                    best_set = temp_set
                    improved = True
                    break
            
            if not improved:
                break
        
        return best
    
    # Try multiple orderings
    best_result = []
    
    # Ordering 1: ascending
    result = greedy(range(1, N+1))
    if len(result) > len(best_result):
        best_result = result
    
    if time.time() > deadline - 20:
        return best_result if best_result else [1]
    
    # Ordering 2: descending
    result = greedy(range(N, 0, -1))
    if len(result) > len(best_result):
        best_result = result
    
    if time.time() > deadline - 15:
        return best_result if best_result else [1]
    
    # Ordering 3: by number of prime factors (fewer first)
    def num_prime_factors(n):
        count = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count
    
    order3 = sorted(range(1, N+1), key=lambda x: (num_prime_factors(x), x))
    result = greedy(order3)
    if len(result) > len(best_result):
        best_result = result
    
    if time.time() > deadline - 10:
        return best_result if best_result else [1]
    
    # Try several random orderings
    rng = random.Random(42)
    for trial in range(20):
        if time.time() > deadline - 5:
            break
        order = list(range(1, N+1))
        rng.shuffle(order)
        result = greedy(order)
        if len(result) > len(best_result):
            best_result = result
    
    if time.time() > deadline - 3:
        return best_result if best_result else [1]
    
    # Local search on best result
    best_result = local_search(best_result, deadline - 1)
    
    return best_result if best_result else [1]
# EVOLVE-BLOCK-END
