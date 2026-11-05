import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(1000)
    
    # Strategy: Use an iterative improvement approach
    # 1. Start with a good initial covering
    # 2. Try to swap primes to improve coverage
    # 3. Use local search to optimize residue assignments
    
    L_target = 1000
    covered = np.zeros(L_target + 1, dtype=bool)
    
    # Initial strategy: use all primes >= 13
    large_primes = [p for p in primes if p >= 13]
    small_primes = [p for p in primes if p < 13]  # [2, 3, 5, 7, 11]
    
    # Compute sum of reciprocals for large primes
    large_sum = Fraction(0, 1)
    for p in large_primes:
        large_sum += Fraction(1, p)
    
    remaining_budget = Fraction(1, 1) - large_sum
    
    # Phase 1: Greedy assignment with better heuristics
    # Use primes in ascending order (smaller primes first for density)
    primes_to_use = sorted(large_primes)
    
    used_primes = set()
    current_sum = Fraction(0, 1)
    classes = []
    
    for p in primes_to_use:
        if time.time() > deadline - 10:
            break
        
        cost = Fraction(1, p)
        if current_sum + cost > Fraction(1, 1):
            continue
        
        # Find best residue: maximize coverage of uncovered numbers
        best_r = 0
        best_gain = -1
        
        if p > L_target:
            # Each residue class has at most one element in [1, L_target]
            for r in range(1, min(p, L_target + 1)):
                if not covered[r]:
                    best_r = r
                    best_gain = 1
                    break
        else:
            # For p <= L_target, count uncovered in each residue class
            for r in range(p):
                if r == 0:
                    start = p
                else:
                    start = r
                
                count = 0
                x = start
                while x <= L_target:
                    if not covered[x]:
                        count += 1
                    x += p
                
                if count > best_gain:
                    best_gain = count
                    best_r = r
        
        if best_gain <= 0:
            continue
        
        # Add this class
        classes.append([p, best_r])
        used_primes.add(p)
        current_sum += cost
        
        # Update covered
        if best_r == 0:
            start = p
        else:
            start = best_r
        x = start
        while x <= L_target:
            covered[x] = True
            x += p
    
    # Phase 2: Try to add small primes if budget allows
    small_primes_sorted = sorted(small_primes, key=lambda p: 1/p, reverse=True)
    
    for p in small_primes_sorted:
        if time.time() > deadline - 5:
            break
        
        cost = Fraction(1, p)
        if current_sum + cost > Fraction(1, 1):
            continue
        
        if p in used_primes:
            continue
        
        # Find best residue
        best_r = 0
        best_gain = -1
        
        for r in range(p):
            if r == 0:
                start = p
            else:
                start = r
            
            count = 0
            x = start
            while x <= L_target:
                if not covered[x]:
                    count += 1
                x += p
            
            if count > best_gain:
                best_gain = count
                best_r = r
        
        if best_gain <= 0:
            continue
        
        classes.append([p, best_r])
        used_primes.add(p)
        current_sum += cost
        
        if best_r == 0:
            start = p
        else:
            start = best_r
        x = start
        while x <= L_target:
            covered[x] = True
            x += p
    
    # Phase 3: Local search - try swapping primes to improve coverage
    # For each pair (used_prime, unused_prime), check if swapping improves L
    unused_primes = [p for p in primes if p not in used_primes]
    
    improved = True
    max_iterations = 50
    iteration = 0
    
    while improved and iteration < max_iterations and time.time() < deadline - 5:
        improved = False
        iteration += 1
        
        for i, (p_used, r_used) in enumerate(classes[:]):
            if time.time() > deadline - 5:
                break
            
            for p_new in unused_primes:
                if time.time() > deadline - 5:
                    break
                
                cost_old = Fraction(1, p_used)
                cost_new = Fraction(1, p_new)
                
                # Check if swap is budget-feasible
                if current_sum - cost_old + cost_new > Fraction(1, 1):
                    continue
                
                # Remove old prime, add new prime with best residue
                # Temporarily remove old prime
                old_indices = []
                for j in range(L_target + 1):
                    if covered[j]:
                        # Check if j is covered only by this prime
                        # This is complex, so we'll just recompute
                        pass
                
                # Simpler: just try all residues for p_new and see if L improves
                best_L_new = 0
                best_r_new = 0
                
                for r_new in range(p_new):
                    # Compute coverage with old prime removed and new prime added
                    temp_covered = np.zeros(L_target + 1, dtype=bool)
                    
                    # Add all other primes
                    for j, (p_j, r_j) in enumerate(classes):
                        if j == i:
                            continue
                        if r_j == 0:
                            start = p_j
                        else:
                            start = r_j
                        x = start
                        while x <= L_target:
                            temp_covered[x] = True
                            x += p_j
                    
                    # Add new prime
                    if r_new == 0:
                        start = p_new
                    else:
                        start = r_new
                    x = start
                    while x <= L_target:
                        temp_covered[x] = True
                        x += p_new
                    
                    # Compute L
                    L_temp = 0
                    for j in range(1, L_target + 1):
                        if temp_covered[j]:
                            L_temp = j
                        else:
                            break
                    
                    if L_temp > best_L_new:
                        best_L_new = L_temp
                        best_r_new = r_new
                
                # Compare with current L
                current_L = 0
                for j in range(1, L_target + 1):
                    if covered[j]:
                        current_L = j
                    else:
                        break
                
                if best_L_new > current_L:
                    # Perform the swap
                    classes[i] = [p_new, best_r_new]
                    used_primes.remove(p_used)
                    used_primes.add(p_new)
                    unused_primes.remove(p_new)
                    unused_primes.append(p_used)
                    current_sum = current_sum - cost_old + cost_new
                    
                    # Update covered
                    covered = np.zeros(L_target + 1, dtype=bool)
                    for p_j, r_j in classes:
                        if r_j == 0:
                            start = p_j
                        else:
                            start = r_j
                        x = start
                        while x <= L_target:
                            covered[x] = True
                            x += p_new
                        x = start
                        while x <= L_target:
                            covered[x] = True
                            x += p_j
                    
                    improved = True
                    break
    
    # Calculate final L
    L = 0
    for i in range(1, L_target + 1):
        if covered[i]:
            L = i
        else:
            break
    
    return classes
