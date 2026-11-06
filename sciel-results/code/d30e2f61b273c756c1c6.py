"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from math import gcd
import time

def construct():
    best_A = None
    best_B = None
    best_count = 0
    
    def check_pairwise_coprime(values):
        vals = sorted(set(values))
        n = len(vals)
        for i in range(n):
            for j in range(i + 1, n):
                if gcd(vals[i], vals[j]) != 1:
                    return False
        return True
    
    def evaluate(A, B):
        sums = [a + b for a in A for b in B]
        distinct_sums = set(sums)
        if check_pairwise_coprime(distinct_sums):
            return len(distinct_sums)
        return 0
    
    # Known baseline
    A_test = [1, 7, 13]
    B_test = [4, 10, 16]
    best_count = evaluate(A_test, B_test)
    best_A = A_test[:]
    best_B = B_test[:]
    
    random.seed(98765)
    start_time = time.time()
    timeout = 55
    
    # Random restarts
    for restart in range(100):
        if time.time() - start_time > timeout:
            break
        if best_count == 9:
            break
        
        # Start from random or from best
        if restart == 0:
            A = best_A[:]
            B = best_B[:]
        else:
            A = sorted(random.sample(range(1, 129), 3))
            B = sorted(random.sample(range(1, 129), 3))
        
        current_count = evaluate(A, B)
        
        # Local search: mutate one element at a time
        for step in range(200):
            if time.time() - start_time > timeout:
                break
            if best_count == 9:
                break
            
            # Pick which set and which position to mutate
            if random.random() < 0.5:
                idx = random.randint(0, 2)
                old_val = A[idx]
                new_val = random.randint(1, 128)
                if new_val in A and new_val != old_val:
                    continue
                A[idx] = new_val
                A.sort()
                new_count = evaluate(A, B)
                if new_count > current_count:
                    current_count = new_count
                    if current_count > best_count:
                        best_count = current_count
                        best_A = A[:]
                        best_B = B[:]
                else:
                    A[idx] = old_val
                    A.sort()
            else:
                idx = random.randint(0, 2)
                old_val = B[idx]
                new_val = random.randint(1, 128)
                if new_val in B and new_val != old_val:
                    continue
                B[idx] = new_val
                B.sort()
                new_count = evaluate(A, B)
                if new_count > current_count:
                    current_count = new_count
                    if current_count > best_count:
                        best_count = current_count
                        best_A = A[:]
                        best_B = B[:]
                else:
                    B[idx] = old_val
                    B.sort()
        
        # Also try pure random perturbations
        for step in range(500):
            if time.time() - start_time > timeout:
                break
            if best_count == 9:
                break
            
            A = sorted(random.sample(range(1, 129), 3))
            B = sorted(random.sample(range(1, 129), 3))
            count = evaluate(A, B)
            if count > best_count:
                best_count = count
                best_A = A[:]
                best_B = B[:]
    
    if best_A is None:
        best_A = [1, 7, 13]
        best_B = [4, 10, 16]
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
