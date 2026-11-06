import random
import time
import math

def construct():
    random.seed(42)
    n = 20
    max_val = 10000
    
    def compute_scores(A):
        sumset = set()
        prodset = set()
        for i in range(n):
            for j in range(i, n):
                sumset.add(A[i] + A[j])
                prodset.add(A[i] * A[j])
        return max(len(sumset), len(prodset))
        
    def mutate(A, mutation_rate=0.5):
        new_A = A[:]
        # Randomly replace some elements
        for i in range(n):
            if random.random() < mutation_rate:
                # Pick a new random value
                new_val = random.randint(1, max_val)
                # Ensure distinctness
                attempts = 0
                while new_val in new_A and attempts < 10:
                    new_val = random.randint(1, max_val)
                    attempts += 1
                if new_val not in new_A:
                    new_A[i] = new_val
                else:
                    # If collision persists, just pick a nearby unique value
                    for delta in range(1, 100):
                        if new_val + delta not in new_A and new_val + delta <= max_val:
                            new_A[i] = new_val + delta
                            break
                        if new_val - delta not in new_A and new_val - delta >= 1:
                            new_A[i] = new_val - delta
                            break
        return new_A

    def local_refine(A):
        # A quick greedy pass to clean up
        current_score = compute_scores(A)
        improved = True
        while improved and time.time() < end_time:
            improved = False
            for i in range(n):
                old_val = A[i]
                # Try small perturbations
                for delta in range(-10, 11):
                    if delta == 0: continue
                    new_val = old_val + delta
                    if new_val < 1 or new_val > max_val or new_val in A:
                        continue
                    A[i] = new_val
                    A_sorted = sorted(A)
                    new_score = compute_scores(A_sorted)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        A = A_sorted
                        break
                    else:
                        A[i] = old_val
                if improved:
                    break
        return A, current_score

    start_time = time.time()
    timeout = 55
    end_time = start_time + timeout
    
    # Initialize with diverse structures
    candidates = []
    
    # 1. Arithmetic Progression
    ap = list(range(1, n + 1))
    candidates.append(ap)
    
    # 2. Geometric Progression (base 2)
    gp = [2**i for i in range(n)]
    if gp[-1] <= max_val:
        candidates.append(gp)
    else:
        # Scale down or use smaller base? Base 2 is smallest. 
        # If 2^19 > 10000, it's invalid. 2^13 = 8192. 2^14 = 16384.
        # So GP of length 20 with base 2 is invalid.
        # Use base 1? That's AP.
        # Use a "mixed" GP?
        pass
        
    # 3. Random
    candidates.append(random.sample(range(1, max_val + 1), n))
    
    # 4. "Square Root" like distribution?
    # Values spread out to minimize product collisions? No, dense is better for sums.
    
    best_A = None
    best_score = float('inf')
    
    for init_A in candidates:
        init_A = sorted(init_A)
        # Check validity
        if len(set(init_A)) != n or init_A[-1] > max_val:
            continue
            
        current_A = init_A[:]
        current_score = compute_scores(current_A)
        
        # Simulated Annealing
        T = 10.0
        cool_rate = 0.95
        steps_per_temp = 50
        
        while T > 0.01 and time.time() < end_time:
            for _ in range(steps_per_temp):
                if time.time() >= end_time:
                    break
                new_A = mutate(current_A, mutation_rate=0.2)
                new_score = compute_scores(new_A)
                
                if new_score < current_score:
                    current_A = new_A
                    current_score = new_score
                else:
                    # Accept worse with probability
                    delta = new_score - current_score
                    if delta < 0:
                        accept_prob = 1.0
                    else:
                        accept_prob = math.exp(-delta / T)
                    
                    if random.random() < accept_prob:
                        current_A = new_A
                        current_score = new_score
            
            T *= cool_rate
        
        # Final local refinement
        refined_A, final_score = local_refine(current_A)
        
        if final_score < best_score:
            best_score = final_score
            best_A = refined_A
            
        if time.time() >= end_time:
            break
            
    if best_A is None:
        return list(range(1, n + 1))
        
    return best_A
