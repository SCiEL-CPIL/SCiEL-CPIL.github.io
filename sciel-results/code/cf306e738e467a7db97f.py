import random
from math import gcd

def score(A, B):
    sums = set()
    for a in A:
        for b in B:
            sums.add(a + b)
    
    if len(sums) == 0:
        return 0
        
    # Check pairwise coprime
    s_list = list(sums)
    for i in range(len(s_list)):
        for j in range(i + 1, len(s_list)):
            if gcd(s_list[i], s_list[j]) != 1:
                return 0
                
    return len(s_list)

def construct():
    # Seed solution
    A = [1, 7, 13]
    B = [4, 10, 16]
    best_score = score(A, B)
    best_A = A[:]
    best_B = B[:]
    
    # Local search
    import time
    start_time = time.time()
    timeout = 55.0  # Leave some buffer
    
    current_A = A[:]
    current_B = B[:]
    current_score = best_score
    
    no_improvement_count = 0
    max_no_improvement = 500
    
    while time.time() - start_time < timeout:
        # Try to improve current solution
        improved = False
        
        # Randomly pick a set to modify
        if random.random() < 0.5:
            idx = random.randint(0, 2)
            old_val = current_A[idx]
            new_val = random.randint(1, 128)
            if new_val not in current_A:
                current_A[idx] = new_val
                new_score = score(current_A, current_B)
                if new_score > current_score:
                    current_score = new_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_A = current_A[:]
                        best_B = current_B[:]
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                else:
                    # Revert if not improved (or keep if equal? usually revert to avoid drift)
                    # But sometimes we want to explore. Let's revert if worse, keep if equal?
                    # Let's revert to be safe and use random restarts for exploration.
                    current_A[idx] = old_val
                    no_improvement_count += 1
        else:
            idx = random.randint(0, 2)
            old_val = current_B[idx]
            new_val = random.randint(1, 128)
            if new_val not in current_B:
                current_B[idx] = new_val
                new_score = score(current_A, current_B)
                if new_score > current_score:
                    current_score = new_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_A = current_A[:]
                        best_B = current_B[:]
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                else:
                    current_B[idx] = old_val
                    no_improvement_count += 1
        
        if no_improvement_count > max_no_improvement:
            # Restart from best or random
            if random.random() < 0.5:
                current_A = best_A[:]
                current_B = best_B[:]
                current_score = best_score
            else:
                current_A = random.sample(range(1, 129), 3)
                current_B = random.sample(range(1, 129), 3)
                current_score = score(current_A, current_B)
                if current_score > best_score:
                    best_score = current_score
                    best_A = current_A[:]
                    best_B = current_B[:]
            no_improvement_count = 0

    return [best_A, best_B]
