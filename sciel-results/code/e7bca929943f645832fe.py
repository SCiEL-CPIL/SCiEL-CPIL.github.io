import time
import random
import math

def calc_score(A):
    """
    Calculate the max of the size of the sumset and product set of A.
    A is a list of distinct positive integers.
    """
    sumset = set(a + b for a in A for b in A)
    prodset = set(a * b for a in A for b in A)
    return max(len(sumset), len(prodset))

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for the 90s limit
    
    rng = random.Random(42)
    
    def calc_score_fast(A):
        A = sorted(A)
        sumset = set()
        prodset = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                sumset.add(A[i] + A[j])
                prodset.add(A[i] * A[j])
        return max(len(sumset), len(prodset))
    
    # Build a diverse candidate pool
    pool = list(range(1, 101))
    # Add smooth numbers: 2^a * 3^b * 5^c
    for a in range(1, 14):
        for b in range(1, 10):
            for c in range(1, 6):
                val = (2 ** a) * (3 ** b) * (5 ** c)
                if val <= 10000:
                    pool.append(val)
    # Add some random numbers
    for _ in range(200):
        pool.append(rng.randint(1, 10000))
    pool = list(set(pool))
    rng.shuffle(pool)
    
    # Generate initial candidates
    candidates = []
    
    # Candidate 1: Arithmetic Progression 1..20
    candidates.append(list(range(1, 21)))
    
    # Candidate 2: Geometric Progression + fillers
    gp = [2**i for i in range(14)]
    for x in range(1, 100):
        if x not in gp and len(gp) < 20:
            gp.append(x)
    candidates.append(gp)
    
    # Candidate 3: Small smooth numbers
    smooth = []
    for a in range(14):
        for b in range(14):
            val = (2**a) * (3**b)
            if val <= 10000:
                smooth.append(val)
    smooth = sorted(set(smooth))
    candidates.append(smooth[:20])
    
    # Candidate 4: Random set of small integers
    rand_set = random.sample(range(1, 101), 20)
    candidates.append(rand_set)
    
    # Candidate 5: Mixed structure
    mixed = list(range(1, 11)) + [12, 14, 16, 20, 25, 30, 40, 50, 60, 80]
    candidates.append(mixed)
    
    best_A = None
    best_score = float('inf')
    
    # Evaluate initial candidates
    for A in candidates:
        if len(set(A)) != 20:
            continue
        if any(x < 1 or x > 10000 for x in A):
            continue
        score = calc_score_fast(A)
        if score < best_score:
            best_score = score
            best_A = list(A)
    
    if best_A is None:
        best_A = list(range(1, 21))
        best_score = calc_score_fast(best_A)
    
    # Simulated Annealing Local Search
    def local_search_with_sa(A, time_limit, rng):
        A = list(A)
        current_score = calc_score_fast(A)
        best_score = current_score
        best_A = list(A)
        
        # Simulated annealing parameters
        temp = 10.0
        cooling = 0.995
        min_temp = 0.1
        
        start_time_sa = time.time()
        
        while time.time() - start_time_sa < time_limit:
            if temp < min_temp:
                break
            
            # Pick a random index to replace
            i = rng.randint(0, len(A) - 1)
            old_val = A[i]
            
            # Pick a random new value from the pool
            new_val = rng.choice(pool)
            
            # Ensure distinctness
            if new_val == old_val or new_val in A:
                continue
            
            new_A = A.copy()
            new_A[i] = new_val
            
            new_score = calc_score_fast(new_A)
            
            # Accept if better or with probability based on temperature
            if new_score < current_score:
                A = new_A
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_A = list(A)
            else:
                delta = new_score - current_score
                if delta < 0:
                    accept_prob = 1.0
                else:
                    accept_prob = math.exp(-delta / temp)
                
                if rng.random() < accept_prob:
                    A = new_A
                    current_score = new_score
            
            temp *= cooling
        
        return best_A, best_score
    
    # Run local search on the best candidate
    time_left = deadline - time.time()
    if time_left > 1.0:
        improved_A, improved_score = local_search_with_sa(best_A, time_left - 1.0, rng)
        if improved_score < best_score:
            best_score = improved_score
            best_A = improved_A
    
    return best_A
