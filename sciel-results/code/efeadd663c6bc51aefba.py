import time
import random
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s margin for 90s limit
    
    # 1. Generate Candidate Pool
    candidates = set()
    
    # Add small integers (good for sums)
    for i in range(1, 101):
        candidates.add(i)
        
    # Add 5-smooth numbers (good for products)
    # 2^a * 3^b * 5^c <= 10000
    for a in range(15): # 2^14 > 10000
        v2 = 2**a
        if v2 > 10000: break
        for b in range(11): # 3^10 > 10000
            v23 = v2 * (3**b)
            if v23 > 10000: break
            for c in range(8): # 5^7 > 10000
                v235 = v23 * (5**c)
                if v235 > 10000: break
                candidates.add(v235)
                
    # Add numbers with small prime factors (7, 11, 13)
    for p in [7, 11, 13]:
        for k in range(1, 10000//p + 1):
            val = p * k
            if val <= 10000:
                candidates.add(val)
            else:
                break
                
    # Add random numbers to ensure diversity
    for _ in range(500):
        candidates.add(random.randint(1, 10000))
        
    candidates = list(candidates)
    random.shuffle(candidates)
    
    # 2. Evaluation Function
    def evaluate(A):
        A = list(A)
        sums = set()
        prods = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                sums.add(A[i] + A[j])
                prods.add(A[i] * A[j])
        return max(len(sums), len(prods))

    # 3. Initialization
    # Start with a mix of small and smooth numbers
    # Sort candidates by value, pick first 20 distinct
    # But we want diversity. Let's start with a random subset biased to small/smooth
    # We'll use a simple greedy init: pick 20 random from first 200 candidates
    initial_pool = candidates[:500]
    current_A = set(random.sample(initial_pool, 20))
    
    best_A = set(current_A)
    best_score = evaluate(best_A)
    current_score = best_score
    
    # 4. Local Search with Simulated Annealing
    # Parameters
    T = 1.0
    cooling = 0.9995
    min_T = 0.01
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        # Check for restart if stuck
        if no_improve_count > max_no_improve:
            # Restart with a new random population from the pool
            current_A = set(random.sample(candidates, 20))
            current_score = evaluate(current_A)
            if current_score < best_score:
                best_A = set(current_A)
                best_score = current_score
            no_improve_count = 0
            T = 1.0 # Reset temperature
            
        # Propose a move: remove x, add y
        x = random.choice(list(current_A))
        y = random.choice(candidates)
        
        if y == x:
            continue
            
        if y in current_A:
            continue
            
        new_A = set(current_A)
        new_A.remove(x)
        new_A.add(y)
        
        new_score = evaluate(new_A)
        
        # Acceptance criterion
        if new_score < current_score:
            current_A = new_A
            current_score = new_score
            no_improve_count = 0
            
            if current_score < best_score:
                best_A = set(current_A)
                best_score = current_score
                # Optional: slightly increase cooling rate if improving fast? No, keep simple.
        else:
            # Simulated annealing acceptance
            if T > min_T:
                delta = new_score - current_score
                # delta is positive
                if random.random() < math.exp(-delta / T):
                    current_A = new_A
                    current_score = new_score
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
                
        T *= cooling
        
    # Final Polish: Try all single swaps on the best set found
    # This is a deterministic local search pass
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_in_pass = best_score
        best_set_in_pass = best_A
        
        for x in best_A:
            for y in candidates:
                if y == x or y in best_A:
                    continue
                new_A = set(best_A)
                new_A.remove(x)
                new_A.add(y)
                new_score = evaluate(new_A)
                if new_score < best_in_pass:
                    best_in_pass = new_score
                    best_set_in_pass = new_A
                    improved = True
                    
        if improved:
            best_A = best_set_in_pass
            best_score = best_in_pass
            
    return list(best_A)

# EVOLVE-BLOCK-END
