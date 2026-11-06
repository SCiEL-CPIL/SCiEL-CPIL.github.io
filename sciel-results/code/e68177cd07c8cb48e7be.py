import time
import random
from math import gcd

def construct():
    # Deadline guard: 90 seconds total, stop at 85s to be safe
    start_time = time.time()
    deadline = start_time + 85.0
    
    MAX_VAL = 10000
    N = 12
    
    # 1. Precompute prime factors for all numbers up to 2 * MAX_VAL
    # Sieve of Eratosthenes variant to store prime factors
    max_sum = 2 * MAX_VAL
    prime_factors = [[] for _ in range(max_sum + 1)]
    
    # Initialize
    for i in range(2, max_sum + 1):
        if not prime_factors[i]:
            # i is prime
            for j in range(i, max_sum + 1, i):
                prime_factors[j].append(i)
    
    # Convert to frozensets for faster union operations if needed, or keep lists
    # Using lists is fine, we just need to union them
    def get_prime_factors(x):
        return prime_factors[x]
    
    def evaluate(arr):
        """
        Returns the number of distinct prime factors of all pairwise sums.
        """
        primes = set()
        n = len(arr)
        for i in range(n):
            for j in range(i + 1, n):
                s = arr[i] + arr[j]
                # Add factors of s to primes
                for p in prime_factors[s]:
                    primes.add(p)
        return len(primes), primes

    # 2. Initial Candidates
    best_arr = None
    best_score = float('inf')
    
    # Candidate 1: Multiples of 2
    cand1 = list(range(2, 2*N+2, 2))
    score1, _ = evaluate(cand1)
    if score1 < best_score:
        best_score = score1
        best_arr = cand1[:]
        
    # Candidate 2: Multiples of 3
    cand2 = list(range(3, 3*N+3, 3))
    score2, _ = evaluate(cand2)
    if score2 < best_score:
        best_score = score2
        best_arr = cand2[:]
        
    # Candidate 3: Multiples of 6
    cand3 = list(range(6, 6*N+6, 6))
    score3, _ = evaluate(cand3)
    if score3 < best_score:
        best_score = score3
        best_arr = cand3[:]
        
    # Candidate 4: Random
    for _ in range(10):
        cand4 = random.sample(range(1, MAX_VAL + 1), N)
        score4, _ = evaluate(cand4)
        if score4 < best_score:
            best_score = score4
            best_arr = cand4[:]
            
    # 3. Local Search (Hill Climbing with random restarts if stuck)
    current_arr = best_arr[:]
    current_score = best_score
    
    # Precompute a set of current values for O(1) uniqueness check
    current_set = set(current_arr)
    
    # Simulated Annealing parameters
    temperature = 1.0
    cooling_rate = 0.99
    
    while time.time() < deadline:
        # Pick a random index to replace
        idx = random.randint(0, N - 1)
        old_val = current_arr[idx]
        
        # Generate a new value
        # Bias towards small numbers? Or random?
        # Small numbers tend to have fewer/larger prime factors? 
        # Actually, sums of small numbers are small, having fewer potential prime factors.
        # Let's try picking from a range biased towards smaller numbers
        new_val = random.randint(1, MAX_VAL)
        
        if new_val in current_set:
            continue
            
        # Try the move
        current_arr[idx] = new_val
        current_set.remove(old_val)
        current_set.add(new_val)
        
        new_score, _ = evaluate(current_arr)
        
        # Acceptance criterion
        if new_score < current_score:
            # Always accept improvement
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_arr = current_arr[:]
        else:
            # Reject (Hill Climbing)
            # Revert
            current_arr[idx] = old_val
            current_set.remove(new_val)
            current_set.add(old_val)
            
        # Optional: Random restart if no improvement for a while
        # For simplicity, we stick to hill climbing on the best found so far
        # We can occasionally perturb the best solution to escape local minima
        if random.random() < 0.01:
             # Perturb best_arr
             temp_arr = best_arr[:]
             for _ in range(2):
                 i = random.randint(0, N-1)
                 temp_arr[i] = random.randint(1, MAX_VAL)
             # Ensure distinct
             while len(set(temp_arr)) < N:
                 i = random.randint(0, N-1)
                 temp_arr[i] = random.randint(1, MAX_VAL)
             s, _ = evaluate(temp_arr)
             if s < best_score:
                 best_score = s
                 best_arr = temp_arr[:]
             # Update current to best if it's better than current
             if best_score < current_score:
                 current_arr = best_arr[:]
                 current_score = best_score
                 current_set = set(current_arr)

    return best_arr

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard: 90 seconds total, stop at 85s to be safe
    start_time = time.time()
    deadline = start_time + 85.0
    
    MAX_VAL = 10000
    N = 12
    
    # 1. Precompute prime factors for all numbers up to 2 * MAX_VAL
    max_sum = 2 * MAX_VAL
    prime_factors = [[] for _ in range(max_sum + 1)]
    
    for i in range(2, max_sum + 1):
        if not prime_factors[i]:
            for j in range(i, max_sum + 1, i):
                prime_factors[j].append(i)
    
    def evaluate(arr):
        primes = set()
        n = len(arr)
        for i in range(n):
            for j in range(i + 1, n):
                s = arr[i] + arr[j]
                for p in prime_factors[s]:
                    primes.add(p)
        return len(primes), primes

    # 2. Initial Candidates
    best_arr = None
    best_score = float('inf')
    
    # Try multiples of small primes
    for m in [2, 3, 4, 5, 6, 10, 12]:
        cand = list(range(m, m*N+m, m))
        if cand[-1] > MAX_VAL:
            continue
        score, _ = evaluate(cand)
        if score < best_score:
            best_score = score
            best_arr = cand[:]
            
    # Random initializations
    for _ in range(20):
        cand = random.sample(range(1, MAX_VAL + 1), N)
        score, _ = evaluate(cand)
        if score < best_score:
            best_score = score
            best_arr = cand[:]
            
    # 3. Local Search
    current_arr = best_arr[:]
    current_score = best_score
    current_set = set(current_arr)
    
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline:
        idx = random.randint(0, N - 1)
        old_val = current_arr[idx]
        
        # Heuristic: try values in the same "scale" as current or small values
        # Small values often lead to smaller sums, which have fewer prime factors
        # But we need distinctness.
        
        # Strategy: Pick a random new value
        new_val = random.randint(1, MAX_VAL)
        
        if new_val in current_set:
            continue
            
        current_arr[idx] = new_val
        current_set.remove(old_val)
        current_set.add(new_val)
        
        new_score, _ = evaluate(current_arr)
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_arr = current_arr[:]
            no_improve_count = 0
        else:
            current_arr[idx] = old_val
            current_set.remove(new_val)
            current_set.add(old_val)
            no_improve_count += 1
            
        if no_improve_count > max_no_improve:
            # Restart from best
            current_arr = best_arr[:]
            current_score = best_score
            current_set = set(current_arr)
            no_improve_count = 0
            max_no_improve = int(max_no_improve * 1.5)

    return best_arr
# EVOLVE-BLOCK-END
