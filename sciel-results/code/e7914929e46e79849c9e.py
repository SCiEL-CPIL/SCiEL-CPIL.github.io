import random
import time
import math

def construct():
    n = 20
    max_val = 10000
    time_limit = 55.0
    
    def get_score(A):
        sums = set()
        prods = set()
        for i in range(n):
            ai = A[i]
            for j in range(i, n):
                # Since A+A and A*A are symmetric and include i=j
                # The set of sums is {a_i + a_j | i, j in [0, n)}
                # The set of prods is {a_i * a_j | i, j in [0, n)}
                # We can just iterate all pairs to be safe and simple, 
                # or optimize by noting symmetry. 
                # For N=20, 400 iterations is trivial.
                pass
        
        # Optimized calculation
        for i in range(n):
            ai = A[i]
            for j in range(n):
                aj = A[j]
                sums.add(ai + aj)
                prods.add(ai * aj)
        
        return max(len(sums), len(prods))

    def generate_initial_set():
        # Strategy: Start with a set that has small product set.
        # Powers of 2 have product set size 2n-1 = 39.
        # Sum set size for powers of 2 is n(n+1)/2 = 210 (all unique).
        # We want to reduce sum set size by introducing collisions.
        # Collisions in sums: a_i + a_j = a_k + a_l.
        # This happens if the set is not "Sidon".
        # We start with a mix of small powers and small integers.
        
        # Base: Powers of 2 up to 2^19 is too big (524288).
        # Max val 10000. 2^13 = 8192. So we can have 14 powers of 2.
        # Let's take 10 powers of 2 and 6 small integers.
        
        A = []
        # 10 powers of 2
        for i in range(10):
            A.append(2 ** i)
        
        # 6 small integers, distinct from powers and each other
        # Choose small numbers to encourage sum collisions with powers?
        # Or just small numbers to keep products small?
        # Small numbers * Powers of 2 are still distinct if the small numbers are odd?
        # If we use small odd numbers, products might stay distinct.
        # Let's use small even numbers? No, distinctness.
        
        candidates = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
        random.shuffle(candidates)
        
        for c in candidates:
            if c not in A:
                A.append(c)
            if len(A) == n:
                break
        
        # Ensure we have n distinct elements
        while len(A) < n:
            # Pick random small numbers
            val = random.randint(1, 50)
            if val not in A:
                A.append(val)
        
        return A[:n]

    def simulated_annealing(initial_set, time_limit):
        current_set = initial_set[:]
        current_score = get_score(current_set)
        best_set = current_set[:]
        best_score = current_score
        
        start_time = time.time()
        T = 1000.0
        cooling_rate = 0.9995
        
        while time.time() - start_time < time_limit:
            if T < 0.01:
                T = 0.01
                
            # Generate a neighbor
            move_type = random.random()
            
            if move_type < 0.4:
                # Replace one element with a random value
                i = random.randint(0, n - 1)
                new_val = random.randint(1, max_val)
                if new_val not in current_set:
                    new_set = current_set[:]
                    new_set[i] = new_val
                else:
                    continue
            elif move_type < 0.7:
                # Replace one element with a value derived from others (sum/diff)
                i = random.randint(0, n - 1)
                x, y = random.sample(range(n), 2)
                if random.random() < 0.5:
                    candidate_val = current_set[x] + current_set[y]
                else:
                    candidate_val = abs(current_set[x] - current_set[y])
                
                if 1 <= candidate_val <= max_val and candidate_val not in current_set:
                    new_set = current_set[:]
                    new_set[i] = candidate_val
                else:
                    continue
            else:
                # Perturb an element by a small amount
                i = random.randint(0, n - 1)
                delta = random.choice([-2, -1, 1, 2, -3, 3, -5, 5])
                new_val = current_set[i] + delta
                if 1 <= new_val <= max_val and new_val not in current_set:
                    new_set = current_set[:]
                    new_set[i] = new_val
                else:
                    continue
            
            new_score = get_score(new_set)
            
            # Acceptance criterion
            if new_score < current_score:
                current_set = new_set
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_set = current_set[:]
            else:
                # Accept worse move with probability exp(-delta/T)
                delta = new_score - current_score
                if delta > 0:
                    prob = math.exp(-delta / T)
                    if random.random() < prob:
                        current_set = new_set
                        current_score = new_score
            
            T *= cooling_rate
        
        return best_set

    # Run multiple independent SA runs with different initial seeds
    best_overall_set = None
    best_overall_score = float('inf')
    
    start_time = time.time()
    num_runs = 5
    
    for run in range(num_runs):
        if time.time() - start_time > time_limit:
            break
            
        initial = generate_initial_set()
        # Add some randomness to initial set
        for _ in range(5):
            i = random.randint(0, n-1)
            val = random.randint(1, max_val)
            if val not in initial:
                initial[i] = val
        
        # Allocate remaining time to this run
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed
        time_per_run = remaining / (num_runs - run)
        
        # Ensure at least 5 seconds per run
        if time_per_run < 5:
            time_per_run = 5
            
        result_set = simulated_annealing(initial, time_per_run)
        result_score = get_score(result_set)
        
        if result_score < best_overall_score:
            best_overall_score = result_score
            best_overall_set = result_set[:]

    if best_overall_set is None:
        best_overall_set = list(range(1, n+1))
        
    return best_overall_set
