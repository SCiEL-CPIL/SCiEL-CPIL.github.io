import random
import math
import time

def construct():
    # Target: 20 distinct positive integers <= 10000
    # Minimize max(|A+A|, |A*A|)
    
    N = 20
    MAX_VAL = 10000
    
    # Precompute small primes for multiplicative mutations
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    def score(A):
        """Compute max(|A+A|, |A*A|) for set A."""
        sums = set()
        prods = set()
        n = len(A)
        # Use list for faster access if needed, but set is fine for n=20
        for i in range(n):
            ai = A[i]
            for j in range(i, n):
                aj = A[j]
                sums.add(ai + aj)
                prods.add(ai * aj)
        return max(len(sums), len(prods))
    
    def valid_set(A):
        """Check if A is a valid set of N distinct positive integers <= MAX_VAL."""
        if len(A) != N:
            return False
        if len(set(A)) != N:
            return False
        if any(x < 1 or x > MAX_VAL for x in A):
            return False
        return True

    def generate_initial_population(pop_size=50):
        """Generate diverse initial candidates."""
        pop = []
        
        # Strategy 1: Geometric Progressions (good for products)
        for r in [1.1, 1.2, 1.3, 1.5, 2.0]:
            start = random.randint(1, 5)
            A = []
            val = start
            for _ in range(N):
                if val > MAX_VAL: break
                A.append(int(val))
                val *= r
            if len(A) == N and valid_set(A):
                pop.append(sorted(A))
        
        # Strategy 2: Arithmetic Progressions (good for sums)
        for d in [1, 2, 3, 5, 10]:
            start = random.randint(1, 10)
            A = [start + i * d for i in range(N)]
            if valid_set(A):
                pop.append(sorted(A))
        
        # Strategy 3: Highly Composite / Smooth Numbers
        # Generate numbers of form 2^a 3^b 5^c
        candidates = set()
        for a in range(15):
            for b in range(10):
                for c in range(8):
                    v = (2 ** a) * (3 ** b) * (5 ** c)
                    if 1 <= v <= MAX_VAL:
                        candidates.add(v)
        candidates = sorted(candidates)
        # Randomly pick N from this smooth pool
        for _ in range(20):
            if len(candidates) >= N:
                A = random.sample(candidates, N)
                if valid_set(A):
                    pop.append(sorted(A))
        
        # Strategy 4: Random Small Integers
        for _ in range(20):
            # Bias towards smaller numbers to encourage collisions
            A = set()
            while len(A) < N:
                # Exponential distribution bias towards small numbers
                x = int(random.expovariate(0.05)) + 1
                if x <= MAX_VAL:
                    A.add(x)
            if valid_set(list(A)):
                pop.append(sorted(list(A)))
                
        # Fill rest with random valid sets if needed
        while len(pop) < pop_size:
            A = random.sample(range(1, MAX_VAL + 1), N)
            if valid_set(A):
                pop.append(sorted(A))
                
        return pop

    def mutate(A, temperature):
        """Apply mutation to set A."""
        A = list(A)
        mutation_type = random.random()
        
        if mutation_type < 0.4:
            # Additive mutation: shift one element
            idx = random.randint(0, N - 1)
            delta = random.randint(-50, 50)
            new_val = A[idx] + delta
            if 1 <= new_val <= MAX_VAL and new_val not in A:
                A[idx] = new_val
            else:
                # Try smaller delta
                for _ in range(10):
                    delta = random.randint(-5, 5)
                    new_val = A[idx] + delta
                    if 1 <= new_val <= MAX_VAL and new_val not in A:
                        A[idx] = new_val
                        break
        elif mutation_type < 0.7:
            # Multiplicative mutation: multiply/divide by small prime
            idx = random.randint(0, N - 1)
            p = random.choice(primes[:5]) # Use small primes
            op = random.choice([1, -1])
            if op == 1:
                new_val = A[idx] * p
            else:
                if A[idx] % p == 0:
                    new_val = A[idx] // p
                else:
                    new_val = A[idx] * p # Fallback to multiply if not divisible
            
            if 1 <= new_val <= MAX_VAL and new_val not in A:
                A[idx] = new_val
            else:
                # Try another prime
                for p2 in primes[:5]:
                    if p2 == p: continue
                    new_val = A[idx] * p2
                    if 1 <= new_val <= MAX_VAL and new_val not in A:
                        A[idx] = new_val
                        break
        else:
            # Replacement: Replace one element with a "good" neighbor
            idx = random.randint(0, N - 1)
            # Generate a candidate that might create collisions
            # Pick a random existing element x, and try x +/- small or x * small
            base = random.choice(A)
            candidates = set()
            for delta in range(-20, 21):
                v = base + delta
                if 1 <= v <= MAX_VAL: candidates.add(v)
            for p in primes[:3]:
                v = base * p
                if 1 <= v <= MAX_VAL: candidates.add(v)
                if base % p == 0:
                    v = base // p
                    if 1 <= v <= MAX_VAL: candidates.add(v)
            
            candidates.difference_update(A)
            if candidates:
                new_val = random.choice(list(candidates))
                A[idx] = new_val
        
        A.sort()
        return A

    def crossover(A, B):
        """Uniform crossover."""
        C = []
        for i in range(N):
            if random.random() < 0.5:
                C.append(A[i])
            else:
                C.append(B[i])
        # Resolve duplicates and validity
        C = list(set(C))
        while len(C) < N:
            # Fill missing with random valid numbers
            # Try to pick numbers that are close to existing ones to maintain structure
            if C:
                base = random.choice(C)
                new_val = base + random.randint(-10, 10)
                if 1 <= new_val <= MAX_VAL and new_val not in C:
                    C.append(new_val)
                else:
                    C.append(random.randint(1, MAX_VAL))
            else:
                C.append(random.randint(1, MAX_VAL))
        
        # Ensure distinctness and bounds
        C = list(set(C))
        if len(C) < N:
             # If we still have duplicates or out of bounds, regenerate entirely
             return None
        C = [x for x in C if 1 <= x <= MAX_VAL]
        if len(C) < N:
            return None
        C = C[:N]
        C.sort()
        if valid_set(C):
            return C
        return None

    # Main GA Loop
    pop_size = 60
    max_generations = 200
    start_time = time.time()
    time_limit = 55.0 # Leave 5s buffer
    
    population = generate_initial_population(pop_size)
    
    # Score initial population
    scored_pop = [(score(ind), ind) for ind in population]
    scored_pop.sort(key=lambda x: x[0])
    best_score, best_set = scored_pop[0]
    
    # Simulated Annealing parameters
    initial_temp = 100.0
    cooling_rate = 0.95
    min_temp = 1.0
    
    for gen in range(max_generations):
        if time.time() - start_time > time_limit:
            break
            
        current_temp = max(initial_temp * (cooling_rate ** gen), min_temp)
        new_population = []
        
        # Elitism: Keep top 10%
        elite_size = max(5, pop_size // 10)
        for i in range(elite_size):
            new_population.append(scored_pop[i][1])
            
        # Generate rest of population
        while len(new_population) < pop_size:
            # Tournament Selection
            s1 = random.randint(0, len(scored_pop) - 1)
            s2 = random.randint(0, len(scored_pop) - 1)
            if scored_pop[s1][0] <= scored_pop[s2][0]:
                parent1 = scored_pop[s1][1]
            else:
                parent1 = scored_pop[s2][1]
                
            s3 = random.randint(0, len(scored_pop) - 1)
            s4 = random.randint(0, len(scored_pop) - 1)
            if scored_pop[s3][0] <= scored_pop[s4][0]:
                parent2 = scored_pop[s3][1]
            else:
                parent2 = scored_pop[s4][1]
                
            # Crossover
            child = crossover(parent1, parent2)
            if child is None:
                child = list(parent1)
                
            # Mutation
            child = mutate(child, current_temp)
            
            if valid_set(child):
                new_population.append(child)
            else:
                # If mutation broke validity, retry or keep parent
                if random.random() < 0.5:
                    new_population.append(list(parent1))
                else:
                    new_population.append(list(parent2))
        
        # Score new population
        scored_pop = [(score(ind), ind) for ind in new_population]
        scored_pop.sort(key=lambda x: x[0])
        
        # Update best
        if scored_pop[0][0] < best_score:
            best_score = scored_pop[0][0]
            best_set = scored_pop[0][1]
            
        # Local Refinement of Best
        # Perform a few local swaps on the best individual to fine-tune
        if gen % 10 == 0:
            local_best = list(best_set)
            local_best_score = best_score
            for _ in range(50):
                idx = random.randint(0, N-1)
                delta = random.randint(-10, 10)
                new_val = local_best[idx] + delta
                if 1 <= new_val <= MAX_VAL and new_val not in local_best:
                    old_val = local_best[idx]
                    local_best[idx] = new_val
                    local_best.sort()
                    s = score(local_best)
                    if s < local_best_score:
                        local_best_score = s
                        best_score = s
                        best_set = local_best
                    else:
                        local_best[idx] = old_val
                        local_best.sort()
                        
    return best_set
