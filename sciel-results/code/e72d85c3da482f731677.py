import numpy as np
import time

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit

    # 1. Precompute Arithmetic Progressions (APs)
    # We focus on "critical" APs: those likely to drive the maximum absolute sum.
    # Singletons (k=1) have sum +/-1, so they are never the max if any k>=2 AP has sum >= 2.
    # For n=256, the max sum will likely be > 2.
    # We select APs with length >= 3.
    # To manage memory and speed, we don't need ALL APs for the search if we verify at the end,
    # but for a GA, we need a consistent fitness function.
    # Let's include all APs with length >= 3.
    # Number of such APs:
    # Sum over step s from 1 to 127 of (n - 2s) roughly.
    # This is manageable.
    
    all_progs = []
    for a in range(n):
        for s in range(1, n):
            idx = list(range(a, n, s))
            if len(idx) >= 3:
                all_progs.append(idx)
                
    Np = len(all_progs)
    
    # Build incidence matrix M (n x Np)
    # M[i, j] = 1 if position i is in AP j
    M = np.zeros((n, Np), dtype=np.float32)
    for j, idx in enumerate(all_progs):
        for i in idx:
            M[i, j] = 1.0
            
    # Precompute row norms? Not strictly needed for max abs sum, but M.T @ signs gives sums.
    
    def evaluate(signs):
        """Compute max absolute sum over all APs."""
        sums = M.T @ signs
        return np.max(np.abs(sums))
        
    def get_worst_violation(signs):
        """Return index of the AP with the largest absolute sum."""
        sums = M.T @ signs
        return np.argmax(np.abs(sums))
        
    rng = np.random.RandomState(42)
    
    # 2. Initialize Population
    # Use diverse initializations:
    # - Random
    # - Thue-Morse (known low discrepancy properties)
    # - Alternating
    # - Random blocks
    
    pop_size = 50
    population = []
    
    # Thue-Morse sequence: t(n) = parity of number of 1s in binary representation of n
    # Map 0 -> 1, 1 -> -1
    thue_morse = np.array([1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)], dtype=np.float32)
    population.append(thue_morse.copy())
    
    # Alternating
    alt = np.array([1 if i % 2 == 0 else -1 for i in range(n)], dtype=np.float32)
    population.append(alt.copy())
    
    # Random
    for _ in range(pop_size - 2):
        population.append(rng.choice([-1, 1], size=n).astype(np.float32))
        
    # Evaluate initial population
    scores = np.array([evaluate(ind) for ind in population])
    
    best_score = np.min(scores)
    best_idx = np.argmin(scores)
    best_individual = population[best_idx].copy()
    
    # 3. Genetic Algorithm Loop
    # Parameters
    tournament_size = 5
    crossover_rate = 0.8
    mutation_rate = 0.05  # Probability per bit
    elitism_count = 5
    
    gen = 0
    max_gens = 500 # Limit by time instead
    
    while time.time() < deadline - 5:
        gen += 1
        
        # Selection: Tournament Selection
        # We want to keep the best individuals.
        # Sort population by score (ascending)
        sorted_indices = np.argsort(scores)
        population = [population[i] for i in sorted_indices]
        scores = scores[sorted_indices]
        
        # Elitism: Copy top elites to next generation
        next_population = [population[i].copy() for i in range(elitism_count)]
        next_scores = list(scores[:elitism_count])
        
        # Fill rest of next population with offspring
        while len(next_population) < pop_size:
            # Select parents via tournament
            def tournament():
                contenders = rng.randint(0, pop_size, size=tournament_size)
                best_c = min(contenders, key=lambda x: scores[x])
                return population[best_c]
            
            p1 = tournament()
            p2 = tournament()
            
            # Crossover: Uniform Crossover
            if rng.rand() < crossover_rate:
                # Mask of which bits come from p1 vs p2
                mask = rng.rand(n) < 0.5
                child = np.where(mask, p1, p2).astype(np.float32)
            else:
                child = p1.copy()
                
            # Mutation
            if rng.rand() < mutation_rate:
                # Flip random bits
                num_flips = rng.randint(1, n // 10 + 1)
                flip_indices = rng.choice(n, size=num_flips, replace=False)
                child[flip_indices] *= -1
            
            # Evaluate child
            child_score = evaluate(child)
            next_population.append(child)
            next_scores.append(child_score)
            
        # Update population and scores
        population = next_population
        scores = np.array(next_scores)
        
        # Update global best
        current_best_idx = np.argmin(scores)
        if scores[current_best_idx] < best_score:
            best_score = scores[current_best_idx]
            best_individual = population[current_best_idx].copy()
            
    # 4. Final Polish with Local Search on the Best Individual
    # The GA might stop near a local optimum. A quick single-bit flip local search can polish it.
    current = best_individual.copy()
    current_score = evaluate(current)
    
    # Simple greedy local search
    improved = True
    while improved and time.time() < deadline - 1:
        improved = False
        # Try flipping each bit
        # To speed up, we can vectorize the evaluation of all single-bit flips
        # sums = M.T @ current
        # For each i, new_sums = sums - 2 * current[i] * M[i, :]
        # We want to minimize max(abs(new_sums))
        
        sums = M.T @ current
        
        # For each position i, compute the new max abs sum if we flip i
        # This is O(n * Np) which is 256 * ~10000 = 2.5M ops. Fast enough.
        
        best_i = -1
        best_new_score = current_score
        
        for i in range(n):
            # Flip i
            delta = -2.0 * current[i] * M[i, :]
            new_sums = sums + delta
            new_score = np.max(np.abs(new_sums))
            
            if new_score < best_new_score:
                best_new_score = new_score
                best_i = i
                
        if best_i != -1:
            current[best_i] *= -1
            current_score = best_new_score
            improved = True
            
    return [int(x) for x in current]

