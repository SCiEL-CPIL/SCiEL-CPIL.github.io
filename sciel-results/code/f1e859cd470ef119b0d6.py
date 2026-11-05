import numpy as np
import time

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    time_limit = 250  # leave margin under 300s

    # Preprocess clauses for NumPy vectorization
    # M[i, j] = 1 if clause i has literal +j, -1 if -j, 0 otherwise
    num_clauses = len(clauses)
    M = np.zeros((num_clauses, n), dtype=np.int8)
    for i, c in enumerate(clauses):
        for l in c:
            v = abs(l) - 1
            s = 1 if l > 0 else -1
            M[i, v] = s

    def get_energy(A):
        # A is int8 array of 1s and -1s
        # L = M @ A
        # Clause satisfied if max(L[i]) == 1
        # Unsatisfied if max(L[i]) == 0
        L = M @ A
        max_L = np.max(L, axis=1)
        return np.sum(max_L == 0)

    def simulated_annealing(seed, max_flips, T_start=10.0, T_end=0.1, cooling_rate=0.999):
        rng = np.random.RandomState(seed)
        # Initialize A with random 1s and -1s
        A = rng.choice([1, -1], size=n).astype(np.int8)
        
        current_energy = get_energy(A)
        if current_energy == 0:
            return A
        
        T = T_start
        
        for flip in range(max_flips):
            if time.time() - start_time > time_limit:
                return None
            
            # Pick a random variable to flip
            # In SA, we usually pick a random neighbor (flip one bit)
            v = rng.randint(n)
            
            # Flip variable v
            A[v] *= -1
            
            new_energy = get_energy(A)
            delta = new_energy - current_energy
            
            # Acceptance criterion
            if delta < 0:
                current_energy = new_energy
            else:
                # Accept with probability exp(-delta / T)
                # Use log to avoid underflow
                # log(exp(-delta/T)) = -delta/T
                # log(rng) < -delta/T  => rng < exp(-delta/T)
                # To avoid calculating exp, we can do:
                # -log(rng) > delta/T  => T > delta / (-log(rng))
                # But simple exp is fine for small delta/T
                if delta == 0:
                    if rng.random() < 0.5:
                        current_energy = new_energy
                    else:
                        A[v] *= -1 # revert
                else:
                    # Calculate probability
                    # If T is very small, prob is 0
                    if T > 0.01:
                        prob = np.exp(-delta / T)
                    else:
                        prob = 0.0
                    
                    if rng.random() < prob:
                        current_energy = new_energy
                    else:
                        A[v] *= -1 # revert
                        
            # Cool down
            T *= cooling_rate
            if T < T_end:
                T = T_end
                
        return None

    # Run multiple restarts with different seeds and temperature profiles
    seeds = [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
    
    for seed in seeds:
        if time.time() - start_time > time_limit:
            break
        
        # Dynamic flip budget: use remaining time
        # Estimate flips per second: ~10000
        # Let's just give a large budget and let the time check handle it
        result_A = simulated_annealing(seed, 100000, T_start=5.0, T_end=0.1, cooling_rate=0.9995)
        if result_A is not None:
            # Convert back to list of booleans
            # A contains 1 (True) and -1 (False)
            return [bool(val) for val in result_A]
            
    # Fallback: Random assignment
    rng = np.random.RandomState(0)
    A = rng.choice([1, -1], size=n).astype(np.int8)
    return [bool(val) for val in A]
