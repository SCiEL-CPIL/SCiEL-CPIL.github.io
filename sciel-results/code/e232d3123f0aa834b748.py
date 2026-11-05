import time
import random
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 85.0  # 85 seconds to be safe (90s limit)
    
    n = 30
    # Precompute exact fractions
    terms = [Fraction(1, i) for i in range(1, n + 1)]
    
    def compute_sum_exact(coeff):
        s = Fraction(0, 1)
        for i in range(n):
            if coeff[i] != 0:
                s += coeff[i] * terms[i]
        return s
    
    def abs_frac(f):
        return abs(f)
    
    # Simulated annealing with exact arithmetic
    def simulated_annealing(coeff, start_time, deadline):
        current_coeff = coeff[:]
        current_sum = compute_sum_exact(current_coeff)
        current_abs = abs_frac(current_sum)
        
        if current_abs == 0:
            return current_coeff, current_sum, current_abs
        
        best_coeff = current_coeff[:]
        best_sum = current_sum
        best_abs = current_abs
        
        # Annealing parameters
        T_start = 0.5
        T_end = 0.001
        alpha = 0.995  # cooling rate per step
        
        T = T_start
        step = 0
        max_steps = 100000
        
        while T > T_end and (time.time() - start_time) < deadline and step < max_steps:
            # Pick a random position
            i = random.randint(0, n - 1)
            old_val = current_coeff[i]
            
            # Pick a new value different from old
            choices = [v for v in [-1, 0, 1] if v != old_val]
            new_val = random.choice(choices)
            
            # Compute new sum efficiently
            delta = (new_val - old_val) * terms[i]
            new_sum = current_sum + delta
            new_abs = abs_frac(new_sum)
            
            # Accept if zero (but we exclude zero, so reject)
            if new_abs == 0:
                T *= alpha
                step += 1
                continue
            
            # Accept if better, or with probability exp(-delta_T / T)
            if new_abs < current_abs:
                current_coeff[i] = new_val
                current_sum = new_sum
                current_abs = new_abs
                
                if current_abs < best_abs:
                    best_coeff = current_coeff[:]
                    best_sum = current_sum
                    best_abs = current_abs
            else:
                # Metropolis criterion: accept worse with probability exp(-(new_abs - current_abs) / T)
                delta_abs = new_abs - current_abs
                # Use log to avoid underflow
                log_prob = -delta_abs / T
                if log_prob > -50:  # avoid underflow
                    if random.random() < min(1.0, 2.718281828459045 ** log_prob):
                        current_coeff[i] = new_val
                        current_sum = new_sum
                        current_abs = new_abs
            
            T *= alpha
            step += 1
        
        return best_coeff, best_sum, best_abs
    
    # Population-based annealing with diversity
    population_size = 5
    best_overall_coeff = None
    best_overall_sum = Fraction(0, 1)
    best_overall_abs = Fraction(1, 0)  # infinity
    
    # Initialize population with diverse starts
    for p in range(population_size):
        if (time.time() - start_time) >= deadline:
            break
        
        if p == 0:
            # Start from seed-like solution
            coeff = [0] * 28 + [1, -1]
        elif p == 1:
            # Random
            coeff = [random.choice([-1, 0, 1]) for _ in range(n)]
        elif p == 2:
            # Greedy: add terms to minimize partial sum
            coeff = [0] * n
            s = Fraction(0, 1)
            for i in range(n - 1, -1, -1):
                for val in [-1, 1]:
                    new_s = s + val * terms[i]
                    if abs(new_s) < abs(s) or s == 0:
                        if new_s != 0:
                            coeff[i] = val
                            s = new_s
                            break
        elif p == 3:
            # All ones except one -1
            coeff = [1] * n
            idx = random.randint(0, n - 1)
            coeff[idx] = -1
        else:
            # All zeros except two random positions
            coeff = [0] * n
            idxs = random.sample(range(n), 2)
            coeff[idxs[0]] = 1
            coeff[idxs[1]] = random.choice([-1, 1])
        
        best_coeff, best_sum, best_abs = simulated_annealing(coeff, start_time, deadline)
        
        if best_abs < best_overall_abs:
            best_overall_coeff = best_coeff
            best_overall_sum = best_sum
            best_overall_abs = best_abs
    
    # If time remains, do more annealing from the best solution found
    if best_overall_coeff is not None and (time.time() - start_time) < deadline:
        # Run additional annealers from the best solution with small perturbations
        for extra in range(3):
            if (time.time() - start_time) >= deadline:
                break
            
            perturbed = best_overall_coeff[:]
            # Perturb 2-3 random positions
            for _ in range(random.randint(2, 3)):
                idx = random.randint(0, n - 1)
                perturbed[idx] = random.choice([-1, 0, 1])
            
            best_coeff, best_sum, best_abs = simulated_annealing(perturbed, start_time, deadline)
            
            if best_abs < best_overall_abs:
                best_overall_coeff = best_coeff
                best_overall_sum = best_sum
                best_overall_abs = best_abs
    
    # Final fallback
    if best_overall_coeff is None:
        return [0] * 28 + [1, -1]
    
    return best_overall_coeff
# EVOLVE-BLOCK-END
