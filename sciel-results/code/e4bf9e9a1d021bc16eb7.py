"""Planted 3-SAT solver using Simulated Annealing with NumPy acceleration."""
import numpy as np
import time
import random

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    time_limit = 290  # Leave margin under 300s

    M = len(clauses)
    if M == 0:
        return [False] * n

    # Preprocess clauses for NumPy
    vars_arr = np.zeros((M, 3), dtype=np.int64)
    signs_arr = np.zeros((M, 3), dtype=bool)
    mask = np.zeros((M, 3), dtype=bool)
    
    for i, c in enumerate(clauses):
        for j, lit in enumerate(c):
            if j < 3: # Safety check
                v = abs(lit) - 1
                vars_arr[i, j] = v
                signs_arr[i, j] = (lit > 0)
                mask[i, j] = True

    def get_energy(assignment):
        """Returns number of unsatisfied clauses."""
        lit_vals = assignment[vars_arr]
        is_sat_lit = (lit_vals == signs_arr) & mask
        clause_sat = np.any(is_sat_lit, axis=1)
        return np.sum(~clause_sat)

    def solve_sa(seed, max_iters):
        rng = np.random.default_rng(seed)
        
        # Initial random assignment
        assignment = rng.random(n) < 0.5
        
        # Simulated Annealing parameters
        T_start = 1.0
        T_end = 0.001
        if max_iters == 0:
            return None
        cooling_rate = (T_end / T_start) ** (1.0 / max_iters)
        
        current_energy = get_energy(assignment)
        best_energy = current_energy
        best_assignment = assignment.copy()
        
        # Track stagnation
        stagnation = 0
        max_stagnation = 5000
        
        for i in range(max_iters):
            if time.time() - start_time > time_limit:
                return None
            
            # Select variable to flip
            v = rng.integers(0, n)
            
            # Calculate delta energy
            assignment[v] = not assignment[v]
            new_energy = get_energy(assignment)
            
            delta_e = new_energy - current_energy
            
            # Acceptance criteria
            if delta_e <= 0:
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_assignment = assignment.copy()
                    stagnation = 0
                else:
                    stagnation += 1
            else:
                T = T_start * (cooling_rate ** i)
                # Avoid division by zero or underflow issues
                if T > 0:
                    prob = np.exp(-delta_e / T)
                else:
                    prob = 0.0
                if rng.random() < prob:
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_assignment = assignment.copy()
                        stagnation = 0
                    else:
                        stagnation += 1
                else:
                    assignment[v] = not assignment[v]
                    stagnation += 1
            
            # Restart if stuck
            if stagnation > max_stagnation:
                assignment = rng.random(n) < 0.5
                current_energy = get_energy(assignment)
                stagnation = 0
                # Optionally reset temperature? 
                # We'll just continue with the current cooling schedule.
                
        if best_energy == 0:
            return best_assignment.tolist()
        return None

    # Try multiple seeds
    # We can afford more iterations due to NumPy speed
    seeds = [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000,
             7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000]

    for i, seed in enumerate(seeds):
        if time.time() - start_time > time_limit:
            break
        # Increase iterations for later seeds? Or keep constant?
        # Constant is fine, different seeds provide diversity.
        max_iters = 50000
        result = solve_sa(seed, max_iters)
        if result is not None:
            return result
            
    # Final fallback: return random assignment
    rng = np.random.default_rng(0)
    return (rng.random(n) < 0.5).tolist()
