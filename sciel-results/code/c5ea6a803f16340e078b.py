import numpy as np
import math
import time
import random

def construct():
    # Start with a simple seed
    best_U = [0, 1, 3]
    best_score = -np.inf
    
    # Time limit
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer
    
    # Initial parameters for Simulated Annealing
    # We will search for sets of varying sizes and max values
    # Let's try to optimize a set of fixed size N? Or variable?
    # Variable size is harder to manage in SA.
    # Let's fix N and M? Or just let M grow?
    # Let's try a fixed N first, say N=10, 20, 50.
    # Or better, start with a small set and grow/shrink.
    
    # Let's use a representation that allows fast evaluation.
    # If M is large, FFT is expensive.
    # Let's restrict M to a reasonable range, e.g., < 10000.
    
    def evaluate(U):
        if not U:
            return -np.inf
        if 0 not in U:
            return -np.inf
        
        U_set = set(U)
        N = len(U_set)
        M = max(U_set)
        
        if M == 0:
            return 1.0
            
        # Direct calculation is faster for N < 200
        # Sumset
        sumset = set()
        diffset = set()
        
        # Optimization: If N is small, nested loops are fine.
        # We can also use numpy for vectorized operations if N is larger, 
        # but for SA, N is likely kept moderate.
        U_list = list(U_set)
        
        # Using sets for O(1) lookup/add
        # To speed up, we can compute sums/diffs using list comprehension and set()
        # sumset = {a+b for a in U_list for b in U_list}
        # diffset = {a-b for a in U_list for b in U_list}
        
        # This is O(N^2) which is acceptable for N up to ~100-200 in Python if optimized?
        # Let's try to be efficient.
        
        sumset = set()
        diffset = set()
        
        for a in U_list:
            for b in U_list:
                sumset.add(a + b)
                diffset.add(a - b)
                
        sumset_size = len(sumset)
        diffset_size = len(diffset)
        
        if sumset_size == 0:
            return -np.inf
            
        ratio = diffset_size / sumset_size
        
        denom = math.log(2 * M + 1)
        if denom == 0:
            return 1.0
            
        score = 1 + math.log(ratio) / denom
        return score

    # Improved Simulated Annealing
    best_overall = -np.inf
    best_U_overall = [0, 1, 3]
    
    # Try multiple random restarts with different initial structures
    # Structures: Geometric, Arithmetic, Random, Modular
    
    def init_geometric(N):
        U = [0]
        curr = 1
        for _ in range(N-1):
            U.append(curr)
            curr *= 2
        return U

    def init_arithmetic(N, step):
        U = [i * step for i in range(N)]
        return U

    def init_random(N, max_val):
        U = [0]
        while len(U) < N:
            v = random.randint(1, max_val)
            if v not in U:
                U.append(v)
        return sorted(U)

    def init_modular(N, base):
        # Set based on residues
        U = [0]
        for i in range(1, N):
            # Try to create structure
            U.append(i * base)
        return sorted(set(U))

    initializers = [
        lambda: init_geometric(10),
        lambda: init_arithmetic(10, 1),
        lambda: init_arithmetic(10, 2),
        lambda: init_random(10, 50),
        lambda: init_modular(10, 3),
        lambda: init_random(20, 100),
        lambda: init_geometric(20),
    ]
    
    for init_func in initializers:
        if time.time() - start_time > time_limit:
            break
            
        U = init_func()
        # Ensure 0 is in U and unique
        U = sorted(list(set(U)))
        if 0 not in U:
            U.append(0)
            U = sorted(U)
            
        current_score = evaluate(U)
        if current_score > best_overall:
            best_overall = current_score
            best_U_overall = U.copy()
            
        T = 0.5
        T_min = 0.01
        cooling_rate = 0.995 # Slower cooling
        
        # Allow variable size
        while T > T_min and (time.time() - start_time) < time_limit:
            move_type = random.random()
            U_temp = U.copy()
            
            # Perturbation strategies
            if move_type < 0.4:
                # Modify an existing element
                if len(U_temp) > 1:
                    idx = random.randint(1, len(U_temp) - 1)
                    # Large jump
                    jump = random.randint(-10, 10)
                    new_val = U_temp[idx] + jump
                    if new_val >= 0 and new_val not in U_temp:
                        U_temp[idx] = new_val
                        U_temp = sorted(U_temp)
                    else:
                        continue
            elif move_type < 0.7:
                # Add a new element
                max_val = max(U_temp)
                # Add near existing or far
                if random.random() < 0.5:
                    new_val = random.randint(1, max_val + 5)
                else:
                    new_val = random.randint(max_val + 1, max_val + 20)
                if new_val not in U_temp:
                    U_temp.append(new_val)
                    U_temp = sorted(U_temp)
                else:
                    continue
            else:
                # Remove an element
                if len(U_temp) > 1:
                    idx = random.randint(1, len(U_temp) - 1)
                    U_temp.pop(idx)
            
            new_score = evaluate(U_temp)
            
            if new_score > current_score:
                U = U_temp
                current_score = new_score
                if current_score > best_overall:
                    best_overall = current_score
                    best_U_overall = U.copy()
            else:
                delta = current_score - new_score
                if delta < 0:
                    prob = 1.0
                else:
                    prob = math.exp(-delta / T)
                if random.random() < prob:
                    U = U_temp
                    current_score = new_score
            
            T *= cooling_rate
            
            if time.time() - start_time > time_limit:
                break

    if best_overall > -np.inf:
        return best_U_overall
    return [0, 1, 3]
