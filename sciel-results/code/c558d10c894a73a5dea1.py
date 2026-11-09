import time
import numpy as np

def construct():
    n = 256
    deadline = time.time() + 110.0
    
    # Thue-Morse initialization
    x = np.array([(-1) ** (bin(i).count('1')) for i in range(n)], dtype=np.int8)
    
    # Precompute APs containing each index
    # This structure allows efficient local updates
    aps_containing = [[] for _ in range(n)]
    for d in range(1, (n - 3) // 3 + 1):
        for a in range(n - 3 * d):
            idxs = [a, a + d, a + 2 * d, a + 3 * d]
            for k, idx in enumerate(idxs):
                others = [idxs[j] for j in range(4) if j != k]
                aps_containing[idx].append(tuple(others))
                
    # Calculate initial energies
    # energy[i] represents the sum of products of the other 3 terms in all 4-APs containing i
    energy = np.zeros(n, dtype=np.int64)
    for i in range(n):
        e = 0
        for others in aps_containing[i]:
            prod = 1
            for j in others:
                prod *= x[j]
            e += prod
        energy[i] = e
        
    # Objective function S = 1/4 * sum(x[i] * energy[i])
    # Minimizing S minimizes the number of monochromatic 4-APs
    current_score = np.sum(x * energy)
    best_score = current_score
    best_x = x.copy()
    
    def flip(i):
        nonlocal x, energy, current_score
        old_xi = x[i]
        x[i] = -old_xi
        
        # Update score
        # Delta S = -2 * x[i] * energy[i]
        current_score += -2 * old_xi * energy[i]
        
        # Update energies for neighbors
        for others in aps_containing[i]:
            j, k, l = others
            energy[j] += -2 * old_xi * x[k] * x[l]
            energy[k] += -2 * old_xi * x[j] * x[l]
            energy[l] += -2 * old_xi * x[j] * x[k]
            
    # Local Search
    # We iterate until no improvement or time limit
    while time.time() < deadline:
        improved = True
        while improved:
            improved = False
            best_gain = 0
            best_idx = -1
            for i in range(n):
                gain = x[i] * energy[i]
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    
            if best_idx != -1 and best_gain > 0:
                flip(best_idx)
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    best_x = x.copy()
                    
        # If stuck in local minimum, apply random perturbation
        if time.time() > deadline - 10:
            break
            
        # Perturb 1-3 random bits to escape local minima
        num_flips = np.random.randint(1, 4)
        for _ in range(num_flips):
            i = np.random.randint(n)
            flip(i)
            
    # Convert -1/1 to 0/1
    return [(1 - xi) // 2 for xi in best_x]
