import numpy as np
import time
import random
import copy

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # 1. Generate Baseline (Schoolbook Rank 72)
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
            # U[i*m+j, col] = 1
            # V[j*p+k, col] = 1
            # W[k*n+i, col] = 1
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best_U = U_sb.tolist()
    best_V = V_sb.tolist()
    best_W = W_sb.tolist()
    best_rank = rank_sb

    def verify(U, V, W, n, m, p):
        """Verify exact decomposition with integer/half-integer tolerance."""
        rank = len(U[0]) if U else 0
        if rank == 0:
            return False
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        # Check all entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def compute_error(U, V, W):
        """Compute sum of squared errors for the tensor decomposition."""
        rank = U.shape[1]
        # Pre-compute products for speed
        # We want T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Let's iterate over i, j, k
        total_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i * m + j, :]
                    v_val = V[j * p + k, :]
                    w_val = W[k * n + i, :]
                    # Dot product of element-wise product
                    # sum_c (u_val[c] * v_val[c]) * w_val[c]
                    prod = np.dot(np.dot(u_val, v_val), w_val)
                    err = prod - 1.0
                    total_err += err * err
        return total_err

    def initialize_population(target_rank, seed):
        """
        Initialize a population of candidate decompositions of target_rank.
        Strategy: Start from schoolbook, then randomly zero out columns to reduce rank,
        then re-initialize non-zero entries from the discrete alphabet.
        """
        np.random.seed(seed)
        random.seed(seed)
        
        pop_size = 50
        population = []
        
        # We need to map the "structure" of schoolbook to a lower rank.
        # A simple heuristic: take the first `target_rank` columns of schoolbook?
        # No, that breaks the sum because we lose terms.
        
        # Better heuristic:
        # Randomly assign each of the n*m*p "required" interactions to one of the `target_rank` columns?
        # This is like a hashing scheme.
        # For each (i,j,k), we need sum_c U[i,j,c]V[j,k,c]W[k,i,c] = 1.
        
        # Let's try a random sparse initialization with values from {-1, 0, 1}
        # and hope GA fills in the rest.
        
        for _ in range(pop_size):
            U = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(n*m, target_rank))
            V = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(m*p, target_rank))
            W = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(p*n, target_rank))
            
            # Sparse initialization: keep only ~50% of entries non-zero
            mask_U = np.random.rand(n*m, target_rank) < 0.5
            mask_V = np.random.rand(m*p, target_rank) < 0.5
            mask_W = np.random.rand(p*n, target_rank) < 0.5
            U[mask_U == False] = 0.0
            V[mask_V == False] = 0.0
            W[mask_W == False] = 0.0
            
            population.append((U, V, W))
            
        return population

    def mutate(U, V, W, mutation_rate=0.1):
        """
        Mutate a candidate by flipping some entries to a random value from the alphabet.
        """
        U_new = U.copy()
        V_new = V.copy()
        W_new = W.copy()
        
        alphabet = [-1.0, -0.5, 0.0, 0.5, 1.0]
        
        # Mutate U
        mask = np.random.rand(*U.shape) < mutation_rate
        U_new[mask] = np.random.choice(alphabet, size=mask.sum())
        
        # Mutate V
        mask = np.random.rand(*V.shape) < mutation_rate
        V_new[mask] = np.random.choice(alphabet, size=mask.sum())
        
        # Mutate W
        mask = np.random.rand(*W.shape) < mutation_rate
        W_new[mask] = np.random.choice(alphabet, size=mask.sum())
        
        return U_new, V_new, W_new

    def crossover(parent1, parent2):
        """
        Uniform crossover between two parents.
        """
        U1, V1, W1 = parent1
        U2, V2, W2 = parent2
        
        mask_U = np.random.rand(*U1.shape) < 0.5
        mask_V = np.random.rand(*V1.shape) < 0.5
        mask_W = np.random.rand(*W1.shape) < 0.5
        
        U_new = np.where(mask_U, U1, U2)
        V_new = np.where(mask_V, V1, V2)
        W_new = np.where(mask_W, W1, W2)
        
        return U_new, V_new, W_new

    def run_ga(target_rank, seed, time_limit):
        """
        Run a simple Genetic Algorithm to find a valid decomposition of target_rank.
        """
        if time.time() > time_limit:
            return None
            
        pop = initialize_population(target_rank, seed)
        # Calculate initial fitness
        fitnesses = [compute_error(U, V, W) for U, V, W in pop]
        
        best_idx = np.argmin(fitnesses)
        best_err = fitnesses[best_idx]
        
        # If we start with a perfect solution (unlikely), return it
        if best_err < 1e-9:
            return pop[best_idx]
            
        pop_size = len(pop)
        elite_size = 5
        max_generations = 1000
        gen = 0
        
        while gen < max_generations:
            if time.time() > time_limit:
                break
                
            # Sort by fitness
            indices = np.argsort(fitnesses)
            sorted_pop = [pop[i] for i in indices]
            sorted_fitnesses = [fitnesses[i] for i in indices]
            
            # Check for convergence
            if sorted_fitnesses[0] < 1e-9:
                return sorted_pop[0]
                
            # Create new population
            new_pop = []
            
            # Elitism
            for i in range(elite_size):
                new_pop.append(sorted_pop[i])
                
            # Fill rest with offspring
            while len(new_pop) < pop_size:
                # Tournament selection
                p1_idx = np.random.randint(0, pop_size)
                p2_idx = np.random.randint(0, pop_size)
                if fitnesses[p1_idx] < fitnesses[p2_idx]:
                    p1 = pop[p1_idx]
                else:
                    p1 = pop[p2_idx]
                    
                p3_idx = np.random.randint(0, pop_size)
                p4_idx = np.random.randint(0, pop_size)
                if fitnesses[p3_idx] < fitnesses[p4_idx]:
                    p2 = pop[p3_idx]
                else:
                    p2 = pop[p4_idx]
                    
                # Crossover
                child = crossover(p1, p2)
                
                # Mutation
                # Higher mutation rate early on? Keep constant for simplicity
                child = mutate(child, mutation_rate=0.15)
                
                new_pop.append(child)
                
            # Calculate new fitnesses
            pop = new_pop
            fitnesses = [compute_error(U, V, W) for U, V, W in pop]
            
            gen += 1
            
            # Adaptive mutation rate? 
            # If stuck, increase mutation?
            # For now, keep it simple.
            
        # Return best found
        best_idx = np.argmin(fitnesses)
        if fitnesses[best_idx] < 1e-9:
            return pop[best_idx]
        return None

    # Try ranks below 72
    # Known lower bounds:
    # 3x4x6. 
    # Schoolbook is 72.
    # Strassen-like reductions might help.
    # Let's try a range of ranks.
    
    # We will try ranks in descending order of likelihood?
    # Or just try a few promising ones.
    # 66 is a common target for 3x4x6 in literature (e.g. 3x3x3 is 23, 3x4x4 is 30?).
    # Actually, 3x4x6 is not a standard cube.
    # Let's try 66, 65, 64, 63...
    
    ranks_to_try = [66, 65, 64, 67, 63, 62, 68, 69, 70, 61, 60]
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
            
        # Try a few seeds
        for seed in [42, 123, 999, 1000, 2000]:
            if time.time() > deadline:
                break
                
            # Set a time limit for this specific GA run
            # GA can be slow. Give it 5 seconds per attempt.
            t_limit = time.time() + 5
            if t_limit > deadline:
                t_limit = deadline
                
            res = run_ga(r, seed, t_limit)
            if res is not None:
                U_r, V_r, W_r = res
                # Final verification
                if verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
                    if r < best_rank:
                        best_U = U_r.tolist()
                        best_V = V_r.tolist()
                        best_W = W_r.tolist()
                        best_rank = r
                    # If we found a solution, we might want to try lower ranks immediately
                    # but for now, let's just record it.
                    # Break seed loop for this rank if we found it? 
                    # Yes, one valid decomposition of rank r is enough to prove rank <= r.
                    # We don't need multiple.
                    break

    return {
        "U": best_U,
        "V": best_V,
        "W": best_W
    }
