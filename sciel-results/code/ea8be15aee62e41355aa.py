import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Target tensor: T[a, b, c, d] = 1 if b == c else 0
    T = np.zeros((n, m, m, p))
    for b in range(m):
        for c in range(m):
            if b == c:
                T[:, b, c, :] = 1.0
    
    # Genetic algorithm to evolve sparse binary support patterns
    # We work over F_2 first to find valid support structures, then assign values
    
    def compute_error_binary(U_bin, V_bin, W_bin, rank):
        """Compute tensor product over F_2 and compare with T mod 2"""
        U_r = U_bin.reshape(n, m, rank)
        V_r = V_bin.reshape(m, p, rank)
        W_r = W_bin.reshape(p, n, rank)
        
        # Compute S mod 2 using integer arithmetic then mod 2
        S_int = np.einsum('abr,cdr,dar->abcd', U_r.astype(np.int64), V_r.astype(np.int64), W_r.astype(np.int64))
        S_bin = S_int % 2
        
        # T mod 2
        T_bin = T % 2
        
        return np.sum((S_bin - T_bin) ** 2)
    
    def fitness(U_bin, V_bin, W_bin, rank):
        """Fitness: lower error is better, also prefer sparser solutions"""
        err = compute_error_binary(U_bin, V_bin, W_bin, rank)
        sparsity_bonus = -0.01 * (np.sum(U_bin) + np.sum(V_bin) + np.sum(W_bin))
        return err + sparsity_bonus
    
    def crossover(U1, V1, W1, U2, V2, W2, rank):
        """Uniform crossover"""
        mask_U = np.random.rand(n * m, rank) < 0.5
        mask_V = np.random.rand(m * p, rank) < 0.5
        mask_W = np.random.rand(p * n, rank) < 0.5
        
        U_child = np.where(mask_U, U1, U2)
        V_child = np.where(mask_V, V1, V2)
        W_child = np.where(mask_W, W1, W2)
        
        return U_child, V_child, W_child
    
    def mutate(U, V, W, rank, mutation_rate=0.1):
        """Flip bits with probability mutation_rate"""
        U_mut = U.copy()
        V_mut = V.copy()
        W_mut = W.copy()
        
        # Mutate U
        mask = np.random.rand(n * m, rank) < mutation_rate
        U_mut[mask] = 1 - U_mut[mask]
        
        # Mutate V
        mask = np.random.rand(m * p, rank) < mutation_rate
        V_mut[mask] = 1 - V_mut[mask]
        
        # Mutate W
        mask = np.random.rand(p * n, rank) < mutation_rate
        W_mut[mask] = 1 - W_mut[mask]
        
        return U_mut, V_mut, W_mut
    
    def initialize_population(pop_size, rank, sparsity=0.3):
        """Initialize population with random sparse binary matrices"""
        pop_U = []
        pop_V = []
        pop_W = []
        
        for _ in range(pop_size):
            U = (np.random.rand(n * m, rank) < sparsity).astype(np.int64)
            V = (np.random.rand(m * p, rank) < sparsity).astype(np.int64)
            W = (np.random.rand(p * n, rank) < sparsity).astype(np.int64)
            pop_U.append(U)
            pop_V.append(V)
            pop_W.append(W)
        
        return pop_U, pop_V, pop_W
    
    # Try ranks from low to high
    for rank in range(15, 37):
        pop_size = 50
        generations = 200
        pop_U, pop_V, pop_W = initialize_population(pop_size, rank, sparsity=0.25)
        
        for gen in range(generations):
            # Evaluate fitness
            fitnesses = []
            for i in range(pop_size):
                f = fitness(pop_U[i], pop_V[i], pop_W[i], rank)
                fitnesses.append(f)
            
            fitnesses = np.array(fitnesses)
            
            # Check if we found a solution
            best_idx = np.argmin(fitnesses)
            if fitnesses[best_idx] < 1e-6:
                # Found a binary solution, now try to assign integer/half-integer values
                U_bin = pop_U[best_idx]
                V_bin = pop_V[best_idx]
                W_bin = pop_W[best_idx]
                
                # Try to find actual values that work
                # Use the binary solution as a guide for which entries are non-zero
                # Initialize with small random values where binary is 1
                U_val = np.zeros((n * m, rank))
                V_val = np.zeros((m * p, rank))
                W_val = np.zeros((p * n, rank))
                
                # Set non-zero entries to random values from {-1, 1, 0.5, -0.5}
                vals = np.array([-1.0, -0.5, 0.5, 1.0])
                U_val[U_bin == 1] = np.random.choice(vals, size=np.sum(U_bin == 1))
                V_val[V_bin == 1] = np.random.choice(vals, size=np.sum(V_bin == 1))
                W_val[W_bin == 1] = np.random.choice(vals, size=np.sum(W_bin == 1))
                
                # Local search to refine values
                for step in range(500):
                    U_r = U_val.reshape(n, m, rank)
                    V_r = V_val.reshape(m, p, rank)
                    W_r = W_val.reshape(p, n, rank)
                    S = np.einsum('abr,cdr,dar->abcd', U_r, V_r, W_r)
                    err = np.sum((S - T) ** 2)
                    
                    if err < 1e-8:
                        # Check if values are already half-integers
                        U_round = np.round(U_val * 2) / 2
                        V_round = np.round(V_val * 2) / 2
                        W_round = np.round(W_val * 2) / 2
                        
                        U_rr = U_round.reshape(n, m, rank)
                        V_rr = V_round.reshape(m, p, rank)
                        W_rr = W_round.reshape(p, n, rank)
                        S_round = np.einsum('abr,cdr,dar->abcd', U_rr, V_rr, W_rr)
                        if np.allclose(S_round, T, atol=1e-6):
                            return {
                                "U": U_round.tolist(),
                                "V": V_round.tolist(),
                                "W": W_round.tolist()
                            }
                    
                    # Mutate one entry
                    factor = np.random.choice(3)
                    if factor == 0:
                        i = np.random.randint(n * m)
                        j = np.random.randint(rank)
                        if U_bin[i, j] == 1:
                            old = U_val[i, j]
                            U_val[i, j] = np.random.choice(vals)
                            U_r = U_val.reshape(n, m, rank)
                            V_r = V_val.reshape(m, p, rank)
                            W_r = W_val.reshape(p, n, rank)
                            S_new = np.einsum('abr,cdr,dar->abcd', U_r, V_r, W_r)
                            new_err = np.sum((S_new - T) ** 2)
                            if new_err >= err:
                                U_val[i, j] = old
                    elif factor == 1:
                        i = np.random.randint(m * p)
                        j = np.random.randint(rank)
                        if V_bin[i, j] == 1:
                            old = V_val[i, j]
                            V_val[i, j] = np.random.choice(vals)
                            U_r = U_val.reshape(n, m, rank)
                            V_r = V_val.reshape(m, p, rank)
                            W_r = W_val.reshape(p, n, rank)
                            S_new = np.einsum('abr,cdr,dar->abcd', U_r, V_r, W_r)
                            new_err = np.sum((S_new - T) ** 2)
                            if new_err >= err:
                                V_val[i, j] = old
                    else:
                        i = np.random.randint(p * n)
                        j = np.random.randint(rank)
                        if W_bin[i, j] == 1:
                            old = W_val[i, j]
                            W_val[i, j] = np.random.choice(vals)
                            U_r = U_val.reshape(n, m, rank)
                            V_r = V_val.reshape(m, p, rank)
                            W_r = W_val.reshape(p, n, rank)
                            S_new = np.einsum('abr,cdr,dar->abcd', U_r, V_r, W_r)
                            new_err = np.sum((S_new - T) ** 2)
                            if new_err >= err:
                                W_val[i, j] = old
                
                # If local search didn't work, try again with different initialization
                continue
            
            # Selection: tournament selection
            new_U = []
            new_V = []
            new_W = []
            
            for _ in range(pop_size):
                # Tournament of 3
                idxs = np.random.randint(0, pop_size, 3)
                best = idxs[np.argmin(fitnesses[idxs])]
                
                if np.random.rand() < 0.5 and len(new_U) >= 2:
                    # Crossover
                    i1 = np.random.randint(0, len(new_U))
                    i2 = np.random.randint(0, len(new_U))
                    child_U, child_V, child_W = crossover(
                        new_U[i1], new_V[i1], new_W[i1],
                        new_U[i2], new_V[i2], new_W[i2],
                        rank
                    )
                    child_U, child_V, child_W = mutate(child_U, child_V, child_W, rank, 0.05)
                    new_U.append(child_U)
                    new_V.append(child_V)
                    new_W.append(child_W)
                else:
                    # Elitism or random
                    new_U.append(pop_U[best].copy())
                    new_V.append(pop_V[best].copy())
                    new_W.append(pop_W[best].copy())
            
            pop_U = new_U
            pop_V = new_V
            pop_W = new_W
        
        # End generations
    
    # If no solution found, return schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
