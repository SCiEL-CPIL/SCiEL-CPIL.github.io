"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 4
    deadline = time.time() + 110.0  # Safety margin
    
    # Schoolbook decomposition as fallback (rank 36)
    def schoolbook():
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

    def verify(U, V, W):
        """Verify that the decomposition exactly represents matrix multiplication."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        
        # Check if sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1 for all i,j,k
        # Vectorized check
        # Indices
        idx_i = np.repeat(np.arange(n), m * p)
        idx_j = np.tile(np.repeat(np.arange(m), p), n)
        idx_k = np.tile(np.arange(p), n * m)
        
        # U indices: i*m + j
        u_idx = idx_i * m + idx_j
        # V indices: j*p + k
        v_idx = idx_j * p + idx_k
        # W indices: k*n + i
        w_idx = idx_k * n + idx_i
        
        # Sum over rank
        # This is a bit tricky to vectorize perfectly without broadcasting huge arrays if rank is large,
        # but for rank < 40, it's fine.
        # We want T[i,j,k] = sum_c U[u_idx,c] * V[v_idx,c] * W[w_idx,c]
        
        # Let's do it in a loop over rank for memory safety, or use einsum if possible.
        # Actually, simple loop over rank is fast enough for small ranks.
        T = np.zeros((n, m, p))
        for c in range(rank):
            T += U[u_idx, c][:, None, None] * V[v_idx, c][None, :, None] * W[w_idx, c][None, None, :]
            
        if not np.allclose(T, 1.0, atol=1e-6):
            return False
        return True

    def check_half_integer(mat):
        """Check if all entries are close to half-integers (multiples of 0.5)."""
        doubled = mat * 2
        rounded = np.round(doubled)
        return np.allclose(doubled, rounded, atol=1e-4)

    def verify_and_round(U, V, W):
        """Try to round to half-integers and verify. Returns (U, V, W) if valid, else None."""
        # Round to nearest half-integer
        U_rounded = np.round(U * 2) / 2.0
        V_rounded = np.round(V * 2) / 2.0
        W_rounded = np.round(W * 2) / 2.0
        
        # Verify the rounded solution
        if verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()):
            return U_rounded, V_rounded, W_rounded
        return None

    # Genetic Algorithm / Local Search Strategy
    # We try to find a low rank decomposition.
    # We start with rank 20 and go up to 35.
    # If we find a valid one, we return it.
    
    best_solution = schoolbook()
    best_rank = n * m * p
    
    # Pre-compute indices for fast evaluation
    idx_i = np.repeat(np.arange(n), m * p)
    idx_j = np.tile(np.repeat(np.arange(m), p), n)
    idx_k = np.tile(np.arange(p), n * m)
    u_idx = idx_i * m + idx_j
    v_idx = idx_j * p + idx_k
    w_idx = idx_k * n + idx_i
    
    def evaluate_fitness(U, V, W):
        """Calculate sum of squared errors."""
        rank = U.shape[1]
        # T[i,j,k]
        T = np.zeros((n, m, p))
        for c in range(rank):
            T += U[u_idx, c][:, None, None] * V[v_idx, c][None, :, None] * W[w_idx, c][None, None, :]
        return np.sum((T - 1.0)**2)

    # Search for lower ranks
    for target_rank in range(20, 36):
        if time.time() > deadline - 5:
            break
            
        # Initialize population
        pop_size = 50
        U_pop = np.random.randn(pop_size, n*m, target_rank)
        V_pop = np.random.randn(pop_size, m*p, target_rank)
        W_pop = np.random.randn(pop_size, p*n, target_rank)
        
        # Add some sparse initializations
        for i in range(pop_size // 2):
            U_pop[i] = np.random.choice([-1, 0, 1], size=(n*m, target_rank))
            V_pop[i] = np.random.choice([-1, 0, 1], size=(m*p, target_rank))
            W_pop[i] = np.random.choice([-1, 0, 1], size=(p*n, target_rank))
            
        # Scale down to help convergence
        U_pop *= 0.5
        V_pop *= 0.5
        W_pop *= 0.5
        
        best_fitness = float('inf')
        best_idx = 0
        best_U, best_V, best_W = U_pop[0], V_pop[0], W_pop[0]
        
        # Simple Evolutionary Loop
        max_generations = 100
        for gen in range(max_generations):
            if time.time() > deadline - 2:
                break
                
            # Evaluate fitness
            fitnesses = []
            for i in range(pop_size):
                f = evaluate_fitness(U_pop[i], V_pop[i], W_pop[i])
                fitnesses.append(f)
                if f < best_fitness:
                    best_fitness = f
                    best_idx = i
                    best_U, best_V, best_W = U_pop[i], V_pop[i], W_pop[i]
                    
            # If we found a very good solution, try to round and verify
            if best_fitness < 1e-6:
                result = verify_and_round(best_U, best_V, best_W)
                if result is not None:
                    U_rounded, V_rounded, W_rounded = result
                    if target_rank < best_rank:
                        best_rank = target_rank
                        best_solution = {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                    return best_solution

            # Selection: Keep top half
            sorted_indices = np.argsort(fitnesses)
            survivors = sorted_indices[:pop_size // 2]
            
            U_survivors = U_pop[survivors]
            V_survivors = V_pop[survivors]
            W_survivors = W_pop[survivors]
            
            # Crossover/Mutation to fill rest of population
            for i in range(pop_size // 2, pop_size):
                # Pick two parents
                p1_idx = np.random.randint(0, len(survivors))
                p2_idx = np.random.randint(0, len(survivors))
                
                # Blend
                alpha = np.random.rand()
                U_pop[i] = alpha * U_survivors[p1_idx] + (1 - alpha) * U_survivors[p2_idx]
                V_pop[i] = alpha * V_survivors[p1_idx] + (1 - alpha) * V_survivors[p2_idx]
                W_pop[i] = alpha * W_survivors[p1_idx] + (1 - alpha) * W_survivors[p2_idx]
                
                # Mutation
                if np.random.rand() < 0.5:
                    U_pop[i] += np.random.randn(n*m, target_rank) * 0.1
                    V_pop[i] += np.random.randn(m*p, target_rank) * 0.1
                    W_pop[i] += np.random.randn(p*n, target_rank) * 0.1
                else:
                    # Sparse mutation
                    mask = np.random.rand(n*m, target_rank) < 0.1
                    U_pop[i][mask] += np.random.choice([-0.5, 0.5], size=np.sum(mask))
                    mask = np.random.rand(m*p, target_rank) < 0.1
                    V_pop[i][mask] += np.random.choice([-0.5, 0.5], size=np.sum(mask))
                    mask = np.random.rand(p*n, target_rank) < 0.1
                    W_pop[i][mask] += np.random.choice([-0.5, 0.5], size=np.sum(mask))
            
            # Also keep the best individual as is (elitism)
            U_pop[0] = best_U
            V_pop[0] = best_V
            W_pop[0] = best_W

    return best_solution
# EVOLVE-BLOCK-END
