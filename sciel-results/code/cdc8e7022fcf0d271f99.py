import time
import random
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify_numpy(U, V, W, num_samples=20):
        # U: (n*m, rank)
        # V: (m*p, rank)
        # W: (p*n, rank)
        rank = U.shape[1]
        if rank == 0:
            return False
        
        A_flat = np.zeros(n * m)
        B_flat = np.zeros(m * p)
        
        for _ in range(num_samples):
            # Generate random integer matrices
            A = np.random.randint(-5, 5, (n, m))
            B = np.random.randint(-5, 5, (m, p))
            A_flat = A.flatten()
            B_flat = B.flatten()
            
            C_true = A @ B
            
            # Compute M_r = sum(U_{ij,r} A_{ij})
            # U is (n*m, rank), A_flat is (n*m,)
            # M is (rank,)
            M = U.T @ A_flat
            N = V.T @ B_flat
            
            # C_fact_{ik} = sum_r M_r N_r W_{ki,r}
            # W is (p*n, rank)
            # C_fact_flat is (p*n,)
            # M*N is (rank,)
            C_fact_flat = W.T @ (M * N)
            C_fact = C_fact_flat.reshape(p, n).T # W index is k*n+i, so reshape to (p, n) then transpose to (n, p)
            
            if not np.array_equal(C_true, C_fact):
                return False
        return True

    # Try to find a lower rank decomposition
    # Start from schoolbook rank 63
    best_result = schoolbook()
    best_rank = n * m * p

    # Search for lower ranks
    # We prioritize ranks close to schoolbook first as they are easier to find
    for target_rank in range(best_rank - 1, 50, -1):
        if time.time() > deadline - 10:
            break
            
        # Try a few random initializations
        for attempt in range(500):
            if time.time() > deadline - 5:
                break
                
            rng = np.random.RandomState(target_rank * 100 + attempt)
            
            # Generate sparse random matrices
            # Density roughly 1/3 for each non-zero element?
            # For rank R, we have R columns.
            # U has 9*R entries.
            # Let's try values in {-1, 0, 1} with prob 0.3 of non-zero
            U = rng.randint(-1, 2, (n * m, target_rank))
            U[rng.rand(n * m, target_rank) < 0.7] = 0
            V = rng.randint(-1, 2, (m * p, target_rank))
            V[rng.rand(m * p, target_rank) < 0.7] = 0
            W = rng.randint(-1, 2, (p * n, target_rank))
            W[rng.rand(p * n, target_rank) < 0.7] = 0
            
            if verify_numpy(U, V, W, num_samples=10):
                # Double check with more samples
                if verify_numpy(U, V, W, num_samples=50):
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = target_rank
                    break # Found one for this rank, go to next lower rank? 
                    # Actually, we want the MINIMAL rank. So we should continue searching lower.
                    # But finding lower is harder. Let's just return the first one found < 63?
                    # The score is seed/value. Lower rank is better.
                    # So we should try to go as low as possible.
                    # But if we find rank 62, and then fail to find 61, we return 62.
                    # If we find 60, we return 60.
                    # So we break the inner loop, but continue outer loop to try lower ranks.
        
        # Optional: If we found a rank, we might want to stop early if time is tight
        # but the loop structure handles it.

    return best_result

