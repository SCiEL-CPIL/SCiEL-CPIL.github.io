"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit (task says 120s)

    # Schoolbook decomposition of rank n*m*p
    rank = n * m * p

    # Initialize matrices as lists of lists for direct return
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index in U: i*m + j
                # Index in V: j*p + k
                # Index in W: k*n + i
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1

    # Best result so far (schoolbook)
    best_rank = rank
    best_U = U
    best_V = V
    best_W = W

    # Try to find a better solution with less rank using structured random search
    # Lower bound: max(n*m, m*p, p*n) = max(15, 35, 21) = 35
    lower_bound = max(n * m, m * p, p * n)

    # Verification function using numpy for speed
    def verify_np(U_arr, V_arr, W_arr, n, m, p):
        """Verify that the decomposition exactly represents matrix multiplication."""
        rank = U_arr.shape[1]
        # Check all entries efficiently
        for i in range(n):
            for j in range(m):
                a = i * m + j
                for j2 in range(m):
                    for k in range(p):
                        b = j2 * p + k
                        c = k * n + i
                        # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
                        val = np.dot(np.dot(U_arr[a, :], V_arr[b, :]), W_arr[c, :])
                        expected = 1.0 if (j == j2) else 0.0
                        if abs(val - expected) > 1e-9:
                            return False
        return True

    # Try to find lower-rank decompositions
    # Strategy: for each target rank, try random half-integer initializations
    # and check if they decompose the tensor exactly.
    # This is very unlikely to succeed for ranks much below 105, but we try.
    
    # We'll try a few specific ranks that might be achievable
    # Known bounds: for 3x5x7, the rank is at least 35 (lower bound)
    # and at most 105 (schoolbook).
    # In practice, finding rank < 105 with integer/half-integer entries is very hard.
    
    # Try ranks from 104 down to lower_bound + 5 (step 5 to save time)
    # But we need to be careful about time.
    
    # First, let's try a quick random search for a few ranks
    np.random.seed(42)
    
    for trial_rank in range(rank - 1, lower_bound, -3):
        if time.time() > deadline - 20:
            break
        
        # Try a few random initializations for this rank
        for attempt in range(3):
            if time.time() > deadline - 20:
                break
            
            # Random half-integer entries in {-1, -0.5, 0, 0.5, 1}
            U_r = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n * m, trial_rank))
            V_r = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m * p, trial_rank))
            W_r = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p * n, trial_rank))
            
            if verify_np(U_r, V_r, W_r, n, m, p):
                best_rank = trial_rank
                best_U = U_r.tolist()
                best_V = V_r.tolist()
                best_W = W_r.tolist()
                break
        
        if best_rank < rank:
            break  # Found a better solution
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
