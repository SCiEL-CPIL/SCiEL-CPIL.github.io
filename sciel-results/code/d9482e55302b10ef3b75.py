"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 6
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Fallback: Schoolbook rank 54
    def generate_schoolbook():
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

    best_solution = generate_schoolbook()
    best_rank = n * m * p

    # Strategy: 
    # 1. Try to find a low-rank decomposition for 3x3x3 (core).
    # 2. If found with rank R_core, the 3x3x6 rank is 2 * R_core.
    # 3. Also try direct random search for 3x3x6 with rank R_total.
    #    R_total can be 2 * R_core (e.g., 46, 48) or potentially lower if sharing is possible?
    #    Actually, for 3x3x6, the standard bound is 2 * mu(3,3,3).
    #    mu(3,3,3) is at most 23 (Landsberg). So 46 is a target.
    #    Schoolbook is 54.
    #    If we find 3x3x3 with rank 23, we get 46 for 3x3x6.
    #    If we find 3x3x3 with rank 24, we get 48.
    #    If we find 3x3x3 with rank 27, we get 54 (no improvement).
    
    # Let's focus on finding 3x3x3 solutions with rank <= 27.
    # Known: Rank 23 exists (Landsberg). Rank 24, 25, 26, 27 also exist.
    # Random search over {-1, 0, 1} often finds rank 27 or 24/25 easily.
    # Rank 23 is harder.
    
    # We will search for 3x3x3 core solutions.
    # If we find one with rank R, we construct the 3x3x6 solution with rank 2R.
    # We iterate R from 23 upwards.
    
    found_core = False
    core_U = core_V = core_W = None
    core_rank = None

    # Search for 3x3x3 core
    for R in range(23, 28):
        if time.time() > deadline:
            break
            
        # Number of attempts depends on R. Lower R is harder.
        attempts = 5000 if R >= 25 else 10000
        
        for attempt in range(attempts):
            if time.time() > deadline:
                break
                
            # Random U, V with entries in {-1, 0, 1}
            # Sparse random matrices often work well.
            # Let's use a density of ~0.5
            U_rand = np.random.randint(-1, 2, size=(9, R))
            V_rand = np.random.randint(-1, 2, size=(9, R))
            
            W_core = np.zeros((9, R))
            valid = True
            
            # For each output index (i, k) in 3x3, we have 3 equations (for j=0,1,2)
            # Sum_r U[i,j,r] * V[j,k,r] * W[i,k,r] = 1
            # This is a linear system for W[i,k, :] given U and V.
            
            for i in range(3):
                for k in range(3):
                    # Build matrix M of size 3 x R
                    # M[j, r] = U[i*3+j, r] * V[j*3+k, r]
                    M = np.zeros((3, R))
                    for j in range(3):
                        # Vectorized product
                        M[j, :] = U_rand[i * 3 + j, :] * V_rand[j * 3 + k, :]
                    
                    # Solve M @ w = ones(3)
                    # We want an exact integer/half-integer solution.
                    # First, check if a solution exists (rank of M must be 3).
                    # If rank < 3, no unique solution or no solution.
                    # If rank == 3, we can find the minimum norm solution or check for integer solutions.
                    
                    # Use lstsq to get a candidate
                    try:
                        w_vec, residuals, rank_m, s = np.linalg.lstsq(M, np.ones(3), rcond=None)
                    except:
                        valid = False
                        break
                    
                    # Check if the solution is close to integers or half-integers
                    # The problem statement says "integer/half-integer entries".
                    # So 2 * w_vec should be integer.
                    w_doubled = w_vec * 2
                    closest_ints = np.round(w_doubled)
                    error = np.max(np.abs(w_doubled - closest_ints))
                    
                    if error < 1e-9:
                        W_core[i * 3 + k, :] = w_vec
                    else:
                        # It's possible lstsq gives a non-integer solution even if an integer one exists
                        # if the system is underdetermined (R > 3).
                        # However, if R is small (23-27) and M is 3xR, it's underdetermined.
                        # The "solution" from lstsq is the min-norm solution.
                        # If the min-norm solution is not integer, there might still be an integer solution.
                        # But checking for existence of integer solution in underdetermined system is hard.
                        # Usually, if a random integer U,V works, the min-norm solution is often the integer one
                        # or very close. If not, we discard.
                        valid = False
                        break
            
            if valid:
                found_core = True
                core_U = U_rand.tolist()
                core_V = V_rand.tolist()
                core_W = W_core.tolist()
                core_rank = R
                break
        
        if found_core:
            break

    if found_core:
        # Construct 3x3x6 solution from core
        # Rank is 2 * core_rank
        R_total = 2 * core_rank
        
        if R_total < best_rank:
            U_total = [[0] * R_total for _ in range(n * m)]
            V_total = [[0] * R_total for _ in range(m * p)]
            W_total = [[0] * R_total for _ in range(p * n)]
            
            # First half of B (columns 0, 1, 2)
            # Uses core indices 0..R-1
            for i in range(3):
                for j in range(3):
                    for r in range(core_rank):
                        U_total[i * 3 + j][r] = core_U[i * 3 + j][r]
                        # V for k=0..2
                        V_total[j * 6 + r % 3][r] = core_V[j * 3 + (r % 3)][r] # Wait, V index is j*p+k
                        # Correct mapping:
                        # V_total[j, k, r] = core_V[j, k, r] for k=0,1,2
                        # W_total[k, i, r] = core_W[k, i, r] for k=0,1,2
                        
                        # Let's do it cleanly
                        pass
            
            # Reset and fill properly
            U_total = [[0] * R_total for _ in range(n * m)]
            V_total = [[0] * R_total for _ in range(m * p)]
            W_total = [[0] * R_total for _ in range(p * n)]
            
            # Block 1: k=0,1,2, tensor indices 0..R-1
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        for r in range(core_rank):
                            U_total[i * 3 + j][r] = core_U[i * 3 + j][r]
                            V_total[j * 6 + k][r] = core_V[j * 3 + k][r]
                            W_total[k * 3 + i][r] = core_W[k * 3 + i][r]
            
            # Block 2: k=3,4,5, tensor indices R..2R-1
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        k_full = k + 3
                        for r in range(core_rank):
                            U_total[i * 3 + j][R_total - core_rank + r] = core_U[i * 3 + j][r]
                            V_total[j * 6 + k_full][R_total - core_rank + r] = core_V[j * 3 + k][r]
                            W_total[k_full * 3 + i][R_total - core_rank + r] = core_W[k * 3 + i][r]
            
            best_solution = {"U": U_total, "V": V_total, "W": W_total}
            best_rank = R_total

    return best_solution
# EVOLVE-BLOCK-END
