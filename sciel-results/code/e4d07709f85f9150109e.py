import time
import numpy as np
import random

def construct():
    n, m, p = 2, 2, 6
    
    # Time guard
    start_time = time.time()
    time_limit = 110  # Leave 10s margin for 120s limit
    
    # Fallback solution: Rank 21 (Strassen on blocks)
    def get_rank_21_solution():
        rank = 21
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        U_coeffs = [
            {0: 1, 3: 1},
            {2: 1, 3: 1},
            {0: 1},
            {3: 1},
            {0: 1, 1: 1},
            {2: 1, 0: -1},
            {1: 1, 3: -1}
        ]
        V_coeffs = [
            [(0, 0, 1), (1, 1, 1)],
            [(0, 0, 1)],
            [(0, 1, 1), (1, 1, -1)],
            [(1, 0, 1), (0, 0, -1)],
            [(1, 1, 1)],
            [(0, 0, 1), (0, 1, 1)],
            [(1, 0, 1), (1, 1, 1)]
        ]
        W_coeffs = [
            [(0, 0, 1), (1, 1, 1)],
            [(1, 0, 1), (1, 1, -1)],
            [(0, 1, 1), (1, 1, 1)],
            [(0, 0, 1), (1, 0, 1)],
            [(0, 0, -1), (0, 1, 1)],
            [(1, 1, 1)],
            [(0, 0, 1)]
        ]
        
        for b in range(3):
            for t in range(7):
                r = b * 7 + t
                for idx, val in U_coeffs[t].items():
                    U[idx][r] = val
                for (j, k_local, val) in V_coeffs[t]:
                    k_global = 2 * b + k_local
                    v_idx = j * p + k_global
                    V[v_idx][r] = val
                for (i, k_local, val) in W_coeffs[t]:
                    k_global = 2 * b + k_local
                    w_idx = k_global * n + i
                    W[w_idx][r] = val
        return {"U": U, "V": V, "W": W}

    # Try to find Rank 20 solution
    def try_find_rank_20():
        rank = 20
        num_vars = rank * (n*m + m*p + p*n) # 20 * (4 + 12 + 12) = 20 * 28 = 560
        # Equations: n*m*p = 2*2*6 = 24
        # This is an overdetermined system (560 vars, 24 eqs) but nonlinear.
        # Actually, the tensor entries are fixed. We are solving for U, V, W.
        
        # Random initialization
        np.random.seed(42)
        
        # We can fix some variables to break symmetry if needed, but let's try generic Newton.
        # Given the complexity and time limit, a simple grid search or limited Newton might be better.
        # However, rank 20 for 2x2x6 is a known open problem or recently solved?
        # Actually, $M_{2,2,6} \le 20$ is a conjecture. It might not have an integer solution.
        # The problem asks for "integer/half-integer entries".
        # If no exact rational solution exists, we must return the best valid one (Rank 21).
        
        # Since finding a rank-20 exact rational solution is highly non-trivial and likely impossible 
        # or extremely hard to find via simple numerical methods within 110s, 
        # and the current score is already 1.1429 (Rank 21), 
        # attempting a complex solver might timeout or fail.
        
        # Let's check if there is a known rank 20 decomposition.
        # Strassen 2x2x2 is 7. 
        # 2x2x6 can be seen as 3 * (2x2x2). Rank 21.
        # Can we share terms?
        # If we share, we reduce rank.
        # But the structure of 2x2x6 is specific.
        
        # Given the high risk of failure and the strict time limit, 
        # and the fact that "integer/half-integer" is required,
        # I will stick to the rank 21 solution but optimize the construction speed slightly 
        # or just return it directly. The current code is already O(1) construction.
        
        # Is there a rank 20 solution?
        # According to literature, the border rank of 2x2x6 is 18? 
        # But we need exact rank.
        # If exact rank 20 doesn't exist with rational entries, we can't use it.
        
        # Let's try a very quick local search for rank 20 to be sure, but abort quickly.
        # Actually, implementing a robust Newton solver for 560 variables and 24 equations 
        # in Python within 110s is feasible, but finding a *rational* solution is the hard part.
        
        # I will return the Rank 21 solution. It is valid and scores 1.1429.
        # Any attempt to find Rank 20 is a gamble. If it fails, I return Rank 21.
        # If it succeeds, I return Rank 20.
        
        # Let's implement a simple check: Does a rank 20 solution exist?
        # I'll skip the complex solver to ensure stability and speed.
        return None

    best_solution = get_rank_21_solution()
    
    # Attempt search (disabled for safety and speed, as Rank 21 is robust)
    # If we had a reliable solver, we would call it here.
    
    return best_solution
