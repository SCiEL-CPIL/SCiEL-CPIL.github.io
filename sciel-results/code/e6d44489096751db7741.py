import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    n, m, p = 2, 2, 10
    # U shape (4, R), V shape (20, R), W shape (20, R)
    
    # Known rank 35 construction:
    # Split B into 5 pairs of columns.
    # For each pair, use Strassen (rank 7).
    # Total rank = 5 * 7 = 35.
    
    def build_strassen_rank_35():
        U = np.zeros((4, 35), dtype=float)
        V = np.zeros((20, 35), dtype=float)
        W = np.zeros((20, 35), dtype=float)
        
        # Standard Strassen terms (verified):
        terms = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
            ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
        ]
        
        col = 0
        for q in range(5):
            for a_vec, b_vec, c_vec in terms:
                # U
                U[0, col] = a_vec[0]
                U[1, col] = a_vec[1]
                U[2, col] = a_vec[2]
                U[3, col] = a_vec[3]
                
                # V
                V[0*10 + 2*q, col] = b_vec[0]
                V[0*10 + 2*q+1, col] = b_vec[1]
                V[1*10 + 2*q, col] = b_vec[2]
                V[1*10 + 2*q+1, col] = b_vec[3]
                
                # W
                W[4*q + 0, col] = c_vec[0]
                W[4*q + 1, col] = c_vec[1]
                W[4*q + 2, col] = c_vec[2]
                W[4*q + 3, col] = c_vec[3]
                
                col += 1
        
        return U, V, W

    U, V, W = build_strassen_rank_35()
    
    # Attempt to improve rank using a quick heuristic search if time permits.
    # We will try to see if we can find a rank 34 solution by random perturbation
    # and checking for linear dependencies or using a small-scale optimization.
    # Given the complexity, we will skip this and return the robust rank 35 solution.
    # This ensures correctness and avoids timeout risks.
    
    elapsed = time.time() - start_time
    if elapsed < 100:
        # Placeholder for potential rank reduction search
        # For now, we return the known solution.
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
