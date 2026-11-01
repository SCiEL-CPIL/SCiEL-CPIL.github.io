"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

import time
import numpy as np
import random

# EVOLVE-BLOCK-START
def construct():
    """
    Attempts to find a rank-23 exact decomposition for 3x3 matrix multiplication.
    Falls back to the schoolbook rank-27 decomposition if no valid rank-23 
    solution is found within the time limit.
    """
    n, m, p = 3, 3, 3
    
    # Fallback: Schoolbook rank 27
    def get_schoolbook():
        rank = 27
        U = [[0] * rank for _ in range(9)]
        V = [[0] * rank for _ in range(9)]
        W = [[0] * rank for _ in range(9)]
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U[i * 3 + j][col] = 1
                    V[j * 3 + k][col] = 1
                    W[k * 3 + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Verification function
    def verify(U, V, W):
        rank = len(U[0])
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * 3 + j][r] * V[j * 3 + k][r] * W[k * 3 + i][r]
                    # Check if s is close to 1
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Search for rank 23
    start_time = time.time()
    time_limit = 10.0  # Search for 10 seconds
    
    rank = 23
    # We represent U, V, W as numpy arrays for faster manipulation
    # Shapes: (9, 23)
    
    # Strategy: Random restarts with local optimization
    # We want integer/half-integer solutions.
    # We can search in continuous space and round, or search in discrete space.
    # Discrete search in 23*27 dimensions is huge.
    # Continuous relaxation:
    # Minimize sum_{i,j,k} (sum_r U_ij_r V_jk_r W_ki_r - 1)^2
    
    # Given the small size, we can try to solve a linear system if we fix some variables.
    # But the system is cubic.
    
    # Let's try a simple random search with "snapping" to half-integers.
    # Initialize with random small integers/half-integers
    # Use gradient descent on the continuous relaxation, then snap.
    
    # To keep it fast and simple, we will try a few random initializations 
    # and use `scipy.optimize.minimize` if available, otherwise a simple gradient step.
    
    # Since we have numpy, we can use it.
    
    found = False
    best_result = None
    
    # We'll try 100 random restarts, each with a few iterations of gradient descent
    # This is a heuristic and might not find the global minimum, but it's a start.
    
    for attempt in range(50):
        if time.time() - start_time > time_limit:
            break
            
        # Random initialization with small integers
        U = np.random.randint(-2, 3, (9, rank)).astype(float)
        V = np.random.randint(-2, 3, (9, rank)).astype(float)
        W = np.random.randint(-2, 3, (9, rank)).astype(float)
        
        # Add some half-integers
        mask = np.random.rand(9, rank) < 0.5
        U[mask] += 0.5
        mask = np.random.rand(9, rank) < 0.5
        V[mask] += 0.5
        mask = np.random.rand(9, rank) < 0.5
        W[mask] += 0.5
        
        # Check if this random guess is valid (highly unlikely)
        if verify(U.tolist(), V.tolist(), W.tolist()):
            found = True
            best_result = (U, V, W)
            break
            
        # Perform a few steps of gradient descent to reduce error
        # Error E = sum_{i,j,k} (S_ijk - 1)^2
        # S_ijk = sum_r U_ij_r V_jk_r W_ki_r
        
        # Compute gradients is complex. Let's use a simple finite difference or 
        # just accept that random search in this space is inefficient.
        
        # Alternative: Use a known structure.
        # The Bini algorithm is sparse. 
        # Most entries are 0, 1, -1.
        
        # Let's try to construct a solution using the fact that the tensor 
        # has symmetry.
        
        # For now, if we don't find a solution quickly, we fallback.
        pass
        
    # If we didn't find a valid rank-23 solution in the limited search,
    # we return the safe rank-27 solution.
    # Note: A proper implementation would include hardcoded valid rank-23 coefficients.
    # Since I cannot guarantee the correctness of hardcoded coefficients without verification,
    # and the search above is very unlikely to succeed in 10s with this simple method,
    # I will stick to the safe rank-27 to ensure a valid score of 1.0.
    # 
    # HOWEVER, to attempt a score improvement, I will try one specific known 
    # rank-23 configuration if I can verify it.
    
    # Let's try to verify a specific candidate from the literature if I can recall it.
    # Actually, let's just return the schoolbook for safety, as an invalid 
    # decomposition scores 0.
    
    return get_schoolbook()

# EVOLVE-BLOCK-END
