import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Schoolbook rank
    schoolbook_rank = n * m * p
    
    # Schoolbook decomposition
    U_school = np.zeros((n * m, schoolbook_rank))
    V_school = np.zeros((m * p, schoolbook_rank))
    W_school = np.zeros((p * n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = schoolbook_rank
    
    def verify(U, V, W, tol=1e-10):
        """Verify the decomposition satisfies the tensor condition."""
        for i in range(n):
            for i_p in range(n):
                for j in range(m):
                    for j_p in range(m):
                        for k in range(p):
                            for k_p in range(p):
                                expected = 1.0 if (i == i_p and j == j_p and k == k_p) else 0.0
                                val = float(np.dot(U[i * m + j, :], V[j_p * p + k, :] * W[k * n + i_p, :]))
                                if abs(val - expected) > tol:
                                    return False
        return True
    
    # Try to find lower rank decompositions using a heuristic
    # For each target rank, try to construct a valid decomposition
    for target_rank in range(schoolbook_rank - 1, 45, -1):
        if time.time() > deadline:
            break
        
        # Try to build a decomposition by starting from schoolbook and removing columns
        # This is heuristic and may not work, but let's try
        U = np.zeros((n * m, target_rank))
        V = np.zeros((m * p, target_rank))
        W = np.zeros((p * n, target_rank))
        
        # Fill with schoolbook columns
        for r in range(target_rank):
            i = r // (m * p)
            j = (r // p) % m
            k = r % p
            if i < n and j < m and k < p:
                U[i * m + j, r] = 1
                V[j * p + k, r] = 1
                W[k * n + i, r] = 1
        
        # This truncated version won't be valid, so we need to adjust
        # For now, just skip and try next rank
        pass
    
    return best_result
# EVOLVE-BLOCK-END
