"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # U: (n*m) x r, V: (m*p) x r, W: (p*n) x r
    # Constraint: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    def verify(U, V, W):
        """Verify that U, V, W correctly decompose the tensor."""
        n, m, p = DIMENSIONS
        r = U.shape[1]
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def to_half_int_list(mat):
        """Convert numpy array to list of lists of half-integers (as floats)."""
        return [[float(mat[i][j]) for j in range(mat.shape[1])] for i in range(mat.shape[0])]
    
    # Try ranks from low to high
    # Known: schoolbook is 105. Try to find something lower.
    # For 3x5x7, try ranks: 70, 75, 80, 85, 90, 95, 100, 105
    
    best_result = None
    
    # First, create the schoolbook solution as fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1.0
                V_school[j*p+k, col] = 1.0
                W_school[k*n+i, col] = 1.0
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find lower rank decompositions
    # Strategy: use random initialization and iterative refinement with rounding
    for target_rank in range(70, rank_school + 1):
        if time.time() > deadline:
            break
        
        found = False
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            np.random.seed(42 + attempt)
            
            # Initialize with random values in {-1, 0, 1}
            U = np.random.choice([-1.0, 0.0, 1.0], size=(n*m, target_rank))
            V = np.random.choice([-1.0, 0.0, 1.0], size=(m*p, target_rank))
            W = np.random.choice([-1.0, 0.0, 1.0], size=(p*n, target_rank))
            
            # Ensure at least some non-zero entries
            if np.all(U == 0) or np.all(V == 0) or np.all(W == 0):
                continue
            
            # Iteratively refine: for each column, try to adjust to satisfy constraints
            # Use a simple approach: compute current tensor and adjust
            for iteration in range(20):
                if time.time() > deadline:
                    break
                
                # Compute current tensor values
                errors = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(target_rank):
                                val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            errors.append((i, j, k, val - 1.0))
                
                max_err = max(abs(e[3]) for e in errors)
                if max_err < 1e-9:
                    # Check if all entries are half-integers
                    all_half = True
                    for mat in [U, V, W]:
                        for row in mat:
                            for val in row:
                                if abs(val * 2 - round(val * 2)) > 1e-9:
                                    all_half = False
                                    break
                            if not all_half:
                                break
                        if not all_half:
                            break
                    if all_half:
                        found = True
                        break
                    else:
                        # Round to nearest half-integer and re-verify
                        U = np.round(U * 2) / 2
                        V = np.round(V * 2) / 2
                        W = np.round(W * 2) / 2
                        if verify(U, V, W):
                            found = True
                            break
                
                if found:
                    break
                
                # Simple gradient-free update: adjust random entries
                # Pick a random (i,j,k) with large error and adjust
                i, j, k, err = errors[np.argmax([abs(e[3]) for e in errors])]
                # Try adjusting random entries in U, V, W
                c = np.random.randint(0, target_rank)
                # Try small adjustments
                for delta in [-0.5, 0.5]:
                    U_old = U[i*m+j, c]
                    U[i*m+j, c] = U_old + delta
                    # Quick check
                    val = 0.0
                    for cc in range(target_rank):
                        val += U[i*m+j, cc] * V[j*p+k, cc] * W[k*n+i, cc]
                    if abs(val - 1.0) < abs(err):
                        pass  # Keep the change
                    else:
                        U[i*m+j, c] = U_old  # Revert
            
            if found:
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                break
        
        if found:
            break
    
    # Convert to lists
    U_list = to_half_int_list(best_U)
    V_list = to_half_int_list(best_V)
    W_list = to_half_int_list(best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
