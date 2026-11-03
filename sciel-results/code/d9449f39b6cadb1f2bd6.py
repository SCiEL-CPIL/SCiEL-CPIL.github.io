"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Build the matrix multiplication tensor T[i,j,k] = 1 if (i,j,k) corresponds to valid indices
    # T is indexed by (i in 0..n-1, j in 0..m-1, k in 0..p-1)
    # The tensor T_{i,j,k} = 1 for all valid (i,j,k) in the multiplication
    # Actually, the tensor for matrix mult C = A*B where A is n x m, B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has indices (i, j, k) and T[i,j,k] = 1 for all i,j,k
    # The rank-r decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Schoolbook rank is n*m*p = 36
    # Let's try to find a lower rank decomposition
    
    best_rank = n * m * p
    best_decomp = None
    
    # Try WFTA-inspired approach: use cyclic group Z_7 structure
    # For 3x3x3, rank 23 is known (Strassen-like). For 3x3x4, let's search.
    
    # Method: Random initialization + local search
    # We want to find U (nm x r), V (mp x r), W (pn x r) such that:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    def validate(U, V, W, r):
        """Check if the decomposition is exact."""
        # For each (i, j, k), compute sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True
    
    def objective(U, V, W, r):
        """Compute error: sum over (i,j,k) of (sum_c U*V*W - 1)^2"""
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                    err += (s - 1.0) ** 2
        return err
    
    # Try ranks from low to high
    for r in range(20, best_rank + 1):
        if time.time() > deadline:
            break
        
        found = False
        num_trials = 50 if r <= 25 else 20
        
        for trial in range(num_trials):
            if time.time() > deadline:
                break
            
            # Random initialization with small values
            np.random.seed(42 + trial * 1000 + r)
            U = np.random.uniform(-1, 1, (n*m, r))
            V = np.random.uniform(-1, 1, (m*p, r))
            W = np.random.uniform(-1, 1, (p*n, r))
            
            # Local optimization: gradient-free Nelder-Mead style or simple coordinate descent
            # Use a simple iterative refinement
            curr_err = objective(U.tolist(), V.tolist(), W.tolist(), r)
            
            for step in range(200):
                if time.time() > deadline:
                    break
                
                # Try small perturbations on each matrix
                improved = False
                for mat_idx in range(3):
                    if mat_idx == 0:
                        rows, cols = n*m, r
                        mat = U
                    elif mat_idx == 1:
                        rows, cols = m*p, r
                        mat = V
                    else:
                        rows, cols = p*n, r
                        mat = W
                    
                    # Try perturbing a random element
                    ri = np.random.randint(0, rows)
                    ci = np.random.randint(0, cols)
                    old_val = mat[ri, ci]
                    for delta in [0.01, -0.01, 0.1, -0.1, 0.001, -0.001]:
                        mat[ri, ci] = old_val + delta
                        new_err = objective(U.tolist(), V.tolist(), W.tolist(), r)
                        if new_err < curr_err - 1e-8:
                            curr_err = new_err
                            improved = True
                            break
                        mat[ri, ci] = old_val
                    if improved:
                        break
                if not improved:
                    break
            
            # Check if we found an exact solution
            if curr_err < 1e-6:
                # Snap to nearest rational values if possible
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                
                # Try to snap to half-integers or integers
                for mat_list in [U_list, V_list, W_list]:
                    for row in mat_list:
                        for idx in range(len(row)):
                            val = row[idx]
                            # Snap to nearest 0.5
                            snapped = round(val * 2) / 2
                            if abs(val - snapped) < 1e-4:
                                row[idx] = snapped
                            else:
                                # Snap to nearest integer
                                snapped_int = round(val)
                                if abs(val - snapped_int) < 1e-4:
                                    row[idx] = snapped_int
                
                if validate(U_list, V_list, W_list, r):
                    best_rank = r
                    best_decomp = (U_list, V_list, W_list)
                    found = True
                    break
                elif validate(U.tolist(), V.tolist(), W.tolist(), r):
                    best_rank = r
                    best_decomp = (U.tolist(), V.tolist(), W.tolist())
                    found = True
                    break
        
        if found:
            break
    
    if best_decomp is not None:
        U, V, W = best_decomp
    else:
        # Fallback to schoolbook
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
