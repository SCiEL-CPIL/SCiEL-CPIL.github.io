import numpy as np
from scipy.optimize import least_squares
import time
import sys

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    size_U = n * m  # 6
    size_V = m * p  # 15
    size_W = p * n  # 10
    
    # Build target tensor indices
    targets_alpha = []
    targets_beta = []
    targets_gamma = []
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                targets_alpha.append(a)
                targets_beta.append(b)
                targets_gamma.append(c)
                
    targets_alpha = np.array(targets_alpha, dtype=int)
    targets_beta = np.array(targets_beta, dtype=int)
    targets_gamma = np.array(targets_gamma, dtype=int)
    
    # Precompute target tensor as a sparse representation for speed?
    # Or just dense since 6x15x10 = 900 is small.
    T_target = np.zeros((size_U, size_V, size_W))
    T_target[targets_alpha, targets_beta, targets_gamma] = 1.0
    
    def compute_residuals(U, V, W):
        """Compute residuals for all entries"""
        T_full = np.einsum('ar,br,cr->abc', U, V, W)
        diff = (T_full - T_target).flatten()
        return diff

    def verify_solution(U, V, W):
        """Verify if the solution is exact (within tolerance)"""
        T_full = np.einsum('ar,br,cr->abc', U, V, W)
        max_err = np.max(np.abs(T_full - T_target))
        return max_err < 1e-9, max_err

    def try_rank_R(R, num_restarts=20, max_nfev=100):
        """Try to find a decomposition of rank R"""
        num_vars = (size_U + size_V + size_W) * R
        
        # Stratified restarts
        for restart in range(num_restarts):
            # Deadline check
            if time.time() > deadline - 5:
                return None
                
            # Initialization
            if restart % 3 == 0:
                # Random integers -2 to 2
                x0 = np.random.randint(-2, 3, num_vars).astype(float)
            elif restart % 3 == 1:
                # Random half-integers
                x0 = np.random.randint(-4, 5, num_vars).astype(float) / 2.0
            else:
                # Structured: Try to embed a known low-rank sub-tensor
                # For M_2,3,5, we can embed M_2,2,2 (rank 7) in the top-left corner
                x0 = np.zeros(num_vars)
                if R >= 7:
                    # Strassen M_2,2,2 decomposition
                    # Indices for M_2,2,2: i in 0,1; j in 0,1; k in 0,1
                    # a = i*3+j, b = j*5+k, c = k*2+i
                    # We map this to the first 7 columns
                    # Strassen steps for M_2,2,2:
                    # 1: A11*(B11-B22) -> C12, C21
                    # 2: (A11+A12)*B22 -> C12, C22
                    # 3: (A21+A22)*B11 -> C11, C21
                    # 4: A22*(B22-B11) -> C11, C22
                    # 5: (A11+A21)*(B12+B21) -> C11, C12, C21, C22
                    # 6: (A12-A21)*(B11+B22) -> C11, C12, C21, C22
                    # 7: (A12+A22)*(B21-B12) -> C11, C12, C21, C22
                    
                    # This mapping is complex to hardcode correctly for M_2,3,5 indices.
                    # Instead, use random but with a specific scale.
                    x0 = np.random.uniform(-1, 1, num_vars)
            
            try:
                # Flatten residuals function
                def residuals_flat(x):
                    U = x[:size_U*R].reshape(size_U, R)
                    V = x[size_U*R:(size_U+size_V)*R].reshape(size_V, R)
                    W = x[(size_U+size_V)*R:].reshape(size_W, R)
                    return compute_residuals(U, V, W)

                res = least_squares(
                    residuals_flat,
                    x0,
                    method='trf',
                    max_nfev=max_nfev,
                    ftol=1e-12,
                    xtol=1e-12,
                    gtol=1e-12
                )
                
                if res.cost < 1e-10:
                    x_sol = res.x
                    # Snap to half-integers
                    x_snapped = np.round(x_sol * 2) / 2.0
                    
                    U = x_snapped[:size_U*R].reshape(size_U, R)
                    V = x_snapped[size_U*R:(size_U+size_V)*R].reshape(size_V, R)
                    W = x_snapped[(size_U+size_V)*R:].reshape(size_W, R)
                    
                    is_valid, max_err = verify_solution(U, V, W)
                    if is_valid:
                        return U.tolist(), V.tolist(), W.tolist()
                        
                    # If snapping failed, check raw solution
                    is_valid, max_err = verify_solution(U, V, W)
                    if is_valid:
                        return U.tolist(), V.tolist(), W.tolist()
                        
            except Exception:
                pass
        
        return None

    start_time = time.time()
    deadline = 115  # Safety margin
    
    # Try rank 26 (known minimal rank for M_2,3,5)
    result = try_rank_R(26, num_restarts=30, max_nfev=50)
    if result is not None:
        U, V, W = result
        return {"U": U, "V": V, "W": W}
        
    # Fallback to schoolbook rank 30
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
