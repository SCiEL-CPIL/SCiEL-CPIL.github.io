import numpy as np
import time
import sys

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    # Pre-compute index mappings
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # Target tensor T is all 1s for the multiplication pattern.
    
    # We search for rank r
    # Lower bound: max(nm, mp, np) = max(9, 21, 21) = 21
    
    best_solution = None
    best_rank = 63
    
    # Schoolbook solution as fallback
    U_school = [[0]*63 for _ in range(9)]
    V_school = [[0]*63 for _ in range(21)]
    W_school = [[0]*63 for _ in range(21)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    best_solution = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = 63

    # Search for smaller ranks
    # Try ranks from 21 to 62
    for r in range(21, 63):
        if time.time() > deadline - 10:
            break
            
        # Number of restarts for this rank
        # Allocate more time to lower ranks
        # If r is close to 21, try more restarts
        max_restarts = 500 if r < 30 else 100
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Random U and V
            # Use a wider set of values to increase chance of finding a structured solution
            # Values: -1, -0.5, 0, 0.5, 1
            vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
            U = vals[np.random.randint(0, len(vals), size=(n*m, r))]
            V = vals[np.random.randint(0, len(vals), size=(m*p, r))]
            
            # Construct A (63 x 21*r)
            # Equation index e = i*p*(m) + j*p + k
            # Variable index v = (k*n + i) * r + c
            
            A = np.zeros((n*m*p, p*n*r))
            b = np.ones(n*m*p)
            
            # Vectorized construction of A
            # We need to fill A[e, v] = U[i*m+j, c] * V[j*p+k, c]
            # e = i*(m*p) + j*p + k
            # v = (k*n + i)*r + c
            
            for i in range(n):
                for j in range(m):
                    u_row = U[i*m + j, :] # shape (r,)
                    for k in range(p):
                        e = i * (m*p) + j * p + k
                        v_row = V[j*p + k, :] # shape (r,)
                        # coeff = u_row * v_row
                        # The variables are W[k*n+i, c]
                        # So for each c, A[e, (k*n+i)*r + c] = coeff[c]
                        coeff = u_row * v_row
                        # Indices in A for this e
                        # v_indices = np.arange(r) + (k*n + i) * r
                        A[e, (k*n + i)*r : (k*n + i)*r + r] = coeff
            
            # Solve A x = b
            try:
                x, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
                
            # Check if x is close to a rational grid
            # Try denominators 1, 2, 4, 8
            found = False
            for denom in [1, 2, 4, 8]:
                x_scaled = x * denom
                x_rounded = np.round(x_scaled)
                diff = np.abs(x_scaled - x_rounded)
                if np.max(diff) < 1e-6:
                    # Snap
                    W_flat = x_rounded / denom
                    W = W_flat.reshape(p*n, r)
                    
                    # Verify exact decomposition
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = 0.0
                                for c in range(r):
                                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                                if abs(val - 1.0) > 1e-9:
                                    valid = False
                                    break
                            if not valid:
                                break
                    if valid:
                        best_rank = r
                        best_solution = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                        return best_solution
                    else:
                        # If verification fails, it might be due to floating point error in lstsq
                        # or the solution is not exact.
                        pass
                        
    # If no solution found, return schoolbook
    return best_solution
