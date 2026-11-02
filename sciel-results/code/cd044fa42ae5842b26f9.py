import numpy as np
from scipy.optimize import least_squares
import time
from itertools import product

DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # The tensor has entries T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Try to find minimal rank solution
    # Known: 2x2x2 has rank 7 (Strassen)
    # For 2x2x6, let's try various ranks
    
    def verify(U, V, W, rank):
        """Verify decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                    if abs(val - 1.0) > 1e-10:
                        return False
        return True
    
    # Try to construct solution using block structure
    # For 2x2x6, we can view it as: output[i,k] = sum_j A[i,j] * B[j,k]
    # Let's try rank = 12 (one per output)
    
    def try_constructive(rank):
        """Try constructive approach for given rank."""
        if time.time() > deadline - 10:
            return None
        
        # Initialize with zeros
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        
        # For each output (i,k), assign a rank-1 term
        # output[i,k] = sum_j A[i,j] * B[j,k]
        # Term r: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should contribute to output[i,k]
        
        # Strategy: for each (i,k) pair, use one rank term
        # But we need sum_j to work out
        
        # Let's try: for term r corresponding to (i0, k0):
        # W[k*n+i, r] = 1 if (i,k) == (i0,k0) else 0
        # Then we need sum_j U[i*m+j, r] * V[j*p+k, r] = 1 if (i,k)==(i0,k0) else 0
        # This means for fixed r, U[:,r] and V[:,r] must satisfy:
        # sum_j U[i*m+j, r] * V[j*p+k, r] = delta_{i,i0} * delta_{k,k0}
        
        # This is equivalent to: the outer product structure
        # Let u_i = U[i*m, r] and u_{i+1} = U[i*m+1, r] for i=0,1
        # Let v_jk = V[j*p+k, r]
        # We need: for all i,k: sum_j u_i[j] * v_j[k] = delta_{i,i0} delta_{k,k0}
        
        # This is a matrix equation. For 2x2 case:
        # [u00 u01] @ [[v00 v01 v02 v03 v04 v05], [v10 v11 v12 v13 v14 v15]]^T should give delta matrix
        
        # Actually, let me think differently. Use Strassen-like decomposition.
        
        # For 2x2 matrix mult, Strassen uses 7 products:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # For 2x2x6, B is 2x6, so B0 = B[0,:] (1x6), B1 = B[1,:] (1x6)
        # We can apply Strassen to each pair of columns, but share the A-side products.
        
        # Let me try: group the 6 columns into 3 pairs
        # For columns (2t, 2t+1), use Strassen with 7 products
        # But A-side products can be shared across all 3 groups
        
        # Strassen products for A (2x2):
        # P1 = (A00+A11) - scalar
        # P2 = (A10+A11) - scalar  
        # P3 = A00 - scalar
        # P4 = A11 - scalar
        # P5 = (A00+A01) - scalar
        # P6 = (A10-A00) - scalar
        # P7 = (A01-A11) - scalar
        
        # For each column pair t, we need B-side products:
        # For group t:
        # Q1_t = B0[2t] + B1[2t+1] (or similar)
        # etc.
        
        # This is getting complex. Let me just solve the linear system.
        
        total_vars = (n*m + m*p + p*n) * rank
        if total_vars > 50000:
            return None
        
        # Set up as least squares
        def residual(params):
            U_flat = params[:n*m*rank]
            V_flat = params[n*m*rank:n*m*rank + m*p*rank]
            W_flat = params[n*m*rank + m*p*rank:]
            
            U = U_flat.reshape(n*m, rank)
            V = V_flat.reshape(m*p, rank)
            W = W_flat.reshape(p*n, rank)
            
            res = np.zeros(n*m*p)
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                        res[idx] = 1.0 - val
                        idx += 1
            return res
        
        # Try random restarts
        best_result = None
        best_cost = np.inf
        
        for restart in range(50):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(1000 + rank * 100 + restart)
            params = np.random.randn(total_vars) * 0.5
            
            try:
                result = least_squares(
                    residual, params,
                    method='lm', max_nfev=2000,
                    xtol=1e-12, ftol=1e-12, gtol=1e-12
                )
                if result.cost < best_cost:
                    best_cost = result.cost
                    best_result = result
                    if best_cost < 1e-10:
                        break
            except Exception:
                continue
        
        if best_result is not None and best_result.cost < 1e-8:
            U_flat = best_result.x[:n*m*rank]
            V_flat = best_result.x[n*m*rank:n*m*rank + m*p*rank]
            W_flat = best_result.x[n*m*rank + m*p*rank:]
            
            U = U_flat.reshape(n*m, rank)
            V = V_flat.reshape(m*p, rank)
            W = W_flat.reshape(p*n, rank)
            
            # Convert to lists
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            
            if verify(U_list, V_list, W_list, rank):
                return {'U': U_list, 'V': V_list, 'W': W_list}
        
        return None
    
    # Try ranks from low to high
    for rank in [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
        if time.time() > deadline - 30:
            break
        
        result = try_constructive(rank)
        if result is not None:
            return result
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
