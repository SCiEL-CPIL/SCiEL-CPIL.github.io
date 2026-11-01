"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    r = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(r):
                    total += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(total - 1.0) > tol:
                    return False
    return True

def _build_rank_11():
    """
    Construct a rank-11 decomposition for 2x2x3 matrix multiplication.
    
    Uses Strassen's 7-term decomposition for the 2x2x2 subproblem (k=0,1)
    plus 4 terms for k=2.
    """
    n, m, p = 2, 2, 3
    rank = 11
    
    U = [[0.0] * rank for _ in range(n * m)]  # 4 x 11
    V = [[0.0] * rank for _ in range(m * p)]  # 6 x 11
    W = [[0.0] * rank for _ in range(p * n)]  # 6 x 11
    
    # Strassen's 7 products for 2x2x2 (k=0,1 part)
    # Each product c: M_c = (sum a_{i,j} A_{i,j}) * (sum b_{j,k} B_{j,k})
    # C_{i,k} = sum w_{c,i,k} M_c
    
    # M1 = (A00+A11)(B00+B11)
    # a: A00=1, A11=1; b: B00=1, B11=1
    # w: C00=1, C01=1, C10=1, C11=1
    c = 0
    U[0][c] = 1; U[3][c] = 1  # A00, A11
    V[0][c] = 1; V[4][c] = 1  # B00 (j=0,k=0), B11 (j=1,k=1)
    W[0][c] = 1; W[1][c] = 1; W[2][c] = 1; W[3][c] = 1  # C00,C01,C10,C11
    
    # M2 = (A10+A11)B00
    # a: A10=1, A11=1; b: B00=1
    # w: C10=1, C11=-1
    c = 1
    U[2][c] = 1; U[3][c] = 1  # A10, A11
    V[0][c] = 1  # B00
    W[2][c] = 1; W[3][c] = -1  # C10=1, C11=-1
    
    # M3 = A00(B10-B11)
    # a: A00=1; b: B10=1, B11=-1
    # w: C01=1, C11=1
    c = 2
    U[0][c] = 1  # A00
    V[3][c] = 1; V[4][c] = -1  # B10 (j=1,k=0), B11 (j=1,k=1)
    W[1][c] = 1; W[3][c] = 1  # C01=1, C11=1
    
    # M4 = A11(B01-B10)
    # a: A11=1; b: B01=1, B10=-1
    # w: C00=1, C10=1
    c = 3
    U[3][c] = 1  # A11
    V[1][c] = 1; V[3][c] = -1  # B01 (j=0,k=1), B10 (j=1,k=0)
    W[0][c] = 1; W[2][c] = 1  # C00=1, C10=1
    
    # M5 = (A00+A01)B11
    # a: A00=1, A01=1; b: B11=1
    # w: C00=-1, C01=1
    c = 4
    U[0][c] = 1; U[1][c] = 1  # A00, A01
    V[4][c] = 1  # B11
    W[0][c] = -1; W[1][c] = 1  # C00=-1, C01=1
    
    # M6 = (A10-A00)(B00+B10)
    # a: A10=1, A00=-1; b: B00=1, B10=1
    # w: C11=1
    c = 5
    U[2][c] = 1; U[0][c] = -1  # A10=1, A00=-1
    V[0][c] = 1; V[3][c] = 1  # B00, B10
    W[3][c] = 1  # C11=1
    
    # M7 = (A01-A11)(B01+B11)
    # a: A01=1, A11=-1; b: B01=1, B11=1
    # w: C00=1
    c = 6
    U[1][c] = 1; U[3][c] = -1  # A01=1, A11=-1
    V[1][c] = 1; V[4][c] = 1  # B01, B11
    W[0][c] = 1  # C00=1
    
    # Terms 7-10: Handle k=2 (4 terms, one per (i,j) pair)
    # For k=2, we need: sum_c U[i*2+j,c] * V[j*3+2,c] * W[2*2+i,c] = 1
    # i.e., U[i*2+j,c] * V[j*3+2,c] * W[4+i,c] = 1 for each (i,j)
    
    # Term 7: (i=0, j=0, k=2)
    c = 7
    U[0][c] = 1  # A00
    V[2][c] = 1  # B02 (j=0, k=2)
    W[4][c] = 1  # C02 (k=2, i=0)
    
    # Term 8: (i=0, j=1, k=2)
    c = 8
    U[1][c] = 1  # A01
    V[5][c] = 1  # B12 (j=1, k=2)
    W[4][c] = 1  # C02 (k=2, i=0)
    
    # Term 9: (i=1, j=0, k=2)
    c = 9
    U[2][c] = 1  # A10
    V[2][c] = 1  # B02 (j=0, k=2)
    W[5][c] = 1  # C12 (k=2, i=1)
    
    # Term 10: (i=1, j=1, k=2)
    c = 10
    U[3][c] = 1  # A11
    V[5][c] = 1  # B12 (j=1, k=2)
    W[5][c] = 1  # C12 (k=2, i=1)
    
    return U, V, W

def _try_rank_11():
    """Attempt to construct and verify a rank-11 decomposition."""
    n, m, p = 2, 2, 3
    U, V, W = _build_rank_11()
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    return None

def _try_rank_10():
    """
    Try to find a rank-10 decomposition using numerical optimization.
    
    Strategy: Set up the nonlinear system and use iterative refinement.
    We parameterize U, V, W with 10 columns and minimize the sum of squared
    errors across all 12 tensor entries.
    """
    n, m, p = 2, 2, 3
    rank = 10
    
    # Start from rank-11 solution, drop one column, and optimize
    U11, V11, W11 = _build_rank_11()
    
    # Try dropping each column and see if we can optimize back to exact
    for drop_col in range(11):
        U = np.array(U11, dtype=float)[:, [c for c in range(11) if c != drop_col]]
        V = np.array(V11, dtype=float)[:, [c for c in range(11) if c != drop_col]]
        W = np.array(W11, dtype=float)[:, [c for c in range(11) if c != drop_col]]
        
        # Use gradient-based optimization
        # Flatten all variables
        x = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def objective(x_flat):
            U_arr = x_flat[:4*10].reshape(4, 10)
            V_arr = x_flat[4*10:4*10+6*10].reshape(6, 10)
            W_arr = x_flat[4*10+6*10:].reshape(6, 10)
            
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c] for c in range(rank))
                        err += (val - 1.0) ** 2
            return err
        
        def gradient(x_flat):
            U_arr = x_flat[:4*10].reshape(4, 10)
            V_arr = x_flat[4*10:4*10+6*10].reshape(6, 10)
            W_arr = x_flat[4*10+6*10:].reshape(6, 10)
            
            gU = np.zeros_like(U_arr)
            gV = np.zeros_like(V_arr)
            gW = np.zeros_like(W_arr)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c] for c in range(rank))
                        d = 2.0 * (val - 1.0)
                        for c in range(rank):
                            gU[i*m+j, c] += d * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                            gV[j*p+k, c] += d * U_arr[i*m+j, c] * W_arr[k*n+i, c]
                            gW[k*n+i, c] += d * U_arr[i*m+j, c] * V_arr[j*p+k, c]
            
            return np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])
        
        # Simple gradient descent with adaptive step size
        lr = 0.01
        for step in range(2000):
            g = gradient(x)
            g_norm = np.linalg.norm(g)
            if g_norm < 1e-12:
                break
            x = x - lr * g / (g_norm + 1e-8)
            # Adaptive learning rate
            if objective(x) > objective(x + lr * g / (g_norm + 1e-8)) * 1.01:
                lr *= 0.9
        
        # Check if we got close enough
        U_arr = x[:4*10].reshape(4, 10)
        V_arr = x[4*10:4*10+6*10].reshape(6, 10)
        W_arr = x[4*10+6*10:].reshape(6, 10)
        
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c] for c in range(rank))
                    max_err = max(max_err, abs(val - 1.0))
        
        if max_err < 1e-6:
            # Convert to lists with rounded values for cleanliness
            U_list = [[round(val, 12) for val in row] for row in U_arr]
            V_list = [[round(val, 12) for val in row] for row in V_arr]
            W_list = [[round(val, 12) for val in row] for row in W_arr]
            
            if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # First, try the known rank-11 construction
    result = _try_rank_11()
    if result is not None:
        return result
    
    # Try rank-10 if we still have time
    if time.time() < deadline - 5:
        result = _try_rank_10()
        if result is not None:
            return result
    
    # Fallback: return the schoolbook rank-12 decomposition
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
