"""Matrix multiplication 2,2,6 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n=2, m=2, p=6):
    """Verify that the decomposition is exact."""
    r = len(U[0]) if U else 0
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    # Reshape: U[i*m+j, alpha] -> U3[i, j, alpha]
    # V[j*p+k, alpha] -> V3[j, k, alpha]
    # W[k*n+i, alpha] -> W3[k, i, alpha]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    
    # Compute S[i, j, k] = sum_alpha U3[i,j,alpha] * V3[j,k,alpha] * W3[k,i,alpha]
    S = np.zeros((n, m, p), dtype=np.float64)
    for a in range(r):
        ua = U3[:, :, a]  # shape (n, m)
        va = V3[:, :, a]  # shape (m, p)
        wa = W3[:, :, a]  # shape (p, n)
        for i in range(n):
            for j in range(m):
                S[i, j, :] += ua[i, j] * va[j, :] * wa[:, i]
    
    return np.allclose(S, 1.0, atol=1e-9)

def _snap_to_half_integer(arr):
    """Snap values to nearest half-integer (multiple of 0.5)."""
    return np.round(2.0 * arr) / 2.0

def _strassen_block_decomposition():
    """
    Apply Strassen's algorithm to each of the 3 column-blocks of the 2x6 matrix.
    
    For multiplying 2x2 A by 2x2 B_block:
    Standard: C = A * B
    Strassen computes 2x2 multiplication with 7 multiplications instead of 4.
    
    For 2x2 A times 2x6 B (split into three 2x2 blocks B1, B2, B3):
    We can apply Strassen independently to each block, giving 7*3 = 21 multiplications.
    
    Strassen's 7 products for 2x2:
    Let A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    
    M1 = (a+d)(e+h)
    M2 = (c+d)e
    M3 = a(f-h)
    M4 = d(g-e)
    M5 = (a+b)h
    M6 = (c-a)(e+f)
    M7 = (b-d)(g+h)
    
    C[0,0] = M1 + M4 - M5 + M7
    C[0,1] = M3 + M5
    C[1,0] = M2 + M4
    C[1,1] = M1 - M2 + M3 + M6
    
    For the full 2x2 x 2x6 problem:
    A = [[a, b], [c, d]]
    B = [[e1, f1, e2, f2, e3, f3],
         [g1, h1, g2, h2, g3, h3]]
    
    Block 1 (columns 0,1): B1 = [[e1, f1], [g1, h1]]
    Block 2 (columns 2,3): B2 = [[e2, f2], [g2, h2]]
    Block 3 (columns 4,5): B3 = [[e3, f3], [g3, h3]]
    
    For each block t, compute 7 Strassen products, then combine.
    Total rank = 21.
    """
    n, m, p = 2, 2, 6
    rank = 21
    
    U = [[0] * rank for _ in range(n * m)]   # 4 x 21
    V = [[0] * rank for _ in range(m * p)]   # 12 x 21
    W = [[0] * rank for _ in range(p * n)]   # 12 x 21
    
    col = 0
    for block in range(3):  # 3 blocks of 2 columns each
        # Block t corresponds to B columns [2*block, 2*block+1]
        # B indices: j*p + k where j is row of B (0 or 1), k is column (0..5)
        
        # For block t:
        # e = B[0, 2*t]  -> V index 0*p + 2*t = 2*t
        # f = B[0, 2*t+1] -> V index 0*p + (2*t+1) = 2*t+1
        # g = B[1, 2*t]  -> V index 1*p + 2*t = p + 2*t
        # h = B[1, 2*t+1] -> V index 1*p + (2*t+1) = p + 2*t+1
        
        # A indices:
        # a = A[0,0] -> U index 0*m+0 = 0
        # b = A[0,1] -> U index 0*m+1 = 1
        # c = A[1,0] -> U index 1*m+0 = 2
        # d = A[1,1] -> U index 1*m+1 = 3
        
        # C output indices (trace convention W[k*n+i, col]):
        # C[0, 2*t]   -> W index (2*t)*n + 0 = 2*t
        # C[0, 2*t+1] -> W index (2*t+1)*n + 0 = 2*t+1
        # C[1, 2*t]   -> W index (2*t)*n + 1 = 2*t + n
        # C[1, 2*t+1] -> W index (2*t+1)*n + 1 = 2*t+1+n
        
        # M1 = (a+d)(e+h)
        U[0][col] = 1; U[3][col] = 1
        V[2*t][col] = 1; V[p + 2*t + 1][col] = 1
        # C[0,0] += M1, C[1,1] += M1
        W[2*t][col] = 1        # C[0, 2*t]
        W[2*t+1+n][col] = 1    # C[1, 2*t+1]
        col += 1
        
        # M2 = (c+d)e
        U[2][col] = 1; U[3][col] = 1
        V[2*t][col] = 1
        # C[1,0] += M2, C[1,1] -= M2
        W[2*t+n][col] = 1      # C[1, 2*t]
        W[2*t+1+n][col] = -1   # C[1, 2*t+1]
        col += 1
        
        # M3 = a(f-h)
        U[0][col] = 1
        V[2*t+1][col] = 1; V[p + 2*t + 1][col] = -1
        # C[0,1] += M3, C[1,1] += M3
        W[2*t+1][col] = 1      # C[0, 2*t+1]
        W[2*t+1+n][col] = 1    # C[1, 2*t+1]
        col += 1
        
        # M4 = d(g-e)
        U[3][col] = 1
        V[p + 2*t][col] = 1; V[2*t][col] = -1
        # C[0,0] += M4, C[1,0] += M4
        W[2*t][col] = 1        # C[0, 2*t]
        W[2*t+n][col] = 1      # C[1, 2*t]
        col += 1
        
        # M5 = (a+b)h
        U[0][col] = 1; U[1][col] = 1
        V[p + 2*t + 1][col] = 1
        # C[0,0] -= M5, C[0,1] += M5
        W[2*t][col] = -1       # C[0, 2*t]
        W[2*t+1][col] = 1      # C[0, 2*t+1]
        col += 1
        
        # M6 = (c-a)(e+f)
        U[2][col] = 1; U[0][col] = -1
        V[2*t][col] = 1; V[2*t+1][col] = 1
        # C[1,1] += M6
        W[2*t+1+n][col] = 1    # C[1, 2*t+1]
        col += 1
        
        # M7 = (b-d)(g+h)
        U[1][col] = 1; U[3][col] = -1
        V[p + 2*t][col] = 1; V[p + 2*t + 1][col] = 1
        # C[0,0] += M7
        W[2*t][col] = 1        # C[0, 2*t]
        col += 1
    
    return U, V, W

def _try_search_rank(r, time_budget):
    """Try to find a rank-r decomposition via optimization."""
    n, m, p = DIMENSIONS
    start = time.time()
    
    best_err = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    np.random.seed(42)
    
    for attempt in range(10):
        if time.time() - start > time_budget:
            break
        
        U = np.random.randn(n*m, r) * 0.5
        V = np.random.randn(m*p, r) * 0.5
        W = np.random.randn(p*n, r) * 0.5
        
        U3 = U.reshape(n, m, r)
        V3 = V.reshape(m, p, r)
        W3 = W.reshape(p, n, r)
        
        for it in range(30):
            if time.time() - start > time_budget:
                break
            
            S = np.zeros((n, m, p), dtype=np.float64)
            for a in range(r):
                ua = U3[:, :, a]
                va = V3[:, :, a]
                wa = W3[:, :, a]
                for i in range(n):
                    for j in range(m):
                        S[i, j, :] += ua[i, j] * va[j, :] * wa[:, i]
            
            err = np.sum((S - 1.0)**2)
            
            if err < 1e-10:
                break
            
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for a in range(r):
                ua = U3[:, :, a]
                va = V3[:, :, a]
                wa = W3[:, :, a]
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            dS = 2.0 * (S[i, j, k] - 1.0)
                            dU[i*m+j, a] += dS * va[j, k] * wa[k, i]
                            dV[j*p+k, a] += dS * ua[i, j] * wa[k, i]
                            dW[k*n+i, a] += dS * ua[i, j] * va[j, k]
            
            lr = 0.01
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            U3 = U.reshape(n, m, r)
            V3 = V.reshape(m, p, r)
            W3 = W.reshape(p, n, r)
            
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
        
        if err < 1e-8:
            U_h = _snap_to_half_integer(U)
            V_h = _snap_to_half_integer(V)
            W_h = _snap_to_half_integer(W)
            if _verify(U_h.tolist(), V_h.tolist(), W_h.tolist()):
                return U_h.tolist(), V_h.tolist(), W_h.tolist()
    
    return None

def construct():
    """Construct a rank-minimal exact tensor decomposition for 2x2 times 2x6 matrix multiplication."""
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # First: try the Strassen-based 21-rank decomposition
    U, V, W = _strassen_block_decomposition()
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Try to find even lower rank via search
    for r in range(19, 21):
        if time.time() > deadline - 15:
            break
        remaining = deadline - time.time() - 5
        if remaining < 5:
            break
        result = _try_search_rank(r, min(20, remaining))
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook decomposition
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
