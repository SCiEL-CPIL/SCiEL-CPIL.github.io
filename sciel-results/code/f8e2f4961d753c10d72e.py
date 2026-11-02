"""Program for 3x3x6 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

def _verify(U, V, W, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = 3, 3, 6
    rank = len(U[0])
    if len(U) != n * m:
        return False
    if len(V) != m * p:
        return False
    if len(W) != p * n:
        return False
    
    # Check all entries of the tensor
    # T[alpha, beta, gamma] = 1 if alpha=(i,j), beta=(j,k), gamma=(k,i) match, else 0
    # alpha index: i*m + j
    # beta index: j*p + k
    # gamma index: k*n + i
    
    for i in range(n):
        for j in range(m):
            alpha = i * m + j
            for jp in range(m):
                for kp in range(p):
                    beta = jp * p + kp
                    for ip in range(n):
                        # gamma index is kp*n + ip
                        gamma = kp * n + ip
                        target = 1.0 if (i == ip and j == jp and # k is implicit in beta/gamma match?
                                         # Wait, beta has k, gamma has k.
                                         # The condition is i==ip, j==jp, and the k in beta must match the k in gamma.
                                         # Let's check indices carefully.
                                         # U[i, j]
                                         # V[j', k']
                                         # W[k'', i'']
                                         # We need i==i'', j==j', k'==k''
                                         # So target is 1 if i==ip (i'') and j==jp (j') and the k in beta (kp) matches the k in gamma.
                                         # In the loop, gamma is constructed with kp. So k in gamma is kp.
                                         # So we need i==ip and j==jp.
                                         True) else 0.0
                        # Wait, the k index in V is kp. The k index in W is also kp (from gamma construction).
                        # So if i==ip and j==jp, the k's match by construction of the loop variable kp.
                        # So target is 1 if i==ip and j==jp.
                        
                        s = 0.0
                        for r in range(rank):
                            s += U[alpha][r] * V[beta][r] * W[gamma][r]
                        if abs(s - target) > tol:
                            return False
    return True

def _schoolbook():
    n, m, p = 3, 3, 6
    rank = n * m * p  # 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify_333(U, V, W, tol=1e-9):
    """Verify 3x3x3 decomposition."""
    n, m, p = 3, 3, 3
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    
    for i in range(n):
        for j in range(m):
            alpha = i * m + j
            for jp in range(m):
                for kp in range(p):
                    beta = jp * p + kp
                    for ip in range(n):
                        gamma = kp * n + ip
                        target = 1.0 if (i == ip and j == jp) else 0.0
                        s = 0.0
                        for r in range(rank):
                            s += U[alpha][r] * V[beta][r] * W[gamma][r]
                        if abs(s - target) > tol:
                            return False
    return True

def _find_333(rank, time_limit=5.0, seed=42):
    """Try to find a rank-r decomposition for 3x3x3."""
    n, m, p = 3, 3, 3
    t0 = time.time()
    np.random.seed(seed)
    
    for attempt in range(100):
        if time.time() - t0 > time_limit:
            break
            
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        
        # Solve for W
        # Equations: sum_r U[alpha, r] V[beta, r] W[gamma, r] = T[alpha, beta, gamma]
        # This is linear in W.
        # Variables: W[gamma, r] for gamma in 0..8, r in 0..rank-1. Total 9*rank.
        # Equations: for each alpha, beta, gamma. Total 27*9 = 243?
        # No, for each alpha, beta, the sum over r is fixed.
        # Let M[alpha, beta, gamma] = sum_r U[alpha, r] V[beta, r] W[gamma, r].
        # We want M = T.
        # This is not linear in W if U, V are fixed?
        # M[alpha, beta, gamma] = sum_r (U[alpha, r] V[beta, r]) W[gamma, r].
        # Let C[alpha, beta, r] = U[alpha, r] V[beta, r].
        # Then M[alpha, beta, gamma] = sum_r C[alpha, beta, r] W[gamma, r].
        # This is a linear equation in W[gamma, :].
        # For each (alpha, beta), we have a vector equation:
        # sum_r C[alpha, beta, r] W[gamma, r] = T[alpha, beta, gamma] for all gamma.
        # This is 9 equations for each (alpha, beta).
        # Total 27 * 9 = 243 equations.
        # Unknowns: 9 * rank.
        
        # We can set up the linear system A x = b.
        # x is flattened W (size 9*rank).
        # For each alpha, beta, and gamma:
        # sum_r C[alpha, beta, r] W[gamma, r] = T[alpha, beta, gamma]
        
        A = np.zeros((243, 9 * rank))
        b = np.zeros(243)
        
        eq_idx = 0
        for alpha in range(n * m):
            for beta in range(m * p):
                for gamma in range(p * n):
                    # Target T
                    i = alpha // m
                    j = alpha % m
                    jp = beta // p
                    kp = beta % p
                    ip = gamma % n
                    # gamma = kp * n + ip
                    target = 1.0 if (i == ip and j == jp) else 0.0
                    b[eq_idx] = target
                    
                    # Coefficients for W[gamma, r]
                    for r in range(rank):
                        A[eq_idx, gamma * rank + r] = U[alpha, r] * V[beta, r]
                    eq_idx += 1
        
        # Solve least squares
        try:
            W_flat, residuals, rank_ls, s = np.linalg.lstsq(A, b, rcond=None)
        except:
            continue
            
        W = W_flat.reshape(9, rank)
        
        # Verify
        U_list = [[float(U[i, r]) for r in range(rank)] for i in range(9)]
        V_list = [[float(V[i, r]) for r in range(rank)] for i in range(9)]
        W_list = [[float(W[i, r]) for r in range(rank)] for i in range(9)]
        
        if _verify_333(U_list, V_list, W_list):
            return {"U": U_list, "V": V_list, "W": W_list}
            
    return None

def _construct_336_from_333(sol_333):
    """Construct 3x3x6 decomposition from two 3x3x3 decompositions."""
    if sol_333 is None:
        return None
        
    U3 = np.array(sol_333["U"])
    V3 = np.array(sol_333["V"])
    W3 = np.array(sol_333["W"])
    r = U3.shape[1]
    
    U = np.zeros((9, 2 * r))
    V = np.zeros((18, 2 * r))
    W = np.zeros((18, 2 * r))
    
    # First half (k=0,1,2)
    U[:, 0:r] = U3
    V[0:9, 0:r] = V3
    W[0:9, 0:r] = W3
    
    # Second half (k=3,4,5)
    U[:, r:2*r] = U3
    V[9:18, r:2*r] = V3
    W[9:18, r:2*r] = W3
    
    U_list = [[float(U[i, r_idx]) for r_idx in range(2*r)] for i in range(9)]
    V_list = [[float(V[i, r_idx]) for r_idx in range(2*r)] for i in range(18)]
    W_list = [[float(W[i, r_idx]) for r_idx in range(2*r)] for i in range(18)]
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 140.0
    
    # Try to find a low rank 3x3x3 decomposition
    best_333 = None
    for r in range(19, 26):
        if time.time() > deadline - 10:
            break
        sol = _find_333(r, time_limit=5.0, seed=r)
        if sol is not None:
            best_333 = sol
            break
            
    if best_333 is not None:
        result = _construct_336_from_333(best_333)
        if result is not None and _verify(result["U"], result["V"], result["W"]):
            return result
            
    # Fallback to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
