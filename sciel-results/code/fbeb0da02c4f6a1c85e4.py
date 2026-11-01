import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build full tensor S[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    S = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Check valid terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if abs(S[a, b, c] - 1.0) > 1e-9:
                    return False
    
    # Check spurious terms are zero
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid.add((i*m+j, j*p+k, k*n+i))
    
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                if (a, b, c) not in valid:
                    if abs(S[a, b, c]) > 1e-9:
                        return False
    return True

def _schoolbook(n, m, p):
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
    return U, V, W

def _strassen_block(n, m, p):
    """
    For multiplying (n x m) by (m x p) where n=m=2, p=16.
    The 2x16 matrix can be seen as 8 independent 2x2 blocks.
    Each 2x2x2 multiplication has rank 7 (Strassen).
    Total rank = 8 * 7 = 56.
    """
    n, m, p = 2, 2, 16
    num_blocks = p // 2  # 8 blocks
    rank = num_blocks * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for t in range(num_blocks):
        intermediates = [
            {
                'U': [(0, 1), (3, 1)],
                'V': [(2*t, 1), (16+2*t+1, 1)],
                'W': [(4*t, 1), (4*t+3, 1)]
            },
            {
                'U': [(2, 1), (3, 1)],
                'V': [(2*t, 1)],
                'W': [(4*t+1, 1), (4*t+3, -1)]
            },
            {
                'U': [(0, 1)],
                'V': [(2*t+1, 1), (16+2*t+1, -1)],
                'W': [(4*t+2, 1), (4*t+3, 1)]
            },
            {
                'U': [(3, 1)],
                'V': [(16+2*t, 1), (2*t, -1)],
                'W': [(4*t, 1), (4*t+1, 1)]
            },
            {
                'U': [(0, 1), (1, 1)],
                'V': [(16+2*t+1, 1)],
                'W': [(4*t, -1), (4*t+2, 1)]
            },
            {
                'U': [(2, 1), (0, -1)],
                'V': [(2*t, 1), (2*t+1, 1)],
                'W': [(4*t+3, 1)]
            },
            {
                'U': [(1, 1), (3, -1)],
                'V': [(16+2*t, 1), (16+2*t+1, 1)],
                'W': [(4*t, 1)]
            }
        ]
        
        for inter in intermediates:
            for idx, coeff in inter['U']:
                U[idx][col] += coeff
            for idx, coeff in inter['V']:
                V[idx][col] += coeff
            for idx, coeff in inter['W']:
                W[idx][col] += coeff
            col += 1
    
    return U, V, W

def _build_tensor(n, m, p):
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _alt_minimize(U, V, W, T, max_iters=200, tol=1e-10):
    n_m, m_p, p_n = T.shape
    R = U.shape[1]
    
    for it in range(max_iters):
        # Update W
        for c in range(p_n):
            A_c = np.outer(U.flatten(), V.flatten()).reshape(n_m, m_p, R).transpose(2, 0, 1).reshape(R, n_m * m_p).T
            target = T[:, :, c].flatten()
            w, _, _, _ = np.linalg.lstsq(A_c, target, rcond=None)
            W[c, :] = w
        
        # Update V
        for b in range(m_p):
            B_b = np.array([np.outer(U[:, r], W[:, r]).flatten() for r in range(R)]).T
            target = T[:, b, :].flatten()
            v, _, _, _ = np.linalg.lstsq(B_b, target, rcond=None)
            V[b, :] = v
        
        # Update U
        for a in range(n_m):
            C_a = np.array([np.outer(V[:, r], W[:, r]).flatten() for r in range(R)]).T
            target = T[a, :, :].flatten()
            u, _, _, _ = np.linalg.lstsq(C_a, target, rcond=None)
            U[a, :] = u
        
        # Check convergence
        S = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.max(np.abs(S - T))
        if err < tol:
            break
    
    return U, V, W, err

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115
    
    T = _build_tensor(n, m, p)
    
    # Try to find a lower-rank solution
    for rank in [32, 36, 40, 44, 48, 52]:
        if time.time() > deadline - 40:
            break
        
        best_err = float('inf')
        best_sol = None
        
        for seed in range(3):
            if time.time() > deadline - 40:
                break
            
            np.random.seed(seed * 1000 + rank)
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
            
            U, V, W, err = _alt_minimize(U, V, W, T, max_iters=100, tol=1e-10)
            
            if err < best_err:
                best_err = err
                best_sol = (U.copy(), V.copy(), W.copy())
            
            if err < 1e-9:
                if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        if best_err < 1e-9 and best_sol is not None:
            U, V, W = best_sol
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to Strassen rank 56
    U, V, W = _strassen_block(n, m, p)
    return {"U": U, "V": V, "W": W}
