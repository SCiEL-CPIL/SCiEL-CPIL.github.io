"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import random

def construct():
    n, m, p = DIMENSIONS
    
    def verify(U, V, W):
        """Verify exact decomposition. Returns True if valid."""
        n, m, p = DIMENSIONS
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        r = U.shape[1]
        if V.shape[1] != r or W.shape[1] != r:
            return False
        # For each (i,j,k), check sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for t in range(r):
                        total += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True
    
    def schoolbook():
        rank = n * m * p
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
    
    def try_structured_rank(r, seed=42):
        """Try to find a decomposition of rank r with structured U and W."""
        random.seed(seed)
        np.random.seed(seed)
        
        # Strategy: Fix U and W to be structured, then solve for V via linear system.
        # The constraint is: for each (i,j,k), sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1
        # This is linear in V for fixed U, W.
        
        # Build U as a structured matrix: each row (i,j) has a pattern of 1s
        # Let's try: U[i*m+j, t] = 1 if (i*m+j + t) % some_pattern == 0
        # Or use a more sophisticated approach.
        
        # Approach 1: U and W are random 0-1 matrices with controlled density
        # Approach 2: U and W are based on modular arithmetic patterns
        
        # Let's try multiple patterns
        patterns = [
            ('mod1', lambda i, j, t, nm: (i * m + j + t) % 3 == 0),
            ('mod2', lambda i, j, t, nm: (i * m + j + 2 * t) % 5 == 0),
            ('mod3', lambda i, j, t, nm: (i * m + j) % 4 == t % 4),
            ('random', lambda i, j, t, nm: random.random() < 0.3),
            ('random2', lambda i, j, t, nm: random.random() < 0.5),
        ]
        
        for pat_name, pat_func in patterns:
            U = np.zeros((n * m, r))
            W = np.zeros((p * n, r))
            
            for i in range(n):
                for j in range(m):
                    for t in range(r):
                        U[i * m + j, t] = 1.0 if pat_func(i, j, t, n * m) else 0.0
            
            for k in range(p):
                for i in range(n):
                    for t in range(r):
                        W[k * n + i, t] = 1.0 if pat_func(k, i, t, p * n) else 0.0
            
            # Check that U and W have enough structure (not too sparse)
            if np.sum(U) < r * 2 or np.sum(W) < r * 2:
                continue
            
            # Build linear system for V
            # Unknowns: V[j*p+k, t] for j in [0,m), k in [0,p), t in [0,r)
            # Total unknowns: m*p*r
            # Equations: n*m*p = 112
            
            A = np.zeros((n * m * p, m * p * r))
            b = np.ones(n * m * p)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for t in range(r):
                            coeff = U[i * m + j, t] * W[k * n + i, t]
                            if coeff != 0:
                                var_idx = (j * p + k) * r + t
                                A[eq_idx, var_idx] += coeff
                        eq_idx += 1
            
            # Solve using least squares
            try:
                V_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
                V = V_flat.reshape(m * p, r)
                
                # Verify the solution
                if verify(U.tolist(), V.tolist(), W.tolist()):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            except:
                pass
        
        return None
    
    # Try ranks from low to high
    # Theoretical lower bound: for 4x4 * 4x7, known results suggest rank around 80-100
    # Let's try from 70 up to 112
    for r in range(70, 113):
        result = try_structured_rank(r, seed=r)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
