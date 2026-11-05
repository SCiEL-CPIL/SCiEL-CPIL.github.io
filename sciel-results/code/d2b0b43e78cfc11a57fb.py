import numpy as np
import time
from itertools import product

def construct():
    start_time = time.time()
    timeout = 110
    n, m, p = 2, 3, 3
    
    def elapsed():
        return time.time() - start_time
    
    def build_tensor():
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        return T
    
    def verify(U, V, W, T, tol=1e-9):
        Re = T - np.einsum('ar,br,cr->abc', U, V, W)
        return np.max(np.abs(Re)) < tol
    
    def build_rank17():
        rank = 17
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        def set_U(i, j, r, val): U[i * m + j, r] = val
        def set_V(j, k, r, val): V[j * p + k, r] = val
        def set_W(k, i, r, val): W[k * n + i, r] = val
        
        set_U(0, 0, 0, 1); set_V(0, 0, 0, 1); set_W(0, 0, 0, 1)
        set_U(1, 0, 1, 1); set_U(1, 1, 1, 1)
        set_V(0, 0, 1, 1)
        set_W(0, 1, 1, 1); set_W(1, 1, 1, -1)
        
        set_U(0, 0, 2, 1)
        set_V(0, 1, 2, 1); set_V(1, 1, 2, -1)
        set_W(1, 0, 2, 1); set_W(1, 1, 2, 1)
        
        set_U(1, 1, 3, 1)
        set_V(1, 0, 3, 1); set_V(0, 0, 3, -1)
        set_W(0, 0, 3, 1); set_W(0, 1, 3, 1)
        
        set_U(0, 0, 4, 1); set_U(0, 1, 4, 1)
        set_V(1, 1, 4, 1)
        set_W(0, 0, 4, -1); set_W(1, 0, 4, 1)
        
        set_U(1, 0, 5, 1); set_U(0, 0, 5, -1)
        set_V(0, 0, 5, 1); set_V(0, 1, 5, 1)
        set_W(1, 1, 5, 1)
        
        set_U(0, 1, 6, 1); set_U(1, 1, 6, -1)
        set_V(1, 0, 6, 1); set_V(1, 1, 6, 1)
        set_W(0, 0, 6, 1)
        
        # Cross terms
        set_U(0, 0, 7, 1); set_V(0, 2, 7, 1); set_W(2, 0, 7, 1)
        set_U(0, 1, 8, 1); set_V(1, 2, 8, 1); set_W(2, 0, 8, 1)
        set_U(1, 0, 9, 1); set_V(0, 2, 9, 1); set_W(2, 1, 9, 1)
        set_U(1, 1, 10, 1); set_V(1, 2, 10, 1); set_W(2, 1, 10, 1)
        set_U(0, 2, 11, 1); set_V(2, 0, 11, 1); set_W(0, 0, 11, 1)
        set_U(0, 2, 12, 1); set_V(2, 1, 12, 1); set_W(1, 0, 12, 1)
        set_U(0, 2, 13, 1); set_V(2, 2, 13, 1); set_W(2, 0, 13, 1)
        set_U(1, 2, 14, 1); set_V(2, 0, 14, 1); set_W(0, 1, 14, 1)
        set_U(1, 2, 15, 1); set_V(2, 1, 15, 1); set_W(1, 1, 15, 1)
        set_U(1, 2, 16, 1); set_V(2, 2, 16, 1); set_W(2, 1, 16, 1)
        
        return U, V, W
    
    def try_rank16():
        """Try to find a rank-16 decomposition using structured search."""
        T = build_tensor()
        
        # Strategy: Fix the first 7 terms (Strassen for 2x2x2 block)
        # Then find 9 terms for the remaining cross terms
        # The cross terms involve indices where j=2 or k=2 or i varies
        
        # Let's try: for the 9 remaining terms, each has a specific structure
        # We'll enumerate possible support patterns for U and V (sparse), 
        # then solve for W linearly and check if it's integer
        
        np.random.seed(123)
        
        # Precompute the tensor structure
        # Non-zero entries: T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
        
        # For the cross terms (involving j=2 or k=2), we need:
        # Terms 7-10: U has i,j in {0,1} with j!=2, V has j=0,1 with k=2, W has k=2
        # Terms 11-13: U has i=0, j=2, V has j=2, W has i=0,1
        # Terms 14-16: U has i=1, j=2, V has j=2, W has i=0,1
        
        # Let's try a different approach: use the full 16-term structure
        # with random sparse U,V and solve for W
        
        def try_one_attempt(seed_offset):
            if elapsed() > timeout - 5:
                return None
            
            np.random.seed(42 + seed_offset)
            rank = 16
            U = np.zeros((6, rank))
            V = np.zeros((9, rank))
            
            # Generate sparse ±1 patterns for U and V
            for r in range(rank):
                # U: 2 nonzero entries
                idx_U = np.random.choice(6, 2, replace=False)
                U[idx_U[0], r] = 1
                U[idx_U[1], r] = np.random.choice([-1, 1])
                
                # V: 2 nonzero entries  
                idx_V = np.random.choice(9, 2, replace=False)
                V[idx_V[0], r] = 1
                V[idx_V[1], r] = np.random.choice([-1, 1])
            
            # Solve for W: for each c in 0..5, solve linear system
            W = np.zeros((6, rank))
            
            for c in range(6):
                # Build system: sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
                # for all (a,b) where the entry should be nonzero
                # This is: A @ W[c,:] = b where A[a*9+b, r] = U[a,r]*V[b,r]
                
                A = np.zeros((54, rank))
                b = np.zeros(54)
                
                for a in range(6):
                    for b_idx in range(9):
                        idx = a * 9 + b_idx
                        A[idx, :] = U[a, :] * V[b_idx, :]
                        
                        # Check if this (a, b_idx) corresponds to a nonzero T entry with output c
                        i = a // 3
                        j = a % 3
                        j2 = b_idx // 3
                        k = b_idx % 3
                        
                        if j == j2:
                            c_expected = 2 * k + i
                            if c_expected == c:
                                b[idx] = 1.0
                            else:
                                b[idx] = 0.0
                        else:
                            b[idx] = 0.0
                
                # Solve least squares
                try:
                    W_c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W[c, :] = W_c
                    
                    # Check if integer
                    if np.max(np.abs(W_c - np.round(W_c))) > 1e-6:
                        return None
                    
                    # Check exactness
                    if np.max(np.abs(A @ W_c - b)) > 1e-9:
                        return None
                except:
                    return None
            
            # Round and verify
            U_r = np.round(U)
            V_r = np.round(V)
            W_r = np.round(W)
            
            if verify(U_r, V_r, W_r, T):
                return U_r, V_r, W_r
            
            return None
        
        # Try multiple random seeds
        for attempt in range(200):
            if elapsed() > timeout - 5:
                break
            result = try_one_attempt(attempt)
            if result is not None:
                return {"U": result[0].tolist(), "V": result[1].tolist(), "W": result[2].tolist()}
        
        return None
    
    T = build_tensor()
    
    # Try rank 16 first
    if elapsed() < timeout - 10:
        result16 = try_rank16()
        if result16 is not None:
            return result16
    
    # Fall back to rank 17
    U, V, W = build_rank17()
    
    if verify(U, V, W, T):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Naive rank 18
    rank = 18
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1.0
                V[j * p + k, idx] = 1.0
                W[k * n + i, idx] = 1.0
                idx += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
