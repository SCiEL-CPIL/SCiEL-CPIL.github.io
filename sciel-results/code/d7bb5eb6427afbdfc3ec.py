import numpy as np
import time

def construct():
    n, m, p = 2, 3, 7
    start = time.time()
    deadline = start + 115
    
    # Index mappings
    def a_idx(i, k): return i * m + k
    def b_idx(k, j): return k * p + j
    def c_idx(j, i): return j * n + i  # column-major: W[k*n+i]
    
    # Build target tensor: T[a, b, c] = 1 if there exists (i,j,k) with a=a_idx(i,k), b=b_idx(k,j), c=c_idx(j,i)
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    triples = []
    for i in range(n):
        for j in range(p):
            for k in range(m):
                a = a_idx(i, k)
                b = b_idx(k, j)
                c = c_idx(j, i)
                T[a, b, c] = 1.0
                triples.append((a, b, c))
    
    target = np.array([T[a, b, c] for a, b, c in triples], dtype=np.float64)
    
    def verify(U, V, W):
        """Check if decomposition is exact (integer/half-integer entries)"""
        U = np.asarray(U, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        W = np.asarray(W, dtype=np.float64)
        for a, b, c in triples:
            val = 0.0
            for r in range(U.shape[1]):
                val += U[a, r] * V[b, r] * W[c, r]
            if abs(val - 1.0) > 1e-6:
                return False
        # Check zero entries
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    if T[a, b, c] == 0:
                        val = 0.0
                        for r in range(U.shape[1]):
                            val += U[a, r] * V[b, r] * W[c, r]
                        if abs(val) > 1e-6:
                            return False
        return True
    
    def alternating_least_squares(U, V, W, max_iter=200, tol=1e-8):
        """ALS to minimize sum of squared residuals over all (a,b,c) entries"""
        R = U.shape[1]
        for it in range(max_iter):
            # Update U: for each a, solve least squares
            for a in range(n*m):
                # T_a[b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # For fixed a: T_a = (V ⊙ W) @ U[a,:]  where (V⊙W)[(b,c),r] = V[b,r]*W[c,r]
                VW = (V[:, :, None] * W[:, None, :]).reshape(m*p*p*n, R)
                Ta = T[a, :, :].reshape(-1)
                # LS: minimize ||VW @ x - Ta||^2
                U[a, :] = np.linalg.lstsq(VW, Ta, rcond=None)[0]
            
            # Update V
            for b in range(m*p):
                UW = (U[:, :, None] * W[:, None, :]).reshape(n*m*p*n, R)
                Tb = T[:, b, :].reshape(-1)
                V[b, :] = np.linalg.lstsq(UW, Tb, rcond=None)[0]
            
            # Update W
            for c in range(p*n):
                UV = (U[:, :, None] * V[:, None, :]).reshape(n*m*m*p, R)
                Tc = T[:, :, c].reshape(-1)
                W[c, :] = np.linalg.lstsq(UV, Tc, rcond=None)[0]
            
            if time.time() > deadline - 5:
                break
        
        return U, V, W
    
    # Build R=39 reference solution
    def build_reference():
        rank = 39
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        
        def add_strassen(k0, k1):
            nonlocal col
            # A: rows 0-1, cols 0-1 => indices 0,1,3,4
            # B: rows 0-1, cols k0,k1
            # C: cols k0,k1, rows 0-1
            b00 = b_idx(0, k0); b01 = b_idx(0, k1)
            b10 = b_idx(1, k0); b11 = b_idx(1, k1)
            c00 = c_idx(k0, 0); c01 = c_idx(k1, 0)
            c10 = c_idx(k0, 1); c11 = c_idx(k1, 1)
            
            # M1 = (A00+A11)(B00+B11)
            U[a_idx(0,0), col] = 1; U[a_idx(1,1), col] = 1
            V[b00, col] = 1; V[b11, col] = 1
            W[c00, col] = 1; W[c11, col] = 1
            col += 1
            # M2 = (A10+A11)B00
            U[a_idx(1,0), col] = 1; U[a_idx(1,1), col] = 1
            V[b00, col] = 1
            W[c10, col] = 1; W[c11, col] = -1
            col += 1
            # M3 = A00(B01-B11)
            U[a_idx(0,0), col] = 1
            V[b01, col] = 1; V[b11, col] = -1
            W[c01, col] = 1; W[c11, col] = 1
            col += 1
            # M4 = A11(B10-B00)
            U[a_idx(1,1), col] = 1
            V[b10, col] = 1; V[b00, col] = -1
            W[c00, col] = 1; W[c10, col] = 1
            col += 1
            # M5 = (A00+A01)B11
            U[a_idx(0,0), col] = 1; U[a_idx(0,1), col] = 1
            V[b11, col] = 1
            W[c00, col] = -1; W[c01, col] = 1
            col += 1
            # M6 = (A10-A00)(B00+B01)
            U[a_idx(1,0), col] = 1; U[a_idx(0,0), col] = -1
            V[b00, col] = 1; V[b01, col] = 1
            W[c11, col] = 1
            col += 1
            # M7 = (A01-A11)(B10+B11)
            U[a_idx(0,1), col] = 1; U[a_idx(1,1), col] = -1
            V[b10, col] = 1; V[b11, col] = 1
            W[c00, col] = 1
            col += 1
        
        add_strassen(0, 1)
        add_strassen(2, 3)
        add_strassen(4, 5)
        
        # Col 6: rows 0-1
        for k in [0, 1]:
            U[a_idx(0, k), col] = 1
            V[b_idx(k, 6), col] = 1
            W[c_idx(6, 0), col] = 1
            col += 1
            U[a_idx(1, k), col] = 1
            V[b_idx(k, 6), col] = 1
            W[c_idx(6, 1), col] = 1
            col += 1
        
        # Row 2 of B: A[:,2] * B[2,:]
        for j in range(7):
            U[a_idx(0, 2), col] = 1
            V[b_idx(2, j), col] = 1
            W[c_idx(j, 0), col] = 1
            col += 1
            U[a_idx(1, 2), col] = 1
            V[b_idx(2, j), col] = 1
            W[c_idx(j, 1), col] = 1
            col += 1
        
        return U, V, W
    
    # First: verify reference
    U_ref, V_ref, W_ref = build_reference()
    assert verify(U_ref, V_ref, W_ref), "Reference R=39 solution failed verification!"
    
    # Try to find lower rank via ALS
    best_result = {"U": U_ref.tolist(), "V": V_ref.tolist(), "W": W_ref.tolist()}
    
    for target_rank in [38, 37, 36, 35, 34]:
        if time.time() > deadline - 20:
            break
        
        found = False
        for attempt in range(3):
            if time.time() > deadline - 10:
                break
            
            # Initialize with small random values
            np.random.seed(42 + target_rank * 100 + attempt)
            U = np.random.randn(n*m, target_rank) * 0.5
            V = np.random.randn(m*p, target_rank) * 0.5
            W = np.random.randn(p*n, target_rank) * 0.5
            
            U, V, W = alternating_least_squares(U, V, W, max_iter=100)
            
            # Round to nearest half-integer
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            if verify(U_r, V_r, W_r):
                found = True
                best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                break
            
            if time.time() > deadline - 15:
                break
        
        if found:
            break
    
    return best_result
