"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 140  # 140s budget, leave 10s margin
    
    n, m, p = DIMENSIONS
    import numpy as np
    
    # Build the target tensor T[a,b,c] where a in [0,4), b in [0,16), c in [0,16)
    # T[2i+j, 8j+k, 2k+i] = 1 for i in {0,1}, j in {0,1}, k in {0..7}
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Helper: verify a decomposition (U, V, W) against T
    def verify(U, V, W, tol=1e-9):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        T_recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        max_err = np.max(np.abs(T_recon - T))
        return max_err < tol, max_err
    
    # Efficient ALS using Kronecker products
    def als_step_U_fast(V, W, T):
        """Solve for U given V, W using Kronecker structure"""
        r = V.shape[1]
        U = np.zeros((4, r))
        # M[(b,c), r'] = V[b, r'] * W[c, r'] = (V ⊗ W)[(b,c), r']
        # Use kron: M = np.kron(V, W) but need to be careful with ordering
        # M has shape (16*16, r), M[b*16+c, r'] = V[b, r'] * W[c, r']
        # This is exactly np.kron(V, W) if we index correctly
        # Actually np.kron(V, W) gives shape (16*16, r) with [b*16+c, r'] = V[b,r']*W[c,r']
        M = np.kron(V, W)  # shape (256, r)
        for a in range(4):
            rhs = T[a, :, :].flatten()
            U[a, :], _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
        return U
    
    def als_step_V_fast(U, W, T):
        """Solve for V given U, W"""
        r = W.shape[1]
        V = np.zeros((16, r))
        # M[(a,c), r'] = U[a, r'] * W[c, r']
        M = np.kron(U, W)  # shape (4*16, r) = (64, r), M[a*16+c, r'] = U[a,r']*W[c,r']
        for b in range(16):
            rhs = T[:, b, :].flatten()
            V[b, :], _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
        return V
    
    def als_step_W_fast(U, V, T):
        """Solve for W given U, V"""
        r = V.shape[1]
        W = np.zeros((16, r))
        # M[(a,b), r'] = U[a, r'] * V[b, r']
        M = np.kron(U, V)  # shape (4*16, r) = (64, r), M[a*16+b, r'] = U[a,r']*V[b,r']
        for c in range(16):
            rhs = T[:, :, c].flatten()
            W[c, :], _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
        return W
    
    def als_decompose(T, rank, max_iter=200, tol=1e-12, seed=0):
        """ALS decomposition to rank r"""
        rng = np.random.RandomState(seed)
        U = rng.randn(4, rank) * 0.1
        V = rng.randn(16, rank) * 0.1
        W = rng.randn(16, rank) * 0.1
        
        for it in range(max_iter):
            U = als_step_U_fast(V, W, T)
            V = als_step_V_fast(U, W, T)
            W = als_step_W_fast(U, V, T)
            
            U_recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            err = np.max(np.abs(U_recon - T))
            if err < tol:
                return U, V, W, err
        
        U_recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        err = np.max(np.abs(U_recon - T))
        return U, V, W, err
    
    # First, verify Strassen works correctly (rank 28)
    # Use Strassen for each 2x2 block of B, giving rank 4*7=28
    num_blocks = p // 2  # 4 blocks of 2x2
    rank_strassen = num_blocks * 7  # 28
    
    U_str = np.zeros((n * m, rank_strassen))  # 4 x 28
    V_str = np.zeros((m * p, rank_strassen))  # 16 x 28
    W_str = np.zeros((p * n, rank_strassen))  # 16 x 28
    
    for b in range(num_blocks):
        base = b * 7
        # For block b, B columns are 2b and 2b+1
        # B[0, 2b] index: 0*8 + 2b = 2b
        # B[0, 2b+1] index: 0*8 + 2b+1 = 2b+1
        # B[1, 2b] index: 1*8 + 2b = 8+2b
        # B[1, 2b+1] index: 1*8 + 2b+1 = 9+2b
        
        # C[i, 2b] index: (2b)*2 + i = 4b + i
        # C[i, 2b+1] index: (2b+1)*2 + i = 4b + 2 + i
        
        # Strassen M1 = (A[0,0]+A[1,1]) * (B[0,2b]+B[1,2b+1])
        # A[0,0] idx=0, A[1,1] idx=3
        U_str[0, base + 0] += 1
        U_str[3, base + 0] += 1
        V_str[2*b, base + 0] += 1        # B[0, 2b]
        V_str[8 + 2*b + 1, base + 0] += 1  # B[1, 2b+1]
        # M1 contributes: C[0,2b] += M1, C[1,2b+1] += M1
        W_str[4*b + 0, base + 0] += 1    # C[0, 2b]
        W_str[4*b + 3, base + 0] += 1    # C[1, 2b+1]
        
        # M2 = (A[1,0]+A[1,1]) * B[0,2b]
        U_str[2, base + 1] += 1  # A[1,0] idx=2
        U_str[3, base + 1] += 1  # A[1,1] idx=3
        V_str[2*b, base + 1] += 1  # B[0, 2b]
        # M2 contributes: C[0,2b+1] += M2, C[1,2b+1] -= M2
        W_str[4*b + 1, base + 1] += 1    # C[0, 2b+1]
        W_str[4*b + 3, base + 1] -= 1    # C[1, 2b+1]
        
        # M3 = A[0,0] * (B[0,2b+1]-B[1,2b+1])
        U_str[0, base + 2] += 1  # A[0,0] idx=0
        V_str[2*b + 1, base + 2] += 1      # B[0, 2b+1]
        V_str[8 + 2*b + 1, base + 2] -= 1  # B[1, 2b+1]
        # M3 contributes: C[0,2b+1] += M3, C[1,2b+1] += M3
        W_str[4*b + 1, base + 2] += 1    # C[0, 2b+1]
        W_str[4*b + 3, base + 2] += 1    # C[1, 2b+1]
        
        # M4 = A[1,1] * (B[1,2b]-B[0,2b])
        U_str[3, base + 3] += 1  # A[1,1] idx=3
        V_str[8 + 2*b, base + 3] += 1    # B[1, 2b]
        V_str[2*b, base + 3] -= 1        # B[0, 2b]
        # M4 contributes: C[0,2b] -= M4, C[1,2b] += M4
        W_str[4*b + 0, base + 3] -= 1    # C[0, 2b]
        W_str[4*b + 2, base + 3] += 1    # C[1, 2b]
        
        # M5 = (A[0,0]+A[0,1]) * B[1,2b+1]
        U_str[0, base + 4] += 1  # A[0,0] idx=0
        U_str[1, base + 4] += 1  # A[0,1] idx=1
        V_str[8 + 2*b + 1, base + 4] += 1  # B[1, 2b+1]
        # M5 contributes: C[0,2b] -= M5, C[1,2b] += M5
        W_str[4*b + 0, base + 4] -= 1    # C[0, 2b]
        W_str[4*b + 2, base + 4] += 1    # C[1, 2b]
        
        # M6 = (A[1,0]-A[0,0]) * (B[0,2b]+B[0,2b+1])
        U_str[2, base + 5] += 1  # A[1,0] idx=2
        U_str[0, base + 5] -= 1  # A[0,0] idx=0
        V_str[2*b, base + 5] += 1      # B[0, 2b]
        V_str[2*b + 1, base + 5] += 1  # B[0, 2b+1]
        # M6 contributes: C[1,2b+1] += M6
        W_str[4*b + 3, base + 5] += 1    # C[1, 2b+1]
        
        # M7 = (A[0,1]-A[1,1]) * (B[1,2b]+B[1,2b+1])
        U_str[1, base + 6] += 1  # A[0,1] idx=1
        U_str[3, base + 6] -= 1  # A[1,1] idx=3
        V_str[8 + 2*b, base + 6] += 1    # B[1, 2b]
        V_str[8 + 2*b + 1, base + 6] += 1  # B[1, 2b+1]
        # M7 contributes: C[0,2b] += M7
        W_str[4*b + 0, base + 6] += 1    # C[0, 2b]
    
    # Verify Strassen
    ok_str, err_str = verify(U_str, V_str, W_str, tol=1e-9)
    
    if not ok_str:
        # Fallback: schoolbook rank 32
        rank_s = n * m * p
        U_s = np.zeros((n * m, rank_s))
        V_s = np.zeros((m * p, rank_s))
        W_s = np.zeros((p * n, rank_s))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j, idx] = 1
                    V_s[j * p + k, idx] = 1
                    W_s[k * n + i, idx] = 1
                    idx += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    # Strassen works, now try to find rank < 28
    best_result = (U_str, V_str, W_str)
    best_rank = 28
    
    # Try ranks from 27 down to 20 with faster ALS
    for target_rank in range(27, 19, -1):
        if time.time() > deadline - 20:
            break
        
        found = False
        # Try multiple random seeds
        for seed in range(10):
            if time.time() > deadline - 20:
                break
            
            U, V, W, err = als_decompose(T, target_rank, max_iter=500, tol=1e-10, seed=seed)
            
            if err < 1e-9:
                ok, max_err = verify(U, V, W, tol=1e-9)
                if ok:
                    best_result = (U, V, W)
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            continue
        else:
            break
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
