"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Find a low-rank exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    Returns dict with U, V, W matrices.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 72
    
    # Build schoolbook decomposition as fallback
    U_school = np.zeros((n * m, schoolbook_rank))
    V_school = np.zeros((m * p, schoolbook_rank))
    W_school = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    best_rank = schoolbook_rank
    
    # Precompute target tensor
    T_expected = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_expected[a, b, c] = 1
    
    def verify_decomposition(U, V, W):
        """Verify exact decomposition."""
        T_recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(T_recon, T_expected, atol=1e-9, rtol=1e-9)
    
    # ALS: fix two, solve for the third
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    def solve_for_U(V, W, T, target_rank):
        """Fix V and W, solve for U."""
        # For each a: T[a,b,c] = sum_r U[a,r] * (V[b,r]*W[c,r])
        # Build matrix B of size (m*p*p*n) x rank
        # B[idx, r] = V[b,r] * W[c,r] where idx = b * (p*n) + c
        B = np.zeros((m * p * p * n, target_rank))
        for b in range(m * p):
            for c in range(p * n):
                idx = b * (p * n) + c
                B[idx, :] = V[b, :] * W[c, :]
        
        T_flat = T.reshape(n * m, -1)
        U = np.zeros((n * m, target_rank))
        for a in range(n * m):
            try:
                U[a, :] = np.linalg.lstsq(B, T_flat[a, :], rcond=None)[0]
            except:
                U[a, :] = 0
        return U
    
    def solve_for_V(U, W, T, target_rank):
        """Fix U and W, solve for V."""
        # For each b: T[a,b,c] = sum_r V[b,r] * (U[a,r]*W[c,r])
        # Build matrix B of size (n*m*p*n) x rank
        # B[idx, r] = U[a,r] * W[c,r] where idx = a * (p*n) + c
        B = np.zeros((n * m * p * n, target_rank))
        for a in range(n * m):
            for c in range(p * n):
                idx = a * (p * n) + c
                B[idx, :] = U[a, :] * W[c, :]
        
        # T needs to be reshaped: for each b, we need T[:,b,:] flattened
        T_reshaped = np.zeros((m * p, n * m * p * n))
        for b in range(m * p):
            T_reshaped[b, :] = T[:, b, :].reshape(-1)
        
        V = np.zeros((m * p, target_rank))
        for b in range(m * p):
            try:
                V[b, :] = np.linalg.lstsq(B, T_reshaped[b, :], rcond=None)[0]
            except:
                V[b, :] = 0
        return V
    
    def solve_for_W(U, V, T, target_rank):
        """Fix U and V, solve for W."""
        # For each c: T[a,b,c] = sum_r W[c,r] * (U[a,r]*V[b,r])
        # Build matrix B of size (n*m*m*p) x rank
        # B[idx, r] = U[a,r] * V[b,r] where idx = a * (m*p) + b
        B = np.zeros((n * m * m * p, target_rank))
        for a in range(n * m):
            for b in range(m * p):
                idx = a * (m * p) + b
                B[idx, :] = U[a, :] * V[b, :]
        
        # T needs to be reshaped: for each c, we need T[:,:,c] flattened
        T_reshaped = np.zeros((p * n, n * m * m * p))
        for c in range(p * n):
            T_reshaped[c, :] = T[:, :, c].reshape(-1)
        
        W = np.zeros((p * n, target_rank))
        for c in range(p * n):
            try:
                W[c, :] = np.linalg.lstsq(B, T_reshaped[c, :], rcond=None)[0]
            except:
                W[c, :] = 0
        return W
    
    # Try rank from lower bound to schoolbook
    # Theoretical lower bound is max(n*m, m*p, p*n) = max(9, 24, 24) = 24
    # But actual minimum is likely higher
    
    for target_rank in range(24, schoolbook_rank + 1):
        if time.time() > deadline - 30:
            break
        
        # Multiple random attempts for each rank
        max_attempts = 8 if target_rank < 40 else 4
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 25:
                break
            
            # Initialize with random values
            rng = np.random.RandomState(42 + attempt * 100 + target_rank)
            
            U = rng.randn(n * m, target_rank) * 0.5
            V = rng.randn(m * p, target_rank) * 0.5
            W = rng.randn(p * n, target_rank) * 0.5
            
            # ALS iterations
            converged = False
            for it in range(30):
                if time.time() > deadline - 20:
                    break
                
                # Solve for U
                U = solve_for_U(V, W, T_expected, target_rank)
                
                # Solve for V
                V = solve_for_V(U, W, T_expected, target_rank)
                
                # Solve for W
                W = solve_for_W(U, V, T_expected, target_rank)
                
                # Check convergence
                T_recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
                error = np.linalg.norm(T_recon - T_expected)
                
                if error < 1e-8:
                    if verify_decomposition(U, V, W):
                        best_U = U.copy()
                        best_V = V.copy()
                        best_W = W.copy()
                        best_rank = target_rank
                        converged = True
                        break
            
            if converged:
                break
        
        if best_rank < schoolbook_rank:
            break
    
    # Convert to lists for return
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
