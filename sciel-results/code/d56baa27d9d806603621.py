import time
import numpy as np
from itertools import product

def construct():
    n, m, p = 3, 3, 4
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill

    # Build target tensor T[a,b,c]
    # a = i*m + j, b = j*p + k, c = k*n + i
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Schoolbook fallback: rank 36
    rank_fb = n * m * p
    U_fb = np.zeros((n*m, rank_fb), dtype=float)
    V_fb = np.zeros((m*p, rank_fb), dtype=float)
    W_fb = np.zeros((p*n, rank_fb), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1.0
                V_fb[j*p+k, col] = 1.0
                W_fb[k*n+i, col] = 1.0
                col += 1

    best_U, best_V, best_W = U_fb, V_fb, W_fb
    best_rank = rank_fb

    def snap_to_half(X):
        return np.round(X * 2) / 2.0

    def check_decomposition(U, V, W, rank, tol=1e-9):
        """Check if the decomposition exactly reproduces T."""
        T_decomp = np.einsum('at,bt,ct->abc', U, V, W, optimize=True)
        return np.allclose(T_decomp, T, atol=tol)

    def als_decompose(rank, U_init, V_init, W_init, max_iter=300, tol=1e-14):
        """Alternating least squares for tensor decomposition with structured updates."""
        U = U_init.copy().astype(float)
        V = V_init.copy().astype(float)
        W = W_init.copy().astype(float)
        
        # Precompute for efficiency
        n_m = n * m  # 9
        m_p = m * p  # 12
        p_n = p * n  # 12
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            # Update U: for each a, solve (V ⊙ W)^T U[a,:]^T = T[a,:,:].reshape(-1)
            # Khatri-Rao product of V and W: (V ⊙ W)[(b,c), t] = V[b,t] * W[c,t]
            VW = np.einsum('bt,ct->bct', V, W)  # shape (m_p, p_n, rank)
            VW_flat = VW.reshape(m_p * p_n, rank)  # (m_p*p_n, rank)
            
            # Precompute VW_flat^T @ VW_flat (shared for all a)
            VWtVW = VW_flat.T @ VW_flat  # (rank, rank)
            
            U_new = np.zeros((n_m, rank))
            for a in range(n_m):
                Ta = T[a, :, :].reshape(-1)
                A = VW_flat.T @ Ta  # (rank,)
                try:
                    U_new[a, :] = np.linalg.solve(WWtVW_reg := VWtVW + 1e-12 * np.eye(rank), A)
                except:
                    U_new[a, :] = np.linalg.lstsq(VWtVW + 1e-12 * np.eye(rank), A, rcond=None)[0]
            U = U_new
            
            # Update V
            UW = np.einsum('at,ct->act', U, W)  # (n_m, p_n, rank)
            UW_flat = UW.reshape(n_m * p_n, rank)
            UWtUW = UW_flat.T @ UW_flat
            
            V_new = np.zeros((m_p, rank))
            for b in range(m_p):
                Tb = T[:, b, :].reshape(-1)
                A = UW_flat.T @ Tb
                try:
                    V_new[b, :] = np.linalg.solve(UWtUW + 1e-12 * np.eye(rank), A)
                except:
                    V_new[b, :] = np.linalg.lstsq(UWtUW + 1e-12 * np.eye(rank), A, rcond=None)[0]
            V = V_new
            
            # Update W
            UV = np.einsum('at,bt->abt', U, V)  # (n_m, m_p, rank)
            UV_flat = UV.reshape(n_m * m_p, rank)
            UVtUV = UV_flat.T @ UV_flat
            
            W_new = np.zeros((p_n, rank))
            for c in range(p_n):
                Tc = T[:, :, c].reshape(-1)
                A = UV_flat.T @ Tc
                try:
                    W_new[c, :] = np.linalg.solve(UVtUV + 1e-12 * np.eye(rank), A)
                except:
                    W_new[c, :] = np.linalg.lstsq(UVtUV + 1e-12 * np.eye(rank), A, rcond=None)[0]
            W = W_new
            
            # Check convergence
            T_decomp = np.einsum('at,bt,ct->abc', U, V, W, optimize=True)
            err = np.linalg.norm(T - T_decomp)
            if err < tol:
                break
        
        return U, V, W, err

    def try_rank_als(rank, num_restarts=15):
        """Try to find a decomposition of given rank using ALS."""
        nonlocal best_U, best_V, best_W, best_rank
        
        for restart in range(num_restarts):
            if time.time() > deadline - 15:
                return False
            
            # Initialize with structured random values from half-integer grid
            # Use a mix of 0, ±0.5, ±1
            vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
            U_init = np.array([[np.random.choice(vals) for _ in range(rank)] for _ in range(n*m)])
            V_init = np.array([[np.random.choice(vals) for _ in range(rank)] for _ in range(m*p)])
            W_init = np.array([[np.random.choice(vals) for _ in range(rank)] for _ in range(p*n)])
            
            # Run ALS
            U, V, W, err = als_decompose(rank, U_init, V_init, W_init, max_iter=200)
            
            # Snap to half-integers and check
            U_s = snap_to_half(U)
            V_s = snap_to_half(V)
            W_s = snap_to_half(W)
            
            if check_decomposition(U_s, V_s, W_s, rank):
                best_U, best_V, best_W = U_s, V_s, W_s
                best_rank = rank
                return True
            
            # Also try without snapping if very close
            if err < 1e-10:
                U_close = np.allclose(U, snap_to_half(U), atol=1e-6)
                V_close = np.allclose(V, snap_to_half(V), atol=1e-6)
                W_close = np.allclose(W, snap_to_half(W), atol=1e-6)
                if U_close and V_close and W_close:
                    best_U, best_V, best_W = snap_to_half(U), snap_to_half(V), snap_to_half(W)
                    best_rank = rank
                    return True
        
        return False

    def try_rank_greedy(rank, num_restarts=10):
        """Greedy rank-1 peeling approach."""
        nonlocal best_U, best_V, best_W, best_rank
        
        for restart in range(num_restarts):
            if time.time() > deadline - 15:
                return False
            
            # Start with random initial decomposition
            vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
            U = np.array([[np.random.choice(vals) for _ in range(rank)] for _ in range(n*m)])
            V = np.array([[np.random.choice(vals) for _ in range(rank)] for _ in range(m*p)])
            W = np.array([[np.random.choice(vals) for _ in range(rank)] for _ in range(p*n)])
            
            # Run ALS to get close
            U, V, W, err = als_decompose(rank, U, V, W, max_iter=100)
            
            # Now do greedy peeling: iteratively find best rank-1 approximation
            # and subtract, refining the remaining factors
            residual = T.copy()
            U_new = np.zeros((n*m, rank))
            V_new = np.zeros((m*p, rank))
            W_new = np.zeros((p*n, rank))
            
            for t in range(rank):
                if time.time() > deadline - 15:
                    return False
                
                # Find best rank-1 approximation to residual
                # Use SVD on flattened residual
                R_flat = residual.reshape(n*m, m*p*p*n)
                try:
                    U_svd, S_svd, Vt_svd = np.linalg.svd(R_flat, full_matrices=False)
                    # Best rank-1: u = U_svd[:,0], v = S_svd[0] * Vt_svd[0].reshape(m*p, p*n)
                    # But we need it in the form u[a] * v[b] * w[c]
                    # This is more complex; let's use a simpler approach
                except:
                    pass
                
                # Simpler: use ALS result and snap
                break
            
            # Just use ALS result
            U_s = snap_to_half(U)
            V_s = snap_to_half(V)
            W_s = snap_to_half(W)
            
            if check_decomposition(U_s, V_s, W_s, rank):
                best_U, best_V, best_W = U_s, V_s, W_s
                best_rank = rank
                return True
        
        return False

    # Try ranks from low to high
    # For 3x3x4 matrix multiplication, the rank is known to be at most 27 (Lader's result for 3x3x3 is 23, 
    # and 3x3x4 should be around 24-27)
    # Let's try a range of ranks
    
    # First, try some structured initializations that might work better
    def try_structured(rank):
        """Try structured initializations based on known patterns."""
        nonlocal best_U, best_V, best_W, best_rank
        
        if time.time() > deadline - 20:
            return False
        
        # Try a few different structured patterns
        patterns = [
            'cyclic', 'block', 'random_half'
        ]
        
        for pattern in patterns:
            if time.time() > deadline - 15:
                return False
            
            if pattern == 'cyclic':
                # Cyclic pattern initialization
                U_init = np.zeros((n*m, rank))
                V_init = np.zeros((m*p, rank))
                W_init = np.zeros((p*n, rank))
                for t in range(rank):
                    for i in range(n):
                        for j in range(m):
                            if (i + j + t) % 3 == 0:
                                U_init[i*m+j, t] = 1.0
                    for j in range(m):
                        for k in range(p):
                            if (j + k + t) % 3 == 0:
                                V_init[j*p+k, t] = 1.0
                    for k in range(p):
                        for i in range(n):
                            if (k + i + t) % 3 == 0:
                                W_init[k*n+i, t] = 1.0
            elif pattern == 'block':
                U_init = np.random.choice([-1.0, 0.0, 1.0], size=(n*m, rank))
                V_init = np.random.choice([-1.0, 0.0, 1.0], size=(m*p, rank))
                W_init = np.random.choice([-1.0, 0.0, 1.0], size=(p*n, rank))
            else:
                U_init = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(n*m, rank))
                V_init = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(m*p, rank))
                W_init = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(p*n, rank))
            
            U, V, W, err = als_decompose(rank, U_init, V_init, W_init, max_iter=150)
            
            U_s = snap_to_half(U)
            V_s = snap_to_half(V)
            W_s = snap_to_half(W)
            
            if check_decomposition(U_s, V_s, W_s, rank):
                best_U, best_V, best_W = U_s, V_s, W_s
                best_rank = rank
                return True
        
        return False

    # Try ranks from low to high
    # Known lower bound: max(9, 12, 12) = 12
    # For 3x3x3, rank is 23 (Lader 1977). For 3x3x4, expect similar or slightly higher.
    # Try from 20 up to 35
    
    for rank in range(20, 36):
        if time.time() > deadline - 20:
            break
        if rank >= best_rank:
            continue
        
        # Try structured initialization first
        if try_structured(rank):
            break
        
        # Try ALS with random restarts
        num_restarts = 10 if rank <= 25 else 8
        if try_rank_als(rank, num_restarts):
            break
    
    # Final snap and verify
    U_out = snap_to_half(best_U)
    V_out = snap_to_half(best_V)
    W_out = snap_to_half(best_W)
    
    if not check_decomposition(U_out, V_out, W_out, best_rank):
        # Fall back to schoolbook
        U_out = U_fb
        V_out = V_fb
        W_out = W_fb
        best_rank = rank_fb
    
    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
