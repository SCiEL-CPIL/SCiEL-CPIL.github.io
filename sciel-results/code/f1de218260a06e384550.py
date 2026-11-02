import time
import numpy as np

DIMENSIONS = [2, 2, 12]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10 second safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    # n=2 (rows of A), m=2 (cols of A, rows of B), p=12 (cols of B)
    
    def build_tensor():
        """Build the matrix multiplication tensor of shape (n*m, m*p, p*n)."""
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                u = i * m + j
                for k in range(p):
                    v = j * p + k
                    w = k * n + i
                    T[u, v, w] = 1.0
        return T
    
    T = build_tensor()
    
    def verify(U, V, W):
        """Check if U, V, W correctly decompose the tensor."""
        T_recon = np.zeros((n*m, m*p, p*n))
        for r in range(U.shape[1]):
            T_recon += np.outer(np.outer(U[:, r], V[:, r]), W[:, r]).reshape(n*m, m*p, p*n)
        return np.allclose(T_recon, T, atol=1e-10)
    
    # Try DFT-based decomposition
    # Map the 12-column dimension to frequency domain using DFT
    # The key insight: for each frequency component, we have a 2x2x2 problem
    # But we can potentially do better by exploiting the structure
    
    # Let's try a direct optimization approach with multiple restarts
    # Start from the Strassen-based rank-42 decomposition and try to reduce
    
    def strassen_decomposition():
        """Construct the Strassen-based rank-42 decomposition."""
        num_blocks = p // 2  # 6 blocks
        rank = num_blocks * 7  # 42
        
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Strassen 7-product decomposition for 2x2x2
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        # M1 = (a+d)(e+h) -> contributes to C[0,0] and C[1,1]
        # M2 = (c+d)e -> contributes to C[1,0] and C[1,1]
        # M3 = a(f-h) -> contributes to C[0,1] and C[1,1]
        # M4 = d(g-e) -> contributes to C[0,0] and C[1,0]
        # M5 = (a+b)h -> contributes to C[0,0] and C[0,1]
        # M6 = (c-a)(e+f) -> contributes to C[1,1]
        # M7 = (b-d)(g+h) -> contributes to C[0,0]
        
        # u index: i*m + j where i is row of A, j is col of A
        # v index: j*p + k where j is row of B, k is col of B  
        # w index: k*n + i where k is col of B/C, i is row of C
        
        # For a 2x2 block at position t (columns 2t, 2t+1 of B):
        # A indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        # B local indices: 0=(0,2t), 1=(0,2t+1), 2=(1,2t), 3=(1,2t+1)
        # C local indices: 0=(2t,0), 1=(2t+1,0), 2=(2t,1), 3=(2t+1,1)
        
        strassen_terms = [
            # (u_coeffs, v_coeffs, w_coeffs)
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),       # M1 = (a+d)(e+h)
            ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),             # M2 = (c+d)e
            ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),             # M3 = a(f-h)
            ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),             # M4 = d(g-e)
            ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),             # M5 = (a+b)h
            ({0: -1, 2: 1}, {0: 1, 1: 1}, {3: 1}),             # M6 = (c-a)(e+f)
            ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),             # M7 = (b-d)(g+h)
        ]
        
        for t in range(num_blocks):
            for s in range(7):
                col = t * 7 + s
                u_coeffs, v_coeffs, w_coeffs = strassen_terms[s]
                
                for a_idx, val in u_coeffs.items():
                    U[a_idx, col] = val
                
                for local_b, val in v_coeffs.items():
                    if local_b == 0:
                        global_b = 0 * p + (2 * t)
                    elif local_b == 1:
                        global_b = 0 * p + (2 * t + 1)
                    elif local_b == 2:
                        global_b = 1 * p + (2 * t)
                    else:
                        global_b = 1 * p + (2 * t + 1)
                    V[global_b, col] = val
                
                for local_c, val in w_coeffs.items():
                    if local_c == 0:
                        global_w = (2 * t) * n + 0
                    elif local_c == 1:
                        global_w = (2 * t + 1) * n + 0
                    elif local_c == 2:
                        global_w = (2 * t) * n + 1
                    else:
                        global_w = (2 * t + 1) * n + 1
                    W[global_w, col] = val
        
        return U, V, W
    
    # Get initial Strassen decomposition
    U_strassen, V_strassen, W_strassen = strassen_decomposition()
    
    if not verify(U_strassen, V_strassen, W_strassen):
        # Fallback: identity-like decomposition (rank 48)
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    U[u, idx] = 1
                    V[v, idx] = 1
                    W[w, idx] = 1
                    idx += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to optimize down from rank 42
    best_U, best_V, best_W = U_strassen, V_strassen, W_strassen
    best_rank = 42
    
    # Try various target ranks
    for target_rank in [41, 40, 39, 38, 37, 36]:
        if time.time() > deadline - 20:
            break
        
        # Initialize from Strassen by zeroing some columns
        U_init = np.zeros((n*m, target_rank))
        V_init = np.zeros((m*p, target_rank))
        W_init = np.zeros((p*n, target_rank))
        
        # Use first target_rank columns of Strassen
        U_init[:, :target_rank] = U_strassen[:, :target_rank]
        V_init[:, :target_rank] = V_strassen[:, :target_rank]
        W_init[:, :target_rank] = W_strassen[:, :target_rank]
        
        # Add some random perturbation to escape local minima
        np.random.seed(target_rank)
        U_init += 0.1 * np.random.randn(*U_init.shape)
        V_init += 0.1 * np.random.randn(*V_init.shape)
        W_init += 0.1 * np.random.randn(*W_init.shape)
        
        # Flatten parameters
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        def objective(x):
            U_p = x[:n*m*target_rank].reshape(n*m, target_rank)
            V_p = x[n*m*target_rank:n*m*target_rank+m*p*target_rank].reshape(m*p, target_rank)
            W_p = x[n*m*target_rank+m*p*target_rank:].reshape(p*n, target_rank)
            T_recon = np.einsum('ir,jr,kr->ijk', U_p, V_p, W_p)
            return float(np.sum((T_recon - T)**2))
        
        try:
            from scipy.optimize import minimize
            result = minimize(objective, x0, method='L-BFGS-B',
                             options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if result.fun < 1e-6:
                U_new = result.x[:n*m*target_rank].reshape(n*m, target_rank)
                V_new = result.x[n*m*target_rank:n*m*target_rank+m*p*target_rank].reshape(m*p, target_rank)
                W_new = result.x[n*m*target_rank+m*p*target_rank:].reshape(p*n, target_rank)
                
                if verify(U_new, V_new, W_new):
                    best_U, best_V, best_W = U_new, V_new, W_new
                    best_rank = target_rank
                    break  # Found a valid lower rank, stop
        except Exception:
            pass
    
    # Try a few more random restarts if time permits
    for restart in range(5):
        if time.time() > deadline - 10:
            break
        
        target_rank = best_rank - 1
        if target_rank < 30:
            break
        
        np.random.seed(1000 + restart)
        U_init = 0.5 * np.random.randn(n*m, target_rank)
        V_init = 0.5 * np.random.randn(m*p, target_rank)
        W_init = 0.5 * np.random.randn(p*n, target_rank)
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        def objective(x):
            U_p = x[:n*m*target_rank].reshape(n*m, target_rank)
            V_p = x[n*m*target_rank:n*m*target_rank+m*p*target_rank].reshape(m*p, target_rank)
            W_p = x[n*m*target_rank+m*p*target_rank:].reshape(p*n, target_rank)
            T_recon = np.einsum('ir,jr,kr->ijk', U_p, V_p, W_p)
            return float(np.sum((T_recon - T)**2))
        
        try:
            from scipy.optimize import minimize
            result = minimize(objective, x0, method='L-BFGS-B',
                             options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if result.fun < 1e-6:
                U_new = result.x[:n*m*target_rank].reshape(n*m, target_rank)
                V_new = result.x[n*m*target_rank:n*m*target_rank+m*p*target_rank].reshape(m*p, target_rank)
                W_new = result.x[n*m*target_rank+m*p*target_rank:].reshape(p*n, target_rank)
                
                if verify(U_new, V_new, W_new):
                    best_U, best_V, best_W = U_new, V_new, W_new
                    best_rank = target_rank
                    break
        except Exception:
            pass
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
