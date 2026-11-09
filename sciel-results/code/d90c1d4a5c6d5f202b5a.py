import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook baseline
    rank_sb = n * m * p  # 54
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
                
    # Build the multiplication tensor T
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i) for some i,j,k
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Pre-compute some useful matrices
    T_by_c = np.zeros((pn, nm, mp))
    for c in range(pn):
        T_by_c[c] = T[:, :, c]
    
    T_by_a = np.zeros((nm, mp, pn))
    for a in range(nm):
        T_by_a[a] = T[a, :, :]
    
    T_by_b = np.zeros((mp, nm, pn))
    for b in range(mp):
        T_by_b[b] = T[:, b, :]
    
    start_time = time.time()
    time_limit = 140  # Leave 10s margin for 150s limit
    
    best_rank = rank_sb
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    
    # Lower bound for rank
    # Trivial lower bound: max(n*m, m*p, p*n) / something?
    # Actually, for matrix multiplication M_{n x m} * M_{m x p}, the rank is at least max(n*m, m*p, p*n) is false.
    # The rank is at least max(n, m, p) * something?
    # Let's use a reasonable lower bound.
    # For 2x3x9, schoolbook is 54.
    # Strassen for 2x2x2 is 7 vs 8.
    # For 2x3x9, we can try to beat 54.
    # Lower bound: max(n, m, p) * min(n, m, p) = 9 * 2 = 18?
    # Let's try down to 30 to be safe and efficient.
    
    lower_bound = 30
    
    # Try ranks from rank_sb-1 down to lower_bound
    for rank in range(rank_sb - 1, lower_bound - 1, -1):
        if time.time() - start_time > time_limit - 20:
            break
        
        found = False
        # Try multiple random initializations with different strategies
        max_restarts = 10
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit - 20:
                break
            
            np.random.seed(42 + restart * 137 + rank * 7)
            
            # Different initialization strategies
            if restart % 3 == 0:
                # Random Gaussian
                U = np.random.randn(nm, rank)
                V = np.random.randn(mp, rank)
                W = np.random.randn(pn, rank)
            elif restart % 3 == 1:
                # Sparse random - many zeros
                U = np.random.choice([0, 1], (nm, rank), p=[0.7, 0.3]).astype(float)
                V = np.random.choice([0, 1], (mp, rank), p=[0.7, 0.3]).astype(float)
                W = np.random.choice([0, 1], (pn, rank), p=[0.7, 0.3]).astype(float)
            else:
                # Mix of random and structured
                U = np.random.randn(nm, rank)
                V = np.random.randn(mp, rank)
                W = np.random.randn(pn, rank)
                # Add some structure based on the tensor
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            r = (i * p + j) % rank
                            U[a, r] += 1
                            V[b, r] += 1
                            W[c, r] += 1
            
            # Scale to reasonable magnitude
            U /= (np.linalg.norm(U) + 1e-10)
            V /= (np.linalg.norm(V) + 1e-10)
            W /= (np.linalg.norm(W) + 1e-10)
            
            # Alternating least squares with adaptive iterations
            max_iter = 300
            prev_error = float('inf')
            for iteration in range(max_iter):
                if time.time() - start_time > time_limit - 20:
                    break
                
                # Solve for U given V, W
                P_flat = np.zeros((rank, mp * pn))
                for r in range(rank):
                    P_flat[r, :] = np.outer(V[:, r], W[:, r]).reshape(-1)
                
                T_flat_a = T_by_a.reshape(nm, mp * pn)
                
                try:
                    U_new, _, _, _ = np.linalg.lstsq(P_flat.T, T_flat_a.T, rcond=None)
                    U = U_new.T
                except:
                    break
                
                # Solve for V given U, W
                P_flat_v = np.zeros((rank, nm * pn))
                for r in range(rank):
                    P_flat_v[r, :] = np.outer(U[:, r], W[:, r]).reshape(-1)
                
                T_flat_b = T_by_b.reshape(mp, nm * pn)
                
                try:
                    V_new, _, _, _ = np.linalg.lstsq(P_flat_v.T, T_flat_b.T, rcond=None)
                    V = V_new.T
                except:
                    break
                
                # Solve for W given U, V
                P_flat_w = np.zeros((rank, nm * mp))
                for r in range(rank):
                    P_flat_w[r, :] = np.outer(U[:, r], V[:, r]).reshape(-1)
                
                T_flat_c = T_by_c.reshape(pn, nm * mp)
                
                try:
                    W_new, _, _, _ = np.linalg.lstsq(P_flat_w.T, T_flat_c.T, rcond=None)
                    W = W_new.T
                except:
                    break
                
                # Check error
                if iteration % 5 == 0 or iteration == max_iter - 1:
                    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                    error = np.linalg.norm(T_recon - T)
                    if error < 1e-8:
                        found = True
                        break
                    # Early termination if not making progress
                    if error > prev_error * 0.999 and iteration > 50:
                        break
                    prev_error = error
            
            if found:
                best_rank = rank
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                break
        
        if found:
            continue
    
    if best_rank < rank_sb:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
