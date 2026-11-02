import numpy as np
import time

DIMENSIONS = [2, 2, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm = n * m  # 4
    mp = m * p  # 12
    pn = p * n  # 12
    
    # Build the target tensor: T[a,b,c] = 1 if (a,b,c) corresponds to valid (i,j,k)
    # a = i*m + j, b = j*p + k, c = k*n + i
    target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0
    
    # Try ranks from low to high
    for rank in range(12, 25):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random restarts
        max_restarts = 200 if rank <= 15 else 500
        best_residual = np.inf
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            # Scale to help convergence
            scale = 1.0 / np.sqrt(rank)
            U *= scale
            V *= scale
            W *= scale
            
            # Use iterative optimization: minimize sum of squared residuals
            # Residual for entry (a,b,c): sum_r U[a,r]*V[b,r]*W[c,r] - target[a,b,c]
            
            # Use a simple gradient-free approach: alternating minimization
            # For fixed V, W, solve for U (linear in each row of U)
            # For fixed U, W, solve for V
            # For fixed U, V, solve for W
            
            converged = False
            for iteration in range(500):
                if time.time() > deadline - 3:
                    break
                
                # Update U: for each a, U[a,:] should satisfy
                # sum_r U[a,r] * V[b,r] * W[c,r] = target[a,b,c] for all (b,c)
                # This is: U[a,:] @ (V[:,b] * W[:,c]) = target[a,b,c] for all b,c
                # Let M[a,b,c] = V[:,b] * W[:,c], then U[a,:] @ M[a,b,c] = target[a,b,c]
                # For fixed a: U[a,:] is a row vector, and we need it to satisfy
                # U[a,r] * sum over (b,c) pairs... 
                # Actually, for fixed a, the equations are:
                # For each (b,c): sum_r U[a,r] * V[b,r] * W[c,r] = target[a,b,c]
                # This is a linear system in U[a,0],...,U[a,rank-1]
                # Coefficient matrix: row (b,c) has entries V[b,r]*W[c,r] for r=0..rank-1
                # Number of equations: mp * pn = 144, unknowns: rank
                # So we have an overdetermined system (144 > rank typically)
                
                # Build the coefficient matrix for U
                # A_U[a] is (mp*pn) x rank matrix where A_U[a][(b,c), r] = V[b,r] * W[c,r]
                # b_target[a] is (mp*pn) vector where b_target[a][(b,c)] = target[a,b,c]
                
                # Similarly for V and W
                
                # For efficiency, precompute
                # A_U[a][(b,c), r] = V[b,r] * W[c,r]
                # We can write this as: for each a, A_U[a] = kron-like structure
                
                # Let's do it explicitly but efficiently
                # V is (mp, rank), W is (pn, rank)
                # V[b,r] * W[c,r] for all b,c,r -> shape (mp, pn, rank)
                # Reshape to (mp*pn, rank)
                
                VW = (V[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(mp * pn, rank)
                # VW[(b*pn + c), r] = V[b,r] * W[c,r]
                
                # target reshaped: target[a, b, c] -> target_flat[a, b*pn+c]
                target_flat = target.reshape(nm, mp * pn)
                
                # For each a, solve: VW @ U[a,:] = target_flat[a,:]
                # Least squares: U[a,:] = pinv(VW) @ target_flat[a,:]
                # But VW is the same for all a! So:
                # U = (target_flat @ pinv(VW.T)).T ... let me think
                # U[a,:] satisfies VW @ U[a,:] = target_flat[a,:]
                # So U.T satisfies VW.T @ U.T = target_flat.T
                # U.T = pinv(VW.T) @ target_flat.T
                # U = (pinv(VW.T) @ target_flat.T).T
                
                # Use lstsq for numerical stability
                try:
                    U_new, _, _, _ = np.linalg.lstsq(VW, target_flat.T, rcond=None)
                    U = U_new.T
                except:
                    pass
                
                # Update V: for each b, V[b,r] * sum_c... 
                # For fixed b: sum_r U[a,r] * V[b,r] * W[c,r] = target[a,b,c] for all a,c
                # Coefficient: U[a,r] * W[c,r] for all a,c -> (nm*pn, rank)
                UW = (U[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(nm * pn, rank)
                target_flat_v = target.transpose(1, 0, 2).reshape(mp, nm * pn)  # target[b, a, c]
                try:
                    V_new, _, _, _ = np.linalg.lstsq(UW, target_flat_v.T, rcond=None)
                    V = V_new.T
                except:
                    pass
                
                # Update W: for each c, W[c,r] * sum_a...
                # For fixed c: sum_r U[a,r] * V[b,r] * W[c,r] = target[a,b,c] for all a,b
                # Coefficient: U[a,r] * V[b,r] for all a,b -> (nm*mp, rank)
                UV = (U[:, np.newaxis, :] * V[np.newaxis, :, :]).reshape(nm * mp, rank)
                target_flat_w = target.transpose(2, 0, 1).reshape(pn, nm * mp)  # target[c, a, b]
                try:
                    W_new, _, _, _ = np.linalg.lstsq(UV, target_flat_w.T, rcond=None)
                    W = W_new.T
                except:
                    pass
                
                # Check convergence
                # Compute residual
                residual = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            val = np.dot(U[a], V[b] * W[c])
                            residual += (val - 1.0) ** 2
                
                if residual < 1e-10:
                    converged = True
                    break
                
                if residual < best_residual:
                    best_residual = residual
            
            if converged:
                # Verify all entries
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            val = np.dot(U[a], V[b] * W[c])
                            if abs(val - 1.0) > 1e-8:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    # Check invalid entries
                    for a in range(nm):
                        for b in range(mp):
                            for c in range(pn):
                                # Check if (a,b,c) is a valid entry
                                i = a // m
                                j = a % m
                                j2 = b // p
                                k = b % p
                                k2 = c // n
                                i2 = c % n
                                if not (j == j2 and i == i2):
                                    val = np.dot(U[a], V[b] * W[c])
                                    if abs(val) > 1e-8:
                                        valid = False
                                        break
                            if not valid:
                                break
                        if not valid:
                            break
                    
                    if valid:
                        # Convert to list of lists
                        return {
                            "U": U.tolist(),
                            "V": V.tolist(),
                            "W": W.tolist()
                        }
        
        # If we didn't converge for this rank, try next
        print(f"Rank {rank}: best residual {best_residual}")
    
    # Fallback: schoolbook rank 24
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
    return {"U": U, "V": V, "W": W}
