import numpy as np
from scipy.optimize import minimize
import time

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    R_schoolbook = n * m * p
    U_sb = [[0] * R_schoolbook for _ in range(n * m)]
    V_sb = [[0] * R_schoolbook for _ in range(m * p)]
    W_sb = [[0] * R_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Build the target tensor T as a sparse representation for quick calculation
    # T is a 3D array of size (n*m, m*p, p*n)
    # We don't need to materialize the full 3D array if we compute residual efficiently.
    # Residual = || sum_r u_r v_r^T w_r^T (outer product) - T ||_F
    # Let's represent T as a list of non-zero indices.
    T_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T_indices.append((u_idx, v_idx, w_idx))
    
    start_time = time.time()
    deadline = start_time + 120.0  # 120s limit, keep safety margin
    
    # Try to find a better rank
    # We scan ranks from lower bounds upwards
    # Lower bound approx: max(n*m, m*p, p*n) is not right.
    # Simple bound: n*p + m*p - 1 ? No.
    # Let's try a range. 30 to 71.
    
    best_rank = R_schoolbook
    best_solution = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # We can also try to optimize for a specific rank
    # Let's try a few candidate ranks
    candidate_ranks = [40, 50, 60, 70]
    # We might find a solution at a rank not in this list, but these are good starting points.
    # Actually, scanning is safer but slower.
    
    # To save time, we will only try to optimize if we have enough time.
    # Optimization is expensive.
    
    for R in candidate_ranks:
        if time.time() > deadline - 10:
            break
            
        # Initialize U, V, W with small random values
        # Symmetry breaking: fix first column of U to be standard basis vectors?
        # Let's just use random initialization.
        
        success = False
        # Multiple restarts for this rank
        for restart in range(5):
            if time.time() > deadline - 10:
                break
                
            np.random.seed(42 + restart)
            U0 = np.random.randn(n * m, R) * 0.1
            V0 = np.random.randn(m * p, R) * 0.1
            W0 = np.random.randn(p * n, R) * 0.1
            
            # Flatten
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            def objective(x):
                U = x[:n*m*R].reshape(n*m, R)
                V = x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                W = x[n*m*R + m*p*R : ].reshape(p*n, R)
                
                # Compute residual
                # C_approx = sum_r U[:,r] V[:,r]^T W[:,r]^T ? No.
                # C_approx is a 3-tensor.
                # We only need to check the non-zero entries of T.
                # T is sparse.
                # Residual^2 = sum_{(u,v,w) in T_indices} (sum_r U[u,r] V[v,r] W[w,r] - 1)^2
                #              + sum_{(u,v,w) NOT in T_indices, but in support?} 
                # Wait, T is 0 everywhere else.
                # The full tensor has (n*m)*(m*p)*(p*n) entries.
                # We must ensure all other entries are 0.
                # This is hard to check without materializing.
                # However, if the rank is low, the support of the reconstructed tensor is limited.
                # We can check the non-zero entries of T must be 1.
                # And we can sample some zero entries to ensure they are close to 0.
                
                # Efficient check:
                # 1. Check T_indices
                # 2. Check a random subset of non-T_indices
                
                if len(T_indices) == 0: return 0.0
                
                res = 0.0
                # Check 1s
                for u, v, w in T_indices:
                    val = 0.0
                    for r in range(R):
                        val += U[u, r] * V[v, r] * W[w, r]
                    diff = val - 1.0
                    res += diff * diff
                
                # Check 0s (sample 1000 random indices not in T)
                # Total indices = n*m * m*p * p*n
                total_indices = (n*m) * (m*p) * (p*n)
                # If R is small, most entries will be 0 naturally if the solution is good.
                # But we need to penalize them.
                # Generating random indices that are NOT in T is tricky.
                # Simpler: Just check a few specific "noise" indices.
                # e.g. u=0, v=0, w=0 (if not in T)
                
                # Let's just rely on the 1s for now? No, that will collapse.
                # We need to penalize the entire tensor.
                # We can compute the full tensor for small dimensions?
                # 9 * 24 * 24 = 5184 entries. R is ~40.
                # 5184 * 40 operations = 200k ops. Very fast.
                
                # Let's compute the full residual every time.
                # It's 5184 * R mults.
                # 5184 * 50 = 259,200.
                # This is very fast in Python? No, Python loops are slow.
                # We should vectorize.
                
                # C_approx_flat = np.zeros(total_indices)
                # for r in range(R):
                #    C_approx_flat += np.outer(np.outer(U[:,r], V[:,r]), W[:,r]).flatten()
                # This is O(R * N*M*P).
                # 50 * 5184 = 259k.
                # This is acceptable if done in NumPy.
                
                C_approx = np.zeros(total_indices)
                for r in range(R):
                    # np.outer(u, v) is (n*m, m*p)
                    # np.outer(outer_result, w) is (n*m, m*p, p*n)
                    # Flatten and add.
                    term = np.outer(np.outer(U[:, r], V[:, r]), W[:, r]).flatten()
                    C_approx += term
                
                T_flat = np.zeros(total_indices)
                for u, v, w in T_indices:
                    idx = (u * (m*p) + v) * (p*n) + w
                    T_flat[idx] = 1.0
                
                diff = C_approx - T_flat
                return np.dot(diff, diff)
            
            # Minimize
            try:
                res = minimize(objective, x0, method='BFGS', options={'maxiter': 200, 'disp': False})
                if res.success and res.fun < 1e-6:
                    # Try to round
                    U_opt = res.x[:n*m*R].reshape(n*m, R)
                    V_opt = res.x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                    W_opt = res.x[n*m*R + m*p*R : ].reshape(p*n, R)
                    
                    # Round to nearest 0.5
                    U_round = np.round(U_opt * 2) / 2
                    V_round = np.round(V_opt * 2) / 2
                    W_round = np.round(W_opt * 2) / 2
                    
                    # Verify exactness
                    C_exact = np.zeros(total_indices)
                    for r in range(R):
                        C_exact += np.outer(np.outer(U_round[:, r], V_round[:, r]), W_round[:, r]).flatten()
                    
                    T_flat = np.zeros(total_indices)
                    for u, v, w in T_indices:
                        idx = (u * (m*p) + v) * (p*n) + w
                        T_flat[idx] = 1.0
                        
                    if np.allclose(C_exact, T_flat, atol=1e-9):
                        # Success
                        best_rank = R
                        best_solution = {
                            "U": U_round.tolist(),
                            "V": V_round.tolist(),
                            "W": W_round.tolist()
                        }
                        success = True
                        break # Break restart loop
            except Exception:
                pass
        
        if success:
            break # Break rank loop
            
    if best_rank < R_schoolbook:
        return best_solution
    else:
        return {"U": U_sb, "V": V_sb, "W": W_sb}

# EVOLVE-BLOCK-END
