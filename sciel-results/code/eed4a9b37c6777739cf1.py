import numpy as np
import time
from scipy.optimize import minimize

def construct():
    n, m, p = 3, 3, 5
    nm, mp, pn = n*m, m*p, p*n
    schoolbook_rank = nm * p  # 45
    
    # 1. Construct Schoolbook Decomposition (Rank 45)
    U_school = np.zeros((nm, schoolbook_rank))
    V_school = np.zeros((mp, schoolbook_rank))
    W_school = np.zeros((pn, schoolbook_rank))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, idx] = 1
                V_school[j*p+k, idx] = 1
                W_school[k*n+i, idx] = 1
                idx += 1
                
    best_U = U_school.tolist()
    best_V = V_school.tolist()
    best_W = W_school.tolist()
    best_rank = schoolbook_rank
    
    start_time = time.time()
    time_limit = 140  # Leave margin for 150s limit
    
    def verify(U, V, W, R):
        # Check if U, V, W of rank R satisfy the tensor equation
        # And if entries are integer/half-integer
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(R):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > 1e-6:
                        return False
        # Check integer/half-integer constraint
        for matrix in [U, V, W]:
            for val in matrix:
                if abs(val * 2 - round(val * 2)) > 1e-6:
                    return False
        return True

    def get_residuals(x, R):
        U = x[:nm*R].reshape(nm, R)
        V = x[nm*R:nm*R+mp*R].reshape(mp, R)
        W = x[nm*R+mp*R:].reshape(pn, R)
        
        res = np.zeros(n*m*p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    res[idx] = val - 1.0
                    idx += 1
        return res

    def objective(x, R):
        res = get_residuals(x, R)
        res_norm = np.sum(res**2)
        
        # Penalty for not being half-integers
        # We want x to be close to 0.5 * integers
        # penalty = sum((2*x - round(2*x))^2)
        # This is non-smooth, but for optimization it acts as a pull towards the grid.
        # To keep it smooth, we can use a soft approximation or just rely on the residual
        # being small and then check the grid.
        # However, adding a small penalty helps guide the optimizer towards discrete solutions.
        # Let's use a smooth approximation: sum(sin(2*pi*x)^2) is 0 at integers, but we want half-integers.
        # sin(2*pi*(x - 0.5))^2 = cos(2*pi*x)^2.
        # This penalizes values far from half-integers.
        penalty = np.sum(np.cos(2 * np.pi * x)**2)
        
        return res_norm + 0.1 * penalty

    # Try ranks 44 down to 40
    for target_rank in range(schoolbook_rank - 1, 39, -1):
        if time.time() - start_time > time_limit:
            break
            
        # We try to remove one column from the schoolbook solution and optimize the rest.
        # This is a heuristic: if a rank R solution exists, it might be a "subspace" 
        # or close to a perturbation of a subset of the schoolbook basis.
        
        # We will try removing each of the 45 columns one by one, 
        # and also try random removals.
        
        # Optimization parameters
        # We use BFGS for local convergence.
        
        found = False
        for restart in range(10): # 10 restarts per rank
            if time.time() - start_time > time_limit:
                break
            if found:
                break
                
            # Initialize
            if restart < schoolbook_rank:
                # Remove column 'restart'
                mask = np.ones(schoolbook_rank, dtype=bool)
                mask[restart] = False
                U0 = U_school[:, mask]
                V0 = V_school[:, mask]
                W0 = W_school[:, mask]
            else:
                # Random permutation removal
                rng = np.random.default_rng(restart)
                perm = rng.permutation(schoolbook_rank)
                mask = np.ones(schoolbook_rank, dtype=bool)
                mask[perm[0]] = False
                U0 = U_school[:, mask]
                V0 = V_school[:, mask]
                W0 = W_school[:, mask]
                
            # Add small noise to escape exact zero-gradient points if any
            rng = np.random.default_rng(restart + 100)
            U0 = U0 + rng.normal(0, 0.01, U0.shape)
            V0 = V0 + rng.normal(0, 0.01, V0.shape)
            W0 = W0 + rng.normal(0, 0.01, W0.shape)
            
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            try:
                res = minimize(objective, x0, args=(target_rank,), method='BFGS', 
                               options={'maxiter': 200, 'gtol': 1e-5})
                
                # Check residual
                final_res = np.sum(get_residuals(res.x, target_rank)**2)
                
                if final_res < 1e-4:
                    U_sol = res.x[:nm*target_rank].reshape(nm, target_rank)
                    V_sol = res.x[nm*target_rank:nm*target_rank+mp*target_rank].reshape(mp, target_rank)
                    W_sol = res.x[nm*target_rank+mp*target_rank:].reshape(pn, target_rank)
                    
                    # Round to half-integers
                    U_rounded = np.round(U_sol * 2) / 2
                    V_rounded = np.round(V_sol * 2) / 2
                    W_rounded = np.round(W_sol * 2) / 2
                    
                    if verify(U_rounded, V_rounded, W_rounded, target_rank):
                        best_U = U_rounded.tolist()
                        best_V = V_rounded.tolist()
                        best_W = W_rounded.tolist()
                        best_rank = target_rank
                        found = True
                        break
                        
            except Exception:
                pass

    return {"U": best_U, "V": best_V, "W": best_W}
