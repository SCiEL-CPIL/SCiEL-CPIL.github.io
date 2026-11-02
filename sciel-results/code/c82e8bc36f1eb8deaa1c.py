"""
Rewritten program for Matrix Multiplication Tensor Decomposition (3,3,6).
Strategy: Systematic search for lower-rank decompositions using structured
initialization and local optimization.
"""
import numpy as np
import time
from fractions import Fraction
from scipy.optimize import minimize
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s limit
    
    n, m, p = 3, 3, 6
    
    # Dimensions
    dim_u = n * m  # 9
    dim_v = m * p  # 18
    dim_w = p * n  # 18
    
    # Construct the target tensor T of shape (n*m, m*p, p*n)
    T_target = np.zeros((dim_u, dim_v, dim_w))
    
    # Schoolbook indices
    schoolbook_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
                schoolbook_indices.append((a, b, c))
                
    rank_sb = len(schoolbook_indices)  # 54

    def verify(U, V, W, tol=1e-8):
        try:
            T_recon = np.einsum('at,bt,ct->abc', U, V, W)
            return np.allclose(T_recon, T_target, atol=tol)
        except Exception:
            return False

    def objective_and_grad(x, target_rank):
        U = x[:dim_u*target_rank].reshape(dim_u, target_rank)
        V = x[dim_u*target_rank : dim_u*target_rank + dim_v*target_rank].reshape(dim_v, target_rank)
        W = x[dim_u*target_rank + dim_v*target_rank : ].reshape(dim_w, target_rank)
        
        T_recon = np.einsum('at,bt,ct->abc', U, V, W)
        error = T_target - T_recon
        loss = np.sum(error**2)
        
        dU = 2 * np.einsum('abc,bt,ct->at', error, V, W)
        dV = 2 * np.einsum('abc,at,ct->bt', error, U, W)
        dW = 2 * np.einsum('abc,at,bt->ct', error, U, V)
        
        grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
        return loss, grad

    def try_rank(target_rank, num_trials=30, seed_base=42):
        """
        Attempts to find a decomposition of target_rank by starting from various
        initializations and optimizing.
        """
        best_solution = None
        
        for trial in range(num_trials):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(seed_base + trial * 13)
            
            # Strategy 1: Start from schoolbook, randomly keep target_rank terms
            if trial < num_trials // 2:
                # Select indices to keep
                keep_indices = np.random.choice(rank_sb, target_rank, replace=False)
                
                U = np.zeros((dim_u, target_rank))
                V = np.zeros((dim_v, target_rank))
                W = np.zeros((dim_w, target_rank))
                
                for new_idx, old_idx in enumerate(keep_indices):
                    a, b, c = schoolbook_indices[old_idx]
                    U[a, new_idx] = 1.0
                    V[b, new_idx] = 1.0
                    W[c, new_idx] = 1.0
            else:
                # Strategy 2: Random initialization with small perturbations
                U = np.random.randn(dim_u, target_rank) * 0.1
                V = np.random.randn(dim_v, target_rank) * 0.1
                W = np.random.randn(dim_w, target_rank) * 0.1
                
                # Add some structure: make each column non-zero in all three matrices
                for t in range(target_rank):
                    a = np.random.randint(0, dim_u)
                    b = np.random.randint(0, dim_v)
                    c = np.random.randint(0, dim_w)
                    U[a, t] = 1.0
                    V[b, t] = 1.0
                    W[c, t] = 1.0
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                # Use L-BFGS-B for large-scale optimization
                res = minimize(
                    lambda x: objective_and_grad(x, target_rank)[0], 
                    x0, 
                    jac=lambda x: objective_and_grad(x, target_rank)[1],
                    method='L-BFGS-B', 
                    options={'maxiter': 50, 'gtol': 1e-10, 'maxfun': 200}
                )
            except Exception:
                continue
            
            if time.time() > deadline - 2:
                break
                
            U_final = res.x[:dim_u*target_rank].reshape(dim_u, target_rank)
            V_final = res.x[dim_u*target_rank : dim_u*target_rank + dim_v*target_rank].reshape(dim_v, target_rank)
            W_final = res.x[dim_u*target_rank + dim_v*target_rank : ].reshape(dim_w, target_rank)
            
            # Check convergence
            if verify(U_final, V_final, W_final, tol=1e-7):
                # Try to snap to simple rational numbers
                for denom in [1, 2, 4, 8]:
                    scale = float(denom)
                    U_int = np.round(U_final * scale) / scale
                    V_int = np.round(V_final * scale) / scale
                    W_int = np.round(W_final * scale) / scale
                    
                    if verify(U_int, V_int, W_int, tol=1e-6):
                        return U_int, V_int, W_int, target_rank
                
                # Return the converged solution
                return U_final, V_final, W_final, target_rank

        return None

    # Fallback schoolbook decomposition
    def schoolbook():
        rank = n * m * p
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1.0
                    V[b, col] = 1.0
                    W[c, col] = 1.0
                    col += 1
        return U, V, W, rank

    U_sb, V_sb, W_sb, r_sb = schoolbook()
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, r_sb

    # Try to find lower-rank decompositions
    # Known results suggest rank 46 might be achievable (2 * 23)
    # Try ranks from 46 up to 53
    
    for target_rank in [46, 47, 48, 49, 50, 51, 52, 53]:
        if time.time() > deadline - 10:
            break
        
        res = try_rank(target_rank, num_trials=15)
        if res is not None:
            best_U, best_V, best_W, best_rank = res
            break  # Found a valid lower-rank solution
    
    # If still at 54, do a few more trials with different strategies
    if best_rank == 54 and time.time() < deadline - 5:
        # Try with more aggressive random initialization
        for target_rank in [46, 48, 50]:
            if time.time() > deadline - 2:
                break
            res = try_rank(target_rank, num_trials=5, seed_base=1000)
            if res is not None:
                best_U, best_V, best_W, best_rank = res
                break

    # Convert to lists with rational approximation
    def to_list(mat):
        res = []
        for row in mat:
            r = []
            for v in row:
                if not np.isfinite(v):
                    r.append(0.0)
                    continue
                try:
                    # Try to find a simple fraction
                    f = Fraction(v).limit_denominator(100)
                    if abs(float(f) - v) < 1e-8:
                        if f.denominator == 1:
                            r.append(int(f.numerator))
                        else:
                            r.append(f.numerator / f.denominator)
                    else:
                        r.append(float(v))
                except (ValueError, OverflowError):
                    r.append(float(v))
            res.append(r)
        return res

    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
