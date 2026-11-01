import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 120s limit, keep margin
    
    # Schoolbook solution as fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = rank_school

    dim_u = n * m  # 6
    dim_v = m * p  # 21
    dim_w = p * n  # 14
    
    # Precompute the "valid" (u, v) pairs and their corresponding w
    valid_pairs = {}  # (u, v) -> w
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                valid_pairs[(u, v)] = w
    
    # Precompute b_w for all w
    b_w = np.zeros((dim_w, dim_u * dim_v))
    for (u, v), w in valid_pairs.items():
        b_w[w, u * dim_v + v] = 1.0
    
    def build_A(U, V, r):
        """Build A matrix where A[(u * dim_v + v), t] = U[u, t] * V[v, t]"""
        A = np.zeros((dim_u * dim_v, r))
        for t in range(r):
            A[:, t] = np.kron(U[:, t], V[:, t])
        return A
    
    def check_exact(U, V, W, r, atol=1e-9):
        """Check if the decomposition is exact"""
        for w in range(dim_w):
            T_recon_w = (U * W[w, :][None, :]) @ V.T
            T_recon_w_flat = T_recon_w.flatten()
            if not np.allclose(T_recon_w_flat, b_w[w], atol=atol):
                return False
        return True
    
    def try_solve(U, V, r, use_lstsq=True):
        """Try to solve for W given U and V. Returns W if exact solution found, else None."""
        A = build_A(U, V, r)
        
        W = np.zeros((dim_w, r))
        for w in range(dim_w):
            try:
                if use_lstsq:
                    W[w, :], residuals, rank_sol, s = np.linalg.lstsq(A, b_w[w], rcond=None)
                else:
                    # Try exact solve
                    W[w, :] = np.linalg.solve(A, b_w[w])
            except np.linalg.LinAlgError:
                return None
            except Exception:
                return None
        
        if check_exact(U, V, W, r):
            return W
        return None
    
    def structured_init_U(r, pattern='block'):
        """Generate structured U initialization exploiting block structure"""
        U = np.zeros((dim_u, r))
        if pattern == 'block':
            # Treat rows as (i, j) where i in 0..n-1, j in 0..m-1
            # For each row, assign a subset of the r dimensions
            for i in range(n):
                for j in range(m):
                    u = i * m + j
                    # Assign a pseudo-random but structured subset
                    # Use a hash-like pattern based on i, j
                    mask = np.zeros(r)
                    for t in range(r):
                        # Structured: use (i * m + j + t) % r < r // 2 + 1
                        if (i * m + j + t) % max(1, r // 2 + 1) < r // 2 + 1:
                            mask[t] = 1
                    U[u, :] = mask
        elif pattern == 'random':
            U = np.random.randint(0, 2, size=(dim_u, r))
        return U
    
    def structured_init_V(r, pattern='block'):
        """Generate structured V initialization exploiting block structure"""
        V = np.zeros((dim_v, r))
        if pattern == 'block':
            # Treat rows as (j, k) where j in 0..m-1, k in 0..p-1
            for j in range(m):
                for k in range(p):
                    v = j * p + k
                    mask = np.zeros(r)
                    for t in range(r):
                        if (j * p + k + t) % max(1, r // 2 + 1) < r // 2 + 1:
                            mask[t] = 1
                    V[v, :] = mask
        elif pattern == 'random':
            V = np.random.randint(0, 2, size=(dim_v, r))
        return V
    
    # Search for lower ranks
    # Try ranks from 21 up to 41
    for r in range(21, best_rank):
        if time.time() > deadline - 10:
            break
            
        # Try different initialization strategies
        patterns = ['block', 'random', 'block', 'random']
        
        for pi, pattern in enumerate(patterns):
            if time.time() > deadline - 10:
                break
            
            # Number of attempts per pattern
            n_attempts = 50 if pattern == 'block' else 80
            
            for attempt in range(n_attempts):
                if time.time() > deadline - 10:
                    break
                
                if pattern == 'block':
                    # Structured initialization with some randomization
                    U = structured_init_U(r, 'block')
                    V = structured_init_V(r, 'block')
                    # Add some random perturbation
                    if attempt > 0:
                        flip_U = np.random.random((dim_u, r)) < 0.3
                        flip_V = np.random.random((dim_v, r)) < 0.3
                        U[flip_U] = 1 - U[flip_U]
                        V[flip_V] = 1 - V[flip_V]
                else:
                    U = np.random.randint(0, 2, size=(dim_u, r))
                    V = np.random.randint(0, 2, size=(dim_v, r))
                
                # Try to solve
                W = try_solve(U, V, r, use_lstsq=True)
                
                if W is not None:
                    # Verify with tighter tolerance
                    if check_exact(U, V, W, r, atol=1e-8):
                        best_rank = r
                        best_result = {
                            "U": U.tolist(),
                            "V": V.tolist(),
                            "W": W.tolist()
                        }
                        break  # Stop searching for this rank
            
            if best_rank < rank_school:
                break
        
        if best_rank < rank_school:
            break
    
    return best_result
