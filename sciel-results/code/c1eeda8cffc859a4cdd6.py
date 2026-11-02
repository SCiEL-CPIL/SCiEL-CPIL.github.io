"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_fast(U, V, W, n, m, p, tol=1e-8):
    """Fast vectorized verification."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    if not np.allclose(T, 1.0, atol=tol):
        return False
    return True

def construct():
    deadline = time.time() + 118
    
    n, m, p = 4, 5, 6
    U_rows = n * m  # 20
    V_rows = m * p  # 30
    W_rows = p * n  # 24
    
    # Start with schoolbook
    rank_school = n * m * p
    U = np.zeros((U_rows, rank_school))
    V = np.zeros((V_rows, rank_school))
    W = np.zeros((W_rows, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank_school
    
    rng = np.random.RandomState(42)
    
    # Try alternating least squares for various ranks
    # For fixed V, W: solve for U (linear)
    # For fixed U, W: solve for V (linear)  
    # For fixed U, V: solve for W (linear)
    
    # Try ranks from high to low
    for target_rank in range(115, 29, -5):
        if time.time() > deadline - 20:
            break
        
        for trial in range(10):
            if time.time() > deadline - 20:
                break
            
            # Initialize
            U_try = rng.randn(U_rows, target_rank)
            V_try = rng.randn(V_rows, target_rank)
            W_try = rng.randn(W_rows, target_rank)
            
            # Iterate: solve for each matrix in turn
            for it in range(50):
                if time.time() > deadline - 20:
                    break
                
                # Solve for W given U, V
                # For each (i,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all j
                # Build system for W
                A_w = np.zeros((n*m*p, n*p*target_rank))
                b_w = np.ones(n*m*p)
                eq_idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for r in range(target_rank):
                                A_w[eq_idx, (k*n+i)*target_rank + r] = U_try[i*m+j, r] * V_try[j*p+k, r]
                            eq_idx += 1
                try:
                    w_vec, _, _, _ = np.linalg.lstsq(A_w, b_w, rcond=None)
                    W_try = w_vec.reshape(p*n, target_rank)
                except:
                    break
                
                # Solve for U given V, W
                # For each (i,j): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all k
                A_u = np.zeros((n*m*p, n*m*target_rank))
                b_u = np.ones(n*m*p)
                eq_idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for r in range(target_rank):
                                A_u[eq_idx, (i*m+j)*target_rank + r] = V_try[j*p+k, r] * W_try[k*n+i, r]
                            eq_idx += 1
                try:
                    u_vec, _, _, _ = np.linalg.lstsq(A_u, b_u, rcond=None)
                    U_try = u_vec.reshape(n*m, target_rank)
                except:
                    break
                
                # Solve for V given U, W
                # For each (j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i
                A_v = np.zeros((n*m*p, m*p*target_rank))
                b_v = np.ones(n*m*p)
                eq_idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for r in range(target_rank):
                                A_v[eq_idx, (j*p+k)*target_rank + r] = U_try[i*m+j, r] * W_try[k*n+i, r]
                            eq_idx += 1
                try:
                    v_vec, _, _, _ = np.linalg.lstsq(A_v, b_v, rcond=None)
                    V_try = v_vec.reshape(m*p, target_rank)
                except:
                    break
                
                # Check convergence
                if _verify_fast(U_try, V_try, W_try, n, m, p, tol=1e-6):
                    if target_rank < best_rank:
                        best_U, best_V, best_W = U_try.copy(), V_try.copy(), W_try.copy()
                        best_rank = target_rank
                    break  # Found valid, try lower rank
            else:
                continue
            break  # Found valid for this rank
    
    if best_rank < rank_school:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

