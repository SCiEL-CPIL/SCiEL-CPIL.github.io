"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 2x5 matrix by a 5x5 matrix.
    
    Strategy: The 2x5x5 multiplication tensor is a direct sum of two
    1x5x5 multiplication tensors. We first try to find a low-rank
    decomposition for 5x5 matrix multiplication, then apply it to
    both blocks.
    
    Fallback: schoolbook decomposition with rank 50.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # n=2, m=5, p=5
    
    # Build the target tensor T
    # T[a, b, c] where a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Schoolbook fallback
    def make_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_result = make_schoolbook()
    best_rank = n * m * p
    
    # Try to find a low-rank decomposition for 5x5 matrix multiplication
    # The 1x5x5 tensor has entries T[i, j, k] = 1 where i indexes row of A,
    # j indexes (col of A, row of B), k indexes (col of B, row of C)
    # For 1x5x5: T[a, b, c] where a=0*5+j=j, b=j*5+k, c=k*1+0=k
    # So T[j, j*5+k, k] = 1
    
    def try_5x5_decomposition(target_rank, max_trials=20):
        """Try to find a rank-r decomposition for 5x5 matrix multiplication."""
        if time.time() > deadline - 15:
            return None
        
        # Build the 1x5x5 tensor
        T_5x5 = np.zeros((5, 25, 5))
        for j in range(5):
            for k in range(5):
                T_5x5[j, j*5 + k, k] = 1.0
        
        # Try random initialization with gradient descent
        def objective(x):
            U = x[:5*r].reshape(5, r)
            V = x[5*r : 5*r + 25*r].reshape(25, r)
            W = x[5*r + 25*r :].reshape(5, r)
            Rec = np.einsum('ik,jk,lk->ijl', U, V, W)
            Res = T_5x5 - Rec
            return np.sum(Res * Res)
        
        def gradient(x):
            U = x[:5*r].reshape(5, r)
            V = x[5*r : 5*r + 25*r].reshape(25, r)
            W = x[5*r + 25*r :].reshape(5, r)
            Rec = np.einsum('ik,jk,lk->ijl', U, V, W)
            Res = T_5x5 - Rec
            Grad_U = np.einsum('ijl,jc,lc->ic', Res, V, W)
            Grad_V = np.einsum('ijl,ic,lc->jc', Res, U, W)
            Grad_W = np.einsum('ijl,ic,jc->lc', Res, U, V)
            return np.concatenate([Grad_U.flatten(), Grad_V.flatten(), Grad_W.flatten()])
        
        r = target_rank
        for trial in range(max_trials):
            if time.time() > deadline - 10:
                break
                
            np.random.seed(42 + r * 100 + trial)
            U0 = np.random.randn(5, r) * 0.3
            V0 = np.random.randn(25, r) * 0.3
            W0 = np.random.randn(5, r) * 0.3
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            # Use Nelder-Mead for small problems, or BFGS
            from scipy.optimize import minimize
            try:
                res = minimize(objective, x0, jac=gradient, method='L-BFGS-B', 
                              options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-12})
                
                if res.fun < 1e-10:
                    U = res.x[:5*r].reshape(5, r)
                    V = res.x[5*r : 5*r + 25*r].reshape(25, r)
                    W = res.x[5*r + 25*r :].reshape(5, r)
                    
                    # Verify
                    Rec = np.einsum('ik,jk,lk->ijl', U, V, W)
                    if np.max(np.abs(Rec - T_5x5)) < 1e-8:
                        return {"U": U, "V": V, "W": W}
            except Exception:
                pass
        
        return None
    
    # Try to find a 5x5 decomposition with rank < 25
    # Known results suggest rank 22 might be achievable
    found_5x5 = None
    for target_rank in [22, 23, 24]:
        if time.time() > deadline - 20:
            break
        
        result = try_5x5_decomposition(target_rank)
        if result is not None:
            found_5x5 = result
            print(f"Found 5x5 decomposition with rank {target_rank}")
            break
    
    if found_5x5 is not None:
        # Use the 5x5 decomposition for both blocks of the 2x5x5 problem
        U_5 = found_5x5["U"]  # shape (5, r)
        V_5 = found_5x5["V"]  # shape (25, r)
        W_5 = found_5x5["W"]  # shape (5, r)
        r_5 = U_5.shape[1]
        
        # For 2x5x5: 
        # Block 1 (i=0): A[0,j] * B[j,k] -> C[0,k]
        # Block 2 (i=1): A[1,j] * B[j,k] -> C[1,k]
        # U has shape (10, 2*r_5): first 5 rows for block 1, next 5 for block 2
        # V has shape (25, 2*r_5): same V for both blocks (since B is shared)
        # W has shape (10, 2*r_5): first 5 rows for block 1 output, next 5 for block 2
        
        U = np.zeros((10, 2 * r_5))
        V = np.zeros((25, 2 * r_5))
        W = np.zeros((10, 2 * r_5))
        
        # Block 1: i=0
        U[0:5, 0:r_5] = U_5
        V[0:25, 0:r_5] = V_5
        W[0:5, 0:r_5] = W_5
        
        # Block 2: i=1
        U[5:10, r_5:2*r_5] = U_5
        V[0:25, r_5:2*r_5] = V_5
        W[5:10, r_5:2*r_5] = W_5
        
        # Verify the full decomposition
        Rec = np.einsum('ik,jk,lk->ijl', U, V, W)
        if np.max(np.abs(Rec - T)) < 1e-8:
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            best_rank = 2 * r_5
    
    return best_result
# EVOLVE-BLOCK-END
