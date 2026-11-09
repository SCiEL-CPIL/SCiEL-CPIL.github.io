"""Program for rank-minimal exact tensor decomposition of 3x3 * 3x8 matrix multiplication."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def _validate(U, V, W, n, m, p, tol=1e-9):
    """Check that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] should equal 1 for all i,j,k
    T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    if not np.allclose(T, np.ones((n, m, p)), atol=tol):
        return False
    return True

def _round_to_half_int(x):
    """Round to nearest half-integer."""
    return np.round(x * 2) / 2

def _solve_for_W(U, V, n, m, p):
    """Given U and V, solve for W using least squares."""
    R = U.shape[1]
    W = np.zeros((p * n, R))
    for i in range(n):
        for k in range(p):
            A = np.zeros((m, R))
            b = np.ones(m)
            for j in range(m):
                A[j, :] = U[i * m + j, :] * V[j * p + k, :]
            w, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            W[k * n + i, :] = w
    return W

def _solve_for_V(U, W, n, m, p):
    """Given U and W, solve for V using least squares."""
    R = U.shape[1]
    V = np.zeros((m * p, R))
    for j in range(m):
        for k in range(p):
            A = np.zeros((n, R))
            b = np.ones(n)
            for i in range(n):
                A[i, :] = U[i * m + j, :] * W[k * n + i, :]
            v, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            V[j * p + k, :] = v
    return V

def _solve_for_U(V, W, n, m, p):
    """Given V and W, solve for U using least squares."""
    R = V.shape[1]
    U = np.zeros((n * m, R))
    for i in range(n):
        for j in range(m):
            A = np.zeros((p, R))
            b = np.ones(p)
            for k in range(p):
                A[k, :] = V[j * p + k, :] * W[k * n + i, :]
            u, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            U[i * m + j, :] = u
    return U

def _alternating_minimization(U, V, W, n, m, p, max_iter=100, tol=1e-10):
    """Run alternating minimization to refine the decomposition."""
    for _ in range(max_iter):
        W_new = _solve_for_W(U, V, n, m, p)
        V_new = _solve_for_V(U, W_new, n, m, p)
        U_new = _solve_for_V(V_new, W_new, n, m, p)  # This is wrong, should be _solve_for_U
        U_new = _solve_for_U(V_new, W_new, n, m, p)
        
        # Check convergence
        if (np.max(np.abs(U_new - U)) < tol and 
            np.max(np.abs(V_new - V)) < tol and 
            np.max(np.abs(W_new - W)) < tol):
            break
        
        U, V, W = U_new, V_new, W_new
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin before 150s limit
    
    n, m, p = 3, 3, 8
    
    # Start with schoolbook decomposition
    rank = n * m * p  # 72
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
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Phase 1: Try to reduce rank by removing columns one at a time
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for r in range(rank):
            if time.time() > deadline:
                break
            
            mask = np.ones(rank, dtype=bool)
            mask[r] = False
            U_trial = U[:, mask]
            V_trial = V[:, mask]
            W_trial = W[:, mask]
            
            # Refine with alternating minimization
            U_trial, V_trial, W_trial = _alternating_minimization(
                U_trial, V_trial, W_trial, n, m, p, max_iter=50
            )
            
            # Round to half-integers
            U_rounded = _round_to_half_int(U_trial)
            V_rounded = _round_to_half_int(V_trial)
            W_rounded = _round_to_half_int(W_trial)
            
            if _validate(U_rounded, V_rounded, W_rounded, n, m, p):
                U, V, W = U_rounded, V_rounded, W_rounded
                rank -= 1
                best_rank = rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                improved = True
                break
    
    # Phase 2: Try lower ranks with random restarts
    if time.time() < deadline:
        for target_rank in range(max(20, rank - 20), best_rank):
            if time.time() > deadline:
                break
            
            found = False
            for restart in range(20):
                if time.time() > deadline:
                    break
                
                np.random.seed(restart * 7 + target_rank)
                
                # Initialize with small random values
                U_trial = np.random.randn(n * m, target_rank) * 0.5
                V_trial = np.random.randn(m * p, target_rank) * 0.5
                W_trial = np.random.randn(p * n, target_rank) * 0.5
                
                # Run alternating minimization
                for iter_num in range(200):
                    if time.time() > deadline:
                        break
                    
                    W_trial = _solve_for_W(U_trial, V_trial, n, m, p)
                    V_trial = _solve_for_V(U_trial, W_trial, n, m, p)
                    U_trial = _solve_for_U(V_trial, W_trial, n, m, p)
                    
                    # Round to half-integers and check
                    U_rounded = _round_to_half_int(U_trial)
                    V_rounded = _round_to_half_int(V_trial)
                    W_rounded = _round_to_half_int(W_trial)
                    
                    if _validate(U_rounded, V_rounded, W_rounded, n, m, p):
                        best_rank = target_rank
                        best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                        found = True
                        break
                
                if found:
                    break
            
            if found:
                break
    
    # Phase 3: Try structured decompositions based on Strassen-like patterns
    if time.time() < deadline:
        # Try various structured rank values
        for target_rank in [24, 25, 26, 27, 28, 29, 30]:
            if target_rank >= best_rank:
                continue
            if time.time() > deadline:
                break
            
            found = False
            for restart in range(10):
                if time.time() > deadline:
                    break
                
                np.random.seed(restart * 13 + target_rank)
                
                # Use a more structured initialization
                U_trial = np.zeros((n * m, target_rank))
                V_trial = np.zeros((m * p, target_rank))
                W_trial = np.zeros((p * n, target_rank))
                
                # Initialize with random ±1 values in a structured way
                for c in range(target_rank):
                    # Random pattern for U
                    for idx in range(n * m):
                        if np.random.rand() < 0.3:
                            U_trial[idx, c] = np.random.choice([-1, 1])
                    
                    # Random pattern for V
                    for idx in range(m * p):
                        if np.random.rand() < 0.3:
                            V_trial[idx, c] = np.random.choice([-1, 1])
                    
                    # Random pattern for W
                    for idx in range(p * n):
                        if np.random.rand() < 0.3:
                            W_trial[idx, c] = np.random.choice([-1, 1])
                
                # Run alternating minimization
                for iter_num in range(300):
                    if time.time() > deadline:
                        break
                    
                    W_trial = _solve_for_W(U_trial, V_trial, n, m, p)
                    V_trial = _solve_for_V(U_trial, W_trial, n, m, p)
                    U_trial = _solve_for_U(V_trial, W_trial, n, m, p)
                    
                    # Round to half-integers and check
                    U_rounded = _round_to_half_int(U_trial)
                    V_rounded = _round_to_half_int(V_trial)
                    W_rounded = _round_to_half_int(W_trial)
                    
                    if _validate(U_rounded, V_rounded, W_rounded, n, m, p):
                        best_rank = target_rank
                        best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                        found = True
                        break
                
                if found:
                    break
            
            if found:
                break
    
    # Convert to lists for return
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
