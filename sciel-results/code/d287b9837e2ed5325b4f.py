import numpy as np
import time
import random

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Try ranks from low to high
    for rank in [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 35, 40, 50, 60, 80]:
        if time.time() > deadline - 15:
            break
        
        result = _find_decomposition(n, m, p, rank, deadline)
        if result is not None:
            return result
    
    # Fallback: schoolbook
    return _schoolbook(n, m, p)

def _schoolbook(n, m, p):
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

def _find_decomposition(n, m, p, rank, deadline):
    """Find a rank decomposition using alternating minimization with exact recovery."""
    nm = n * m  # 16
    mp = m * p  # 20
    pn = p * n  # 20
    
    np.random.seed(42)
    random.seed(42)
    
    max_restarts = 30
    
    for restart in range(max_restarts):
        if time.time() > deadline - 10:
            return None
        
        # Different initialization strategies
        if restart < 10:
            # Random integer initialization
            U = np.random.randint(-3, 4, size=(nm, rank)).astype(float)
            V = np.random.randint(-3, 4, size=(mp, rank)).astype(float)
            W = np.random.randint(-3, 4, size=(pn, rank)).astype(float)
        elif restart < 20:
            # Sparse random initialization
            U = np.zeros((nm, rank))
            V = np.zeros((mp, rank))
            W = np.zeros((pn, rank))
            for c in range(rank):
                U[np.random.randint(0, nm), c] = np.random.choice([-1, 1])
                V[np.random.randint(0, mp), c] = np.random.choice([-1, 1])
                W[np.random.randint(0, pn), c] = np.random.choice([-1, 1])
        else:
            # Gaussian initialization
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
        
        # Alternating minimization
        for iteration in range(300):
            if time.time() > deadline - 10:
                break
            
            # Solve for U: fix V, W
            U_new = _solve_factor_U(n, m, p, rank, V, W)
            
            # Solve for V: fix U_new, W
            V_new = _solve_factor_V(n, m, p, rank, U_new, W)
            
            # Solve for W: fix U_new, V_new
            W_new = _solve_factor_W(n, m, p, rank, U_new, V_new)
            
            U, V, W = U_new, V_new, W_new
        
        # Try to recover exact rational solution
        result = _recover_exact(n, m, p, U, V, W, deadline)
        if result is not None:
            return result
        
        # Also try direct verification with tolerance
        if _verify_numpy(n, m, p, U, V, W, tol=1e-8):
            # Round to nearest half-integer
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            if _verify_numpy(n, m, p, U_r, V_r, W_r, tol=1e-6):
                return {
                    "U": U_r.tolist(),
                    "V": V_r.tolist(),
                    "W": W_r.tolist()
                }
    
    return None

def _solve_factor_U(n, m, p, rank, V, W):
    """Solve for U given V and W."""
    nm = n * m
    U = np.zeros((nm, rank))
    for i in range(n):
        for j in range(m):
            # A[c] = sum_k V[j*p+k, c] * W[k*n+i, c]
            A = np.zeros(rank)
            for k in range(p):
                A += V[j * p + k, :] * W[k * n + i, :]
            # Solve: U[i*m+j, :] @ A = 1 (minimum norm solution)
            norm_A = np.linalg.norm(A)
            if norm_A < 1e-12:
                U[i * m + j, :] = 0
            else:
                # Minimum norm solution: U_row = A / ||A||^2
                U[i * m + j, :] = A / (norm_A ** 2)
    return U

def _solve_factor_V(n, m, p, rank, U, W):
    """Solve for V given U and W."""
    mp = m * p
    V = np.zeros((mp, rank))
    for j in range(m):
        for k in range(p):
            # B[c] = sum_i U[i*m+j, c] * W[k*n+i, c]
            B = np.zeros(rank)
            for i in range(n):
                B += U[i * m + j, :] * W[k * n + i, :]
            norm_B = np.linalg.norm(B)
            if norm_B < 1e-12:
                V[j * p + k, :] = 0
            else:
                V[j * p + k, :] = B / (norm_B ** 2)
    return V

def _solve_factor_W(n, m, p, rank, U, V):
    """Solve for W given U and V."""
    pn = p * n
    W = np.zeros((pn, rank))
    for k in range(p):
        for i in range(n):
            # C[c] = sum_j U[i*m+j, c] * V[j*p+k, c]
            C = np.zeros(rank)
            for j in range(m):
                C += U[i * m + j, :] * V[j * p + k, :]
            norm_C = np.linalg.norm(C)
            if norm_C < 1e-12:
                W[k * n + i, :] = 0
            else:
                W[k * n + i, :] = C / (norm_C ** 2)
    return W

def _recover_exact(n, m, p, U, V, W, deadline):
    """Try to recover an exact rational (half-integer) solution."""
    # First check if current solution is close to exact
    if _verify_numpy(n, m, p, U, V, W, tol=1e-6):
        # Try rounding to nearest half-integer
        for scale in [1, 2, 4]:
            U_r = np.round(U * scale) / scale
            V_r = np.round(V * scale) / scale
            W_r = np.round(W * scale) / scale
            if _verify_numpy(n, m, p, U_r, V_r, W_r, tol=1e-6):
                return {
                    "U": U_r.tolist(),
                    "V": V_r.tolist(),
                    "W": W_r.tolist()
                }
    
    # Try local refinement around current solution
    # Use gradient-free perturbation
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_err = _max_error(n, m, p, U, V, W)
    
    if best_err < 1e-4:
        # Do a few rounds of perturbation and re-solve
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
            
            # Perturb one factor at a time
            for factor_idx in range(3):
                if factor_idx == 0:
                    U_pert = U + 0.01 * np.random.randn(*U.shape)
                    V_pert, W_pert = V.copy(), W.copy()
                elif factor_idx == 1:
                    U_pert, W_pert = U.copy(), W.copy()
                    V_pert = V + 0.01 * np.random.randn(*V.shape)
                else:
                    U_pert, V_pert = U.copy(), V.copy()
                    W_pert = W + 0.01 * np.random.randn(*W.shape)
                
                # One round of alternating minimization from perturbed start
                U_new = _solve_factor_U(n, m, p, U_pert.shape[1], V_pert, W_pert)
                V_new = _solve_factor_V(n, m, p, U_new.shape[1], U_new, W_pert)
                W_new = _solve_factor_W(n, m, p, U_new.shape[1], U_new, V_new)
                
                err = _max_error(n, m, p, U_new, V_new, W_new)
                if err < best_err:
                    best_U, best_V, best_W = U_new, V_new, W_new
                    best_err = err
        
        if best_err < 1e-6:
            for scale in [1, 2, 4]:
                U_r = np.round(best_U * scale) / scale
                V_r = np.round(best_V * scale) / scale
                W_r = np.round(best_W * scale) / scale
                if _verify_numpy(n, m, p, U_r, V_r, W_r, tol=1e-6):
                    return {
                        "U": U_r.tolist(),
                        "V": V_r.tolist(),
                        "W": W_r.tolist()
                    }
    
    return None

def _max_error(n, m, p, U, V, W):
    """Compute maximum error over all tensor entries."""
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err

def _verify_numpy(n, m, p, U, V, W, tol=1e-6):
    """Verify decomposition using numpy."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True
# EVOLVE-BLOCK-END
