"""Matrix multiplication 2x3x7 rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction
import itertools
import random
from collections import defaultdict

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START

def _validate_decomposition(U, V, W, n, m, p, rank):
    """Check if U, V, W form a valid exact decomposition using Fraction arithmetic."""
    Uf = [[Fraction(U[i][r]) for r in range(rank)] for i in range(n*m)]
    Vf = [[Fraction(V[j][r]) for r in range(rank)] for j in range(m*p)]
    Wf = [[Fraction(W[k][r]) for r in range(rank)] for k in range(p*n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = sum(Uf[i*m+j][r] * Vf[j*p+k][r] * Wf[k*n+i][r] for r in range(rank))
                if total != 1:
                    return False
    return True

def _validate_decomposition_float(U, V, W, n, m, p, rank, tol=1e-9):
    """Faster validation using float arithmetic."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U_arr[i*m+j]
                v_row = V_arr[j*p+k]
                w_row = W_arr[k*n+i]
                s = np.sum(u_row * v_row * w_row)
                if abs(s - 1.0) > tol:
                    return False
    return True

def _to_rational_matrix(U, V, W, n, m, p, rank, max_denom=1000):
    """Try to convert float matrices to exact rationals with small denominators."""
    def _to_rational_matrix_single(M, rows, cols):
        result = []
        for i in range(rows):
            row = []
            for j in range(cols):
                val = M[i][j]
                if abs(val) < 1e-12:
                    row.append(0)
                else:
                    found = False
                    for d in range(1, max_denom+1):
                        num = round(val * d)
                        if abs(val - num/d) < 1e-10:
                            row.append(num/d)
                            found = True
                            break
                    if not found:
                        return None
            result.append(row)
        return result
    
    U_r = _to_rational_matrix_single(U, n*m, rank)
    if U_r is None:
        return None
    V_r = _to_rational_matrix_single(V, m*p, rank)
    if V_r is None:
        return None
    W_r = _to_rational_matrix_single(W, p*n, rank)
    if W_r is None:
        return None
    
    return U_r, V_r, W_r

def _try_solve_for_W(U, V, n, m, p, rank):
    """Given U and V, try to find W that satisfies the tensor equation.
    
    For each (i,k), we need W[k*n+i, :] such that:
    For all j: sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] = 1
    
    This is a linear system for each (i,k) pair.
    """
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for k in range(p):
            # For fixed i,k, we need to find w = W[k*n+i, :]
            # such that for all j in [0,m):
            # sum_r U[i*m+j][r] * V[j*p+k][r] * w[r] = 1
            #
            # This is: A_j . w = 1 for j=0..m-1
            # where A_j[r] = U[i*m+j][r] * V[j*p+k][r]
            
            # Build the system: m equations, rank unknowns
            # A is m x rank, b is m-vector of 1s
            A = np.zeros((m, rank), dtype=np.float64)
            for j in range(m):
                for r in range(rank):
                    A[j][r] = U[i*m+j][r] * V[j*p+k][r]
            
            b = np.ones(m, dtype=np.float64)
            
            # Solve the underdetermined system A @ w = b
            # Use least squares, then check if it's exact
            try:
                # If rank >= m, we can find a solution
                w, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                
                # Check if the solution is exact
                if np.allclose(A @ w, b, atol=1e-8):
                    # Check if w has small rational values
                    for r in range(rank):
                        val = w[r]
                        if abs(val) < 1e-12:
                            W[k*n+i][r] = 0
                        else:
                            # Try to find rational with small denominator
                            found = False
                            for d in range(1, 200):
                                num = round(val * d)
                                if abs(val - num/d) < 1e-8:
                                    W[k*n+i][r] = num/d
                                    found = True
                                    break
                            if not found:
                                return None
                else:
                    return None
            except:
                return None
    
    return W

def _try_rank_with_linear_solve(n, m, p, rank, seed=42):
    """Try to find a valid rank-r decomposition by fixing U,V and solving for W."""
    np.random.seed(seed)
    random.seed(seed)
    
    # Initialize U and V with structured random values
    # Use values from {-1, 0, 1, 0.5, -0.5} for simplicity
    entries = [0, 1, -1, 0.5, -0.5]
    
    U = [[random.choice(entries) for _ in range(rank)] for _ in range(n * m)]
    V = [[random.choice(entries) for _ in range(rank)] for _ in range(m * p)]
    
    # Ensure no row is all zeros
    for i in range(n * m):
        if all(U[i][r] == 0 for r in range(rank)):
            U[i][random.randint(0, rank-1)] = 1.0
    for j in range(m * p):
        if all(V[j][r] == 0 for r in range(rank)):
            V[j][random.randint(0, rank-1)] = 1.0
    
    W = _try_solve_for_W(U, V, n, m, p, rank)
    
    if W is not None:
        # Validate
        if _validate_decomposition_float(U, V, W, n, m, p, rank, tol=1e-8):
            # Try to rationalize
            rational = _to_rational_matrix(U, V, W, n, m, p, rank, max_denom=200)
            if rational is not None:
                U_r, V_r, W_r = rational
                if _validate_decomposition(U_r, V_r, W_r, n, m, p, rank):
                    return U_r, V_r, W_r
            return U, V, W
    
    return None

def _gradient_descent_3tensor(n, m, p, rank, max_iter=5000, lr=0.01, tol=1e-10):
    """Use gradient descent on the non-convex objective to find a rank-r decomposition."""
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    
    # Structured initialization
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if idx < rank:
                    U[i*m+j][idx] = 1.0
                    V[j*p+k][idx] = 1.0
                    W[k*n+i][idx] = 1.0
                    idx += 1
    
    # Add small random perturbation
    U += np.random.randn(n*m, rank) * 0.01
    V += np.random.randn(m*p, rank) * 0.01
    W += np.random.randn(p*n, rank) * 0.01
    
    def _objective_and_grad(U, V, W):
        obj = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ui = U[i*m+j]
                    vk = V[j*p+k]
                    wk = W[k*n+i]
                    
                    s = np.sum(ui * vk * wk)
                    err = s - 1.0
                    obj += err * err
                    
                    grad_U[i*m+j] += 2 * err * vk * wk
                    grad_V[j*p+k] += 2 * err * ui * wk
                    grad_W[k*n+i] += 2 * err * ui * vk
        
        return obj, grad_U, grad_V, grad_W
    
    # Adam optimizer
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    m_U = np.zeros_like(U); v_U = np.zeros_like(U)
    m_V = np.zeros_like(V); v_V = np.zeros_like(V)
    m_W = np.zeros_like(W); v_W = np.zeros_like(W)
    t = 0
    best_obj = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        t += 1
        obj, gU, gV, gW = _objective_and_grad(U, V, W)
        
        if obj < best_obj:
            best_obj = obj
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if obj < tol:
            break
        
        m_U = beta1*m_U + (1-beta1)*gU; v_U = beta2*v_U + (1-beta2)*gU**2
        m_U_hat = m_U/(1-beta1**t); v_U_hat = v_U/(1-beta2**t)
        m_V = beta1*m_V + (1-beta1)*gV; v_V = beta2*v_V + (1-beta2)*gV**2
        m_V_hat = m_V/(1-beta1**t); v_V_hat = v_V/(1-beta2**t)
        m_W = beta1*m_W + (1-beta1)*gW; v_W = beta2*v_W + (1-beta2)*gW**2
        m_W_hat = m_W/(1-beta1**t); v_W_hat = v_W/(1-beta2**t)
        
        U -= lr * m_U_hat / (np.sqrt(v_U_hat) + eps)
        V -= lr * m_V_hat / (np.sqrt(v_V_hat) + eps)
        W -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)
    
    return best_U, best_V, best_W, best_obj

def _try_rank_with_gradient_descent(n, m, p, rank, max_iter=3000, lr=0.005, tol=1e-10, seed=42):
    """Try to find a valid rank-r decomposition using gradient descent."""
    np.random.seed(seed)
    
    U, V, W, obj = _gradient_descent_3tensor(n, m, p, rank, max_iter=max_iter, lr=lr, tol=tol)
    
    if obj < 1e-6:
        if _validate_decomposition_float(U.tolist(), V.tolist(), W.tolist(), n, m, p, rank, tol=1e-8):
            rational = _to_rational_matrix(U.tolist(), V.tolist(), W.tolist(), n, m, p, rank, max_denom=200)
            if rational is not None:
                U_r, V_r, W_r = rational
                if _validate_decomposition(U_r, V_r, W_r, n, m, p, rank):
                    return U_r, V_r, W_r
            return U.tolist(), V.tolist(), W.tolist()
    
    return None

def _construct_structured(n, m, p, rank):
    """Try to construct a structured decomposition exploiting the 7-column sharing."""
    # For 2x3x7, we can potentially share across k values
    # The key insight: for fixed i,j, the equation is:
    # sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] = 1 for all k
    
    # This suggests that V[j*p+k] and W[k*n+i] should be related across k
    
    # Try a circulant-like structure for the k dimension
    # For each (i,j), define a vector u_{i,j}
    # For each j, define vectors v_{j,k} for k=0..p-1
    # For each (k,i), define vectors w_{k,i}
    
    # If we set v_{j,k} = v_j rotated by k, and w_{k,i} = w_i rotated by k
    # with appropriate scaling, we might get sharing
    
    # For now, try a simpler approach: use the same rank indices for all k
    # This means we need rank >= n*m = 6, but with constraints
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Assign each (i,j) to a unique rank index
    for i in range(n):
        for j in range(m):
            idx = i * m + j
            if idx < rank:
                U[i*m+j][idx] = 1.0
    
    # For each (j,k), assign to a rank index
    # Try to reuse indices across k
    for j in range(m):
        for k in range(p):
            # Use a pattern: rank index = (j * p + k) % rank
            idx = (j * p + k) % rank
            V[j*p+k][idx] = 1.0
    
    # For each (k,i), assign to a rank index
    for k in range(p):
        for i in range(n):
            idx = (k * n + i) % rank
            W[k*n+i][idx] = 1.0
    
    # Check if this works
    if _validate_decomposition_float(U, V, W, n, m, p, rank, tol=1e-8):
        return U, V, W
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition: rank = n*m*p = 42
    rank_schoolbook = n * m * p
    
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_schoolbook
    
    # Lower bound: max(n*m, m*p, p*n) = 21
    min_rank = max(n*m, m*p, p*n)
    
    # Try ranks from lower bound upward
    # Focus on ranks that might have structure
    ranks_to_try = list(range(min_rank, rank_schoolbook))
    
    # Prioritize certain ranks
    priority_ranks = [21, 22, 23, 24, 25, 26, 27, 28, 30, 32, 35, 36, 37, 38, 39, 40, 41]
    ranks_to_try = [r for r in priority_ranks if min_rank <= r < rank_schoolbook]
    ranks_to_try += [r for r in range(min_rank, rank_schoolbook) if r not in ranks_to_try]
    
    for r in ranks_to_try:
        if time.time() > deadline - 20:
            break
        
        if r >= best_rank:
            continue
        
        # Try structured construction first
        if time.time() < deadline - 30:
            structured = _construct_structured(n, m, p, r)
            if structured is not None:
                U, V, W = structured
                best_result = {"U": U, "V": V, "W": W}
                best_rank = r
                continue
        
        # Try linear solve approach with multiple seeds
        found = False
        for seed in [42, 123, 789, 2024, 555, 1001, 2025, 7777, 9999, 31337]:
            if time.time() > deadline - 15:
                break
            
            result = _try_rank_with_linear_solve(n, m, p, r, seed=seed)
            if result is not None:
                U, V, W = result
                best_result = {"U": U, "V": V, "W": W}
                best_rank = r
                found = True
                break
        
        if found:
            continue
        
        # Try gradient descent if time permits
        if time.time() < deadline - 10:
            for seed in [42, 123, 789]:
                if time.time() > deadline - 10:
                    break
                
                result = _try_rank_with_gradient_descent(n, m, p, r, max_iter=1500, lr=0.005, seed=seed)
                if result is not None:
                    U, V, W = result
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = r
                    break
    
    return best_result
# EVOLVE-BLOCK-END
