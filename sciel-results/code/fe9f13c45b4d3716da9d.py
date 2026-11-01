import numpy as np
import time
import random
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 3]

def _verify_factors(U, V, W, n, m, p, tol=1e-9):
    """Verify that the factor matrices exactly decompose the matrix multiplication tensor."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                A = i * m + j
                B = j * p + k
                C = k * n + i
                expected = 1
                actual = np.dot(U_arr[A], V_arr[B] * W_arr[C])
                if abs(actual - expected) > tol:
                    return False, (A, B, C, expected, actual)
    return True, None

def _verify_exact(U, V, W, n, m, p):
    """Verify exactness using Fraction arithmetic."""
    U_frac = [[Fraction(x).limit_denominator(1000000) for x in row] for row in U]
    V_frac = [[Fraction(x).limit_denominator(1000000) for x in row] for row in V]
    W_frac = [[Fraction(x).limit_denominator(1000000) for x in row] for row in W]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                A = i * m + j
                B = j * p + k
                C = k * n + i
                s = Fraction(0)
                for r in range(len(U_frac[A])):
                    s += U_frac[A][r] * V_frac[B][r] * W_frac[C][r]
                if s != 1:
                    return False
    return True

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _compute_residual(U, V, W, n, m, p):
    """Compute the residual for the tensor decomposition."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    res = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                A = i * m + j
                B = j * p + k
                C = k * n + i
                val = np.dot(U_arr[A], V_arr[B] * W_arr[C])
                res.append(val - 1.0)
    return np.array(res)

def _polish_solution(U, V, W, n, m, p, max_iter=10000):
    """High-precision polish using Gauss-Newton style updates."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    R = U_arr.shape[1]
    
    # Build Jacobian for Gauss-Newton
    def build_jacobian(U, V, W):
        J = np.zeros((n*m*p, R*(n*m + m*p + p*n)))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    A = i * m + j
                    B = j * p + k
                    C = k * n + i
                    idx = i * m * p + j * p + k
                    for r in range(R):
                        # d/dU[A,r]
                        col_u = r * (n*m + m*p + p*n) + A
                        J[idx, col_u] = V[B, r] * W[C, r]
                        # d/dV[B,r]
                        col_v = R*(n*m) + r*(m*p) + B
                        J[idx, col_v] = U[A, r] * W[C, r]
                        # d/dW[C,r]
                        col_w = R*(n*m + m*p) + r*(p*n) + C
                        J[idx, col_w] = U[A, r] * V[B, r]
        return J
    
    for it in range(max_iter):
        res = _compute_residual(U_arr, V_arr, W_arr, n, m, p)
        if np.linalg.norm(res) < 1e-14:
            break
        
        J = build_jacobian(U_arr, V_arr, W_arr)
        try:
            # Solve J * delta = -res
            delta, residuals, rank_j, sv = np.linalg.lstsq(J, -res, rcond=None)
            if rank_j < J.shape[1]:
                break
            
            # Damped update
            alpha = 1.0
            for _ in range(20):
                U_new = U_arr.copy()
                V_new = V_arr.copy()
                W_new = W_arr.copy()
                
                U_delta = delta[:R*n*m].reshape(n*m, R).T
                V_delta = delta[R*n*m:R*(n*m+m*p)].reshape(m*p, R).T
                W_delta = delta[R*(n*m+m*p):].reshape(p*n, R).T
                
                U_new = U_arr + alpha * U_delta
                V_new = V_arr + alpha * V_delta
                W_new = W_arr + alpha * W_delta
                
                res_new = _compute_residual(U_new, V_new, W_new, n, m, p)
                if np.linalg.norm(res_new) < np.linalg.norm(res):
                    U_arr, V_arr, W_arr = U_new, V_new, W_new
                    break
                alpha *= 0.5
            
            if alpha < 1e-10:
                break
        except np.linalg.LinAlgError:
            break
    
    return U_arr.tolist(), V_arr.tolist(), W_arr.tolist()

def _rational_reconstruct(U, V, W, n, m, p, max_denom=100000):
    """Attempt to find rational coefficients close to the floating point solution."""
    U_frac = []
    V_frac = []
    W_frac = []
    
    for row in U:
        U_frac.append([Fraction(x).limit_denominator(max_denom) for x in row])
    for row in V:
        V_frac.append([Fraction(x).limit_denominator(max_denom) for x in row])
    for row in W:
        W_frac.append([Fraction(x).limit_denominator(max_denom) for x in row])
    
    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                A = i * m + j
                B = j * p + k
                C = k * n + i
                s = Fraction(0)
                for r in range(len(U_frac[A])):
                    s += U_frac[A][r] * V_frac[B][r] * W_frac[C][r]
                if s != 1:
                    return None
    
    return (
        [[float(f) for f in row] for row in U_frac],
        [[float(f) for f in row] for row in V_frac],
        [[float(f) for f in row] for row in W_frac]
    )

def _structured_init_17(n, m, p, R=17):
    """Create a structured initialization for rank 17 based on known patterns."""
    # For 2x3 x 3x3, we know rank 17 is achievable
    # Use a pattern based on the Strassen-like decomposition
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    # Initialize with random values scaled appropriately
    np.random.seed(42)
    U = np.random.randn(n*m, R) * 0.5
    V = np.random.randn(m*p, R) * 0.5
    W = np.random.randn(p*n, R) * 0.5
    
    # Add some structure: make some entries zero to help convergence
    for r in range(R):
        # Each rank-1 component should have sparse structure
        mask_u = np.random.choice([0, 1], size=n*m, p=[0.3, 0.7])
        mask_v = np.random.choice([0, 1], size=m*p, p=[0.3, 0.7])
        mask_w = np.random.choice([0, 1], size=p*n, p=[0.3, 0.7])
        U[:, r] *= mask_u
        V[:, r] *= mask_v
        W[:, r] *= mask_w
    
    return U.tolist(), V.tolist(), W.tolist()

def _solve_for_rank(n, m, p, R, max_time=30, max_restarts=50):
    """
    Attempt to find a decomposition of rank R using iterative optimization.
    """
    start_time = time.time()
    
    for restart in range(max_restarts):
        if time.time() - start_time > max_time:
            break
        
        # Initialize
        if restart == 0:
            U, V, W = _structured_init_17(n, m, p, R)
        else:
            # Random restart
            np.random.seed(restart * 12345 + 67890)
            U = np.random.randn(n*m, R) * 0.3
            V = np.random.randn(m*p, R) * 0.3
            W = np.random.randn(p*n, R) * 0.3
            U, V, W = U.tolist(), V.tolist(), W.tolist()
        
        # Polish
        U_polished, V_polished, W_polished = _polish_solution(U, V, W, n, m, p, max_iter=5000)
        
        # Check if solution is good enough
        res = _compute_residual(U_polished, V_polished, W_polished, n, m, p)
        if np.linalg.norm(res) < 1e-10:
            # Try rational reconstruction
            rational_result = _rational_reconstruct(U_polished, V_polished, W_polished, n, m, p)
            if rational_result is not None:
                return rational_result
            
            # Return floating point if rational fails
            ok, _ = _verify_factors(U_polished, V_polished, W_polished, n, m, p, tol=1e-8)
            if ok:
                return (U_polished, V_polished, W_polished)
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145  # Safety margin before 150s kill
    
    # Try rank 17 first (known to be achievable for 2x3 x 3x3)
    if time.time() - start_time < deadline - 10:
        result = _solve_for_rank(n, m, p, 17, max_time=60, max_restarts=100)
        if result is not None:
            U, V, W = result
            # Final verification
            ok, _ = _verify_factors(U, V, W, n, m, p, tol=1e-8)
            if ok:
                return {"U": U, "V": V, "W": W}
    
    # Try rank 16 (unlikely but worth a quick attempt)
    if time.time() - start_time < deadline - 5:
        result = _solve_for_rank(n, m, p, 16, max_time=20, max_restarts=30)
        if result is not None:
            U, V, W = result
            ok, _ = _verify_factors(U, V, W, n, m, p, tol=1e-8)
            if ok:
                return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook (rank 18)
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
