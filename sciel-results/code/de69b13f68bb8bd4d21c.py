"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START

def _schoolbook():
    n, m, p = DIMENSIONS
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


def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast vectorized verification using einsum."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    
    T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    
    max_err = np.max(np.abs(T - 1.0))
    return max_err < tol, max_err


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition exactly reconstructs the matrix multiplication tensor."""
    ok, _ = _verify_fast(U, V, W, n, m, p, tol)
    return ok


def _objective_and_grad(x, n, m, p, r):
    """Compute objective and gradient for L-BFGS."""
    U = x[:n*m*r].reshape(n, m, r)
    V = x[n*m*r:2*n*m*r].reshape(m, p, r)
    W = x[2*n*m*r:].reshape(p, n, r)
    
    T = np.einsum('ijc,jkc,kic->ijk', U, V, W)
    residual = T - 1.0
    obj = np.sum(residual**2)
    
    # Gradients
    dU = 2 * np.einsum('ijk,jkc,kic->ijc', residual, V, W)
    dV = 2 * np.einsum('ijk,ijc,kic->jkc', residual, U, W)
    dW = 2 * np.einsum('ijk,ijc,jkc->kic', residual, U, V)
    
    grad = np.concatenate([dU.ravel(), dV.ravel(), dW.ravel()])
    return obj, grad


def _lbfgs_solve(n, m, p, r, max_time=15, seed=42):
    """Use L-BFGS-B to find a rank-r decomposition."""
    start_time = time.time()
    rng = np.random.RandomState(seed)
    
    # Try multiple initializations
    for init_type in ['random', 'structured', 'sparse']:
        if time.time() - start_time > max_time:
            break
        
        if init_type == 'random':
            U = rng.randn(n * m, r) * 0.5
            V = rng.randn(m * p, r) * 0.5
            W = rng.randn(p * n, r) * 0.5
        elif init_type == 'structured':
            # Initialize with some structure
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            for c in range(r):
                i = c % n
                j = (c // n) % m
                k = (c // (n*m)) % p
                U[i*m+j, c] = 1.0
                V[j*p+k, c] = 1.0
                W[k*n+i, c] = 1.0
            # Add small random perturbation
            U += rng.randn(n*m, r) * 0.1
            V += rng.randn(m*p, r) * 0.1
            W += rng.randn(p*n, r) * 0.1
        else:  # sparse
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            for c in range(r):
                idx = rng.randint(0, n*m-1)
                U[idx, c] = 1.0
                idx = rng.randint(0, m*p-1)
                V[idx, c] = 1.0
                idx = rng.randint(0, p*n-1)
                W[idx, c] = 1.0
            # Add small random perturbation to all
            U += rng.randn(n*m, r) * 0.05
            V += rng.randn(m*p, r) * 0.05
            W += rng.randn(p*n, r) * 0.05
        
        x0 = np.concatenate([U.ravel(), V.ravel(), W.ravel()])
        
        def obj_and_grad(x):
            return _objective_and_grad(x, n, m, p, r)
        
        # Use L-BFGS-B with a time limit via callback
        class TimeLimitExceeded(Exception):
            pass
        
        def callback(xk):
            if time.time() - start_time > max_time - 2:
                raise TimeLimitExceeded()
            return False
        
        try:
            result = minimize(
                lambda x: _objective_and_grad(x, n, m, p, r)[0],
                x0,
                jac=lambda x: _objective_and_grad(x, n, m, p, r)[1],
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 500, 'callback': callback}
            )
            
            if result.success or result.fun < 1e-8:
                U_opt = result.x[:n*m*r].reshape(n*m, r)
                V_opt = result.x[n*m*r:2*n*m*r].reshape(m*p, r)
                W_opt = result.x[2*n*m*r:].reshape(p*n, r)
                
                ok, err = _verify_fast(U_opt.tolist(), V_opt.tolist(), W_opt.tolist(), n, m, p, tol=1e-8)
                if ok or err < 1e-6:
                    return U_opt, V_opt, W_opt, r, err
        except TimeLimitExceeded:
            continue
        except Exception:
            continue
    
    return None


def _try_polish(U, V, W, n, m, p, max_time=5):
    """Try to polish a near-exact solution to exact fractions."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Try multiple denominator limits
    for max_den in [10, 50, 100, 500, 1000, 5000, 10000]:
        if time.time() > start_time + max_time:
            break
        
        U_frac = [[Fraction(x).limit_denominator(max_den) for x in row] for row in U_arr.tolist()]
        V_frac = [[Fraction(x).limit_denominator(max_den) for x in row] for row in V_arr.tolist()]
        W_frac = [[Fraction(x).limit_denominator(max_den) for x in row] for row in W_arr.tolist()]
        
        ok, _ = _verify_fast(U_frac, V_frac, W_frac, n, m, p, tol=1e-9)
        if ok:
            return U_frac, V_frac, W_frac
    
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Start with schoolbook
    result = _schoolbook()
    best_rank = n * m * p
    best_result = result
    
    # Try L-BFGS for various ranks, starting from lower ranks
    # Theoretical lower bound is max(n*m, m*p, p*n) = max(12, 16, 12) = 16
    # But structured decompositions may do better
    for r in range(16, 35):
        if time.time() > deadline - 15:
            break
        
        remaining = deadline - time.time() - 10
        if remaining < 8:
            break
        
        # Allocate time per seed based on remaining time
        num_seeds = min(8, max(3, int(remaining / 8)))
        time_per_seed = max(3, (remaining - 10) / num_seeds)
        
        found = False
        for seed in range(num_seeds):
            if time.time() > deadline - 10:
                break
            
            res = _lbfgs_solve(n, m, p, r, max_time=time_per_seed, seed=seed + 100)
            if res is not None:
                U, V, W, rank, err = res
                if rank < best_rank:
                    best_rank = rank
                    # Try to polish
                    polished = _try_polish(U, V, W, n, m, p, max_time=min(3, deadline - time.time() - 10))
                    if polished is not None:
                        U_f, V_f, W_f = polished
                        best_result = {"U": U_f, "V": V_f, "W": W_f}
                    else:
                        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    found = True
                    break
        
        if found:
            # If we found a good rank, try to go lower
            if best_rank <= 18:
                break
    
    return best_result

# EVOLVE-BLOCK-END
