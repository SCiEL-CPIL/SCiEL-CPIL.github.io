"""Matrix multiplication 3,3,4 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 3, 4]

import time
import math
import numpy as np
import scipy.optimize as opt
from fractions import Fraction

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k."""
    rank = U.shape[1]
    T = np.zeros((n, m, p))
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        T += np.einsum('ij,jk,ki->ijk', u, v, w)
    return np.allclose(T, 1.0, atol=tol)

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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
    return U, V, W

def _snap_and_verify(U, V, W, n, m, p, tol=1e-6):
    """
    Try to snap U, V, W to half-integers and verify.
    Returns (U_snapped, V_snapped, W_snapped) if valid, else None.
    """
    # First try snapping to nearest 0.5
    U_s = np.round(U * 2) / 2
    V_s = np.round(V * 2) / 2
    W_s = np.round(W * 2) / 2
    
    if _verify(U_s, V_s, W_s, n, m, p, tol=1e-5):
        return U_s, V_s, W_s
    
    # Try snapping to nearest integer
    U_i = np.round(U)
    V_i = np.round(V)
    W_i = np.round(W)
    
    if _verify(U_i, V_i, W_i, n, m, p, tol=1e-5):
        return U_i, V_i, W_i
        
    # Try a local search around the snapped values? 
    # For now, just return None. The caller can try to polish.
    return None

def _polish_solution(U, V, W, n, m, p, time_budget=5.0):
    """
    High-precision polish pass: refine a near-solution with tighter tolerances.
    Returns polished (U, V, W) if it snaps to half-integers and verifies, else None.
    """
    start = time.time()
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    R = U.shape[1]
    
    def objective_polish(params):
        U_flat = params[:dim_U * R].reshape(dim_U, R)
        V_flat = params[dim_U * R : dim_U * R + dim_V * R].reshape(dim_V, R)
        W_flat = params[dim_U * R + dim_V * R :].reshape(dim_W, R)
        
        T = np.zeros((n, m, p))
        for r in range(R):
            u = U_flat[:, r].reshape(n, m)
            v = V_flat[:, r].reshape(m, p)
            w = W_flat[:, r].reshape(p, n)
            T += np.einsum('ij,jk,ki->ijk', u, v, w)
        
        E = T - 1.0
        loss = np.sum(E * E)
        
        dU = np.zeros_like(U_flat)
        dV = np.zeros_like(V_flat)
        dW = np.zeros_like(W_flat)
        
        for r in range(R):
            u = U_flat[:, r].reshape(n, m)
            v = V_flat[:, r].reshape(m, p)
            w = W_flat[:, r].reshape(p, n)
            
            dU_r = 2 * np.einsum('ijk,jk,ki->ij', E, v, w)
            dV_r = 2 * np.einsum('ijk,ij,ki->jk', E, u, w)
            dW_r = 2 * np.einsum('ijk,ij,jk->ki', E, u, v)
            
            dU[:, r] = dU_r.flatten()
            dV[:, r] = dV_r.flatten()
            dW[:, r] = dW_r.flatten()
        
        grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
        return loss, grad
    
    params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    try:
        res = opt.minimize(objective_polish, params0, method='L-BFGS-B', jac=True,
                           options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-12,
                                    'maxfun': 10000, 'maxls': 50})
    except Exception:
        return None
    
    if res.fun < 1e-10:
        U_sol = res.x[:dim_U * R].reshape(dim_U, R)
        V_sol = res.x[dim_U * R : dim_U * R + dim_V * R].reshape(dim_V, R)
        W_sol = res.x[dim_U * R + dim_V * R :].reshape(dim_W, R)
        
        snapped = _snap_and_verify(U_sol, V_sol, W_sol, n, m, p, tol=1e-8)
        if snapped:
            return snapped
    
    return None

def _structured_init(n, m, p, R, rng):
    """
    Create a structured initialization based on partial schoolbook patterns.
    This exploits the tensor structure to provide better starting points.
    """
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    U = np.zeros((dim_U, R))
    V = np.zeros((dim_V, R))
    W = np.zeros((dim_W, R))
    
    # Fill with a mix of structured and random patterns
    for r in range(R):
        # Choose random indices with some structure
        i = rng.randint(n)
        j = rng.randint(m)
        k = rng.randint(p)
        
        # Add structured components
        U[i * m + j, r] = rng.choice([0.5, 1.0, -0.5, -1.0])
        V[j * p + k, r] = rng.choice([0.5, 1.0, -0.5, -1.0])
        W[k * n + i, r] = rng.choice([0.5, 1.0, -0.5, -1.0])
        
        # Add some random perturbation
        U[:, r] += rng.randn(dim_U) * 0.1
        V[:, r] += rng.randn(dim_V) * 0.1
        W[:, r] += rng.randn(dim_W) * 0.1
    
    return U, V, W

def _try_optimize(n, m, p, target_rank, time_limit, seed=42):
    """
    Use scipy.optimize.minimize to find a rank-`target_rank` decomposition.
    """
    rng = np.random.RandomState(seed)
    start = time.time()
    
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    R = target_rank
    
    def objective_full(params):
        U_flat = params[:dim_U * R].reshape(dim_U, R)
        V_flat = params[dim_U * R : dim_U * R + dim_V * R].reshape(dim_V, R)
        W_flat = params[dim_U * R + dim_V * R :].reshape(dim_W, R)
        
        T = np.zeros((n, m, p))
        for r in range(R):
            u = U_flat[:, r].reshape(n, m)
            v = V_flat[:, r].reshape(m, p)
            w = W_flat[:, r].reshape(p, n)
            T += np.einsum('ij,jk,ki->ijk', u, v, w)
        
        E = T - 1.0
        loss = np.sum(E * E)
        
        dU = np.zeros_like(U_flat)
        dV = np.zeros_like(V_flat)
        dW = np.zeros_like(W_flat)
        
        for r in range(R):
            u = U_flat[:, r].reshape(n, m)
            v = V_flat[:, r].reshape(m, p)
            w = W_flat[:, r].reshape(p, n)
            
            dU_r = 2 * np.einsum('ijk,jk,ki->ij', E, v, w)
            dV_r = 2 * np.einsum('ijk,ij,ki->jk', E, u, w)
            dW_r = 2 * np.einsum('ijk,ij,jk->ki', E, u, v)
            
            dU[:, r] = dU_r.flatten()
            dV[:, r] = dV_r.flatten()
            dW[:, r] = dW_r.flatten()
        
        grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
        return loss, grad

    best_result = None
    best_loss = float('inf')
    best_params = None
    
    # We will try multiple initializations: random and structured
    num_random = 6
    num_structured = 4
    total_attempts = num_random + num_structured
    
    for attempt in range(total_attempts):
        if time.time() > start + time_limit - 3:
            break
            
        if attempt < num_random:
            # Random initialization with varying scales
            scale = 1.0 / math.sqrt(R)
            if attempt == 0:
                s = scale
            elif attempt == 1:
                s = 1.0
            elif attempt == 2:
                s = 0.5
            else:
                s = scale * (0.5 + attempt * 0.1)
                
            U0 = rng.randn(dim_U, R) * s
            V0 = rng.randn(dim_V, R) * s
            W0 = rng.randn(dim_W, R) * s
        else:
            # Structured initialization
            U0, V0, W0 = _structured_init(n, m, p, R, rng)
        
        params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        try:
            # Use L-BFGS-B with a generous number of iterations
            res = opt.minimize(objective_full, params0, method='L-BFGS-B', jac=True, 
                               options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-10,
                                        'maxfun': 5000, 'maxls': 30})
        except Exception as e:
            continue
            
        if res.fun < best_loss:
            best_loss = res.fun
            best_params = res.x
            
            if best_loss < 1e-8:
                U_sol = res.x[:dim_U * R].reshape(dim_U, R)
                V_sol = res.x[dim_U * R : dim_U * R + dim_V * R].reshape(dim_V, R)
                W_sol = res.x[dim_U * R + dim_V * R :].reshape(dim_W, R)
                
                # Try direct snap first
                snapped = _snap_and_verify(U_sol, V_sol, W_sol, n, m, p)
                if snapped:
                    return snapped
                
                # If not snapped, try polish
                if time.time() < start + time_limit - 5:
                    polished = _polish_solution(U_sol, V_sol, W_sol, n, m, p, time_budget=5.0)
                    if polished:
                        return polished
                    
    # Final check on best solution found
    if best_params is not None and best_loss < 1e-6:
        U_sol = best_params[:dim_U * R].reshape(dim_U, R)
        V_sol = best_params[dim_U * R : dim_U * R + dim_V * R].reshape(dim_V, R)
        W_sol = best_params[dim_U * R + dim_V * R :].reshape(dim_W, R)
        
        if time.time() < start + time_limit - 3:
            polished = _polish_solution(U_sol, V_sol, W_sol, n, m, p, time_budget=3.0)
            if polished:
                return polished
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Start with schoolbook
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = n * m * p
    
    # Theoretical lower bound for 3x3x4 is likely 24 or 25.
    # We try from 24 upwards.
    # If we find a valid decomposition, we return it.
    
    # Allocate time more carefully: give more time to lower ranks
    rank_time_weights = {
        24: 0.35,
        25: 0.30,
        26: 0.20,
        27: 0.10,
        28: 0.05
    }
    
    total_weight = sum(rank_time_weights.values())
    available_time = deadline - start_time - 10  # Reserve 10s for overhead
    
    for target_rank in range(24, best_rank + 1):
        if time.time() > deadline - 5:
            break
        
        weight = rank_time_weights.get(target_rank, 0.02)
        time_for_rank = (weight / total_weight) * available_time
        time_for_rank = max(10.0, min(time_for_rank, deadline - time.time() - 5))
        
        if time_for_rank < 10:
            break
            
        seed = 42 + target_rank * 7
        result = _try_optimize(n, m, p, target_rank, time_for_rank, seed=seed)
        
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p):
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                best_rank = target_rank
                break
                
    return best_result

# EVOLVE-BLOCK-END
