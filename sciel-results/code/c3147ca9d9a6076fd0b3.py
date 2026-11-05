"""Matrix multiplication 3,5,6 - rank-minimal tensor decomposition."""
import numpy as np
import time
import random
import itertools
from collections import defaultdict

DIMENSIONS = [3, 5, 6]
DEADLINE = 115.0
START_TIME = time.time()

def _elapsed():
    return time.time() - START_TIME

def _remaining():
    return DEADLINE - _elapsed() - 5.0

def _snap_to_half(x):
    return round(2 * x) / 2.0

def _verify_fast(U, V, W, n, m, p):
    if not U or not V or not W:
        return False
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    if Ua.shape != (n*m, rank) or Va.shape != (m*p, rank) or Wa.shape != (p*n, rank):
        return False
    
    for arr in [Ua, Va, Wa]:
        if not np.allclose(2*arr, np.round(2*arr), atol=1e-9):
            return False
    
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    if not np.allclose(T, np.ones((n, m, p)), atol=1e-9):
        return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _build_tensor_matrix(n, m, p, rank, seed=None):
    """
    Build the linear system for a given support set of rank-one terms.
    Each rank-one term (i,j,k) contributes a row to the system.
    We need sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all (i,j,k).
    
    This is nonlinear, but we can linearize by fixing two of the three matrices.
    Strategy: fix U and V randomly, then solve for W linearly.
    Or: use the fact that for each (i,j), the equation is:
        sum_k sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
    
    Better approach: For a fixed support pattern, we can set up a system where
    we treat the product as bilinear and use alternating linear solves.
    
    Here we use a "perfect hashing" approach:
    - Choose a small set of (i,j,k) triples as the support
    - For each triple, assign a unique "hash" value
    - Solve the resulting linear system
    """
    pass

def _try_perfect_hash(n, m, p, rank, max_attempts=50, seed=None):
    """
    Use a minimal perfect hashing strategy:
    1. Generate candidate support sets of size 'rank'
    2. For each support, set up and solve the linear system
    3. Check if the solution has half-integer entries
    
    The key insight: if we can find rank rank-one terms whose
    "hashes" (index triples) form a collision-free mapping to
    the n*m*p tensor entries, we can solve a linear system.
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    all_triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    for attempt in range(max_attempts):
        if _remaining() < 5.0:
            break
        
        # Choose a random support set
        support = rng.sample(all_triples, min(rank, len(all_triples)))
        
        # For each support element, we'll assign a "weight"
        # The constraint is: for each (i,j,k), sum of weights of support
        # elements that "cover" (i,j,k) equals 1
        # 
        # A support element (i',j',k') "covers" (i,j,k) if:
        # i'=i, j'=j, k'=k (exact match) OR we use a more general covering
        #
        # Actually, the tensor decomposition requires:
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
        #
        # If we restrict to rank-one terms where each term has non-zero
        # entries only at specific (i,j), (j,k), (k,i) positions, then
        # term r contributes to T[i,j,k] only if U[i,j,r], V[j,k,r], W[k,i,r]
        # are all non-zero.
        #
        # So if term r has U non-zero only at (i_r, j_r), V non-zero only at
        # (j_r, k_r), W non-zero only at (k_r, i_r), then term r contributes
        # U[i_r,j_r,r]*V[j_r,k_r,r]*W[k_r,i_r,r] to T[i_r,j_r,k_r].
        #
        # This means each term covers exactly one tensor entry!
        # So we need at least n*m*p terms... unless we allow terms to cover
        # multiple entries.
        #
        # Let's try a different approach: allow each term to have multiple
        # non-zero entries in U, V, W.
        
        # Random sparse initialization
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        for r in range(rank):
            # Each term gets 2-4 non-zero entries in each matrix
            n_nonzero_U = rng.randint(2, 4)
            n_nonzero_V = rng.randint(2, 4)
            n_nonzero_W = rng.randint(2, 4)
            
            U_indices = rng.sample(range(n*m), n_nonzero_U)
            V_indices = rng.sample(range(m*p), n_nonzero_V)
            W_indices = rng.sample(range(p*n), n_nonzero_W)
            
            for idx in U_indices:
                U[idx, r] = rng.choice([-1, 1])
            for idx in V_indices:
                V[idx, r] = rng.choice([-1, 1])
            for idx in W_indices:
                W[idx, r] = rng.choice([-1, 1])
        
        # Now run alternating linear solve to converge
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        
        for it in range(100):
            if _remaining() < 3.0:
                break
            
            # Solve for U
            for i in range(n):
                for j in range(m):
                    M = np.zeros((p, rank))
                    for k in range(p):
                        M[k, :] = V3[j, k, :] * W3[k, i, :]
                    try:
                        U3[i, j, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Solve for V
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, rank))
                    for i in range(n):
                        M[i, :] = U3[i, j, :] * W3[k, i, :]
                    try:
                        V3[j, k, :] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Solve for W
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, rank))
                    for j in range(m):
                        M[j, :] = U3[i, j, :] * V3[j, k, :]
                    try:
                        W3[k, i, :] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
            error = np.max(np.abs(T - 1.0))
            
            if error < 1e-10:
                Ua = U3.reshape(n*m, rank)
                Va = V3.reshape(m*p, rank)
                Wa = W3.reshape(p*n, rank)
                
                # Try snapping
                Ua_s = np.vectorize(_snap_to_half)(Ua)
                Va_s = np.vectorize(_snap_to_half)(Va)
                Wa_s = np.vectorize(_snap_to_half)(Wa)
                
                U_out = Ua_s.tolist()
                V_out = Va_s.tolist()
                W_out = Wa_s.tolist()
                
                if _verify_fast(U_out, V_out, W_out, n, m, p):
                    return U_out, V_out, W_out
                
                U_out_raw = Ua.tolist()
                V_out_raw = Va.tolist()
                W_out_raw = Wa.tolist()
                
                if _verify_fast(U_out_raw, V_out_raw, W_out_raw, n, m, p):
                    return U_out_raw, V_out_raw, W_out_raw
        
    return None

def _try_systematic_support(n, m, p, rank, seed=None):
    """
    Systematically try different support patterns.
    Use the structure of the problem to guide the search.
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    # Strategy: use the fact that n=3, m=5, p=6
    # Try to find a pattern where each term covers multiple entries
    
    # Initialize with a structured pattern
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Assign terms in a round-robin fashion
    for r in range(rank):
        i = r % n
        j = (r // n) % m
        k = (r // (n*m)) % p
        
        # Each term primarily covers (i,j,k) but also has secondary entries
        U[i*m+j, r] = 1.0
        
        # Add secondary entries to cover other positions
        for delta in range(1, min(3, m)):
            j2 = (j + delta) % m
            if rng.random() < 0.5:
                U[i*m+j2, r] = rng.choice([-1, 1])
        
        V[j*p+k, r] = 1.0
        for delta in range(1, min(3, p)):
            k2 = (k + delta) % p
            if rng.random() < 0.5:
                V[j*p+k2, r] = rng.choice([-1, 1])
        
        W[k*n+i, r] = 1.0
        for delta in range(1, min(3, n)):
            i2 = (i + delta) % n
            if rng.random() < 0.5:
                W[k*n+i2, r] = rng.choice([-1, 1])
    
    # Run alternating solve
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    for it in range(200):
        if _remaining() < 3.0:
            break
        
        for i in range(n):
            for j in range(m):
                M = np.zeros((p, rank))
                for k in range(p):
                    M[k, :] = V3[j, k, :] * W3[k, i, :]
                try:
                    U3[i, j, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, rank))
                for i in range(n):
                    M[i, :] = U3[i, j, :] * W3[k, i, :]
                try:
                    V3[j, k, :] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                except:
                    pass
        
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, rank))
                for j in range(m):
                    M[j, :] = U3[i, j, :] * V3[j, k, :]
                try:
                    W3[k, i, :] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                except:
                    pass
        
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        error = np.max(np.abs(T - 1.0))
        
        if error < 1e-10:
            Ua = U3.reshape(n*m, rank)
            Va = V3.reshape(m*p, rank)
            Wa = W3.reshape(p*n, rank)
            
            Ua_s = np.vectorize(_snap_to_half)(Ua)
            Va_s = np.vectorize(_snap_to_half)(Va)
            Wa_s = np.vectorize(_snap_to_half)(Wa)
            
            U_out = Ua_s.tolist()
            V_out = Va_s.tolist()
            W_out = Wa_s.tolist()
            
            if _verify_fast(U_out, V_out, W_out, n, m, p):
                return U_out, V_out, W_out
            
            U_out_raw = Ua.tolist()
            V_out_raw = Va.tolist()
            W_out_raw = Wa.tolist()
            
            if _verify_fast(U_out_raw, V_out_raw, W_out_raw, n, m, p):
                return U_out_raw, V_out_raw, W_out_raw
    
    return None

def _try_rank_altsolve(n, m, p, rank, max_iter=500, seed=None, tol=1e-10):
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    U = np.array([[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(n*m)], dtype=float)
    V = np.array([[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(m*p)], dtype=float)
    W = np.array([[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(p*n)], dtype=float)
    
    best_error = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if _remaining() < 2.0:
            break
        
        for i in range(n):
            for j in range(m):
                M = np.zeros((p, rank))
                for k in range(p):
                    M[k, :] = V[j*p + k] * W[k*n + i]
                try:
                    U[i*m + j] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, rank))
                for i in range(n):
                    M[i, :] = U[i*m + j] * W[k*n + i]
                try:
                    V[j*p + k] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                except:
                    pass
        
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, rank))
                for j in range(m):
                    M[j, :] = U[i*m + j] * V[j*p + k]
                try:
                    W[k*n + i] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                except:
                    pass
        
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        error = np.max(np.abs(T - 1.0))
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if error < tol:
            break
    
    if best_error < tol:
        Ua_snapped = np.vectorize(_snap_to_half)(best_U)
        Va_snapped = np.vectorize(_snap_to_half)(best_V)
        Wa_snapped = np.vectorize(_snap_to_half)(best_W)
        
        U_out = Ua_snapped.tolist()
        V_out = Va_snapped.tolist()
        W_out = Wa_snapped.tolist()
        
        if _verify_fast(U_out, V_out, W_out, n, m, p):
            return U_out, V_out, W_out
        
        U_out_raw = best_U.tolist()
        V_out_raw = best_V.tolist()
        W_out_raw = best_W.tolist()
        
        if _verify_fast(U_out_raw, V_out_raw, W_out_raw, n, m, p):
            return U_out_raw, V_out_raw, W_out_raw
    
    return None

def _try_rank_gradient_smooth(n, m, p, rank, max_iter=5000, seed=None):
    """
    Gradient descent with smooth max-norm surrogate using log-sum-exp.
    Uses adaptive learning rate and momentum for better convergence.
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    U = np.array([[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(n*m)], dtype=float)
    V = np.array([[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(m*p)], dtype=float)
    W = np.array([[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(p*n)], dtype=float)
    
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    # Momentum buffers
    mU = np.zeros_like(U3)
    mV = np.zeros_like(V3)
    mW = np.zeros_like(W3)
    
    lr = 0.02
    momentum = 0.9
    beta = 50.0  # Temperature for log-sum-exp (higher = smoother)
    
    best_error = np.inf
    best_U, best_V, best_W = U3.copy(), V3.copy(), W3.copy()
    
    for it in range(max_iter):
        if _remaining() < 2.0:
            break
        
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        err = T - 1.0
        abs_err = np.abs(err)
        
        # Smooth max-norm using log-sum-exp
        max_abs = np.max(abs_err)
        if max_abs > 1e-12:
            # Compute smooth approximation: (1/beta) * log(sum(exp(beta*abs_err)))
            scaled = beta * abs_err
            max_scaled = np.max(scaled)
            log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
            smooth_max = log_sum_exp / beta
            
            # Gradient of smooth max w.r.t. err
            weights = np.exp(scaled - max_scaled) / np.sum(np.exp(scaled - max_scaled))
            grad_smooth = (weights * np.sign(err)) / 1.0  # d/derr of smooth_max
        else:
            smooth_max = 0.0
            grad_smooth = np.zeros_like(err)
        
        error = np.max(abs_err)
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U3.copy(), V3.copy(), W3.copy()
        
        if error < 1e-10:
            break
        
        # Compute gradients using smooth surrogate
        grad_U = np.einsum('ijk,jkr,kir->ijr', grad_smooth, V3, W3)
        grad_V = np.einsum('ijk,ijr,kir->jkr', grad_smooth, U3, W3)
        grad_W = np.einsum('ijk,ijr,jkr->kir', grad_smooth, U3, V3)
        
        # Update with momentum
        mU = momentum * mU + (1 - momentum) * grad_U
        mV = momentum * mV + (1 - momentum) * grad_V
        mW = momentum * mW + (1 - momentum) * grad_W
        
        U3 -= lr * mU
        V3 -= lr * mV
        W3 -= lr * mW
        
        # Adaptive learning rate
        if it % 200 == 0:
            lr *= 0.9
            beta *= 0.95  # Gradually reduce smoothness
        
        # Periodic exact gradient step
        if it % 500 == 499:
            err_exact = T - 1.0
            grad_U_exact = np.einsum('ijk,jkr,kir->ijr', err_exact, V3, W3)
            grad_V_exact = np.einsum('ijk,ijr,kir->jkr', err_exact, U3, W3)
            grad_W_exact = np.einsum('ijk,ijr,jkr->kir', err_exact, U3, V3)
            
            U3 -= 0.01 * grad_U_exact
            V3 -= 0.01 * grad_V_exact
            W3 -= 0.01 * grad_W_exact
    
    Ua = best_U.reshape(n*m, rank)
    Va = best_V.reshape(m*p, rank)
    Wa = best_W.reshape(p*n, rank)
    
    # Try snapping to half-integers
    Ua_snapped = np.vectorize(_snap_to_half)(Ua)
    Va_snapped = np.vectorize(_snap_to_half)(Va)
    Wa_snapped = np.vectorize(_snap_to_half)(Wa)
    
    U_out = Ua_snapped.tolist()
    V_out = Va_snapped.tolist()
    W_out = Wa_snapped.tolist()
    
    if _verify_fast(U_out, V_out, W_out, n, m, p):
        return U_out, V_out, W_out
    
    # Try raw values
    U_out_raw = Ua.tolist()
    V_out_raw = Va.tolist()
    W_out_raw = Wa.tolist()
    
    if _verify_fast(U_out_raw, V_out_raw, W_out_raw, n, m, p):
        return U_out_raw, V_out_raw, W_out_raw
    
    # Final high-precision polish
    if best_error < 0.1:
        Ua_polished, Va_polished, Wa_polished = _polish_high_precision(
            Ua.tolist(), Va.tolist(), Wa.tolist(), n, m, p
        )
        if Ua_polished is not None:
            Ua_snapped = np.vectorize(_snap_to_half)(Ua_polished)
            Va_snapped = np.vectorize(_snap_to_half)(Va_polished)
            Wa_snapped = np.vectorize(_snap_to_half)(Wa_polished)
            
            U_out = Ua_snapped.tolist()
            V_out = Va_snapped.tolist()
            W_out = Wa_snapped.tolist()
            
            if _verify_fast(U_out, V_out, W_out, n, m, p):
                return U_out, V_out, W_out
            
            U_out_raw = Ua_polished.tolist()
            V_out_raw = Va_polished.tolist()
            W_out_raw = Wa_polished.tolist()
            
            if _verify_fast(U_out_raw, V_out_raw, W_out_raw, n, m, p):
                return U_out_raw, V_out_raw, W_out_raw
    
    return None

def _polish_high_precision(U, V, W, n, m, p, max_iter=500, tol=1e-14):
    """
    High-precision alternating least squares polish with very tight tolerance.
    """
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    rank = Ua.shape[1]
    
    for it in range(max_iter):
        if _remaining() < 1.0:
            break
        
        # Solve for U
        for i in range(n):
            for j in range(m):
                M = np.zeros((p, rank))
                for k in range(p):
                    M[k, :] = Va[j*p + k] * Wa[k*n + i]
                try:
                    Ua[i*m + j] = np.linalg.lstsq(M, np.ones(p), rcond=1e-15)[0]
                except:
                    pass
        
        # Solve for V
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, rank))
                for i in range(n):
                    M[i, :] = Ua[i*m + j] * Wa[k*n + i]
                try:
                    Va[j*p + k] = np.linalg.lstsq(M, np.ones(n), rcond=1e-15)[0]
                except:
                    pass
        
        # Solve for W
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, rank))
                for j in range(m):
                    M[j, :] = Ua[i*m + j] * Va[j*p + k]
                try:
                    Wa[k*n + i] = np.linalg.lstsq(M, np.ones(m), rcond=1e-15)[0]
                except:
                    pass
        
        U3 = Ua.reshape(n, m, rank)
        V3 = Va.reshape(m, p, rank)
        W3 = Wa.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        error = np.max(np.abs(T - 1.0))
        
        if error < tol:
            break
    
    return Ua, Va, Wa

def _try_rank_structure(n, m, p, rank, seed=None):
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for r in range(rank):
        i = rng.randint(0, n-1)
        j = rng.randint(0, m-1)
        k = rng.randint(0, p-1)
        
        U[i*m+j, r] = rng.uniform(0.5, 1.5)
        V[j*p+k, r] = rng.uniform(0.5, 1.5)
        W[k*n+i, r] = rng.uniform(0.5, 1.5)
        
        if rng.random() < 0.3:
            i2 = rng.randint(0, n-1)
            j2 = rng.randint(0, m-1)
            k2 = rng.randint(0, p-1)
            U[i2*m+j2, r] += rng.uniform(-0.5, 0.5)
            V[j2*p+k2, r] += rng.uniform(-0.5, 0.5)
            W[k2*n+i2, r] += rng.uniform(-0.5, 0.5)
    
    return U, V, W

def _polish(U, V, W, n, m, p, max_iter=100, tol=1e-12):
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    rank = Ua.shape[1]
    
    for it in range(max_iter):
        if _remaining() < 1.0:
            break
        
        for i in range(n):
            for j in range(m):
                M = np.zeros((p, rank))
                for k in range(p):
                    M[k, :] = Va[j*p + k] * Wa[k*n + i]
                try:
                    Ua[i*m + j] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, rank))
                for i in range(n):
                    M[i, :] = Ua[i*m + j] * Wa[k*n + i]
                try:
                    Va[j*p + k] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                except:
                    pass
        
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, rank))
                for j in range(m):
                    M[j, :] = Ua[i*m + j] * Va[j*p + k]
                try:
                    Wa[k*n + i] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                except:
                    pass
        
        U3 = Ua.reshape(n, m, rank)
        V3 = Va.reshape(m, p, rank)
        W3 = Wa.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        error = np.max(np.abs(T - 1.0))
        
        if error < tol:
            break
    
    return Ua, Va, Wa

def _try_kronecker_structure(n, m, p, rank, seed=None):
    """
    Try a Kronecker-product-based structured initialization.
    Decompose indices into binary-like sub-indices and use structured patterns.
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Try to use a block structure
    # For 3x5x6, we can think of 3 = 1*3, 5 = 1*5, 6 = 2*3
    # Or use binary representations
    
    for r in range(rank):
        # Assign a random "type" to this rank component
        comp_type = rng.randint(0, 3)
        
        if comp_type == 0:
            # Type 0: affects a single (i,j,k)
            i = rng.randint(0, n-1)
            j = rng.randint(0, m-1)
            k = rng.randint(0, p-1)
            U[i*m+j, r] = rng.uniform(0.5, 1.5)
            V[j*p+k, r] = rng.uniform(0.5, 1.5)
            W[k*n+i, r] = rng.uniform(0.5, 1.5)
        elif comp_type == 1:
            # Type 1: affects a row of i (all j, k)
            i = rng.randint(0, n-1)
            for j in range(m):
                for k in range(p):
                    U[i*m+j, r] += rng.uniform(0.1, 0.5)
                    V[j*p+k, r] += rng.uniform(0.1, 0.5)
                    W[k*n+i, r] += rng.uniform(0.1, 0.5)
        elif comp_type == 2:
            # Type 2: affects a column of j
            j = rng.randint(0, m-1)
            for i in range(n):
                for k in range(p):
                    U[i*m+j, r] += rng.uniform(0.1, 0.5)
                    V[j*p+k, r] += rng.uniform(0.1, 0.5)
                    W[k*n+i, r] += rng.uniform(0.1, 0.5)
        else:
            # Type 3: affects a slice of k
            k = rng.randint(0, p-1)
            for i in range(n):
                for j in range(m):
                    U[i*m+j, r] += rng.uniform(0.1, 0.5)
                    V[j*p+k, r] += rng.uniform(0.1, 0.5)
                    W[k*n+i, r] += rng.uniform(0.1, 0.5)
    
    return U, V, W

def _try_low_rank_exact(n, m, p, rank, max_iter=200, seed=None):
    """
    Try to find an exact low-rank decomposition using a combination of
    structured initialization and alternating linear solve.
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    # Try multiple initializations
    for init_type in range(4):
        if _remaining() < 5.0:
            break
        
        if init_type == 0:
            U = np.array([[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(n*m)], dtype=float)
            V = np.array([[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(m*p)], dtype=float)
            W = np.array([[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(p*n)], dtype=float)
        elif init_type == 1:
            U, V, W = _try_rank_structure(n, m, p, rank, seed=seed+init_type)
        elif init_type == 2:
            U, V, W = _try_kronecker_structure(n, m, p, rank, seed=seed+init_type)
        else:
            # Sparse random initialization
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            for r in range(rank):
                for _ in range(rng.randint(1, 3)):
                    i = rng.randint(0, n-1)
                    j = rng.randint(0, m-1)
                    k = rng.randint(0, p-1)
                    U[i*m+j, r] = rng.choice([-1, 1])
                    V[j*p+k, r] = rng.choice([-1, 1])
                    W[k*n+i, r] = rng.choice([-1, 1])
        
        # Run alternating solve
        for it in range(max_iter):
            if _remaining() < 3.0:
                break
            
            for i in range(n):
                for j in range(m):
                    M = np.zeros((p, rank))
                    for k in range(p):
                        M[k, :] = V[j*p + k] * W[k*n + i]
                    try:
                        U[i*m + j] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, rank))
                    for i in range(n):
                        M[i, :] = U[i*m + j] * W[k*n + i]
                    try:
                        V[j*p + k] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, rank))
                    for j in range(m):
                        M[j, :] = U[i*m + j] * V[j*p + k]
                    try:
                        W[k*n + i] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            U3 = U.reshape(n, m, rank)
            V3 = V.reshape(m, p, rank)
            W3 = W.reshape(p, n, rank)
            T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
            error = np.max(np.abs(T - 1.0))
            
            if error < 1e-10:
                U_snapped = np.vectorize(_snap_to_half)(U).tolist()
                V_snapped = np.vectorize(_snap_to_half)(V).tolist()
                W_snapped = np.vectorize(_snap_to_half)(W).tolist()
                if _verify_fast(U_snapped, V_snapped, W_snapped, n, m, p):
                    return U_snapped, V_snapped, W_snapped
                else:
                    return U.tolist(), V.tolist(), W.tolist()
    
    return None

def _strassen_2x2x2():
    """Returns U, V, W for 2x2x2 multiplication with rank 7."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen's 7 multiplications
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    
    # M1 = (A00 + A11) * (B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1; W[1, 0] = -1; W[2, 0] = -1
    
    # M2 = (A10 + A11) * B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[0, 1] = -1; W[2, 1] = 1
    
    # M3 = A00 * (B01 - B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[0, 2] = 1; W[2, 2] = -1
    
    # M4 = A11 * (B10 - B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[1, 3] = 1; W[3, 3] = -1
    
    # M5 = (A00 + A01) * B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = -1; W[3, 4] = 1
    
    # M6 = (A10 - A00) * (B00 + B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[0, 5] = 1; W[1, 5] = 1
    
    # M7 = (A01 - A11) * (B10 + B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[2, 6] = 1; W[3, 6] = 1
    
    return U, V, W

def _try_strassen_init(n, m, p, rank, seed=None):
    """
    Try to initialize using Strassen blocks where possible.
    This provides a structured starting point for the alternating solver.
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
        
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # This is a placeholder for a more complex block construction
    # For now, we just use random init but with a bias towards non-zero values
    # that might help the solver converge faster
    
    for r in range(rank):
        i = rng.randint(0, n-1)
        j = rng.randint(0, m-1)
        k = rng.randint(0, p-1)
        U[i*m+j, r] = rng.uniform(-1, 1)
        V[j*p+k, r] = rng.uniform(-1, 1)
        W[k*n+i, r] = rng.uniform(-1, 1)
        
        # Add some structure
        if rng.random() < 0.5:
            i2 = (i + 1) % n
            j2 = (j + 1) % m
            k2 = (k + 1) % p
            U[i2*m+j2, r] += rng.uniform(-0.5, 0.5)
            V[j2*p+k2, r] += rng.uniform(-0.5, 0.5)
            W[k2*n+i2, r] += rng.uniform(-0.5, 0.5)
            
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    schoolbook = _schoolbook(n, m, p)
    best = schoolbook
    best_rank = n * m * p
    
    # Lower bound for rank: max(n*m, m*p, n*p) = max(15, 30, 18) = 30
    min_rank = max(n*m, m*p, n*p)
    
    # Strategy 1: Perfect hashing approach - try to find sparse support
    found = False
    for attempt in range(20):
        if _remaining() < 20.0:
            break
        
        result = _try_perfect_hash(n, m, p, min_rank, max_attempts=5, seed=min_rank*100+attempt*7)
        if result is not None:
            U, V, W = result
            if _verify_fast(U, V, W, n, m, p):
                best = {"U": U, "V": V, "W": W}
                best_rank = min_rank
                found = True
                break
    
    # Strategy 2: Systematic support patterns
    if not found:
        for attempt in range(15):
            if _remaining() < 15.0:
                break
            
            result = _try_systematic_support(n, m, p, min_rank, seed=min_rank*200+attempt*13)
            if result is not None:
                U, V, W = result
                if _verify_fast(U, V, W, n, m, p):
                    best = {"U": U, "V": V, "W": W}
                    best_rank = min_rank
                    found = True
                    break
    
    # Strategy 3: Smooth gradient descent
    if not found:
        for attempt in range(8):
            if _remaining() < 15.0:
                break
            
            result = _try_rank_gradient_smooth(n, m, p, min_rank, max_iter=3000, seed=min_rank*100+attempt)
            if result is not None:
                U, V, W = result
                if _verify_fast(U, V, W, n, m, p):
                    best = {"U": U, "V": V, "W": W}
                    best_rank = min_rank
                    found = True
                    break
    
    # Strategy 4: Try low-rank exact with alternating solve
    if not found:
        for attempt in range(10):
            if _remaining() < 12.0:
                break
            
            result = _try_low_rank_exact(n, m, p, min_rank, max_iter=200, seed=min_rank*100+attempt)
            if result is not None:
                U, V, W = result
                if _verify_fast(U, V, W, n, m, p):
                    best = {"U": U, "V": V, "W": W}
                    best_rank = min_rank
                    found = True
                    break
    
    # Strategy 5: Try slightly higher ranks if needed
    if not found:
        for rank in range(min_rank + 1, min(min_rank + 8, best_rank + 1)):
            if _remaining() < 12.0:
                break
            
            found = False
            for attempt in range(5):
                if _remaining() < 6.0:
                    break
                
                if attempt % 2 == 0:
                    result = _try_perfect_hash(n, m, p, rank, max_attempts=3, seed=rank*100+attempt*3)
                else:
                    result = _try_low_rank_exact(n, m, p, rank, max_iter=150, seed=rank*100+attempt)
                
                if result is not None:
                    U, V, W = result
                    if _verify_fast(U, V, W, n, m, p):
                        best = {"U": U, "V": V, "W": W}
                        best_rank = rank
                        found = True
                        break
            
            if found:
                break
    
    # Strategy 6: Try altsolve as fallback
    if best_rank == n * m * p:
        for attempt in range(5):
            if _remaining() < 8.0:
                break
            
            result = _try_rank_altsolve(n, m, p, min_rank, max_iter=200, seed=min_rank*200+attempt)
            if result is not None:
                U, V, W = result
                if _verify_fast(U, V, W, n, m, p):
                    best = {"U": U, "V": V, "W": W}
                    best_rank = min_rank
                    break
    
    # Final check: ensure the best result is valid
    if not _verify_fast(best["U"], best["V"], best["W"], n, m, p):
        best = schoolbook
    
    return best
