"""Tensor decomposition search for 3x4 * 4x6 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 6]
N, M, P = DIMENSIONS
RANK_SCHOOLBOOK = N * M * P  # 72
DEADLINE = 130.0  # safety margin before 150s kill


def _build_tensors(rank):
    """Return (A, B, C) as (n*m*p, rank) arrays for the matrix mult tensor."""
    A = np.zeros((N * M * P, rank))
    B = np.zeros((N * M * P, rank))
    C = np.zeros((N * M * P, rank))
    for i in range(N):
        for j in range(M):
            for k in range(P):
                idx = i * M * P + j * P + k
                A[idx, 0] = 1.0
                B[idx, 0] = 1.0
                C[idx, 0] = 1.0
    # Actually we need A[i,j,c], B[j,k,c], C[k,i,c] but flattened differently.
    # Let me redo: the tensor T[i,j,k] = e_i e_j^T e_k^T.
    # Decomposition: T = sum_c u_c ⊗ v_c ⊗ w_c
    # A[i*m+j, c] = u_c[i*m+j], B[j*p+k, c] = v_c[j*p+k], C[k*n+i, c] = w_c[k*n+i]
    # T[i,j,k] = sum_c A[i*m+j,c]*B[j*p+k,c]*C[k*n+i,c]
    # For schoolbook: A[i*m+j, i*m*p+j*p+k]=1, B[j*p+k, ...]=1, C[k*n+i,...]=1
    # So T[i,j,k] = 1 for the matching (i,j,k) term.
    return None  # placeholder, we'll verify differently


def verify(U, V, W, rank):
    """Check if decomposition is exact. Returns True if valid."""
    n, m, p = N, M, P
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # and T[i,j,k] = 1 for all (i,j,k) in the multiplication tensor
    # Wait - the matrix multiplication tensor is T[i,j,k] = 1 if the (i,j) entry of A and (j,k) of B
    # contribute to (i,k) of C. So T[i,j,k] = 1 for all valid (i,j,k).
    # Actually the tensor for matrix mult is: T(i,j,k) = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # because C[i,k] = sum_j A[i,j]*B[j,k], so each (i,j,k) triple has coefficient 1.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def verify_fast(U, V, W, rank):
    """Faster verification using numpy."""
    n, m, p = N, M, P
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Build the full tensor T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # This is a 3-way contraction.
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    # T[i,j,k] = sum_c Ur[i,j,c] * Vr[j,k,c] * Wr[k,i,c]
    # Use einsum
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    # T should be all ones
    return np.allclose(T, 1.0, atol=1e-9)


def residual_fast(U, V, W, rank):
    """Compute the residual tensor T[i,j,k] - 1 for error measurement."""
    n, m, p = N, M, P
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    return T - 1.0


def schoolbook():
    """Return the schoolbook decomposition of rank n*m*p."""
    n, m, p = N, M, P
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
    return {"U": U, "V": V, "W": W}


def try_gradient_descent(target_rank, max_iters, time_budget, seed=42):
    """Try gradient descent to minimize ||T - 1||_F^2 for given rank with vectorized gradients."""
    n, m, p = N, M, P
    start = time.time()
    rng = np.random.RandomState(seed)
    
    best_res = float('inf')
    best_U = best_V = best_W = None
    
    n_attempts = max(5, int(time_budget / 2))
    for attempt in range(n_attempts):
        if time.time() - start > time_budget or time.time() > DEADLINE:
            break
        
        # Initialize with various scales
        scale = 0.3 + 0.4 * (attempt % 3) / 2.0
        U = rng.randn(n * m, target_rank) * scale
        V = rng.randn(m * p, target_rank) * scale
        W = rng.randn(p * n, target_rank) * scale
        
        lr = 0.05
        prev_res_sq = float('inf')
        stalled = 0
        
        for it in range(max_iters):
            if time.time() - start > time_budget or time.time() > DEADLINE:
                break
            
            Ur = U.reshape(n, m, target_rank)
            Vr = V.reshape(m, p, target_rank)
            Wr = W.reshape(p, n, target_rank)
            
            # Compute T[i,j,k] = sum_c Ur[i,j,c]*Vr[j,k,c]*Wr[k,i,c]
            T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
            res = T - 1.0
            res_sq = float(np.sum(res * res))
            
            if res_sq < best_res:
                best_res = res_sq
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if res_sq < 1e-12:
                break
            
            # Vectorized gradient computation
            # grad_U[i*m+j, c] = sum_k res[i,j,k] * V[j*p+k, c] * W[k*n+i, c]
            # grad_V[j*p+k, c] = sum_i res[i,j,k] * U[i*m+j, c] * W[k*n+i, c]
            # grad_W[k*n+i, c] = sum_j res[i,j,k] * U[i*m+j, c] * V[j*p+k, c]
            
            # For grad_U: res[i,j,k] * V[j,k,c] * W[k,i,c]
            # Reshape: res (n,m,p), V (m,p,r), W (p,n,r)
            # grad_U[i,j,c] = einsum('ijk,jkc,kic->ijc', res, Vr, Wr)
            grad_U_3d = np.einsum('ijk,jkc,kic->ijc', res, Vr, Wr)
            grad_U = grad_U_3d.reshape(n * m, target_rank)
            
            # grad_V[j,k,c] = einsum('ijk,ijc,kic->jkc', res, Ur, Wr)
            grad_V_3d = np.einsum('ijk,ijc,kic->jkc', res, Ur, Wr)
            grad_V = grad_V_3d.reshape(m * p, target_rank)
            
            # grad_W[k,i,c] = einsum('ijk,ijc,jkc->kic', res, Ur, Vr)
            grad_W_3d = np.einsum('ijk,ijc,jkc->kic', res, Ur, Vr)
            grad_W = grad_W_3d.reshape(p * n, target_rank)
            
            # Adaptive learning rate
            if res_sq > prev_res_sq:
                lr *= 0.5
                stalled += 1
            else:
                lr = min(lr * 1.05, 0.1)
                stalled = 0
            
            prev_res_sq = res_sq
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Break if stalled too much
            if stalled > 50:
                break
    
    if best_res < 1e-10:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    return None


def try_algebraic_search(target_rank, time_budget):
    """Try structured algebraic approaches for lower rank."""
    n, m, p = N, M, P
    start = time.time()
    
    # Try gradient descent with more iterations
    remaining = min(time_budget - (time.time() - start), 40)
    if remaining > 5:
        result = try_gradient_descent(target_rank, 1000, remaining)
        if result is not None:
            return result
    
    # Try random with restricted values for small ranks
    if target_rank <= 45:
        rng = np.random.RandomState(123)
        for trial in range(100):
            if time.time() - start > time_budget or time.time() > DEADLINE:
                break
            
            U = rng.randint(-2, 3, size=(n * m, target_rank)).astype(float)
            V = rng.randint(-2, 3, size=(m * p, target_rank)).astype(float)
            W = rng.randint(-2, 3, size=(p * n, target_rank)).astype(float)
            
            if verify_fast(U, V, W, target_rank):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None


def try_random_search(max_rank, time_budget):
    """Try random search for a decomposition of rank max_rank."""
    n, m, p = N, M, P
    start = time.time()
    rng = np.random.RandomState(42)
    
    # Use iterative refinement: start with random matrices and try to project
    # onto the constraint manifold.
    
    for attempt in range(200):
        if time.time() - start > time_budget:
            break
        if time.time() > DEADLINE:
            break
        
        # Random initialization
        U = rng.randn(n * m, max_rank)
        V = rng.randn(m * p, max_rank)
        W = rng.randn(p * n, max_rank)
        
        # Try a few iterations of gradient-like updates
        # The constraint is: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # We can try to use the fact that if we fix two, we can solve for the third.
        
        # Alternative: use the approach of trying to find rank-r decomposition
        # by solving a system of equations.
        
        # For small ranks, try brute force with structured entries.
        # Try entries from {-1, 0, 1} or {-1, 0, 1, 2}
        
        if max_rank <= 40:
            # Try random integer entries
            for trial in range(50):
                if time.time() > DEADLINE:
                    break
                U = rng.randint(-2, 3, size=(n * m, max_rank))
                V = rng.randint(-2, 3, size=(m * p, max_rank))
                W = rng.randint(-2, 3, size=(p * n, max_rank))
                if verify_fast(U, V, W, max_rank):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None


def try_structured_search(time_budget):
    """Try structured/analytic approaches."""
    n, m, p = N, M, P
    start = time.time()
    
    # Try decreasing ranks with increasing time allocation
    # Focus on ranks that might be achievable
    target_ranks = [68, 66, 64, 62, 60, 58, 56, 54, 52, 50]
    
    for target_rank in target_ranks:
        if time.time() - start > time_budget or time.time() > DEADLINE:
            break
        remaining = time_budget - (time.time() - start)
        if remaining < 5:
            break
        # Allocate more time to higher ranks (easier to find)
        alloc = min(remaining * 0.3, 15)
        result = try_algebraic_search(target_rank, alloc)
        if result is not None:
            return result
    
    return None


def construct():
    start_time = time.time()
    
    # Start with schoolbook as fallback
    best = schoolbook()
    best_rank = RANK_SCHOOLBOOK
    
    # Try to find better decompositions
    # Strategy 1: Algebraic/gradient search for lower rank
    remaining = min(100, DEADLINE - (time.time() - start_time) - 10)
    if remaining > 5:
        result = try_structured_search(remaining)
        if result is not None:
            best = result
            best_rank = len(result["U"][0])
    
    # Strategy 2: If we still have time, try more aggressive search
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 15 and best_rank > 60:
        for target_rank in [best_rank - 2, best_rank - 4, best_rank - 6]:
            if time.time() > DEADLINE - 10:
                break
            result = try_random_search(target_rank, 8)
            if result is not None:
                best = result
                best_rank = target_rank
                break
    
    # Verify the best result
    if not verify_fast(best["U"], best["V"], best["W"], len(best["U"][0])):
        # Fallback to schoolbook
        best = schoolbook()
    
    return best
