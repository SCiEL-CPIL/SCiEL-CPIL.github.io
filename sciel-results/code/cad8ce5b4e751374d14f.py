"""Rewritten program: search for a lower-rank exact decomposition of the 2x3 * 3x10 matmul tensor."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 10]
N, M, P = DIMENSIONS
RANK_SCHOOLBOOK = N * M * P  # 60

# Deadline guard: keep a safety margin before the 150s wall-clock limit.
DEADLINE = 135.0  # seconds
_start_time = time.time()

def _elapsed():
    return time.time() - _start_time

def _time_left():
    return DEADLINE - _elapsed()

def _verify(U, V, W, tol=1e-6):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Convention: W[k*n + i, r]
    Tensor T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    must equal 1 for all i, j, k (since (A*B)[i,k] = sum_j A[i,j]*B[j,k]).
    """
    rank = U.shape[1]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U.reshape(N, M, rank)
    V3 = V.reshape(M, P, rank)
    W3 = W.reshape(P, N, rank)
    
    # Use einsum: T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    # The matmul tensor is all ones
    expected = np.ones((N, M, P))
    return np.allclose(T, expected, atol=tol)

def _schoolbook():
    """Return the trivial rank-60 schoolbook decomposition."""
    U = np.zeros((N * M, RANK_SCHOOLBOOK))
    V = np.zeros((M * P, RANK_SCHOOLBOOK))
    W = np.zeros((P * N, RANK_SCHOOLBOOK))
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j, col] = 1.0
                V[j * P + k, col] = 1.0
                W[k * N + i, col] = 1.0
                col += 1
    return U, V, W

def _try_random_search(max_rank, n_iter=5000, seed=42):
    """Random search for a decomposition of given rank using integer/half-integer entries."""
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    
    for it in range(n_iter):
        if _time_left() < 5.0:
            break
        
        # Random integer/half-integer entries in range [-2, 2]
        # Values: -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2
        vals = np.array([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
        
        U = np_rng.choice(vals, size=(N * M, max_rank))
        V = np_rng.choice(vals, size=(M * P, max_rank))
        W = np_rng.choice(vals, size=(P * N, max_rank))
        
        if _verify(U, V, W, tol=1e-8):
            return U, V, W
    return None

def _try_structured_search(max_rank):
    """Try structured low-rank decompositions based on known matmul tensor structure."""
    for rank in range(15, max_rank + 1):
        if _time_left() < 10.0:
            break
        result = _try_random_search(rank, n_iter=2000, seed=rank * 1000 + 42)
        if result is not None:
            return result
    return None

def _try_alternating_least_squares(rank, n_iter=200, tol=1e-10, seed=0):
    """
    Alternating Least Squares (ALS) to minimize ||T - sum_r U_r ⊗ V_r ⊗ W_r||_F.
    T is the matmul tensor of shape (N*M, M*P, P*N) with T[i*m+j, j*p+k, k*n+i] = 1.
    We reshape T to (N*M, M*P, P*N) and solve for U (N*M, rank), V (M*P, rank), W (P*N, rank).
    
    The objective is: minimize sum_{a,b,c} (T[a,b,c] - sum_r U[a,r] V[b,r] W[c,r])^2
    This is equivalent to finding a rank-r approximation of the 3-way tensor T.
    """
    np_rng = np.random.RandomState(seed)
    
    # Initialize randomly with small values
    U = np_rng.randn(N * M, rank) * 0.1
    V = np_rng.randn(M * P, rank) * 0.1
    W = np_rng.randn(P * N, rank) * 0.1
    
    # Precompute T as a 3D array
    T = np.zeros((N * M, M * P, P * N))
    for i in range(N):
        for j in range(M):
            for k in range(P):
                a = i * M + j
                b = j * P + k
                c = k * N + i
                T[a, b, c] = 1.0
    
    # T_flat is not needed, we work with the 3D structure via einsum or matrix operations.
    # ALS updates:
    # Fix V, W -> solve for U: U[:,r] = argmin ||T_(1) - U (V_r ⊙ W_r)^T||_F
    # where T_(1) is the mode-1 unfolding of T (shape N*M, M*P * P*N)
    # Actually, it's easier to view the problem as:
    # Minimize || T - sum_r u_r v_r^T w_r^T ||_F
    # This is not standard matrix ALS. We can use the "unfolding" trick.
    # Let's reshape T to (N*M, M*P, P*N).
    # Update U: For each column r of U, we want to minimize || T - sum_s u_s v_s^T w_s^T ||_F
    # This is equivalent to minimizing || T_(1) - U V_(1)^T ||_F where V_(1) is not quite right.
    
    # A better way: use the fact that the tensor is separable in the rank decomposition.
    # Let's use a simpler iterative approach:
    # 1. Fix V, W. Compute the "effective" target for U.
    #    T = sum_r U[:,r] (V[:,r] ⊗ W[:,r])^T ? No.
    #    T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    #    For fixed V, W, define M_r[b,c] = V[b,r] W[c,r]. Then T[a,b,c] = sum_r U[a,r] M_r[b,c].
    #    This is a linear system in U. We can solve for U by flattening the (b,c) indices.
    #    Let T_flat be T reshaped to (N*M, M*P*P*N).
    #    Let B be a matrix of size (M*P*P*N, rank) where B[(b,c), r] = V[b,r] W[c,r].
    #    Then T_flat = U B^T. So U = T_flat B^{-T} (if B is square and invertible) or least squares.
    
    # Precompute B matrix: B has shape (M*P * P*N, rank)
    # B[(b * P*N + c), r] = V[b, r] * W[c, r]
    # This is the outer product of V and W, but with specific index mapping.
    # Actually, B is the Khatri-Rao product of V and W, but transposed/reshaped.
    # Let's construct B explicitly.
    # Size of B: (M*P * P*N) x rank = (30 * 20) x rank = 600 x rank.
    # This is manageable.
    
    B = np.zeros((M * P * P * N, rank))
    for b in range(M * P):
        for c in range(P * N):
            for r in range(rank):
                B[b * P * N + c, r] = V[b, r] * W[c, r]
    
    # T_flat: (N*M, M*P*P*N)
    T_flat = T.reshape(N * M, M * P * P * N)
    
    for it in range(n_iter):
        if _time_left() < 10.0:
            break
        
        # Update U
        # Solve U B^T = T_flat => U = T_flat B^{-T} (least squares)
        # Use np.linalg.lstsq
        try:
            U_new, _, _, _ = np.linalg.lstsq(B.T, T_flat.T, rcond=None)
            U_new = U_new.T
        except:
            break
        
        # Update V
        # Similarly, T_flat2 = T reshaped to (M*P, N*M * P*N)
        # C[(a, c), r] = U[a, r] W[c, r]
        C = np.zeros((N * M * P * N, rank))
        for a in range(N * M):
            for c in range(P * N):
                for r in range(rank):
                    C[a * P * N + c, r] = U_new[a, r] * W[c, r]
        
        T_flat2 = T.reshape(M * P, N * M * P * N)
        try:
            V_new, _, _, _ = np.linalg.lstsq(C.T, T_flat2.T, rcond=None)
            V_new = V_new.T
        except:
            break
        
        # Update W
        # T_flat3 = T reshaped to (P*N, N*M * M*P)
        # D[(a, b), r] = U[a, r] V[b, r]
        D = np.zeros((N * M * M * P, rank))
        for a in range(N * M):
            for b in range(M * P):
                for r in range(rank):
                    D[a * M * P + b, r] = U_new[a, r] * V_new[b, r]
        
        T_flat3 = T.reshape(P * N, N * M * M * P)
        try:
            W_new, _, _, _ = np.linalg.lstsq(D.T, T_flat3.T, rcond=None)
            W_new = W_new.T
        except:
            break
        
        U, V, W = U_new, V_new, W_new
        
        # Check convergence
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        if np.allclose(T_approx, T, atol=tol):
            break
    
    # Snap to nearest half-integers? No, the problem allows any real numbers, but says "integer/half-integer entries".
    # Wait, the prompt says "with integer/half-integer entries".
    # So we must snap the result to the nearest half-integer.
    # However, ALS might not converge to a half-integer solution.
    # If we snap, the decomposition might be invalid.
    # Let's check if the ALS solution is close to a half-integer solution.
    
    # For now, let's just return the ALS solution and let the verifier check it.
    # If it's not valid, we discard it.
    # But the requirement is "integer/half-integer entries".
    # So we should snap U, V, W to the nearest half-integers.
    U_snap = np.round(U * 2) / 2
    V_snap = np.round(V * 2) / 2
    W_snap = np.round(W * 2) / 2
    
    if _verify(U_snap, V_snap, W_snap, tol=1e-6):
        return U_snap, V_snap, W_snap
    
    # If snapping fails, try the raw solution (though it might violate the entry constraint)
    # But the problem statement says "with integer/half-integer entries".
    # So we should only return if it satisfies the constraint.
    # If ALS doesn't find a half-integer solution, we return None.
    
    return None

def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 * 3x10 matrix multiplication."""
    global _start_time
    _start_time = time.time()
    
    # First, get the schoolbook solution as a fallback
    U_fb, V_fb, W_fb = _schoolbook()
    
    best = (U_fb, V_fb, W_fb)
    best_rank = RANK_SCHOOLBOOK
    
    # Try ALS for lower ranks
    # Start from a reasonable lower bound
    # Theoretical lower bound is max(n*m, m*p, p*n) = 30? No, that's not a lower bound for tensor rank.
    # But let's try ranks from 30 up to 59.
    # We'll try a few ranks and a few random initializations for each.
    
    for rank in range(30, best_rank):
        if _time_left() < 15.0:
            break
        
        # Try a few random initializations for this rank
        for seed in range(5):
            if _time_left() < 10.0:
                break
            
            result = _try_alternating_least_squares(rank, n_iter=50, tol=1e-10, seed=seed)
            if result is not None:
                U, V, W = result
                if _verify(U, V, W, tol=1e-8):
                    best = (U, V, W)
                    best_rank = rank
                    break
        
        if best_rank < RANK_SCHOOLBOOK:
            # Found a valid decomposition, return it
            # We could continue to search for even lower ranks, but let's be safe.
            # Actually, let's try to go lower if time permits.
            # But for now, break to save time.
            pass
    
    # Also try random search for very low ranks (just in case)
    for rank in range(20, best_rank):
        if _time_left() < 15.0:
            break
        result = _try_random_search(rank, n_iter=1000, seed=rank * 7919 + 13)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, tol=1e-8):
                best = (U, V, W)
                best_rank = rank
                break
    
    U, V, W = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
