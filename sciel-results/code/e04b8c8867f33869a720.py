"""
Matrix multiplication 3x4x8 tensor decomposition with low-rank exact solution.

Strategy: Use the Kronecker product structure.
- The 3x4x8 multiplication tensor can be viewed as combining:
  - A 3x2x2 sub-problem (3 rows, 2 inner, 2 outer)
  - A 2x2x4 sub-problem (2 rows, 2 inner, 4 outer)
- We use schoolbook for the 3x2x2 part (rank 12) and Strassen for the 2x2x2 part (rank 7),
  but this doesn't directly work due to dimension mismatches.

Better strategy:
- Note that 4 = 2*2 and 8 = 4*2
- We can decompose the inner dimension j=0..3 into (j1,j2) with j1 in {0,1}, j2 in {0,1}
- And the output dimension k=0..7 into (k1,k2) with k1 in {0..3}, k2 in {0,1}
- For fixed i (0..2), we need to compute C[i, k1, k2] for k1 in 0..3, k2 in 0..1
- This is like computing a 3x4 (i x k1) matrix from A[i, j1, j2] and B[j1, j2, k1, k2]
- But B has shape 4x8 = (2,2)x(4,2), so B[j1,j2,k1,k2] where j1,j2 in {0,1}, k1 in 0..3, k2 in 0..1

Actually, let's think differently. The schoolbook rank is 3*4*8 = 96.
We can use the fact that the tensor is a Kronecker product of smaller tensors.

The 3x4x8 matrix multiplication tensor T satisfies:
T[(i,j), (j',k)] = 1 if j==j' else 0, for i in 0..2, j in 0..3, k in 0..7

This is the standard matrix multiplication tensor for C = A*B where A is 3x4, B is 4x8.

We can write this as a tensor product:
- Decompose j = 2*j1 + j2 where j1 in {0,1}, j2 in {0,1}
- Decompose k = 4*k1 + k2 where k1 in {0,1}, k2 in {0..3}
- Then T is related to the tensor product of a 3x2x2 tensor and a 2x2x4 tensor? 

No, let's just use a direct approach: try to find a rank-r decomposition using
gradient-based optimization with multiple random restarts, and use the known fact that
the minimum rank for matrix multiplication is at least max(n,m,p) = 8, but in practice
for rectangular matrices it's higher.

For 3x4x8, known lower bounds give rank >= 24 (from slice rank arguments).
The schoolbook rank is 96. We aim to find something significantly better.

Let me try a structured approach:
- For each of the 4 values of j, we have a 3x8 sub-tensor that is rank 24 (schoolbook).
- But we can potentially share factors across different j values.

Actually, the simplest exact decomposition with potentially lower rank:
Use the fact that we can combine multiple (i,j,k) triples in each factor.

Let's try: for each j, use a rank-24 decomposition (which is schoolbook for that slice),
but see if we can share columns.

Actually, let me just implement a robust gradient descent approach with many restarts
and hope to find a lower-rank solution. If that fails, return schoolbook.
"""

import time
import numpy as np

DIMENSIONS = [3, 4, 8]
START_TIME = time.time()
TIME_LIMIT = 120.0  # Safety margin before 150s kill

def _build_true_tensor(n, m, p):
    """Build the true matrix multiplication tensor."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify(U, V, W, T_true, atol=1e-8):
    """Verify that U, V, W correctly decompose the tensor."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    return np.allclose(T_recon, T_true, atol=atol)

def _snap_to_half_inters(U, V, W):
    """Try to snap values to nearest half-integer and verify."""
    for scale in [1, 2]:
        U_r = np.round(U * scale) / scale
        V_r = np.round(V * scale) / scale
        W_r = np.round(W * scale) / scale
        return U_r.tolist(), V_r.tolist(), W_r.tolist()
    return U.tolist(), V.tolist(), W.tolist()

def _gradient_descent(U, V, W, T_true, eta, max_iters, tol=1e-10):
    """Run gradient descent to minimize ||T_recon - T_true||^2."""
    n, m, p = T_true.shape
    nm, mp, pn = U.shape[0], V.shape[0], W.shape[0]
    rank = U.shape[1]
    
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        R = T_true - T_recon
        err = np.max(np.abs(R))
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < tol:
            break
        
        # Compute gradients
        grad_U = np.einsum('abc,br,cr->ar', R, V, W)
        grad_V = np.einsum('abc,ar,cr->br', R, U, W)
        grad_W = np.einsum('abc,ar,br->cr', R, U, V)
        
        U += eta * grad_U
        V += eta * grad_V
        W += eta * grad_W
    
    return best_U, best_V, best_W, best_err

def _try_sparse_schoolbook(T_true, nm, mp, pn, target_rank, time_limit):
    """Start from schoolbook and try to find a lower-rank solution by
    randomly selecting a subset of schoolbook terms and optimizing."""
    n, m, p = 3, 4, 8
    schoolbook_rank = n * m * p  # 96
    
    if target_rank >= schoolbook_rank:
        return None
    
    # Build schoolbook factors
    U_sb = np.zeros((nm, schoolbook_rank))
    V_sb = np.zeros((mp, schoolbook_rank))
    W_sb = np.zeros((pn, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    
    # Try multiple random subsets
    for restart in range(5):
        if time.time() - START_TIME > time_limit:
            break
        
        rng = np.random.RandomState(restart * 777 + target_rank)
        
        # Select a random subset of schoolbook columns
        selected = rng.choice(schoolbook_rank, size=target_rank, replace=False)
        U = U_sb[:, selected].copy()
        V = V_sb[:, selected].copy()
        W = W_sb[:, selected].copy()
        
        # Add small random perturbation to help escape local minima
        U += rng.randn(nm, target_rank) * 0.1
        V += rng.randn(mp, target_rank) * 0.1
        W += rng.randn(pn, target_rank) * 0.1
        
        # Optimize
        eta = 0.01
        max_iters = 2000
        for phase in range(3):
            U, V, W, err = _gradient_descent(U, V, W, T_true, eta, max_iters, tol=1e-10)
            if err < 1e-8:
                U_s, V_s, W_s = _snap_to_half_inters(U, V, W)
                if _verify(U_s, V_s, W_s, T_true):
                    return U_s, V_s, W_s
                if _verify(U.tolist(), V.tolist(), W.tolist(), T_true):
                    return U.tolist(), V.tolist(), W.tolist()
            eta *= 0.5
            max_iters = max(100, max_iters // 2)
    
    return None



def construct():
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    T_true = _build_true_tensor(n, m, p)
    
    # Check time
    if time.time() - START_TIME > TIME_LIMIT - 5:
        return schoolbook()
    
    best_result = schoolbook()
    best_rank = n * m * p  # 96
    
    # Strategy 1: Try sparse schoolbook approach (start from schoolbook, select subset)
    candidate_ranks_sparse = list(range(40, best_rank, 8)) + [best_rank - 1]
    candidate_ranks_sparse = sorted(set(candidate_ranks_sparse))
    
    for r in candidate_ranks_sparse:
        if time.time() - START_TIME > TIME_LIMIT - 20:
            break
        if r >= best_rank:
            break
        
        result = _try_sparse_schoolbook(T_true, nm, mp, pn, r, TIME_LIMIT - 20)
        if result is not None:
            U, V, W = result
            if _verify(U.tolist(), V.tolist(), W.tolist(), T_true, atol=1e-6):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Strategy 2: Direct gradient descent with random initialization
    candidate_ranks = list(range(32, best_rank, 8)) + [best_rank - 1]
    candidate_ranks = sorted(set(candidate_ranks))
    
    for r in candidate_ranks:
        if time.time() - START_TIME > TIME_LIMIT - 15:
            break
        if r >= best_rank:
            break
        
        found = False
        num_restarts = 3
        
        for restart in range(num_restarts):
            if time.time() - START_TIME > TIME_LIMIT - 15:
                break
            
            rng = np.random.RandomState(restart * 1000 + r)
            
            # Initialize with small random values
            U = rng.randn(nm, r) * 0.5
            V = rng.randn(mp, r) * 0.5
            W = rng.randn(pn, r) * 0.5
            
            # Adaptive learning rate
            eta = 0.01
            max_iters = 2000
            
            for phase in range(3):
                U, V, W, err = _gradient_descent(U, V, W, T_true, eta, max_iters, tol=1e-10)
                
                if err < 1e-8:
                    # Try to snap to half-integers
                    U_s, V_s, W_s = _snap_to_half_inters(U, V, W)
                    if _verify(U_s, V_s, W_s, T_true):
                        found = True
                        break
                    # Also try without snapping
                    if _verify(U.tolist(), V.tolist(), W.tolist(), T_true):
                        found = True
                        break
                
                # Reduce learning rate for next phase
                eta *= 0.5
                max_iters = max(100, max_iters // 2)
            
            if found:
                break
        
        if found:
            # Final verification with tighter tolerance
            if _verify(U.tolist(), V.tolist(), W.tolist(), T_true, atol=1e-6):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return best_result
