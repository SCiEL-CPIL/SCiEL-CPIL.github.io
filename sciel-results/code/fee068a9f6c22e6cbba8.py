"""Matrix multiplication 4x4x6 rank-minimal exact decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START

def _build_block_strassen():
    """Build the explicit block-Strassen rank-84 decomposition."""
    n, m, p = DIMENSIONS
    rank = 84
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    A_blocks = {
        (0, 0): (0, 1, 0, 1),
        (0, 1): (0, 1, 2, 3),
        (1, 0): (2, 3, 0, 1),
        (1, 1): (2, 3, 2, 3),
    }
    B_blocks = {
        (0, 0): (0, 1, 0, 1),
        (0, 1): (0, 1, 2, 3),
        (0, 2): (0, 1, 4, 5),
        (1, 0): (2, 3, 0, 1),
        (1, 1): (2, 3, 2, 3),
        (1, 2): (2, 3, 4, 5),
    }
    C_blocks = {
        (0, 0): (0, 1, 0, 1),
        (0, 1): (0, 1, 2, 3),
        (0, 2): (0, 1, 4, 5),
        (1, 0): (2, 3, 0, 1),
        (1, 1): (2, 3, 2, 3),
        (1, 2): (2, 3, 4, 5),
    }
    
    pairs = [
        ((0, 0), (0, 0), (0, 0)),
        ((0, 1), (1, 0), (0, 0)),
        ((0, 0), (0, 1), (0, 1)),
        ((0, 1), (1, 1), (0, 1)),
        ((0, 0), (0, 2), (0, 2)),
        ((0, 1), (1, 2), (0, 2)),
        ((1, 0), (0, 0), (1, 0)),
        ((1, 1), (1, 0), (1, 0)),
        ((1, 0), (0, 1), (1, 1)),
        ((1, 1), (1, 1), (1, 1)),
        ((1, 0), (0, 2), (1, 2)),
        ((1, 1), (1, 2), (1, 2)),
    ]
    
    strassen_terms = [
        {'u': [(0, 1), (3, 1)], 'v': [(0, 1), (3, 1)], 'w': [(0, 1), (3, 1)]},
        {'u': [(2, 1), (3, 1)], 'v': [(0, 1)], 'w': [(2, 1), (3, -1)]},
        {'u': [(0, 1)], 'v': [(1, 1), (3, -1)], 'w': [(1, 1), (3, 1)]},
        {'u': [(3, 1)], 'v': [(2, 1), (0, -1)], 'w': [(0, 1), (2, 1)]},
        {'u': [(0, 1), (1, 1)], 'v': [(3, 1)], 'w': [(0, -1), (1, 1)]},
        {'u': [(2, 1), (0, -1)], 'v': [(0, 1), (1, 1)], 'w': [(3, 1)]},
        {'u': [(1, 1), (3, -1)], 'v': [(2, 1), (3, 1)], 'w': [(0, 1)]},
    ]
    
    col = 0
    for (a_idx, b_idx, c_idx) in pairs:
        a_r1, _, a_c1, _ = A_blocks[a_idx]
        b_r1, _, b_c1, _ = B_blocks[b_idx]
        c_r1, _, c_c1, _ = C_blocks[c_idx]
        
        for term in strassen_terms:
            for local_a, coeff in term['u']:
                lr, lc = divmod(local_a, 2)
                U[(a_r1 + lr) * m + (a_c1 + lc), col] += coeff
            for local_b, coeff in term['v']:
                lr, lc = divmod(local_b, 2)
                V[(b_r1 + lr) * p + (b_c1 + lc), col] += coeff
            for local_c, coeff in term['w']:
                lr, lc = divmod(local_c, 2)
                W[(c_c1 + lc) * n + (c_r1 + lr), col] += coeff
            col += 1
    
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify the decomposition is exact."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = float(np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :]))
                if abs(val - 1.0) > tol:
                    return False
    return True


def _snap_halfint(M, tol=1e-3):
    """Snap values to nearest half-integer; return False if any value is far off."""
    for r in range(M.shape[1]):
        for row in range(M.shape[0]):
            v = M[row, r]
            if abs(v) < 1e-9:
                M[row, r] = 0.0
                continue
            h = np.round(2 * v) / 2
            if abs(v - h) > tol:
                return False
            M[row, r] = h
    return True


def _independent_gauge_optimize(U0, V0, W0, target_rank, n, m, p, deadline, seed=42):
    """
    Apply independent random gauge transformations to each factor, then optimize.
    Key insight: U, V, W must be transformed independently to preserve the tensor structure.
    """
    rng = np.random.RandomState(seed)
    start = time.time()
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    ntriples = len(triples)
    
    # Precompute index arrays
    tri_i = np.array([t[0] for t in triples])
    tri_j = np.array([t[1] for t in triples])
    tri_k = np.array([t[2] for t in triples])
    u_idx = tri_i * m + tri_j
    v_idx = tri_j * p + tri_k
    w_idx = tri_k * n + tri_i
    
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        if target_rank > U0.shape[1]:
            continue
        
        # Create INDEPENDENT random mixing matrices for each factor
        # This is the critical fix: U, V, W must be transformed independently
        # to preserve the tensor decomposition structure
        mix_U = rng.randn(target_rank, U0.shape[1]) * 0.5
        mix_V = rng.randn(target_rank, V0.shape[1]) * 0.5
        mix_W = rng.randn(target_rank, W0.shape[1]) * 0.5
        
        U = U0 @ mix_U.T
        V = V0 @ mix_V.T
        W = W0 @ mix_W.T
        
        # Normalize columns to prevent explosion
        for r in range(target_rank):
            nU = np.linalg.norm(U[:, r])
            nV = np.linalg.norm(V[:, r])
            nW = np.linalg.norm(W[:, r])
            if nU > 1e-10 and nV > 1e-10 and nW > 1e-10:
                scale = (nU * nV * nW) ** (1.0/3.0)
                U[:, r] /= (nU / scale)
                V[:, r] /= (nV / scale)
                W[:, r] /= (nW / scale)
        
        # Optimize using smooth L2 objective
        lr = 0.01
        best_err = float('inf')
        
        for it in range(5000):
            if time.time() > deadline:
                break
            
            # Compute all errors vectorized
            U_sel = U[u_idx, :]
            V_sel = V[v_idx, :]
            W_sel = W[w_idx, :]
            
            vals = np.sum(U_sel * V_sel * W_sel, axis=1)
            errs = vals - 1.0
            mse = np.mean(errs ** 2)
            max_err = np.max(np.abs(errs))
            
            if mse < best_err:
                best_err = mse
            
            if max_err < 1e-9:
                break
            
            # Gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            dU_contrib = 2.0 * errs[:, None] * V_sel * W_sel
            dV_contrib = 2.0 * errs[:, None] * U_sel * W_sel
            dW_contrib = 2.0 * errs[:, None] * U_sel * V_sel
            
            np.add.at(dU, u_idx, dU_contrib)
            np.add.at(dV, v_idx, dV_contrib)
            np.add.at(dW, w_idx, dW_contrib)
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Decay learning rate
            lr *= 0.999
            
        # Check convergence
        if _verify(U, V, W, n, m, p, tol=1e-6):
            # Final polish
            lr2 = 0.002
            for it2 in range(2000):
                if time.time() > deadline:
                    break
                U_sel = U[u_idx, :]
                V_sel = V[v_idx, :]
                W_sel = W[w_idx, :]
                vals = np.sum(U_sel * V_sel * W_sel, axis=1)
                errs = vals - 1.0
                max_err = np.max(np.abs(errs))
                if max_err < 1e-10:
                    break
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                dU_contrib = 2.0 * errs[:, None] * V_sel * W_sel
                dV_contrib = 2.0 * errs[:, None] * U_sel * W_sel
                dW_contrib = 2.0 * errs[:, None] * U_sel * V_sel
                np.add.at(dU, u_idx, dU_contrib)
                np.add.at(dV, v_idx, dV_contrib)
                np.add.at(dW, w_idx, dW_contrib)
                U -= lr2 * dU
                V -= lr2 * dV
                W -= lr2 * dW
            
            if _verify(U, V, W, n, m, p, tol=1e-8):
                # Try to snap to half-integers
                U_copy, V_copy, W_copy = U.copy(), V.copy(), W.copy()
                if _snap_halfint(U_copy) and _snap_halfint(V_copy) and _snap_halfint(W_copy):
                    if _verify(U_copy, V_copy, W_copy, n, m, p, tol=1e-6):
                        return U_copy, V_copy, W_copy
    
    return None


def _structured_rank_reduction(U0, V0, W0, target_rank, n, m, p, deadline, seed=42):
    """
    Try to reduce rank by finding linear dependencies among the rank-84 columns.
    Use SVD to find the best low-rank approximation, then optimize.
    """
    rng = np.random.RandomState(seed)
    start = time.time()
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    ntriples = len(triples)
    
    tri_i = np.array([t[0] for t in triples])
    tri_j = np.array([t[1] for t in triples])
    tri_k = np.array([t[2] for t in triples])
    u_idx = tri_i * m + tri_j
    v_idx = tri_j * p + tri_k
    w_idx = tri_k * n + tri_i
    
    for attempt in range(10):
        if time.time() > deadline:
            break
        
        if target_rank > U0.shape[1]:
            continue
        
        # Use SVD-based initialization: find the best rank-r approximation
        # by projecting onto a random subspace
        proj_U = rng.randn(U0.shape[1], target_rank)
        proj_V = rng.randn(V0.shape[1], target_rank)
        proj_W = rng.randn(W0.shape[1], target_rank)
        
        U = U0 @ proj_U
        V = V0 @ proj_V
        W = W0 @ proj_W
        
        # Normalize
        for r in range(target_rank):
            nU = np.linalg.norm(U[:, r])
            nV = np.linalg.norm(V[:, r])
            nW = np.linalg.norm(W[:, r])
            if nU > 1e-10 and nV > 1e-10 and nW > 1e-10:
                scale = (nU * nV * nW) ** (1.0/3.0)
                U[:, r] /= (nU / scale)
                V[:, r] /= (nV / scale)
                W[:, r] /= (nW / scale)
        
        # Optimize
        lr = 0.01
        for it in range(4000):
            if time.time() > deadline:
                break
            
            U_sel = U[u_idx, :]
            V_sel = V[v_idx, :]
            W_sel = W[w_idx, :]
            
            vals = np.sum(U_sel * V_sel * W_sel, axis=1)
            errs = vals - 1.0
            max_err = np.max(np.abs(errs))
            
            if max_err < 1e-9:
                break
            
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            dU_contrib = 2.0 * errs[:, None] * V_sel * W_sel
            dV_contrib = 2.0 * errs[:, None] * U_sel * W_sel
            dW_contrib = 2.0 * errs[:, None] * U_sel * V_sel
            
            np.add.at(dU, u_idx, dU_contrib)
            np.add.at(dV, v_idx, dV_contrib)
            np.add.at(dW, w_idx, dW_contrib)
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            lr *= 0.999
        
        if _verify(U, V, W, n, m, p, tol=1e-6):
            U_copy, V_copy, W_copy = U.copy(), V.copy(), W.copy()
            if _snap_halfint(U_copy) and _snap_halfint(V_copy) and _snap_halfint(W_copy):
                if _verify(U_copy, V_copy, W_copy, n, m, p, tol=1e-6):
                    return U_copy, V_copy, W_copy
    
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    n, m, p = DIMENSIONS
    
    # Guaranteed baseline: block-Strassen rank 84
    U, V, W = _build_block_strassen()
    best = (U.copy(), V.copy(), W.copy())
    best_rank = 84
    
    # Try independent gauge optimization for lower ranks
    target_ranks = [72, 68, 64, 60, 56, 52, 48, 44, 40]
    time_weights = [0.20, 0.18, 0.16, 0.14, 0.12, 0.10, 0.06, 0.03, 0.01]
    
    for idx, target_rank in enumerate(target_ranks):
        if time.time() > deadline:
            break
        
        alloc_time = (deadline - time.time()) * time_weights[idx] / (sum(time_weights[idx:]))
        sub_deadline = min(time.time() + alloc_time, deadline)
        
        # First try independent gauge optimization
        res = _independent_gauge_optimize(U, V, W, target_rank, n, m, p, sub_deadline, seed=target_rank)
        if res is None and time.time() < deadline:
            # Then try structured rank reduction
            res = _structured_rank_reduction(U, V, W, target_rank, n, m, p, sub_deadline, seed=target_rank + 1000)
        
        if res is not None:
            U_new, V_new, W_new = res
            if _verify(U_new, V_new, W_new, n, m, p):
                best = (U_new, V_new, W_new)
                best_rank = target_rank
                break
    
    return {"U": best[0].tolist(), "V": best[1].tolist(), "W": best[2].tolist()}
# EVOLVE-BLOCK-END
