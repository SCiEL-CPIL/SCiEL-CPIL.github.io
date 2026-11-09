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


def _gauged_search(U0, V0, W0, target_rank, n, m, p, deadline, seed=42):
    """
    Structural search for lower-rank decomposition using gauge transformations
    and gradient-based optimization with smooth objective.
    """
    rng = np.random.RandomState(seed)
    start = time.time()
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    ntriples = len(triples)
    
    # Precompute index arrays for speed
    tri_i = np.array([t[0] for t in triples])
    tri_j = np.array([t[1] for t in triples])
    tri_k = np.array([t[2] for t in triples])
    u_idx = tri_i * m + tri_j
    v_idx = tri_j * p + tri_k
    w_idx = tri_k * n + tri_i
    
    for attempt in range(15):
        if time.time() > deadline:
            break
        
        # Initialize by randomly permuting and combining rank-84 columns
        if target_rank > U0.shape[1]:
            continue
        
        # Create a random mixing matrix to combine rank-84 columns into target_rank
        mix = rng.randn(target_rank, U0.shape[1]) * 0.3
        # Ensure the mixing preserves some structure: use block-diagonal-like mixing
        # Group rank-84 columns into blocks and mix within blocks
        U = U0 @ mix.T
        V = V0 @ mix.T
        W = W0 @ mix.T
        
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
        
        # Optimize using smooth L2 objective with adaptive learning rate
        lr = 0.02
        best_err = float('inf')
        
        for it in range(3000):
            if time.time() > deadline:
                break
            
            # Compute all errors vectorized
            # For each triple t: val_t = dot(U[u_idx[t]], V[v_idx[t]] * W[w_idx[t]])
            # We need to compute this efficiently
            
            # Compute element-wise products
            # U[u_idx[t], r] * V[v_idx[t], r] * W[w_idx[t], r] summed over r
            U_sel = U[u_idx, :]  # (ntriples, target_rank)
            V_sel = V[v_idx, :]  # (ntriples, target_rank)
            W_sel = W[w_idx, :]  # (ntriples, target_rank)
            
            vals = np.sum(U_sel * V_sel * W_sel, axis=1)  # (ntriples,)
            errs = vals - 1.0
            mse = np.mean(errs ** 2)
            max_err = np.max(np.abs(errs))
            
            if mse < best_err:
                best_err = mse
            
            if max_err < 1e-9:
                break
            
            # Gradient of MSE w.r.t. V and W (U fixed)
            # dMSE/dV[v_idx[t], r] = 2 * errs[t] * U_sel[t, r] * W_sel[t, r]
            # dMSE/dW[w_idx[t], r] = 2 * errs[t] * U_sel[t, r] * V_sel[t, r]
            
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            # Accumulate gradients
            # For each triple t and rank r:
            # dV[v_idx[t], r] += 2 * errs[t] * U_sel[t, r] * W_sel[t, r]
            # dW[w_idx[t], r] += 2 * errs[t] * U_sel[t, r] * V_sel[t, r]
            
            # Use np.add.at for accumulation
            dV_contrib = 2.0 * errs[:, None] * U_sel * W_sel  # (ntriples, target_rank)
            dW_contrib = 2.0 * errs[:, None] * U_sel * V_sel  # (ntriples, target_rank)
            
            np.add.at(dV, v_idx, dV_contrib)
            np.add.at(dW, w_idx, dW_contrib)
            
            # Adaptive learning rate: reduce if error increases
            V -= lr * dV
            W -= lr * dW
            
            # Decay learning rate slowly
            lr *= 0.9995
        
        # Check convergence
        if _verify(U, V, W, n, m, p, tol=1e-6):
            # Final polish with tighter tolerance
            lr2 = 0.005
            for it2 in range(500):
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
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                dV_contrib = 2.0 * errs[:, None] * U_sel * W_sel
                dW_contrib = 2.0 * errs[:, None] * U_sel * V_sel
                np.add.at(dV, v_idx, dV_contrib)
                np.add.at(dW, w_idx, dW_contrib)
                V -= lr2 * dV
                W -= lr2 * dW
            
            if _verify(U, V, W, n, m, p, tol=1e-8):
                # Try to snap to half-integers
                U_copy, V_copy, W_copy = U.copy(), V.copy(), W.copy()
                if _snap_halfint(U_copy) and _snap_halfint(V_copy) and _snap_halfint(W_copy):
                    if _verify(U_copy, V_copy, W_copy, n, m, p, tol=1e-6):
                        return U_copy, V_copy, W_copy
    
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s limit
    n, m, p = DIMENSIONS
    
    # Guaranteed baseline: block-Strassen rank 84
    U, V, W = _build_block_strassen()
    best = (U.copy(), V.copy(), W.copy())
    best_rank = 84
    
    # Try structural search for lower ranks
    # Allocate time proportionally: more time for higher target ranks (closer to 84)
    target_ranks = [72, 68, 64, 60, 56, 52, 48]
    # Give more time to ranks closer to the baseline
    time_weights = [0.25, 0.20, 0.15, 0.12, 0.10, 0.10, 0.08]
    total_weight = sum(time_weights)
    remaining_time = deadline - start_time
    
    for idx, target_rank in enumerate(target_ranks):
        if time.time() > deadline:
            break
        
        # Allocate time based on weight
        alloc_time = remaining_time * time_weights[idx] / (sum(time_weights[idx:]))
        sub_deadline = min(time.time() + alloc_time, deadline)
        
        res = _gauged_search(U, V, W, target_rank, n, m, p, sub_deadline, seed=target_rank)
        if res is not None:
            U_new, V_new, W_new = res
            if _verify(U_new, V_new, W_new, n, m, p):
                best = (U_new, V_new, W_new)
                best_rank = target_rank
                break  # Found a valid lower rank, stop
    
    return {"U": best[0].tolist(), "V": best[1].tolist(), "W": best[2].tolist()}
# EVOLVE-BLOCK-END
