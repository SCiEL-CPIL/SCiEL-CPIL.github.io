import numpy as np
import time
import itertools

DIMENSIONS = [3, 4, 5]
TIME_LIMIT = 140  # safety margin from 150s

def _verify(U, V, W):
    """Check if U,V,W exactly decompose the 3x4x5 matmul tensor."""
    n, m, p = 3, 4, 5
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    T = np.einsum('ur,vr,wr->uvw', U, V, W, optimize=True)
    
    # Expected: T[i*m+l, l*p+k, k*n+i] = 1, else 0
    indices = []
    for i in range(n):
        for l in range(m):
            for k in range(p):
                u_idx = i * m + l
                v_idx = l * p + k
                w_idx = k * n + i
                indices.append((u_idx, v_idx, w_idx))
    
    for idx in indices:
        if abs(T[idx] - 1.0) > 1e-9:
            return False
    
    T_target = np.zeros((n*m, m*p, p*n))
    for idx in indices:
        T_target[idx] = 1.0
        
    if not np.allclose(T, T_target, atol=1e-9):
        return False
    
    return True

def _seed_solution():
    """Return the seed 56-rank solution as U,V,W numpy arrays."""
    n, m, p = 3, 4, 5
    U_parts = []
    V_parts = []
    W_parts = []
    
    def strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        return U, V, W
    
    def schoolbook_2x2x1():
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
        U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
        return U, V, W
    
    def schoolbook_1x2x2():
        U = np.zeros((2, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[2, 1] = 1; W[0, 1] = 1
        U[0, 2] = 1; V[1, 2] = 1; W[1, 2] = 1
        U[1, 3] = 1; V[3, 3] = 1; W[1, 3] = 1
        return U, V, W
    
    def schoolbook_1x2x1():
        U = np.zeros((2, 2))
        V = np.zeros((2, 2))
        W = np.zeros((1, 2))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        return U, V, W
    
    def add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk):
        for r in range(U_blk.shape[1]):
            U_parts.append(np.zeros(n * m))
            for i, idx in enumerate(u_idx):
                U_parts[-1][idx] = U_blk[i, r]
            V_parts.append(np.zeros(m * p))
            for i, idx in enumerate(v_idx):
                V_parts[-1][idx] = V_blk[i, r]
            W_parts.append(np.zeros(p * n))
            for i, idx in enumerate(w_idx):
                W_parts[-1][idx] = W_blk[i, r]
    
    add_block([0,1,4,5], [0,1,5,6], [0,3,1,4], *strassen_2x2x2())
    add_block([0,1,4,5], [2,3,7,8], [6,9,7,10], *strassen_2x2x2())
    add_block([0,1,4,5], [4,9], [12,13], *schoolbook_2x2x1())
    add_block([2,3,6,7], [10,11,15,16], [0,3,1,4], *strassen_2x2x2())
    add_block([2,3,6,7], [12,13,17,18], [6,9,7,10], *strassen_2x2x2())
    add_block([2,3,6,7], [14,19], [12,13], *schoolbook_2x2x1())
    add_block([8,9], [0,1,5,6], [2,5], *schoolbook_1x2x2())
    add_block([8,9], [2,3,7,8], [8,11], *schoolbook_1x2x2())
    add_block([8,9], [4,9], [14], *schoolbook_1x2x1())
    add_block([10,11], [10,11,15,16], [2,5], *schoolbook_1x2x2())
    add_block([10,11], [12,13,17,18], [8,11], *schoolbook_1x2x2())
    add_block([10,11], [14,19], [14], *schoolbook_1x2x1())
    
    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    return U, V, W

def _try_remove_columns(U, V, W):
    """Try to remove one column (rank term) by checking if it's linearly dependent."""
    rank = U.shape[1]
    if rank <= 1:
        return None
    
    for r in range(rank):
        U_test = np.delete(U, r, axis=1)
        V_test = np.delete(V, r, axis=1)
        W_test = np.delete(W, r, axis=1)
        if _verify(U_test, V_test, W_test):
            return U_test, V_test, W_test
    return None

def _try_merge_columns(U, V, W, max_subset_size=3):
    """Try to find a column that can be expressed as a linear combination of others."""
    rank = U.shape[1]
    if rank < 2:
        return None
    
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    # First, try simple: can we express column j as a single other column i (with scaling)?
    for j in range(rank):
        for i in range(rank):
            if i == j:
                continue
            if np.linalg.norm(U[:,i]) > 1e-10:
                a = np.dot(U[:,j], U[:,i]) / np.dot(U[:,i], U[:,i])
                if np.allclose(U[:,j], a * U[:,i], atol=1e-9):
                    if np.allclose(V[:,j], a * V[:,i], atol=1e-9) and np.allclose(W[:,j], a * W[:,i], atol=1e-9):
                        U_test = U.copy()
                        V_test = V.copy()
                        W_test = W.copy()
                        U_test[:,i] += U_test[:,j]
                        V_test[:,i] += V_test[:,j]
                        W_test[:,i] += W_test[:,j]
                        U_test = np.delete(U_test, j, axis=1)
                        V_test = np.delete(V_test, j, axis=1)
                        W_test = np.delete(W_test, j, axis=1)
                        if _verify(U_test, V_test, W_test):
                            return U_test, V_test, W_test
    
    # Try subsets of size 2 and 3
    for j in range(rank):
        other_cols = [i for i in range(rank) if i != j]
        for subset_size in range(2, min(max_subset_size, len(other_cols)) + 1):
            for subset in itertools.combinations(other_cols, subset_size):
                A_mat = np.column_stack([U[:,c] for c in subset])
                try:
                    coeffs, residuals, rnk, sv = np.linalg.lstsq(A_mat, U[:,j], rcond=None)
                    if rnk < subset_size:
                        continue
                    if len(residuals) > 0 and residuals[0] < 1e-9:
                        # Check V and W
                        v_pred = A_mat_v = np.column_stack([V[:,c] for c in subset])
                        v_pred = v_pred @ coeffs
                        w_pred = np.column_stack([W[:,c] for c in subset]) @ coeffs
                        if np.allclose(V[:,j], v_pred, atol=1e-9) and np.allclose(W[:,j], w_pred, atol=1e-9):
                            U_test = U.copy()
                            V_test = V.copy()
                            W_test = W.copy()
                            for k, c in enumerate(subset):
                                U_test[:,c] += coeffs[k] * U_test[:,j]
                                V_test[:,c] += coeffs[k] * V_test[:,j]
                                W_test[:,c] += coeffs[k] * W_test[:,j]
                            U_test = np.delete(U_test, j, axis=1)
                            V_test = np.delete(V_test, j, axis=1)
                            W_test = np.delete(W_test, j, axis=1)
                            if _verify(U_test, V_test, W_test):
                                return U_test, V_test, W_test
                except:
                    pass
    return None

def _randomized_rank_reduction(U, V, W, n_restarts=50, time_budget=60):
    """Try randomized perturbations to find lower-rank equivalent decompositions."""
    rank = U.shape[1]
    best = (U.copy(), V.copy(), W.copy())
    best_rank = rank
    start = time.time()
    
    rng = np.random.RandomState(42)
    
    for restart in range(n_restarts):
        if time.time() - start > time_budget:
            break
        
        # Try to find a random linear transformation that preserves the tensor
        # but might reveal lower-rank structure
        # We try: for a random subset of columns, check if they can be replaced by fewer columns
        
        # Strategy: pick a random column j, try to express it as combination of random subset
        j = rng.randint(rank)
        subset_size = rng.randint(2, min(5, rank - 1) + 1)
        other_cols = [i for i in range(rank) if i != j]
        if len(other_cols) < subset_size:
            continue
        subset = list(rng.choice(other_cols, size=subset_size, replace=False))
        
        A_mat = np.column_stack([U[:,c] for c in subset])
        try:
            coeffs, residuals, rnk, sv = np.linalg.lstsq(A_mat, U[:,j], rcond=None)
            if rnk < subset_size:
                continue
            if len(residuals) > 0 and residuals[0] < 1e-8:
                v_pred = np.column_stack([V[:,c] for c in subset]) @ coeffs
                w_pred = np.column_stack([W[:,c] for c in subset]) @ coeffs
                if np.allclose(V[:,j], v_pred, atol=1e-8) and np.allclose(W[:,j], w_pred, atol=1e-8):
                    U_test = U.copy()
                    V_test = V.copy()
                    W_test = W.copy()
                    for k, c in enumerate(subset):
                        U_test[:,c] += coeffs[k] * U_test[:,j]
                        V_test[:,c] += coeffs[k] * V_test[:,j]
                        W_test[:,c] += coeffs[k] * W_test[:,j]
                    U_test = np.delete(U_test, j, axis=1)
                    V_test = np.delete(V_test, j, axis=1)
                    W_test = np.delete(W_test, j, axis=1)
                    if _verify(U_test, V_test, W_test):
                        best_rank = U_test.shape[1]
                        best = (U_test, V_test, W_test)
        except:
            pass
    
    return best

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Start from seed
    U, V, W = _seed_solution()
    best = (U.copy(), V.copy(), W.copy())
    best_rank = U.shape[1]
    
    # Verify seed
    if not _verify(best[0], best[1], best[2]):
        raise ValueError("Seed solution failed verification")
    
    # Iterative column removal and merging
    for iteration in range(200):
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - 30:
            break
        
        # First try simple removal
        result = _try_remove_columns(best[0], best[1], best[2])
        if result is None:
            # Try merging with larger subsets
            result = _try_merge_columns(best[0], best[1], best[2], max_subset_size=3)
        
        if result is not None:
            best = result
            best_rank = best[0].shape[1]
            # Continue searching
        else:
            # Try randomized approach
            if time.time() - start_time < TIME_LIMIT - 40:
                randomized_result = _randomized_rank_reduction(best[0], best[1], best[2], 
                                                                 n_restarts=30, 
                                                                 time_budget=20)
                if randomized_result[0].shape[1] < best_rank:
                    best = randomized_result
                    best_rank = best[0].shape[1]
                else:
                    break
    
    # Final attempt: try more aggressive merging
    if time.time() - start_time < TIME_LIMIT - 10:
        for iteration in range(50):
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            result = _try_merge_columns(best[0], best[1], best[2], max_subset_size=4)
            if result is not None:
                best = result
                best_rank = best[0].shape[1]
            else:
                break
    
    # Return best solution
    U_out, V_out, W_out = best
    return {
        "U": U_out.tolist(),
        "V": V_out.tolist(),
        "W": W_out.tolist()
    }
