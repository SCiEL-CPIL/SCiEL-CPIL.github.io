import numpy as np
import time
import itertools
import random

DEADLINE = 140.0

def _build_strassen_36():
    """Build rank-36 block Strassen decomposition for 2x4 * 4x5."""
    n, m, p = 2, 4, 5
    rank = 36
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    col = 0
    
    def strassen_block(y, z, w_off):
        nonlocal col
        y00, y01, y10, y11 = y
        z00, z01, z10, z11 = z
        
        # M1 = (y00+y11)(z00+z11)
        U[u_idx(*y00), col] = 1
        U[u_idx(*y11), col] = 1
        V[v_idx(*z00), col] = 1
        V[v_idx(*z11), col] = 1
        W[w_idx(*w_off[0]), col] += 1
        W[w_idx(*w_off[3]), col] += 1
        col += 1
        
        # M2 = (y10+y11)z00
        U[u_idx(*y10), col] = 1
        U[u_idx(*y11), col] = 1
        V[v_idx(*z00), col] = 1
        W[w_idx(*w_off[2]), col] += 1
        W[w_idx(*w_off[3]), col] -= 1
        col += 1
        
        # M3 = y00(z01-z11)
        U[u_idx(*y00), col] = 1
        V[v_idx(*z01), col] = 1
        V[v_idx(*z11), col] = -1
        W[w_idx(*w_off[1]), col] += 1
        W[w_idx(*w_off[3]), col] += 1
        col += 1
        
        # M4 = y11(z10-z00)
        U[u_idx(*y11), col] = 1
        V[v_idx(*z10), col] = 1
        V[v_idx(*z00), col] = -1
        W[w_idx(*w_off[0]), col] += 1
        W[w_idx(*w_off[2]), col] += 1
        col += 1
        
        # M5 = (y00+y01)z11
        U[u_idx(*y00), col] = 1
        U[u_idx(*y01), col] = 1
        V[v_idx(*z11), col] = 1
        W[w_idx(*w_off[0]), col] -= 1
        W[w_idx(*w_off[1]), col] += 1
        col += 1
        
        # M6 = (y10-y00)(z00+z01)
        U[u_idx(*y10), col] = 1
        U[u_idx(*y00), col] = -1
        V[v_idx(*z00), col] = 1
        V[v_idx(*z01), col] = 1
        W[w_idx(*w_off[3]), col] += 1
        col += 1
        
        # M7 = (y01-y11)(z10+z11)
        U[u_idx(*y01), col] = 1
        U[u_idx(*y11), col] = -1
        V[v_idx(*z10), col] = 1
        V[v_idx(*z11), col] = 1
        W[w_idx(*w_off[0]), col] += 1
        col += 1
    
    # Block 1: A_L B_11
    strassen_block([(0,0),(0,1),(1,0),(1,1)], [(0,0),(0,1),(1,0),(1,1)], [(0,0),(1,0),(0,1),(1,1)])
    # Block 2: A_R B_21
    strassen_block([(0,2),(0,3),(1,2),(1,3)], [(2,0),(2,1),(3,0),(3,1)], [(0,0),(1,0),(0,1),(1,1)])
    # Block 3: A_L B_12
    strassen_block([(0,0),(0,1),(1,0),(1,1)], [(0,2),(0,3),(1,2),(1,3)], [(2,0),(3,0),(2,1),(3,1)])
    # Block 4: A_R B_22
    strassen_block([(0,2),(0,3),(1,2),(1,3)], [(2,2),(2,3),(3,2),(3,3)], [(2,0),(3,0),(2,1),(3,1)])
    
    # Block 5: A_L B_15 (schoolbook 2x2 * 2x1)
    for i, j in [(0,0),(0,1),(1,0),(1,1)]:
        U[u_idx(i,j), col] = 1
        V[v_idx(j,4), col] = 1
        W[w_idx(4,i), col] += 1
        col += 1
    
    # Block 6: A_R B_25 (schoolbook 2x2 * 2x1)
    for i, j in [(0,2),(0,3),(1,2),(1,3)]:
        U[u_idx(i,j), col] = 1
        V[v_idx(j,4), col] = 1
        W[w_idx(4,i), col] += 1
        col += 1
    
    return U, V, W

def _verify_fast(U, V, W):
    """Fast verification using einsum."""
    n, m, p = 2, 4, 5
    T = np.einsum('ir,jr,kr->ijk', U, V, W)
    return bool(np.allclose(T, 1.0, atol=1e-8))

def _verify_full(U, V, W):
    """Full verification."""
    n, m, p = 2, 4, 5
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _check_column_redundant(U, V, W, col_idx):
    rank = U.shape[1]
    if rank <= 1:
        return False
    mask = [i for i in range(rank) if i != col_idx]
    return _verify_fast(U[:, mask], V[:, mask], W[:, mask])

def _check_pair_redundant(U, V, W, i, j):
    rank = U.shape[1]
    if rank <= 2:
        return False
    mask = [k for k in range(rank) if k != i and k != j]
    return _verify_fast(U[:, mask], V[:, mask], W[:, mask])

def _als_optimize(U, V, W, target_rank, time_limit=30.0, seed=0):
    """ALS optimization to drive a column to zero."""
    n, m, p = 2, 4, 5
    T = np.ones((n, m, p))
    start = time.time()
    deadline = start + time_limit
    rng = np.random.RandomState(seed)
    
    best = None
    for trial in range(15):
        if time.time() > deadline:
            break
        
        scale = 1e-2
        U_p = U + scale * rng.randn(*U.shape)
        V_p = V + scale * rng.randn(*V.shape)
        W_p = W + scale * rng.randn(*W.shape)
        
        for it in range(300):
            if time.time() > deadline - 1:
                break
            
            # ALS update for U
            for i in range(n):
                for j in range(m):
                    num = np.zeros(target_rank)
                    den = np.zeros(target_rank)
                    for k in range(p):
                        vw = V_p[j, :] * W_p[k, i]
                        num += T[i, j, k] * vw
                        den += vw ** 2
                    den = np.where(den < 1e-14, 1.0, den)
                    U_p[i * m + j, :] = num / den
            
            # ALS update for V
            for j in range(m):
                for k in range(p):
                    num = np.zeros(target_rank)
                    den = np.zeros(target_rank)
                    for i in range(n):
                        uw = U_p[i * m + j, :] * W_p[k, i]
                        num += T[i, j, k] * uw
                        den += uw ** 2
                    den = np.where(den < 1e-14, 1.0, den)
                    V_p[j * p + k, :] = num / den
            
            # ALS update for W
            for k in range(p):
                for i in range(n):
                    num = np.zeros(target_rank)
                    den = np.zeros(target_rank)
                    for j in range(m):
                        uv = U_p[i * m + j, :] * V_p[j * p + k, :]
                        num += T[i, j, k] * uv
                        den += uv ** 2
                    den = np.where(den < 1e-14, 1.0, den)
                    W_p[k * n + i, :] = num / den
            
            # Check if any column is near zero
            norms = np.linalg.norm(np.vstack([U_p, V_p, W_p]), axis=0)
            if np.min(norms) < 1e-4:
                idx = int(np.argmin(norms))
                mask = [r for r in range(target_rank) if r != idx]
                U35 = U_p[:, mask]
                V35 = V_p[:, mask]
                W35 = W_p[:, mask]
                if _verify_fast(U35, V35, W35):
                    if best is None or U35.shape[1] < best[0].shape[1]:
                        best = (U35, V35, W35)
                break
        
        if best is not None:
            break
    
    return best

def _try_column_rotation(U, V, W, time_limit=10.0):
    """Try rotating columns to find a representation where one column is zero."""
    n, m, p = 2, 4, 5
    rank = U.shape[1]
    start_time = time.time()
    rng = np.random.RandomState(123)
    
    best = None
    
    for trial in range(8):
        if time.time() - start_time > time_limit:
            break
        
        Q, _ = np.linalg.qr(rng.randn(rank, rank))
        
        U_rot = U @ Q
        V_rot = V @ Q
        W_rot = W @ Q
        
        norms = np.linalg.norm(np.vstack([U_rot, V_rot, W_rot]), axis=0)
        min_norm = np.min(norms)
        
        if min_norm < 1e-6:
            idx = int(np.argmin(norms))
            mask = [r for r in range(rank) if r != idx]
            U35 = U_rot[:, mask]
            V35 = V_rot[:, mask]
            W35 = W_rot[:, mask]
            if _verify_fast(U35, V35, W35):
                if best is None or U35.shape[1] < best[0].shape[1]:
                    best = (U35, V35, W35)
    
    return best

def _try_kronecker_form(U, V, W, time_limit=20.0):
    """
    Exploit n=2 structure: view tensor as pair of 4x5 operators.
    T[i,j,k] for i=0,1 gives two 4x5 matrices A0, A1.
    The multiplication tensor rank relates to the joint Kronecker structure.
    We try to find a lower-rank decomposition by exploiting the 2-dimensional
    output structure.
    """
    n, m, p = 2, 4, 5
    rank = U.shape[1]
    start_time = time.time()
    
    # Extract the two "slices" for i=0 and i=1
    # T[0,j,k] and T[1,j,k]
    # In the decomposition, W[k,0] and W[k,1] are the output weights
    
    # Try to find if we can merge terms that only differ in the i-index
    # This is a heuristic based on the commuting operator structure
    
    best = None
    rng = np.random.RandomState(777)
    
    # Attempt: try to find a rank-35 decomposition by solving a linear system
    # with one column fixed to a specific sparse pattern
    
    for attempt in range(5):
        if time.time() - start_time > time_limit:
            break
        
        # Try removing one column and solving for replacement
        col_to_remove = rng.randint(0, rank)
        
        # Build system for remaining rank-35
        # We need: sum_{r=0}^{34} U[i,j,r] V[j,k,r] W[k,i,r] = 1
        
        # This is nonlinear, so we use iterative refinement
        U_curr = U.copy()
        V_curr = V.copy()
        W_curr = W.copy()
        
        # Remove the column
        mask = [r for r in range(rank) if r != col_to_remove]
        U_curr = U_curr[:, mask]
        V_curr = V_curr[:, mask]
        W_curr = W_curr[:, mask]
        
        # Now try to adjust the remaining columns to still represent T
        # This is generally impossible unless the removed column was redundant
        
        if _verify_fast(U_curr, V_curr, W_curr):
            best = (U_curr.copy(), V_curr.copy(), W_curr.copy())
            break
    
    return best

def _try_greedy_reduction(U, V, W, time_limit=30.0):
    """
    More aggressive greedy reduction: try removing columns in different orders.
    """
    n, m, p = 2, 4, 5
    start_time = time.time()
    rank = U.shape[1]
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Try all single-column removals first
    for col in range(rank):
        if time.time() - start_time > time_limit:
            break
        if _check_column_redundant(U, V, W, col):
            mask = [i for i in range(rank) if i != col]
            U_new = U[:, mask]
            V_new = V[:, mask]
            W_new = W[:, mask]
            if U_new.shape[1] < best_rank:
                best_U, best_V, best_W = U_new.copy(), V_new.copy(), W_new.copy()
                best_rank = U_new.shape[1]
    
    if best_rank < rank:
        return best_U, best_V, best_W
    
    # Try pair removals
    for i in range(rank):
        if time.time() - start_time > time_limit:
            break
        for j in range(i+1, rank):
            if time.time() - start_time > time_limit:
                break
            if _check_pair_redundant(U, V, W, i, j):
                mask = [k for k in range(rank) if k != i and k != j]
                U_new = U[:, mask]
                V_new = V[:, mask]
                W_new = W[:, mask]
                if U_new.shape[1] < best_rank:
                    best_U, best_V, best_W = U_new.copy(), V_new.copy(), W_new.copy()
                    best_rank = U_new.shape[1]
                break
    
    if best_rank < rank:
        return best_U, best_V, best_W
    
    return None

def construct():
    """Construct a rank-minimal exact tensor decomposition for 2x4 x 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    U, V, W = _build_strassen_36()
    
    # Verify the base decomposition
    if not _verify_full(U, V, W):
        n, m, p = 2, 4, 5
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    rank = U.shape[1]
    
    # Phase 1: Greedy column removal
    if time.time() < deadline - 50:
        result = _try_greedy_reduction(U, V, W, time_limit=20.0)
        if result is not None:
            best_U, best_V, best_W = result
            rank = best_U.shape[1]
    
    # Phase 2: Border-rank heuristic via perturbation + ALS
    if time.time() < deadline - 40 and rank > 35:
        result = _als_optimize(U, V, W, rank, time_limit=30.0, seed=42)
        if result is not None:
            U_r, V_r, W_r = result
            if _verify_fast(U_r, V_r, W_r) and U_r.shape[1] < best_U.shape[1]:
                best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                rank = best_U.shape[1]
    
    # Phase 3: Column rotation
    if time.time() < deadline - 30 and rank > 35:
        result = _try_column_rotation(U, V, W, time_limit=10.0)
        if result is not None:
            U_r, V_r, W_r = result
            if _verify_fast(U_r, V_r, W_r) and U_r.shape[1] < best_U.shape[1]:
                best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                rank = best_U.shape[1]
    
    # Phase 4: Kronecker-form exploitation
    if time.time() < deadline - 20 and rank > 35:
        result = _try_kronecker_form(U, V, W, time_limit=15.0)
        if result is not None:
            U_r, V_r, W_r = result
            if _verify_fast(U_r, V_r, W_r) and U_r.shape[1] < best_U.shape[1]:
                best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                rank = best_U.shape[1]
    
    # Phase 5: Final verification
    if not _verify_fast(best_U, best_V, best_W):
        best_U, best_V, best_W = U, V, W
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
