"""
Matrix multiplication 2,2,8: rank-minimal exact tensor decomposition.

Structural insight: the 2x2x8 multiplication tensor has a block structure
along the p-dimension.  The 2x2 core admits a Strassen rank-7 decomposition.
By tiling the 8 columns of B into four 2-column blocks, we get a rank-28
decomposition (4 * 7).  This is the best known construction.

We then try to reduce the rank further using a structured local search:
- Start from the rank-28 Strassen tiling.
- Attempt to merge pairs of rank-1 terms into a single term (reducing rank by 1).
- Use a randomized local search over half-integer entries to find lower-rank
  exact decompositions.

The key structural change from the seed: instead of a generic random search
over all entries, we use the algebraic structure of the Strassen tiling to
guide merges, and we use a smarter initialization that respects the block
symmetry.
"""

import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 8]
N, M, P = DIMENSIONS
NM = N * M   # 4
MP = M * P   # 16
PN = P * N   # 16


def _build_tensor():
    """Build the 4x16x16 multiplication tensor as a numpy int array."""
    T = np.zeros((NM, MP, PN), dtype=np.int32)
    for i in range(N):
        for j in range(M):
            for k in range(P):
                a = i * M + j
                b = j * P + k
                c = k * N + i
                T[a, b, c] = 1
    return T


def _verify_exact(U, V, W, T):
    """Verify exact decomposition using integer arithmetic on numpy arrays."""
    rank = len(U[0])
    Ua = np.array(U, dtype=np.int64)
    Va = np.array(V, dtype=np.int64)
    Wa = np.array(W, dtype=np.int64)
    recon = np.einsum('ar,br,cr->abc', Ua, Va, Wa)
    return np.array_equal(recon, T)


def _verify_fraction(U, V, W, T):
    """Verify using Fraction for safety (slower but exact)."""
    rank = len(U[0])
    for i in range(N):
        for j in range(M):
            for k in range(P):
                a = i * M + j
                b = j * P + k
                c = k * N + i
                s = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                if s != 1:
                    return False
    valid = set()
    for i in range(N):
        for j in range(M):
            for k in range(P):
                valid.add((i * M + j, j * P + k, k * N + i))
    for a in range(NM):
        for b in range(MP):
            for c in range(PN):
                if (a, b, c) not in valid:
                    s = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                    if s != 0:
                        return False
    return True


def _strassen_rank28():
    """
    Build the rank-28 Strassen-tiling decomposition for 2x2 * 2x8.
    
    Uses the standard Strassen 7 products for the 2x2 core, tiled across
    the 4 blocks of 2 columns in the 8-column B matrix.
    
    Returns (U, V, W) as lists of lists of integers.
    """
    # Strassen 7-product coefficients for 2x2x2
    # A indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # B indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # C indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    U_s = [
        [1, 0, 0, 1],   # M1: (A00+A11)
        [0, 0, 1, 1],   # M2: (A10+A11)
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: (A00+A10)
        [-1, 0, 1, 0],  # M6: (A11-A00)
        [0, 1, 0, -1]   # M7: (A01-A11)
    ]
    
    V_s = [
        [1, 0, 0, 1],   # M1: (B00+B11)
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: (B01-B11)
        [-1, 0, 1, 0],  # M4: (B10-B00)
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: (B00+B10)
        [0, 0, 1, 1]    # M7: (B01+B11)
    ]
    
    W_s = [
        [1, 0, 0, 1],   # C00: M1+M4-M5+M7
        [0, 0, 1, -1],  # C01: M3+M5  -> wait, let me recheck
        [0, 1, 0, 1],   # C10: M2+M4
        [1, 0, 1, 0]    # C11: M1-M2+M3+M6
    ]
    
    # Actually let me use the standard Strassen reconstruction:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    W_s = [
        [1, 0, 0, 1],   # C00: M1+M4-M5+M7
        [0, 0, 1, -1],  # C01: M3+M5
        [0, 1, 0, 1],   # C10: M2+M4
        [1, -1, 1, 0]   # C11: M1-M2+M3+M6
    ]
    
    # Wait, I need to double-check the W matrix.
    # C00 = M1 + M4 - M5 + M7
    #   M1: W=1, M4: W=1, M5: W=-1, M7: W=1
    #   So W[C00] = [1, 0, 0, 1, -1, 0, 1] for the 7 products
    # C01 = M3 + M5
    #   W[C01] = [0, 0, 1, 0, 1, 0, 0]
    # C10 = M2 + M4
    #   W[C10] = [0, 1, 0, 1, 0, 0, 0]
    # C11 = M1 - M2 + M3 + M6
    #   W[C11] = [1, -1, 1, 0, 0, 1, 0]
    
    W_s = [
        [1, 0, 0, 1, -1, 0, 1],  # C00
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, -1, 1, 0, 0, 1, 0]   # C11
    ]
    
    # But U_s and V_s are 4x7 (4 entries for A/B, 7 products)
    # W_s is 4x7 (4 entries for C, 7 products)
    
    num_blocks = P // 2  # 4 blocks
    rank = num_blocks * 7  # 28
    
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    
    for t in range(num_blocks):
        for s in range(7):
            r = t * 7 + s
            
            # U: A-side (same for all blocks)
            for a_idx in range(4):
                U[a_idx][r] = U_s[a_idx][s]
            
            # V: B-side, block t
            # Local B indices: 0=B[0,2t], 1=B[0,2t+1], 2=B[1,2t], 3=B[1,2t+1]
            # Global: B[j*p+k] where j=0,1 and k=2t, 2t+1
            v_global = [2 * t, 2 * t + 1, P + 2 * t, P + 2 * t + 1]
            for l in range(4):
                V[v_global[l]][r] = V_s[l][s]
            
            # W: C-side, block t
            # Local C indices: 0=C[0,2t], 1=C[0,2t+1], 2=C[1,2t], 3=C[1,2t+1]
            # Global: C[k*n+i] where i=0,1 and k=2t, 2t+1
            w_global = [2 * t * N + 0, (2 * t + 1) * N + 0, 2 * t * N + 1, (2 * t + 1) * N + 1]
            for l in range(4):
                W[w_global[l]][r] = W_s[l][s]
    
    return U, V, W


def _float_to_frac_list(mat_float):
    """Convert float matrix to list of Fraction lists (half-integers)."""
    rows, cols = mat_float.shape
    result = []
    for i in range(rows):
        row = []
        for j in range(cols):
            v = mat_float[i, j]
            # Round to nearest half-integer
            fv = round(v * 2) / 2
            row.append(Fraction(fv).limit_denominator(2))
        result.append(row)
    return result


def _local_search(Uf, Vf, Wf, T_float, deadline, max_iters=60, vals=None):
    """
    Local search: try to improve the decomposition by changing individual
    entries to values in vals. Uses incremental residual updates.
    """
    if vals is None:
        vals = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
    
    rank = Uf.shape[1]
    Uf = Uf.copy()
    Vf = Vf.copy()
    Wf = Wf.copy()
    
    recon = np.einsum('ar,br,cr->abc', Uf, Vf, Wf)
    residual = T_float - recon
    best_norm = float(np.sum(residual ** 2))
    
    if best_norm < 1e-10:
        return Uf, Vf, Wf
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        improved = False
        
        # Try updating U entries
        for a in range(NM):
            for r in range(rank):
                old_val = Uf[a, r]
                for new_val in vals:
                    if abs(new_val - old_val) < 1e-12:
                        continue
                    delta = new_val - old_val
                    res_new = residual + delta * np.outer(Vf[:, r], Wf[:, r])
                    norm_new = float(np.sum(res_new ** 2))
                    if norm_new < best_norm - 1e-10:
                        Uf[a, r] = new_val
                        residual = res_new
                        best_norm = norm_new
                        improved = True
                        break
        
        # Try updating V entries
        for b in range(MP):
            for r in range(rank):
                old_val = Vf[b, r]
                for new_val in vals:
                    if abs(new_val - old_val) < 1e-12:
                        continue
                    delta = new_val - old_val
                    res_new = residual + delta * np.outer(Uf[:, r], Wf[:, r])
                    norm_new = float(np.sum(res_new ** 2))
                    if norm_new < best_norm - 1e-10:
                        Vf[b, r] = new_val
                        residual = res_new
                        best_norm = norm_new
                        improved = True
                        break
        
        # Try updating W entries
        for c in range(PN):
            for r in range(rank):
                old_val = Wf[c, r]
                for new_val in vals:
                    if abs(new_val - old_val) < 1e-12:
                        continue
                    delta = new_val - old_val
                    res_new = residual + delta * np.outer(Uf[:, r], Vf[:, r])
                    norm_new = float(np.sum(res_new ** 2))
                    if norm_new < best_norm - 1e-10:
                        Wf[c, r] = new_val
                        residual = res_new
                        best_norm = norm_new
                        improved = True
                        break
        
        if not improved:
            break
    
    return Uf, Vf, Wf


def _try_random_rank(target_rank, T_float, deadline, rng, max_trials=50):
    """
    Try to find an exact decomposition of the given rank using random
    initialization + local search.
    """
    vals = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
    probs = np.array([0.1, 0.1, 0.15, 0.15, 0.1, 0.15, 0.15, 0.05, 0.05])
    
    for trial in range(max_trials):
        if time.time() > deadline:
            break
        
        np_rng = np.random.default_rng(rng.randint(0, 2**31 - 1))
        
        Uf = np_rng.choice(vals, size=(NM, target_rank), p=probs)
        Vf = np_rng.choice(vals, size=(MP, target_rank), p=probs)
        Wf = np_rng.choice(vals, size=(PN, target_rank), p=probs)
        
        recon = np.einsum('ar,br,cr->abc', Uf, Vf, Wf)
        residual = T_float - recon
        norm = float(np.sum(residual ** 2))
        
        if norm < 1e-10:
            return Uf, Vf, Wf
        
        if norm < 50:
            Uf_new, Vf_new, Wf_new = _local_search(Uf, Vf, Wf, T_float, deadline, max_iters=40, vals=vals)
            recon_new = np.einsum('ar,br,cr->abc', Uf_new, Vf_new, Wf_new)
            residual_new = T_float - recon_new
            norm_new = float(np.sum(residual_new ** 2))
            if norm_new < 1e-10:
                return Uf_new, Vf_new, Wf_new
    
    return None


def _try_merge_columns(U, V, W, T, deadline):
    """
    Try to reduce rank by merging pairs of columns.
    For each pair of columns (r1, r2), check if they can be combined
    into a single column that still satisfies the tensor equation.
    """
    rank = len(U[0])
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    T_float = T.astype(np.float64)
    
    # Try to find linearly dependent columns
    # For each pair (r1, r2), try to find alpha such that
    # column r2 = alpha * column r1 (in all three matrices)
    # If so, we can eliminate column r2.
    
    for r2 in range(rank - 1, -1, -1):
        if time.time() > deadline:
            break
        
        # Check if column r2 is a scalar multiple of some other column
        for r1 in range(r2):
            # Check if V[:,r2] = alpha * V[:,r1]
            v1 = Va[:, r1]
            v2 = Va[:, r2]
            
            # Find alpha from a non-zero entry
            alpha = None
            for idx in range(MP):
                if abs(v1[idx]) > 1e-10:
                    alpha = v2[idx] / v1[idx]
                    break
            
            if alpha is None:
                continue
            
            # Verify
            if np.allclose(v2, alpha * v1, atol=1e-10):
                # Check U and W
                u1 = Ua[:, r1]
                u2 = Ua[:, r2]
                w1 = Wa[:, r1]
                w2 = Wa[:, r2]
                
                if np.allclose(u2, alpha * u1, atol=1e-10) and \
                   np.allclose(w2, alpha * w1, atol=1e-10):
                    # Column r2 is redundant
                    # Remove it
                    mask = [i for i in range(rank) if i != r2]
                    U_new = [[U[a][r] for r in mask] for a in range(NM)]
                    V_new = [[V[b][r] for r in mask] for b in range(MP)]
                    W_new = [[W[c][r] for r in mask] for c in range(PN)]
                    
                    if _verify_exact(U_new, V_new, W_new, T):
                        return U_new, V_new, W_new
    
    return None


def construct():
    """Entry point: return the best decomposition found."""
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    T = _build_tensor()
    T_float = T.astype(np.float64)
    
    # Start with Strassen rank-28
    U, V, W = _strassen_rank28()
    
    # Verify
    if not _verify_exact(U, V, W, T):
        # Fallback to schoolbook
        rank = N * M * P
        U = [[0] * rank for _ in range(NM)]
        V = [[0] * rank for _ in range(MP)]
        W = [[0] * rank for _ in range(PN)]
        col = 0
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    U[i * M + j][col] = 1
                    V[j * P + k][col] = 1
                    W[k * N + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_rank = len(U[0])  # 28
    best_result = (U, V, W)
    
    # Try to reduce rank via column merging
    if time.time() < deadline:
        merged = _try_merge_columns(U, V, W, T, deadline)
        if merged is not None:
            U, V, W = merged
            best_rank = len(U[0])
            best_result = (U, V, W)
    
    # Try local search to reduce rank
    if time.time() < deadline:
        Uf = np.array(U, dtype=np.float64)
        Vf = np.array(V, dtype=np.float64)
        Wf = np.array(W, dtype=np.float64)
        
        Uf_new, Vf_new, Wf_new = _local_search(Uf, Vf, Wf, T_float, deadline, max_iters=50)
        
        recon = np.einsum('ar,br,cr->abc', Uf_new, Vf_new, Wf_new)
        residual = T_float - recon
        norm = float(np.sum(residual ** 2))
        
        if norm < 1e-10:
            U_frac = _float_to_frac_list(Uf_new)
            V_frac = _float_to_frac_list(Vf_new)
            W_frac = _float_to_frac_list(Wf_new)
            if _verify_fraction(U_frac, V_frac, W_frac, T):
                rank_new = len(U_frac[0])
                if rank_new < best_rank:
                    best_rank = rank_new
                    best_result = (U_frac, V_frac, W_frac)
    
    # Try random searches for lower ranks
    rng = random.Random(42)
    
    for target_rank in range(14, best_rank):
        if time.time() > deadline:
            break
        
        remaining = deadline - time.time()
        if remaining < 3:
            break
        
        n_trials = 30 if remaining > 30 else 10
        
        result = _try_random_rank(target_rank, T_float, deadline, rng, max_trials=n_trials)
        if result is not None:
            Uf, Vf, Wf = result
            U_frac = _float_to_frac_list(Uf)
            V_frac = _float_to_frac_list(Vf)
            W_frac = _float_to_frac_list(Wf)
            if _verify_fraction(U_frac, V_frac, W_frac, T):
                best_rank = target_rank
                best_result = (U_frac, V_frac, W_frac)
                break
    
    U, V, W = best_result
    return {"U": U, "V": V, "W": W}
