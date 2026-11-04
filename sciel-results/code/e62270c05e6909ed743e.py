import time
import numpy as np
from fractions import Fraction
from itertools import combinations

N, M, P = 3, 4, 4
NM, MP, PN = N * M, M * P, P * N


def _verify(U, V, W, tol=1e-10):
    """Verify using einsum."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float).reshape(N, M, rank)
    V_arr = np.array(V, dtype=float).reshape(M, P, rank)
    W_arr = np.array(W, dtype=float).reshape(P, N, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U_arr, V_arr, W_arr)
    max_err = np.max(np.abs(T - 1.0))
    return max_err < tol, max_err


def _verify_exact(U, V, W):
    """Exact verification using Fractions."""
    rank = len(U[0])
    U_frac = [[Fraction(x) for x in row] for row in U]
    V_frac = [[Fraction(x) for x in row] for row in V]
    W_frac = [[Fraction(x) for x in row] for row in W]
    for i in range(N):
        for j in range(M):
            for k in range(P):
                val = Fraction(0)
                for r in range(rank):
                    val += U_frac[i * M + j][r] * V_frac[j * P + k][r] * W_frac[k * N + i][r]
                if val != 1:
                    return False
    return True


def _schoolbook():
    rank = NM * MP * PN
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    c = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][c] = 1
                V[j * P + k][c] = 1
                W[k * N + i][c] = 1
                c += 1
    return {'U': U, 'V': V, 'W': W}


def _strassen_44():
    """Rank 44: 16 (schoolbook for row 0) + 28 (Strassen for rows 1,2)."""
    rank = 44
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    
    col = 0
    # Part 1: Row 0 of A, schoolbook: 4*4 = 16 terms
    for j in range(M):
        for k in range(P):
            U[0 * M + j][col] = 1
            V[j * P + k][col] = 1
            W[k * N + 0][col] = 1
            col += 1
    
    # Part 2: Rows 1,2 of A, Strassen on 4 blocks: 4*7 = 28 terms
    strassen_terms = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]
    
    blocks = [
        (1, 0, 0, 0, 1, 0),
        (1, 0, 0, 2, 1, 2),
        (1, 2, 2, 0, 1, 0),
        (1, 2, 2, 2, 1, 2)
    ]
    
    for (ar, ac, br, bc, cr, cc) in blocks:
        for (x_c, y_c, z_c) in strassen_terms:
            for (a, b), val in x_c.items():
                U[(ar + a) * M + (ac + b)][col] += val
            for (b, c), val in y_c.items():
                V[(br + b) * P + (bc + c)][col] += val
            for (a, b), val in z_c.items():
                W[(cc + b) * N + (cr + a)][col] += val
            col += 1
    
    assert col == rank
    return {'U': U, 'V': V, 'W': W}


def _strassen_44_alt():
    """Alternative: rows 0,1 Strassen + row 2 schoolbook."""
    rank = 44
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    
    col = 0
    # Part 1: Rows 0,1 Strassen
    strassen_terms = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]
    
    blocks = [
        (0, 0, 0, 0, 0, 0),
        (0, 0, 0, 2, 0, 2),
        (0, 2, 2, 0, 0, 0),
        (0, 2, 2, 2, 0, 2)
    ]
    
    for (ar, ac, br, bc, cr, cc) in blocks:
        for (x_c, y_c, z_c) in strassen_terms:
            for (a, b), val in x_c.items():
                U[(ar + a) * M + (ac + b)][col] += val
            for (b, c), val in y_c.items():
                V[(br + b) * P + (bc + c)][col] += val
            for (a, b), val in z_c.items():
                W[(cc + b) * N + (cr + a)][col] += val
            col += 1
    
    # Part 2: Row 2 schoolbook
    for j in range(M):
        for k in range(P):
            U[2 * M + j][col] = 1
            V[j * P + k][col] = 1
            W[k * N + 2][col] = 1
            col += 1
    
    assert col == rank
    return {'U': U, 'V': V, 'W': W}


def _lax_reduce(U, V, W, deadline, max_iters=200):
    """
    Lax subspace reduction: iteratively check if any column's tensor
    U[:,c]⊗V[:,c]⊗W[:,c] is in the span of the others.
    Uses randomized linear combination search.
    """
    rank = len(U[0])
    U_f = np.array(U, dtype=float)
    V_f = np.array(V, dtype=float)
    W_f = np.array(W, dtype=float)
    
    # Build tensor matrix: each column c gives vector T_c = U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    # Shape: (NM*MP*PN, rank)
    # T[i,j,k,c] = U[i,c]*V[j,c]*W[k,c]
    
    def build_tensor_matrix(U_f, V_f, W_f):
        r = U_f.shape[1]
        T = np.zeros((NM * MP * PN, r))
        for c in range(r):
            T[:, c] = (U_f[:, c].reshape(N, M) * V_f[:, c].reshape(M, P) * W_f[:, c].reshape(P, N)).flatten()
        return T
    
    T = build_tensor_matrix(U_f, V_f, W_f)
    # Target: T should equal the multiplication tensor (all entries = 1 for valid (i,j,k))
    
    best_rank = rank
    best_U, best_V, best_W = U_f.copy(), V_f.copy(), W_f.copy()
    
    for iteration in range(max_iters):
        if time.time() > deadline:
            break
        
        r = best_U.shape[1]
        if r <= 1:
            break
        
        # For each column, check if it can be removed
        # by seeing if T[:,c] is in span of T[:,other_cols]
        # Use a quick rank check: rank(T) should equal r if all independent
        # If rank < r, some columns are dependent
        
        # Efficient check: for each column c, compute residual after projecting
        # onto the space spanned by all other columns
        # This is O(r * NM*MP*PN * r) which is expensive for large r
        # Instead, use the fact that we can check via SVD of the full matrix
        
        # Quick method: compute rank of T
        try:
            sv = np.linalg.svd(T, compute_uv=False)
            tol_svd = 1e-8 * sv[0] if sv[0] > 0 else 1e-8
            current_rank = np.sum(sv > tol_svd)
        except:
            break
        
        if current_rank < best_rank:
            # There's a dependency. Find which column to remove.
            # Try removing each column and check if rank stays the same
            for c_remove in range(best_rank - 1, -1, -1):
                if time.time() > deadline:
                    break
                
                T_reduced = np.delete(T, c_remove, axis=1)
                try:
                    sv_r = np.linalg.svd(T_reduced, compute_uv=False)
                    tol_r = 1e-8 * sv_r[0] if sv_r[0] > 0 else 1e-8
                    rank_r = np.sum(sv_r > tol_r)
                except:
                    continue
                
                if rank_r == current_rank:
                    # Column c_remove is redundant
                    best_U = np.delete(best_U, c_remove, axis=1)
                    best_V = np.delete(best_V, c_remove, axis=1)
                    best_W = np.delete(best_W, c_remove, axis=1)
                    T = T_reduced
                    best_rank = best_rank - 1
                    break
        else:
            # No simple dependency found. Try randomized linear combinations.
            # Pick a random column and try to express it as combination of others
            c_target = np.random.randint(0, best_rank)
            
            # Build matrix of all other columns
            other_cols = [c for c in range(best_rank) if c != c_target]
            T_others = T[:, other_cols]
            
            # Try to solve T_others @ alpha = T[:, c_target]
            # Least squares
            try:
                alpha, residuals, rank_ls, sv = np.linalg.lstsq(T_others, T[:, c_target], rcond=1e-8)
            except:
                continue
            
            residual = T[:, c_target] - T_others @ alpha
            if np.linalg.norm(residual) < 1e-7 * max(1.0, np.linalg.norm(T[:, c_target])):
                # Column c_target is in span of others
                # Remove it
                best_U = np.delete(best_U, c_target, axis=1)
                best_V = np.delete(best_V, c_target, axis=1)
                best_W = np.delete(best_W, c_target, axis=1)
                T = np.delete(T, c_target, axis=1)
                best_rank = best_rank - 1
    
    # Convert back to lists
    best_U_list = [[float(x) for x in row] for row in best_U]
    best_V_list = [[float(x) for x in row] for row in best_V]
    best_W_list = [[float(x) for x in row] for row in best_W]
    
    return {'U': best_U_list, 'V': best_V_list, 'W': best_W_list}, best_rank


def _try_merge_columns(U, V, W, deadline):
    """
    Try to find pairs of columns that can be merged.
    Two columns c1, c2 can be merged if there exist scalars a, b, c with a*b*c=1
    such that U[:,c2]=a*U[:,c1], V[:,c2]=b*V[:,c1], W[:,c2]=c*W[:,c1].
    """
    rank = len(U[0])
    U_f = np.array(U, dtype=float)
    V_f = np.array(V, dtype=float)
    W_f = np.array(W, dtype=float)
    
    best_rank = rank
    best_U, best_V, best_W = U_f.copy(), V_f.copy(), W_f.copy()
    
    for c1 in range(best_rank - 1):
        if time.time() > deadline:
            break
        for c2 in range(c1 + 1, best_rank):
            # Check proportionality
            u1, u2 = best_U[:, c1], best_U[:, c2]
            v1, v2 = best_V[:, c1], best_V[:, c2]
            w1, w2 = best_W[:, c1], best_W[:, c2]
            
            # Find scalar a: u2 = a * u1
            mask_u = u1 != 0
            if not np.any(mask_u):
                continue
            a_vals = u2[mask_u] / u1[mask_u]
            if not np.allclose(a_vals, a_vals[0], atol=1e-10):
                continue
            a = a_vals[0]
            if a == 0:
                continue
            if not np.allclose(u2, a * u1, atol=1e-10):
                continue
            
            mask_v = v1 != 0
            if not np.any(mask_v):
                continue
            b_vals = v2[mask_v] / v1[mask_v]
            if not np.allclose(b_vals, b_vals[0], atol=1e-10):
                continue
            b = b_vals[0]
            if b == 0:
                continue
            if not np.allclose(v2, b * v1, atol=1e-10):
                continue
            
            mask_w = w1 != 0
            if not np.any(mask_w):
                continue
            c_vals = w2[mask_w] / w1[mask_w]
            if not np.allclose(c_vals, c_vals[0], atol=1e-10):
                continue
            c = c_vals[0]
            if c == 0:
                continue
            if not np.allclose(w2, c * w1, atol=1e-10):
                continue
            
            if abs(a * b * c - 1) < 1e-10:
                # Merge: add c2 to c1
                best_U[:, c1] += best_U[:, c2]
                best_V[:, c1] += best_V[:, c2]
                best_W[:, c1] += best_W[:, c2]
                best_U = np.delete(best_U, c2, axis=1)
                best_V = np.delete(best_V, c2, axis=1)
                best_W = np.delete(best_W, c2, axis=1)
                best_rank -= 1
                c2 = c1  # adjust index after deletion
    
    best_U_list = [[float(x) for x in row] for row in best_U]
    best_V_list = [[float(x) for x in row] for row in best_V]
    best_W_list = [[float(x) for x in row] for row in best_W]
    
    return {'U': best_U_list, 'V': best_V_list, 'W': best_W_list}, best_rank


def _randomized_polish(U, V, W, deadline, n_trials=50):
    """
    Randomized search: try random linear combinations of columns to see
    if we can reduce the effective rank.
    """
    rank = len(U[0])
    U_f = np.array(U, dtype=float)
    V_f = np.array(V, dtype=float)
    W_f = np.array(W, dtype=float)
    
    best_rank = rank
    best_U, best_V, best_W = U_f.copy(), V_f.copy(), W_f.copy()
    
    def build_tensor(U_f, V_f, W_f):
        r = U_f.shape[1]
        T = np.zeros((NM * MP * PN, r))
        for c in range(r):
            T[:, c] = (U_f[:, c].reshape(N, M) * V_f[:, c].reshape(M, P) * W_f[:, c].reshape(P, N)).flatten()
        return T
    
    for trial in range(n_trials):
        if time.time() > deadline:
            break
        
        r = best_U.shape[1]
        if r <= 1:
            break
        
        # Pick a random column to try to eliminate
        c_target = np.random.randint(0, r)
        
        # Try random linear combinations of 2-3 other columns
        other_cols = [c for c in range(r) if c != c_target]
        
        for combo_size in [2, 3]:
            if len(other_cols) < combo_size:
                continue
            
            for combo in combinations(other_cols, combo_size):
                if time.time() > deadline:
                    break
                
                # Try to find coefficients alpha such that
                # sum(alpha_i * T[:, combo_i]) ≈ T[:, c_target]
                T_target = (best_U[:, c_target].reshape(N, M) * 
                          best_V[:, c_target].reshape(M, P) * 
                          best_W[:, c_target].reshape(P, N)).flatten()
                
                T_combo = np.column_stack([
                    (best_U[:, c].reshape(N, M) * 
                     best_V[:, c].reshape(M, P) * 
                     best_W[:, c].reshape(P, N)).flatten()
                    for c in combo
                ])
                
                try:
                    alpha, _, _, _ = np.linalg.lstsq(T_combo, T_target, rcond=1e-10)
                except:
                    continue
                
                residual = T_target - T_combo @ alpha
                if np.linalg.norm(residual) < 1e-8 * max(1.0, np.linalg.norm(T_target)):
                    # Column c_target is in span of combo
                    # Remove it
                    best_U = np.delete(best_U, c_target, axis=1)
                    best_V = np.delete(best_V, c_target, axis=1)
                    best_W = np.delete(best_W, c_target, axis=1)
                    best_rank -= 1
                    break
            
            if best_rank < rank:
                break
    
    best_U_list = [[float(x) for x in row] for row in best_U]
    best_V_list = [[float(x) for x in row] for row in best_V]
    best_W_list = [[float(x) for x in row] for row in best_W]
    
    return {'U': best_U_list, 'V': best_V_list, 'W': best_W_list}, best_rank


def construct():
    start = time.time()
    deadline = start + 130.0  # Safety margin before 150s limit
    
    # Generate candidate decompositions
    candidates = []
    
    # Primary: row 0 schoolbook + rows 1,2 Strassen
    c1 = _strassen_44()
    ok, err = _verify(c1['U'], c1['V'], c1['W'])
    if ok:
        candidates.append(('strassen_44', c1))
    
    # Alternative: rows 0,1 Strassen + row 2 schoolbook
    c2 = _strassen_44_alt()
    ok, err = _verify(c2['U'], c2['V'], c2['W'])
    if ok:
        candidates.append(('strassen_44_alt', c2))
    
    # Fallback: schoolbook
    c3 = _schoolbook()
    candidates.append(('schoolbook', c3))
    
    if not candidates:
        return c3
    
    # Sort by rank
    candidates.sort(key=lambda x: len(x[1]['U'][0]))
    
    best_result = candidates[0][1]
    best_rank = len(best_result['U'][0])
    
    # Try to reduce each candidate
    for name, cand in candidates[:3]:  # Try top 3 candidates
        if time.time() > deadline - 20:
            break
        
        cand_rank = len(cand['U'][0])
        if cand_rank >= best_rank:
            continue
        
        # Try merge
        if time.time() < deadline - 15:
            merged, merged_rank = _try_merge_columns(cand['U'], cand['V'], cand['W'], deadline - 5)
            if merged_rank < cand_rank:
                ok, err = _verify(merged['U'], merged['V'], merged['W'])
                if ok and merged_rank < best_rank:
                    best_result = merged
                    best_rank = merged_rank
        
        # Try Lax reduction
        if time.time() < deadline - 10:
            lax, lax_rank = _lax_reduce(cand['U'], cand['V'], cand['W'], deadline - 5, max_iters=100)
            if lax_rank < best_rank:
                ok, err = _verify(lax['U'], lax['V'], lax['W'])
                if ok:
                    best_result = lax
                    best_rank = lax_rank
        
        # Try randomized polish
        if time.time() < deadline - 5:
            polished, polish_rank = _randomized_polish(best_result['U'], best_result['V'], best_result['W'], deadline - 2, n_trials=30)
            if polish_rank < best_rank:
                ok, err = _verify(polished['U'], polished['V'], polished['W'])
                if ok:
                    best_result = polished
                    best_rank = polish_rank
    
    # Final verification
    ok, err = _verify(best_result['U'], best_result['V'], best_result['W'])
    if not ok:
        # Fallback to first valid candidate
        for name, cand in candidates:
            ok, err = _verify(cand['U'], cand['V'], cand['W'])
            if ok:
                best_result = cand
                break
    
    return best_result
